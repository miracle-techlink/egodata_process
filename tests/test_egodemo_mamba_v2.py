"""Contract tests for genuine missingness, anchoring, and future isolation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.mamba.egodemo_v2 import (
    EgoDemoV2Dataset, EgoDemoV2Encoder, collate_egodemo_v2,
    rotation_6d_to_matrix, decode_events,
)


def put_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def make_fixture(root):
    root = Path(root)
    T, N = 8, 3
    camera = np.tile(np.eye(4), (T, 1, 1))
    camera[:, 0, 3] = 10
    hands = np.zeros((T, 2, 21, 3), np.float32)
    hands[..., 0] = 10 + np.arange(T)[:, None, None]
    hand_valid = np.ones((T, 2), bool)
    hand_valid[1, 1] = False
    hands[1, 1] = np.nan
    pose = np.full((T, N, 4, 4), np.nan, np.float32)
    valid = np.zeros((T, N), bool)
    pose[[1, 2, 5], 0] = np.eye(4)
    pose[[1, 2, 5], 0, 0, 3] = 12
    valid[[1, 2, 5], 0] = True
    pose[5, 2] = np.eye(4)
    valid[5, 2] = True
    observed = valid.copy()
    observed[[0, 2, 5], 1] = True
    frames = np.array([0, 2, 5, 5, 5], np.int32)
    points = np.array([0, 0, 0, 1, 2], np.int32)
    objects = np.array([1, 1, 1, 1, 2], np.int32)
    positions = np.array([[11, 0, 1], [12, 0, 1], [13, 0, 1], [77, 0, 1], [88, 0, 1]], np.float32)
    arrays = dict(timestamps=np.arange(T, dtype=np.float64) / 3,
                  frame_index=np.arange(T, dtype=np.int32), camera_to_world=camera,
                  hands_world=hands, hand_valid=hand_valid,
                  object_ids=np.array(["rigid", "cloth", "future"]),
                  object_kind=np.array([1, 2, 1], np.int8), object_pose_world=pose,
                  object_pose_valid=valid, object_observed=observed,
                  surface_frame_index=frames, surface_object_index=objects,
                  surface_point_index=points, surface_world=positions,
                  surface_frame_ptr=np.searchsorted(frames, np.arange(T + 1)).astype(np.int64),
                  point_keys=np.array(["cloth/old", "cloth/future", "future/first"]),
                  point_object_index=np.array([1, 1, 2], np.int32))
    np.savez(root / "sequence.npz", **arrays)
    put_json(root / "metadata.json", {"training_allowed": False, "allowed_as_model_input": False})
    put_json(root / "events.json", {"records": [
        {"layer": "operation", "type": "fold", "start_frame": 3, "end_frame": 7,
         "object_index": -1, "subject": -1, "weak_positive": True, "label_valid": True},
        {"layer": "operation", "type": "fold", "start_frame": 3, "end_frame": 7,
         "object_index": 2, "subject": 0, "weak_positive": True, "label_valid": False},
        {"layer": "transition", "type": "release", "start_frame": 3, "end_frame": 4,
         "object_index": 0, "subject": 1, "weak_positive": True, "label_valid": False},
        {"layer": "transition", "type": "release", "start_frame": 2, "end_frame": 3,
         "object_index": 0, "subject": 0, "weak_positive": True, "label_valid": False},
    ]})
    put_json(root / "vocab.json", {
        "operation": {"codes": ["fold"], "names_zh": ["折叠"]},
        "relation": {"codes": [], "names_zh": []},
        "transition": {"codes": ["release"], "names_zh": ["释放"]},
        "cooperation": {"codes": [], "names_zh": []},
    })
    row = dict(sequence_id="actual_fixture", start_frame=0, end_frame=3, anchor_frame=2,
               target_frame=5, target_timestamp=5 / 3, actual_horizon_s=1,
               sequence_path="sequence.npz", metadata_path="metadata.json", events_path="events.json")
    (root / "windows.jsonl").write_text(json.dumps(row) + "\n")
    put_json(root / "dataset_index.json", {
        "schema_version": "egodemo_mamba_v2", "training_allowed": False,
        "allowed_as_model_input": False, "vocab_path": "vocab.json",
        "splits": {"train": {"window_path": "windows.jsonl"}},
    })
    return arrays


class V2ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.arrays = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return EgoDemoV2Dataset(self.root, diagnostic=True, n_max=3,
                                points_per_object=2, **kwargs)

    def test_formal_loading_refuses_pending_noncausal_data(self):
        with self.assertRaisesRegex(PermissionError, "diagnostic"):
            EgoDemoV2Dataset(self.root)

    def test_observed_slots_point_ids_and_future_targets(self):
        sample = self.dataset()[0]
        self.assertEqual(sample["metadata"]["object_ids"], ["cloth", "rigid", ""])
        self.assertEqual(sample["metadata"]["point_keys"][0], ["cloth/old", ""])
        self.assertEqual(sample["object_kind"].tolist(), [2, 1, 0])
        self.assertEqual(sample["object_mask"].tolist(), [True, True, False])
        self.assertEqual(sample["target_surface_valid"][0].tolist(), [True, False])
        np.testing.assert_allclose(sample["target_surface"][0, 0], [3, 0, 1])
        self.assertFalse(sample["obj_valid"][:, 0].any())  # cloth never gets fabricated 6D
        self.assertTrue(torch.isnan(sample["obj_pose_seq"][:, 0]).all())
        self.assertEqual(sample["target_obj_valid"].tolist(), [False, True, False])

    def test_future_mutation_does_not_change_input(self):
        first = self.dataset()[0]
        self.arrays["surface_world"][2:] = -999
        self.arrays["hands_world"][3:] = 333
        self.arrays["object_pose_world"][3:] = 444
        np.savez(self.root / "sequence.npz", **self.arrays)
        second = self.dataset()[0]
        for key in ("hand_seq", "obj_pose_seq", "surface_seq", "object_mask", "point_mask"):
            torch.testing.assert_close(first[key], second[key], equal_nan=True)
        self.assertEqual(first["metadata"]["object_ids"], second["metadata"]["object_ids"])
        self.assertEqual(first["metadata"]["point_keys"], second["metadata"]["point_keys"])

    def test_one_shared_camera_anchor_and_missingness(self):
        sample = self.dataset()[0]
        np.testing.assert_allclose(sample["hand_seq"][2, 0, 0], [2, 0, 0])
        np.testing.assert_allclose(sample["target_hand"][0, 0], [5, 0, 0])
        np.testing.assert_allclose(sample["obj_pose_seq"][1, 1, :3, 3], [2, 0, 0])
        self.assertTrue(torch.isnan(sample["hand_seq"][1, 1]).all())
        self.assertFalse(sample["hand_valid"][1, 1])

    def test_event_unknowns_weak_masks_and_transition_horizon(self):
        sample = self.dataset()[0]
        self.assertEqual(sample["event_candidates"].shape, (4, 5, 2))
        self.assertEqual(sample["event_candidates"][3, 4, 0], 1)
        self.assertEqual(sample["event_candidates"][1, 1, 1], 1)
        self.assertEqual(sample["event_candidates"][1, 0, 1], 0)  # at anchor, not future
        self.assertEqual(int(sample["event_candidates"].sum()), 2)
        self.assertEqual(int(sample["event_label_valid"].sum()), 0)
        self.assertEqual(sample["metadata"]["dropped_target_events_unselected_objects"], 1)

    def test_encoder_finite_on_missing_data_and_ignores_targets(self):
        batch = collate_egodemo_v2([self.dataset()[0]])
        encoder = EgoDemoV2Encoder(d_model=16, n_heads=4, dropout=0).eval()
        with torch.no_grad():
            result = encoder(batch)
            batch["target_hand"].fill_(999)
            batch["event_candidates"].fill_(888)
            torch.testing.assert_close(result, encoder(batch))
            self.assertEqual(result.shape, (1, 3, 6, 16))
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue((result[:, :, -1] == 0).all())
            for key in ("hand_seq", "obj_pose_seq", "surface_seq"):
                batch[key].fill_(float("nan"))
            self.assertTrue(torch.isfinite(encoder(batch)).all())

    def test_zero_rotation_decoder_is_proper_so3(self):
        matrix = rotation_6d_to_matrix(torch.zeros(2, 6))
        torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, torch.eye(3).expand(2, 3, 3))
        torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(2))

    def test_decode_padding_and_unknown_chinese(self):
        dataset = self.dataset()
        sample = dataset[0]
        logits = torch.full((1, 4, 5, 2), -100.)
        logits[0, 2, 0, 0] = 100  # padded slot must be dropped
        logits[0, 3, 4, 0] = 100
        decoded = decode_events(logits, dataset.event_types, [sample["metadata"]])
        self.assertEqual(len(decoded[0]), 1)
        self.assertIsNone(decoded[0][0]["object_id"])
        self.assertEqual(decoded[0][0]["subject_name_zh"], "未知主体")
        self.assertEqual(decoded[0][0]["name_zh"], "折叠")


if __name__ == "__main__":
    unittest.main()
