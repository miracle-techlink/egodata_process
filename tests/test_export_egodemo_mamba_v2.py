import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset.complete_egodemo import digest, save
from src.dataset.export_egodemo_mamba_v2 import export_dataset, export_sequence, native_windows


def make_source(tmp_path, metric_allowed=True):
    root = tmp_path / "source"
    uid = "sample"
    t = np.arange(80, dtype=np.float64) / 10
    c2w = np.broadcast_to(np.eye(4), (80, 4, 4)).copy()
    c2w[:, 0, 3] = 5
    hand = np.ones((80, 2, 21, 3), np.float32)
    valid = np.ones((80, 2), bool)
    valid[4, 1] = False
    (root / "sequences" / uid).mkdir(parents=True)
    np.savez(root / "sequences" / uid / "hands.npz", timestamps=t, frame_index=np.arange(80),
             hands_world=hand, hand_valid=valid, camera_to_world=c2w)
    save(root / "sequences" / uid / "metadata.json", {
        "source_selection_sha256": "sourcehash", "task_name": "测试任务", "frame_count": 80})
    entity = {"id": "obj_001", "name_zh": "杯子", "rigidity": "rigid", "tracklets": ["tracklet_0001"],
              "metric_geometry_allowed": True, "label_valid": False}
    proposal = {"source_selection_sha256": "sourcehash", "entities": [entity],
                "interactions": [{"layer": "relation", "type": "contact", "subject": "left_hand",
                                  "object_id": "obj_001", "start_frame": 0, "end_frame": 30,
                                  "label_valid": True, "label_source": "geometry_proposal"}],
                "source_semantic_intervals": [{"source_interval_id": "source_0000", "start_frame": 0,
                    "end_frame": 80, "description_zh": "提起杯子", "description_original": "Lift the cup",
                    "operations": [{"type": "lift", "name_zh": "提起", "label_valid": True}]}]}
    save(root / "joint_annotations" / uid / "proposal.json", proposal)
    save(root / "joint_annotations" / uid / "receipt.json", {"metric_geometry_allowed": metric_allowed})
    pose = np.broadcast_to(np.eye(4), (1, 80, 4, 4)).copy()
    pose[:, :, 1, 3] = 2
    np.savez(root / "joint_annotations" / uid / "objects.npz", object_ids=np.array(["obj_001"]),
             frame_index=np.arange(80), poses=pose, pose_valid=np.ones((1, 80), bool))
    trackdir = root / "native_tracks" / uid
    trackdir.mkdir(parents=True)
    points = np.array([[[5, 1, 1], [5, 2, 1]], [[6, 1, 1], [np.nan, 0, 0]]], dtype=float)
    np.savez(trackdir / "tracklet_0001.npz", frame_index=np.array([1, 20]), timestamps=t[[1, 20]],
             points_world=points, point_valid=np.ones((2, 2), bool), point_ids=np.array([7, 25]))
    # This raw background track was rejected by the proposal and must never leak.
    np.savez(trackdir / "tracklet_9999.npz", points_world=np.array([999]))
    save(trackdir / "tracklets.json", {"records": [
        {"tracklet_id": "tracklet_0001", "sha256": digest(trackdir / "tracklet_0001.npz")},
        {"tracklet_id": "tracklet_9999", "sha256": digest(trackdir / "tracklet_9999.npz")}],
        "training_allowed": False})
    return root, uid


def test_export_preserves_coordinate_frame_point_identity_and_unknown_labels(tmp_path):
    root, uid = make_source(tmp_path)
    out = tmp_path / "out"
    receipt = export_sequence(root, out, uid)
    with np.load(out / "sequences" / uid / "sequence.npz", allow_pickle=False) as z:
        np.testing.assert_allclose(z["object_pose_world"][:, 0, :3, 3], np.tile([5, 2, 0], (80, 1)))
        assert np.isnan(z["hands_world"][4, 1]).all()
        assert z["point_keys"].tolist() == ["tracklet_0001/7", "tracklet_0001/25"]
        assert z["surface_frame_index"].tolist() == [1, 1, 20]
        assert z["surface_point_index"].tolist() == [0, 1, 0]
        assert z["surface_frame_ptr"][-1] == 3
        assert np.isfinite(z["surface_world"]).all()
    events = json.loads((out / "sequences" / uid / "events.json").read_text())
    assert events["records"][0]["subject"] == -1
    assert events["records"][0]["object_index"] == -1
    assert all(not row["label_valid"] and row["weak_positive"] for row in events["records"])
    assert receipt["surface_observations"] == 3
    assert not receipt["training_allowed"]


def test_inconsistent_calibration_blocks_pose_and_surface_but_preserves_hands(tmp_path):
    root, uid = make_source(tmp_path, metric_allowed=False)
    out = tmp_path / "out"
    export_sequence(root, out, uid)
    with np.load(out / "sequences" / uid / "sequence.npz") as z:
        assert not z["object_pose_valid"].any()
        assert np.isnan(z["object_pose_world"]).all()
        assert not z["object_observed"].any()
        assert len(z["surface_world"]) == 0
        assert z["surface_frame_ptr"].shape == (81,)
        assert z["hand_valid"].sum() == 159


def test_native_horizon_rejects_gap_without_interpolating():
    times = np.array([0, .1, .2, .3, .4, .8, .9, 1.0])
    rows = list(native_windows(times, 2, 1, .2))
    assert all(row["target_timestamp"] == times[row["target_frame"]] for row in rows)
    assert all(row["actual_horizon_s"] >= .2 - 1e-9 for row in rows)
    assert not any(row["anchor_frame"] == 3 for row in rows)
    assert not any(row["anchor_frame"] == 4 for row in rows)


def test_dataset_manifest_hashes_and_frozen_assignments(tmp_path):
    source, uid = make_source(tmp_path)
    manifest = tmp_path / "frozen.json"
    manifest.write_text('{"assignments": {"sample": "test"}, "source_selection_sha256": "sourcehash"}')
    output = tmp_path / "out"
    report = export_dataset(source, output, manifest, workers=1)
    assert digest(manifest) == digest(output / "splits" / "split_manifest.json")
    index = json.loads((output / "dataset_index.json").read_text())
    assert report["windows_by_split"]["test"] > 0
    assert index["splits"]["train"]["window_count"] == 0
    row = json.loads((output / "windows" / "test.jsonl").read_text().splitlines()[0])
    assert row["end_frame"] == 32 and row["anchor_frame"] == 31
    assert row["sequence_sha256"] == digest(output / row["sequence_path"])
    assert row["split_manifest_sha256"] == digest(manifest)
    assert row["target_frame"] > row["anchor_frame"]
    assert not index["training_allowed"] and not index["allowed_as_model_input"]
    assert report["gold_label_count"] == 0


def test_changed_source_cannot_reuse_completed_shard(tmp_path):
    source, uid = make_source(tmp_path)
    output = tmp_path / "out"
    export_sequence(source, output, uid)
    # A stable rerun is safe; changed source requires a separate export.
    export_sequence(source, output, uid)
    path = source / "joint_annotations" / uid / "proposal.json"
    data = json.loads(path.read_text())
    data["entities"][0]["name_zh"] = "另一个杯子"
    save(path, data)
    with pytest.raises(ValueError, match="source changed"):
        export_sequence(source, output, uid)
