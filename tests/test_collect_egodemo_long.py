"""Offline collector invariants: no network or credentials required."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/dataset/collect_egodemo_long.py"
SPEC = importlib.util.spec_from_file_location("collect_egodemo_long", MODULE_PATH)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CollectEgoDemoTests(unittest.TestCase):
    def test_raw_or_wrong_camera_is_not_a_paired_video(self):
        self.assertIsNone(collector.identify_role("EgoProRaw/mp4/task/uuid/head_left/video.mp4"))
        self.assertIsNone(collector.identify_role("videos/observation.images.head_right/chunk-000/file-000.mp4"))
        self.assertEqual(collector.identify_role("videos/observation.images.head_left/chunk-000/file-000.mp4"), "head_left_video")

    def test_diverse_selection_and_budget(self):
        records = [{"acquisition_id": str(index), "task_name": task, "scene_id": task,
                    "manifest_duration_seconds": duration, "planned_bytes": 100}
                   for index, (task, duration) in enumerate([("stapler", 25), ("stapler", 28), ("tape", 30), ("short", 12)])]
        selected = collector.choose_candidates(records, 3, 200)
        self.assertEqual(len(selected), 2)
        self.assertEqual({record["task_name"] for record in selected}, {"stapler", "tape"})

    def test_source_pts_absolute_alignment_not_only_normalized(self):
        self.assertTrue(collector.compare_timestamps([0, 0.033333335], [0, 0.033333], 30)["pass"])
        self.assertFalse(collector.compare_timestamps([10, 10.0333], [0, 0.0333], 30)["pass"])
        self.assertFalse(collector.compare_timestamps([0, 0], [0, 0.0333], 30)["pass"])
        self.assertFalse(collector.compare_timestamps([0], [0, 0.0333], 30)["pass"])
        self.assertFalse(collector.compare_timestamps([float("nan")], [0], 30)["pass"])

    def test_frozen_catalog_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            collector.save_json(path, {"frozen": True}, exclusive=True)
            with self.assertRaises(FileExistsError):
                collector.save_json(path, {"frozen": False}, exclusive=True)
            self.assertTrue(json.loads(path.read_text())["frozen"])

    def test_lfs_pointer_is_catalog_metadata_not_payload_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pointer = root / "file.mp4"
            pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "ab" * 32 + "\nsize 99999\n")
            asset = collector.asset_entry(pointer, root, "head_left_video")
            self.assertEqual(asset["expected_size"], 99999)
            self.assertEqual(asset["expected_sha256"], "ab" * 32)
            self.assertEqual(asset["source_storage"], "git_lfs")

    def test_export_real_source_timestamps_for_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = {"acquisition_id": "fixture", "assets": [{"role": "data", "sha256": "ab" * 32}],
                      "timestamp_alignment": {"pass": True}}
            collector.write_timestamp_asset(record, Path(temporary), [0.0, 0.04, 0.08])
            asset = record["assets"][-1]
            self.assertEqual(asset["role"], "timestamps")
            self.assertEqual(record["timestamp_source"], "source_timestamps")
            self.assertEqual(json.loads(Path(asset["path"]).read_text())["timestamps"], [0.0, 0.04, 0.08])
            self.assertEqual(asset["sha256"], collector.sha256(asset["path"]))


if __name__ == "__main__":
    unittest.main()
