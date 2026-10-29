"""Pure-stdlib synthetic tests: python -S tests/test_egodemo_manifest.py."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from uuid import UUID


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/dataset/egodemo_manifest.py"
SPEC = importlib.util.spec_from_file_location("egodemo_manifest", MODULE_PATH)
manifest_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_tool)

CSV_FIELDS = (
    "source_episode_uuid", "variant", "task_name", "lerobot_path", "fps", "frame_count",
    "duration_s", "has_hand_pose", "has_semantic_annotations",
)
HASH = "ab" * 32
POINTER = f"version https://git-lfs.github.com/spec/v1\noid sha256:{HASH}\nsize 123456\n".encode()
MP4 = (24).to_bytes(4, "big") + b"ftypisom" + bytes(4) + b"isommp42"
PARQUET = b"PAR1fixture" + (7).to_bytes(4, "little") + b"PAR1"


class EgoDemoManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "EgoDemo"
        self.raw = self.root / "egodemo_subset"
        self.dataset.mkdir()
        self.raw.mkdir()
        self.rows = {variant: [] for variant in manifest_tool.VARIANTS}
        self.write_manifests()

    def write_file(self, relative, content, root=None):
        path = (self.dataset if root is None else root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_manifests(self):
        for variant, rows in self.rows.items():
            path = self.dataset / variant / "manifests/train.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

    def episode(self, number, task="Refill Stapler", variant="EgoProStandard", raw=True, pointers=True):
        episode_uuid = str(UUID(int=number))
        base = f"{variant}/lerobot/{task}/{episode_uuid}"
        row = dict(zip(CSV_FIELDS, (
            episode_uuid, variant, task, base + "/", "29.964285714285715", "840", "28.0", "true", "true",
        )))
        self.rows[variant].append(row)
        payload = POINTER if pointers else PARQUET
        for suffix in (
            "data/chunk-000/file-000.parquet", "meta/episodes/chunk-000/file-000.parquet", "meta/tasks.parquet",
        ):
            self.write_file(f"{base}/{suffix}", payload)
        self.write_file(f"{base}/meta/info.json", json.dumps({"fps": 30, "total_frames": 840}).encode())
        self.write_file(f"{base}/annotation.json", json.dumps({
            "subtask_annotations": [{"start": 0, "end": 2, "description": "Grasp and replace paper roll"}],
        }).encode())
        self.write_file(f"{base}/bad_frame_ratio.json", b'{"ratio":0}')
        self.write_file(f"{base}/meta/stats.json", b'{"present":true}')
        for camera in ("head_left", "head_right", "left_wrist", "right_wrist"):
            self.write_file(
                f"{base}/videos/observation.images.{camera}/chunk-000/file-000.mp4",
                POINTER if pointers else MP4,
            )
            if raw:
                self.write_file(
                    f"EgoProRaw/mp4/{task}/{episode_uuid}/{camera}_camera/{camera}_camera_undistorted.mp4",
                    MP4, root=self.raw,
                )
        self.write_manifests()
        return row

    def build(self, limit=10):
        return manifest_tool.build_manifest(self.dataset, self.raw, limit)

    def test_inventory_coverage_82_unique_episodes_across_variants(self):
        for number in range(1, 83):
            self.episode(number, f"Task {number % 15:02d}", manifest_tool.VARIANTS[number > 61])
        result = self.build()
        inventory = result["inventory"]
        self.assertEqual(inventory["raw_episode_count"], 82)
        self.assertEqual(inventory["raw_video_count"], 328)
        self.assertEqual(inventory["raw_payload_video_count"], 328)
        self.assertEqual(inventory["matched_episode_count"], 82)
        self.assertEqual(inventory["coverage_ratio"], 1)
        self.assertEqual(inventory["by_variant"]["EgoProStandard"]["matched_episodes"], 61)
        self.assertEqual(inventory["by_variant"]["EgoProStandard-body"]["matched_episodes"], 21)
        self.assertEqual(len({episode["source_episode_uuid"] for episode in result["selected_episodes"]}), 10)

    def test_selection_balances_tasks_and_is_independent_of_csv_order(self):
        for number in range(1, 9):
            self.episode(number)
        for number, task in ((30, "Adjust Labels"), (31, "Adjust Labels"), (40, "Replace Paper Roll")):
            self.episode(number, task)
        original = self.build(limit=5)
        for rows in self.rows.values():
            rows.reverse()
        self.write_manifests()
        self.assertEqual(original, self.build(limit=5))
        self.assertEqual(
            [episode["task_name"] for episode in original["selected_episodes"]],
            ["Adjust Labels", "Refill Stapler", "Replace Paper Roll", "Adjust Labels", "Refill Stapler"],
        )

    def test_unmatched_and_annotated_only_episodes_are_counted(self):
        self.episode(1)
        self.episode(2, raw=False)
        unmatched = str(UUID(int=3))
        self.write_file(f"EgoProRaw/mp4/task/{unmatched}/head_left/file.mp4", MP4, root=self.raw)
        self.write_file("flat_video.mp4", MP4, root=self.raw)
        inventory = self.build()["inventory"]
        self.assertEqual(inventory["unmatched_raw_uuids"], [unmatched])
        self.assertEqual(inventory["annotated_only_episode_count"], 1)
        self.assertEqual(inventory["coverage_ratio"], 0.5)
        self.assertEqual(inventory["ignored_raw_videos"], ["flat_video.mp4"])

    def test_lfs_expected_metadata_and_exact_fetch_allowlist(self):
        row = self.episode(1, "Replace Paper Roll (Label Printer)")
        result = self.build()
        episode = result["selected_episodes"][0]
        pointers = [record for record in episode["files"] if record["status"] == "lfs_pointer"]
        self.assertEqual(len(pointers), 4)
        self.assertEqual(len(episode["files"]), 7)
        self.assertEqual(set(record["role"] for record in episode["files"]), set(manifest_tool.REQUIRED_ROLES))
        self.assertEqual(result["fetch_files"], sorted(record["path"] for record in pointers))
        for record in pointers:
            self.assertEqual(record["expected_size"], 123456)
            self.assertEqual(record["expected_sha256"], HASH)
            self.assertEqual(record["local_size"], len(POINTER))
            self.assertFalse(record["ready"])
            self.assertFalse(record["downloaded"])
        self.assertFalse(episode["readiness"]["ready_for_staging"])
        self.assertTrue(episode["readiness"]["fetch_plan_complete"])
        self.assertEqual(episode["duration_s"], 28.0)
        self.assertEqual(episode["fps"], float(row["fps"]))
        self.assertNotIn("head_right", json.dumps(episode["files"]))
        self.assertNotIn("meta/stats.json", json.dumps(episode["files"]))

    def test_real_payloads_are_ready_for_staging_not_training(self):
        row = self.episode(1, pointers=False)
        annotation = self.dataset / row["lerobot_path"] / "annotation.json"
        original = annotation.read_bytes()
        result = self.build()
        episode = result["selected_episodes"][0]
        self.assertTrue(all(record["ready"] for record in episode["files"]))
        self.assertTrue(episode["readiness"]["ready_for_staging"])
        self.assertFalse(episode["readiness"]["training_ready"])
        self.assertEqual(result["fetch_files"], [])
        self.assertFalse(result["annotation_semantics"]["hoi_ground_truth"])
        self.assertFalse(result["annotation_semantics"]["automatic_label_mapping"])
        self.assertNotIn("label_subject", json.dumps(result))
        self.assertEqual(annotation.read_bytes(), original)

    def test_empty_text_json_and_malformed_lfs_are_not_payloads(self):
        row = self.episode(1, pointers=False)
        base = row["lerobot_path"]
        fixtures = (
            ("meta/tasks.parquet", b"", "placeholder"),
            ("annotation.json", b'{"placeholder":true}', "placeholder"),
            ("bad_frame_ratio.json", b"{}", "placeholder"),
            ("data/chunk-000/file-000.parquet", b"placeholder", "placeholder"),
            ("meta/info.json", b"not downloaded", "placeholder"),
            ("meta/episodes/chunk-000/file-000.parquet", b"PAR1incomplete", "invalid"),
            ("videos/observation.images.head_left/chunk-000/file-000.mp4", POINTER.replace(b"ab", b"zz"), "invalid"),
        )
        for suffix, payload, expected in fixtures:
            with self.subTest(suffix=suffix):
                self.write_file(base + suffix, payload)
                record = manifest_tool.inspect_file(self.dataset, base + suffix, "test")
                self.assertEqual(record["status"], expected)
                self.assertFalse(record["ready"])
                self.assertFalse(record["downloaded"])
        self.assertFalse(self.build()["selected_episodes"][0]["readiness"]["ready_for_staging"])

    def test_json_can_also_be_an_lfs_pointer(self):
        row = self.episode(1)
        self.write_file(row["lerobot_path"] + "meta/info.json", POINTER)
        episode = self.build()["selected_episodes"][0]
        record = next(record for record in episode["files"] if record["role"] == "info")
        self.assertEqual(record["status"], "lfs_pointer")
        self.assertEqual(episode["frame_count"], 840)

    def test_raw_pointer_is_not_counted_as_downloaded_video(self):
        row = self.episode(1)
        raw_path = f"EgoProRaw/mp4/{row['task_name']}/{row['source_episode_uuid']}/head_left_camera/head_left_camera_undistorted.mp4"
        self.write_file(raw_path, POINTER, root=self.raw)
        result = self.build()
        self.assertEqual(result["inventory"]["raw_video_count"], 4)
        self.assertEqual(result["inventory"]["raw_payload_video_count"], 3)

    def test_identical_duplicate_rows_are_deduplicated(self):
        row = self.episode(1)
        self.rows[row["variant"]].append(dict(row))
        self.write_manifests()
        result = self.build()
        self.assertEqual(result["inventory"]["identical_duplicate_rows"], 1)
        self.assertEqual(len(result["selected_episodes"]), 1)

    def test_conflicting_duplicates_rejected_even_when_outside_selected_limit(self):
        self.episode(1, "A first task")
        row = self.episode(2)
        self.rows[row["variant"]].append({**row, "frame_count": "839"})
        self.write_manifests()
        with self.assertRaisesRegex(manifest_tool.ManifestError, "Conflicting duplicate UUID"):
            self.build(limit=1)

    def test_same_uuid_in_different_variants_is_a_conflict(self):
        self.episode(1)
        self.episode(1, variant="EgoProStandard-body")
        with self.assertRaisesRegex(manifest_tool.ManifestError, "Conflicting duplicate UUID"):
            self.build()

    def test_absolute_traversal_windows_and_url_paths_are_rejected(self):
        row = self.episode(1)
        for unsafe in (".", "./", "", "/etc/passwd", "../outside", "EgoProStandard/../../outside", "C:\\outside", "https://example.test/data"):
            with self.subTest(path=unsafe):
                row["lerobot_path"] = unsafe
                self.write_manifests()
                with self.assertRaises(manifest_tool.ManifestError):
                    self.build()

    def test_metadata_symlink_escape_is_rejected(self):
        row = self.episode(1)
        outside = self.root / "outside.json"
        outside.write_text('{"outside":true}')
        path = self.dataset / row["lerobot_path"] / "annotation.json"
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(manifest_tool.ManifestError, "escapes dataset root"):
            self.build()

    def test_directory_and_raw_symlink_escape_is_rejected(self):
        row = self.episode(1)
        outside = self.root / "outside"
        outside.mkdir()
        directory = self.dataset / row["lerobot_path"] / "data/escape"
        directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(manifest_tool.ManifestError, "escapes dataset root"):
            self.build()
        directory.unlink()
        (self.raw / "escape").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(manifest_tool.ManifestError, "escapes dataset root"):
            self.build()

    def test_missing_metadata_reports_incomplete_plan_without_invented_shards(self):
        row = self.episode(1)
        base = self.dataset / row["lerobot_path"]
        for suffix in ("meta/info.json", "annotation.json", "data/chunk-000/file-000.parquet"):
            (base / suffix).unlink()
        episode = self.build()["selected_episodes"][0]
        self.assertEqual(episode["readiness"]["missing_file_roles"], ["data"])
        self.assertFalse(episode["readiness"]["fetch_plan_complete"])
        self.assertFalse(episode["readiness"]["ready_for_staging"])
        self.assertNotIn("data", [record["role"] for record in episode["files"]])
        self.assertEqual(next(record["status"] for record in episode["files"] if record["role"] == "annotation"), "missing")

    def test_missing_manifest_is_visible_in_coverage(self):
        self.episode(1)
        (self.dataset / "EgoProStandard-body/manifests/train.csv").unlink()
        self.assertFalse(self.build()["inventory"]["manifest_inventory_complete"])
        (self.dataset / "EgoProStandard/manifests/train.csv").write_bytes(POINTER)
        with self.assertRaisesRegex(manifest_tool.ManifestError, "No readable"):
            self.build()

    def test_missing_or_duplicated_csv_columns_are_rejected(self):
        self.episode(1)
        path = self.dataset / "EgoProStandard/manifests/train.csv"
        for content in (
            "source_episode_uuid,task_name\n",
            "source_episode_uuid,task_name,lerobot_path,task_name\n",
        ):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaisesRegex(manifest_tool.ManifestError, "Missing or duplicate CSV columns"):
                    self.build()

    def test_unicode_metadata_crossing_read_boundary_is_not_a_placeholder(self):
        path = "unicode.csv"
        self.write_file(path, b"column\n" + b"a" * (8191 - 7) + "纸卷".encode())
        self.assertTrue(manifest_tool.inspect_file(self.dataset, path, "source_manifest")["ready"])

    def test_mismatched_variant_or_uuid_path_is_rejected(self):
        row = self.episode(1)
        original_path = row["lerobot_path"]
        row["lerobot_path"] = original_path.replace(str(UUID(int=1)), str(UUID(int=2)))
        self.write_manifests()
        with self.assertRaisesRegex(manifest_tool.ManifestError, "Variant/UUID disagrees"):
            self.build()
        row["lerobot_path"] = original_path
        row["variant"] = "EgoProStandard-body"
        self.write_manifests()
        with self.assertRaisesRegex(manifest_tool.ManifestError, "Invalid task or variant"):
            self.build()

    def test_multiple_shards_are_explicit_and_only_head_left_is_fetched(self):
        row = self.episode(1)
        base = row["lerobot_path"]
        self.write_file(base + "data/chunk-001/file-002.parquet", POINTER)
        self.write_file(base + "meta/episodes/chunk-001/file-002.parquet", POINTER)
        self.write_file(base + "videos/observation.images.head_left/chunk-001/file-002.mp4", POINTER)
        result = self.build()
        self.assertEqual(len(result["fetch_files"]), 7)
        videos = [path for path in result["fetch_files"] if path.endswith(".mp4")]
        self.assertEqual(len(videos), 2)
        self.assertTrue(all("observation.images.head_left/" in path for path in videos))

    def test_missing_numeric_metadata_falls_back_to_info_with_provenance(self):
        row = self.episode(1)
        row.update(fps="", frame_count="", duration_s="")
        self.write_manifests()
        episode = self.build()["selected_episodes"][0]
        self.assertEqual((episode["fps"], episode["frame_count"], episode["duration_s"]), (30.0, 840, 28.0))
        self.assertEqual(episode["metadata_sources"]["duration_s"], "derived:frame_count/fps")
        (self.dataset / row["lerobot_path"] / "meta/info.json").unlink()
        episode = self.build()["selected_episodes"][0]
        self.assertEqual(episode["readiness"]["missing_metadata_fields"], ["fps", "frame_count", "duration_s"])
        self.assertIsNone(episode["fps"])

    def test_invalid_numeric_metadata_and_limits_are_rejected(self):
        row = self.episode(1)
        for value in ("nan", "inf", "0", "-1", "not-a-number"):
            with self.subTest(fps=value):
                row["fps"] = value
                self.write_manifests()
                with self.assertRaisesRegex(manifest_tool.ManifestError, "Invalid fps"):
                    self.build()
        for value in (0, -1, 1.5, True):
            with self.subTest(limit=value), self.assertRaisesRegex(manifest_tool.ManifestError, "positive integer"):
                self.build(limit=value)

    def test_cli_works_without_site_packages_and_is_byte_deterministic(self):
        for number in range(1, 12):
            self.episode(number, f"Task {number}")
        output = self.root / "selection.json"
        command = [
            sys.executable, "-S", "-B", str(MODULE_PATH), "--dataset-root", str(self.dataset),
            "--raw-root", str(self.raw), "--output", str(output),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        original = output.read_bytes()
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(json.loads(original)["selection"]["count"], 10)
        failed = subprocess.run(command + ["--limit", "0"], capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_refuses_overwriting_input_file(self):
        row = self.episode(1)
        path = self.dataset / row["lerobot_path"] / "annotation.json"
        original = path.read_bytes()
        result = subprocess.run([
            sys.executable, "-S", "-B", str(MODULE_PATH), "--dataset-root", str(self.dataset),
            "--raw-root", str(self.raw), "--output", str(path),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
