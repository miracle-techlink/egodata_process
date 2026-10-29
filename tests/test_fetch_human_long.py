"""Data-only acquisition tests: python -S tests/test_fetch_human_long.py."""

import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "fetch_human_long", Path(__file__).resolve().parents[1] / "src/dataset/fetch_human_long.py"
)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class AcquisitionTests(unittest.TestCase):
    def test_primary_selection_attributes_assistant_not_user_approval(self):
        self.assertEqual(
            fetch.CAMERA_SELECTION_BASIS,
            "Primary assistant visual inspection of RGB-only contact sheet before training; no GT/model error selection",
        )

    def test_rejects_traversal(self):
        for name in ("../evil", "/absolute", "a/../b", "a\\b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fetch.safe_name(name)

    def test_preserves_safe_name(self):
        self.assertEqual(fetch.safe_name("a/b.jpg"), "a/b.jpg")

    def test_test_overrides_train(self):
        info = {"subject_ids": ["subject_1"], "sequence_ids": ["capture"]}
        splits = {"train": [[0, 0, 0, 0]], "test": [[0, 0, 1, 30]]}
        self.assertEqual(fetch.capture_split("subject_1", "capture", info, splits), "test")

    def test_unknown_is_not_train(self):
        info = {"subject_ids": ["subject_1"], "sequence_ids": ["capture"]}
        self.assertEqual(fetch.capture_split("subject_1", "capture", info, {}), "unknown")

    def test_validation_not_train(self):
        info = {"subject_ids": ["subject_1"], "sequence_ids": ["capture"]}
        self.assertEqual(fetch.capture_split("subject_1", "capture", info,
                                            {"train": [[0, 0]], "valid": [[0, 0]]}), "valid")

    def test_selection_excludes_heldout_cheaper_capture(self):
        info = {"subject_ids": ["subject_1"], "sequence_ids": ["20230101_000000", "20230101_000001"]}
        splits = {"train": [[0, 1]], "test": [[0, 0]]}
        tree = [{"path": f"subject_1-{sequence}.tar", "size": size, "lfs": {"oid": "a" * 64}}
                for sequence, size in zip(info["sequence_ids"], (1, 100))]
        self.assertEqual(fetch.select_archives(tree, info, splits, ["subject_1"])[0]["size"], 100)

    def test_selection_requires_training(self):
        info = {"subject_ids": ["subject_1"], "sequence_ids": []}
        with self.assertRaises(ValueError):
            fetch.select_archives([], info, {}, ["subject_1"])

    def test_duration_uses_elapsed_time(self):
        self.assertFalse(fetch.continuity(list(range(300)), 30)["eligible_long_continuous"])
        self.assertTrue(fetch.continuity(list(range(301)), 30)["eligible_long_continuous"])

    def test_no_stitching_or_duplicate_frames(self):
        for frames in ([0, 1, 3], [0, 1, 1, 2], []):
            self.assertFalse(fetch.continuity(frames, 30)["contiguous"])

    def test_rejects_invalid_fps(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fetch.continuity([0], fps)

    def test_range_rejects_whole_file_fallback(self):
        response = io.BytesIO(b"data")
        response.status = 200
        response.headers = {}
        with patch.object(fetch.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(ValueError):
                fetch.read_range("https://example.org/a", 0, 4, 100)

    def test_range_validates_interval_and_length(self):
        for status, header, payload in ((206, "bytes 4-7/100", b"data"),
                                        (206, "bytes 0-3/100", b"ab")):
            response = io.BytesIO(payload)
            response.status = status
            response.headers = {"Content-Range": header}
            with patch.object(fetch.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    fetch.read_range("https://example.org/a", 0, 4, 100)

    def test_range_success(self):
        response = io.BytesIO(b"data")
        response.status = 206
        response.headers = {"Content-Range": "bytes 0-3/100"}
        with patch.object(fetch.urllib.request, "urlopen", return_value=response):
            self.assertEqual(fetch.read_range("https://example.org/a", 0, 4, 100), b"data")

    def make_archive(self, gap=False, duplicate=False):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for frame in ([0, 2] if gap else [0, 1]):
                for modality in ("color.jpg", "depth.png", "label.npy"):
                    member = tarfile.TarInfo(f"037522251142-{frame:06d}.{modality}")
                    member.size = 4
                    archive.addfile(member, io.BytesIO(b"data"))
            member = tarfile.TarInfo("037522251142-000001.color.jpg" if duplicate else "043422252387-000000.color.jpg")
            member.size = 4
            archive.addfile(member, io.BytesIO(b"data"))
        stream.seek(0)
        return stream

    def test_extract_only_one_camera_and_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            result = fetch.extract_first_camera(self.make_archive(), Path(folder), time.monotonic() + 10)
            self.assertTrue(result["contiguous_all_modalities"])
            self.assertEqual(result["frame_count"], 2)
            self.assertEqual(len(result["files"]), 6)
            self.assertFalse((Path(folder) / "043422252387").exists())
            for asset in result["files"]:
                self.assertEqual(fetch.sha256_file(Path(folder) / asset["path"]), asset["sha256"])

    def test_gap_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            result = fetch.extract_first_camera(self.make_archive(gap=True), Path(folder), time.monotonic() + 10)
            self.assertFalse(result["contiguous_all_modalities"])

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            fetch.extract_first_camera(self.make_archive(duplicate=True), Path(folder), time.monotonic() + 10)

    def test_reader_enforces_time_and_byte_limits(self):
        reader = fetch.BoundedReader(io.BytesIO(b"123456"), 3, time.monotonic() + 10)
        self.assertEqual(reader.read(10), b"123")
        with self.assertRaises(ValueError):
            reader.read(1)
        with self.assertRaises(TimeoutError):
            fetch.BoundedReader(io.BytesIO(b"data"), 4, time.monotonic() - 1).read(1)

    def test_camera_start_binary_search(self):
        stream = io.BytesIO()
        cameras = ("037522251142", "043422252387", "046122250168", "105322251225",
                   "105322251564", "108222250342", "115422250549", "117222250549")
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for camera in cameras:
                for frame in range(20):
                    for modality in ("color.jpg", "depth.png", "label.npy"):
                        member = tarfile.TarInfo(f"{camera}-{frame:06d}.{modality}")
                        member.size = 3072
                        member.pax_headers = {"mtime": "1.1"}
                        archive.addfile(member, io.BytesIO(b"x" * member.size))
        payload = stream.getvalue()
        calls = []

        def read_at(offset, size):
            calls.append((offset, size))
            return payload[offset:offset + size]

        offset = fetch.locate_camera_start(read_at, len(payload), "105322251564", window=32768)
        member = tarfile.TarInfo.frombuf(payload[offset:offset + 512], "utf-8", "strict")
        self.assertEqual(member.name, "105322251564-000000.color.jpg")
        self.assertLess(sum(size for _, size in calls), len(payload) // 2)
        with self.assertRaises(ValueError):
            fetch.locate_camera_start(read_at, len(payload), "999999999999", window=32768)

    def test_header_parser_rejects_payload_lookalikes(self):
        name = b"105322251564-000000.color.jpg\x00"
        self.assertEqual(fetch.tar_headers(name + bytes(1024), 0), [])
        member = tarfile.TarInfo("105322251564-000000.color.jpg")
        payload = member.tobuf()
        self.assertEqual(fetch.tar_headers(payload, 1), [])
        self.assertEqual(fetch.tar_headers(payload, 512)[0]["offset"], 512)

    def test_primary_offsets_remain_absolute(self):
        with tempfile.TemporaryDirectory() as folder:
            result = fetch.extract_first_camera(self.make_archive(), Path(folder),
                                                time.monotonic() + 10, archive_offset=8192)
            self.assertEqual(result["files"][0]["archive_offset"], 8192 + 512)

    def test_shared_request_budget_counts_reserved_response(self):
        budget = fetch.NetworkBudget(100, time.monotonic() + 10)
        budget.reserve(60)
        budget.reserve(40)
        with self.assertRaises(ValueError):
            budget.reserve(1)
        self.assertEqual(budget.reserved, 100)
        with self.assertRaises(TimeoutError):
            fetch.NetworkBudget(100, time.monotonic() - 1).reserve(1)

    def test_sparse_holes_not_used_as_index_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sparse.tar"
            with path.open("wb") as stream:
                stream.seek(1024 * 1024)
                stream.write(b"x" * 65536)
            self.assertIsNone(fetch.local_index_range(path, 32768, 512))
            if hasattr(os, "SEEK_DATA"):
                self.assertEqual(fetch.local_index_range(path, 1024 * 1024 + 8192, 512), b"x" * 512)

    def test_primary_candidate_preserves_previous_manifest_and_unknown_fps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "hocap_candidates.json").write_text("old immutable manifest")
            (root / "fps_evidence.json").write_text(json.dumps({
                "capture_fps_verified": False, "nominal_fps": 30, "fps": None,
            }))
            directory = root / "captures_primary/subject_1/20230101_000000"
            directory.mkdir(parents=True)
            path = directory / "105322251564-000000.color.jpg"
            path.write_bytes(b"image fixture")
            fetch.write_json(directory / "receipt.json", {
                "status": "complete", "subject_id": "subject_1", "acquisition_id": "20230101_000000",
                "camera_id": "105322251564", "official_split": "train", "view": "third_person_fixed",
                "url": "https://example.org/archive.tar", "frame_count": 1,
                "files": [{"path": path.name, "sha256": fetch.sha256_file(path), "bytes": path.stat().st_size}],
            })
            metadata = root / "labels_selected"
            metadata.mkdir()
            fetch.write_json(metadata / "subject_1-20230101_000000.meta.yaml", {
                "num_frames": 1, "object_ids": ["G01_1"], "task_id": 1,
            })
            with patch.dict("sys.modules", {"yaml": SimpleNamespace(safe_load=json.loads)}):
                fetch.candidate_manifest(root, primary=True)
            self.assertEqual((root / "hocap_candidates.json").read_text(), "old immutable manifest")
            result = json.loads((root / "primary_candidates.json").read_text())
            self.assertFalse(result["training_allowed"])
            record = result["records"][0]
            self.assertIsNone(record["fps"])
            self.assertIsNone(record["duration_seconds"])
            self.assertIsNone(record["split"])
            self.assertEqual(record["nominal_fps"], 30)
            self.assertEqual(record["quality"]["visual_review"], "pending")

    def test_exact_camera_interval_needs_no_next_camera_payload(self):
        original = self.make_archive().getvalue()
        with tarfile.open(fileobj=io.BytesIO(original)) as archive:
            members = list(archive)
        boundary = next(member.offset_data for member in members
                        if member.name.startswith("043422252387"))
        with tempfile.TemporaryDirectory() as folder:
            bounded = fetch.BoundedReader(io.BytesIO(original[:boundary]), boundary, time.monotonic() + 10)
            result = fetch.extract_first_camera(bounded, Path(folder), time.monotonic() + 10)
            self.assertTrue(result["contiguous_all_modalities"])
            self.assertEqual(result["frame_count"], 2)
            self.assertLessEqual(bounded.received, boundary)

    def test_reuse_checks_hash_and_does_not_count_old_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = root / "frame.jpg"
            payload.write_bytes(b"source bytes")
            receipt = {"status": "complete", "camera_id": "105322251564", "url": "https://example.org/a",
                       "network_bytes_read": 1234, "files": [{"path": payload.name,
                       "bytes": payload.stat().st_size, "sha256": fetch.sha256_file(payload)}]}
            fetch.write_json(root / "receipt.json", receipt)
            before = (root / "receipt.json").read_bytes()
            reused = fetch.reusable_camera(root, {"url": receipt["url"]}, "105322251564")
            self.assertTrue(reused["reused_existing"])
            self.assertEqual(reused["network_bytes_read"], 0)
            self.assertEqual(reused["network_bytes_previously_read"], 1234)
            self.assertEqual((root / "receipt.json").read_bytes(), before)
            self.assertEqual(reused["camera_selection_basis"], fetch.CAMERA_SELECTION_BASIS)
            payload.write_bytes(b"wrong bytes!")
            with self.assertRaises(ValueError):
                fetch.reusable_camera(root, {"url": receipt["url"]}, "105322251564")

    def test_reuse_rejects_different_camera_or_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(fetch.reusable_camera(root, {"url": "source"}, "105322251564"))
            fetch.write_json(root / "receipt.json", {"status": "complete", "camera_id": "other",
                                                    "url": "source", "files": [{}]})
            with self.assertRaises(ValueError):
                fetch.reusable_camera(root, {"url": "source"}, "105322251564")


if __name__ == "__main__":
    unittest.main()
