"""Independent annotated-pilot QA, including a tiny fully decoded video."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from src.dataset import audit_egodemo as auditor


EPISODE_UUID = "d057beac-dab1-43a1-9192-6b8b06abddaf"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_timeline(path, sequence, events):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_index", "timestamp", "subtask_ids"])
        writer.writeheader()
        for frame, timestamp in zip(sequence["frame_index"], sequence["timestamps"]):
            writer.writerow({"frame_index": int(frame), "timestamp": float(timestamp),
                             "subtask_ids": json.dumps([event["subtask_id"] for event in events["subtask_annotations"]
                                                       if event["start_time"] <= timestamp < event["end_time"]])})


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    root, staging = tmp_path / "payload", tmp_path / "staging"
    directory = staging / EPISODE_UUID
    directory.mkdir(parents=True)
    base = f"EgoProStandard/lerobot/task/{EPISODE_UUID}"
    sequence = {"timestamps": np.arange(6, dtype=np.float64) / 10,
                "frame_index": np.arange(6), "index": np.arange(6),
                "episode_index": np.zeros(6, dtype=np.int64), "task_index": np.zeros(6, dtype=np.int64),
                "hand_valid": np.ones((6, 2), dtype=bool),
                "head_left_camera_position": np.zeros((6, 3), dtype=np.float32),
                "head_left_camera_rotation": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (6, 1))}
    for side in ("left", "right"):
        raw = np.broadcast_to(np.arange(6, dtype=np.float32)[:, None, None] * .01, (6, 21, 3)).copy()
        sequence[f"{side}_hand_world_raw"] = raw
        sequence[f"{side}_hand_world"] = raw.copy()
    sequence["hand_valid"][2, 0] = False
    sequence["left_hand_world"][2] = np.nan
    quality = {"total_frames": 6, "hand_bad_frame_details": [{"frame": 2, "left_bad_reason": "occlusion", "right_bad_reason": None}]}
    events = {"episode_description": "fixture only", "subtask_annotations": [
        {"start_time": 0, "end_time": .3, "subtask_id": "first", "description": "first phase"},
        {"start_time": .3, "end_time": .6, "subtask_id": "second", "description": "second phase"},
    ]}
    info = {"fps": 10, "total_frames": 6, "features": {}}
    files, sources = [], {}
    for role, suffix, content in (
        ("info", "meta/info.json", json.dumps(info).encode()),
        ("annotation", "annotation.json", json.dumps(events).encode()),
        ("bad_frame_ratio", "bad_frame_ratio.json", json.dumps(quality).encode()),
        ("data", "data/chunk-000/file-000.parquet", b"PAR1test-fixture-not-real-parquetPAR1"),
        ("head_left_video", "videos/observation.images.head_left/chunk-000/file-000.mp4", b"mock-video"),
    ):
        path = root / base / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append({"role": role, "path": f"{base}/{suffix}", "expected_size": len(content),
                      "expected_sha256": hashlib.sha256(content).hexdigest(), "ready": False})
        sources[role] = path
    episode = {"source_episode_uuid": EPISODE_UUID, "lerobot_path": base, "variant": "EgoProStandard",
               "task_name": "fixture", "frame_count": 6, "fps": 9.99, "files": files}
    manifest = tmp_path / "selection.json"
    write_json(manifest, {"selected_episodes": [episode]})
    meta = {"source_episode_uuid": EPISODE_UUID, "frame_count": 6, "coordinate_system": "world",
            "provenance": {"manifest": {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                           "verified_source_files": [{"path": record["path"], "sha256": record["expected_sha256"]}
                                                     for record in files if record["role"] != "head_left_video"]}}
    np.savez_compressed(directory / "sequence.npz", **sequence)
    for filename, value in (("meta.json", meta), ("events.json", events), ("bad_frame_ratio.json", quality)):
        write_json(directory / filename, value)
    write_timeline(directory / "event_timeline.csv", sequence, events)
    original_probe = auditor.probe_video

    def probe(path):
        assert path == sources["head_left_video"]
        return {"decoded_frame_count": 6, "pts": np.arange(6) / 10, "method": "test mock"}

    monkeypatch.setattr(auditor, "probe_video", probe)
    return SimpleNamespace(root=root, staging=staging, directory=directory, manifest=manifest,
                           sequence=sequence, quality=quality, events=events, info=info, episode=episode,
                           sources=sources, meta=meta, original_probe=original_probe, probe=probe)


def audit(pilot):
    return auditor.audit_manifest(pilot.manifest, pilot.root, pilot.staging)


def codes(report, level="errors"):
    return {item["code"] for item in report[level]}


def save_sequence(pilot):
    np.savez_compressed(pilot.directory / "sequence.npz", **pilot.sequence)


def replace_source(pilot, role, value):
    content = value if isinstance(value, bytes) else json.dumps(value).encode()
    pilot.sources[role].write_bytes(content)
    for record in pilot.episode["files"]:
        if record["role"] == role:
            record.update(expected_size=len(content), expected_sha256=hashlib.sha256(content).hexdigest())
    write_json(pilot.manifest, {"selected_episodes": [pilot.episode]})
    pilot.meta["provenance"] = {
        "manifest": {"sha256": hashlib.sha256(pilot.manifest.read_bytes()).hexdigest()},
        "verified_source_files": [{"path": record["path"], "sha256": record["expected_sha256"]}
                                  for record in pilot.episode["files"] if record["role"] != "head_left_video"],
    }
    write_json(pilot.directory / "meta.json", pilot.meta)


def test_valid_pilot_is_reproducible_and_does_not_trust_ready_or_nominal_fps(pilot):
    before = {path: path.read_bytes() for path in pilot.directory.iterdir()}
    report = audit(pilot)
    assert report["summary"] == {"episode_count": 1, "error_count": 0, "warning_count": 0,
                                 "passed": True, "decoded_frames": 6, "native_frames": 6}
    assert audit(pilot) == report
    episode = report["episodes"][0]
    assert episode["sequence"]["valid_hand_counts"] == [5, 6]
    assert episode["events"]["covered_frames"] == 6
    assert episode["events"]["overlap_frames"] == 0
    assert all(len(source["sha256"]) == 64 and Path(source["path"]).is_absolute() for source in episode["sources"])
    assert {path: path.read_bytes() for path in pilot.directory.iterdir()} == before
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("filename", ["sequence.npz", "events.json", "meta.json", "bad_frame_ratio.json", "event_timeline.csv"])
def test_missing_staging_file_is_explicit_failure(pilot, filename):
    (pilot.directory / filename).unlink()
    report = audit(pilot)
    assert not report["summary"]["passed"]
    assert "staging_invalid" in codes(report)
    assert report["summary"]["decoded_frames"] == 6


@pytest.mark.parametrize("role", ["info", "annotation", "bad_frame_ratio", "data", "head_left_video"])
@pytest.mark.parametrize("damage", ["missing", "pointer", "hash", "empty"])
def test_missing_or_damaged_payload_never_passes(pilot, role, damage):
    path = pilot.sources[role]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes({"pointer": b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n", "hash": b"altered", "empty": b""}[damage])
    assert {"source_asset", "missing_source"} <= codes(audit(pilot))


@pytest.mark.parametrize("change", ["reverse", "duplicate", "nan", "negative", "shape", "integer"])
def test_invalid_native_timestamps(pilot, change):
    native = pilot.sequence["timestamps"]
    if change == "reverse":
        native[3] = .15
    elif change == "duplicate":
        native[3] = native[2]
    elif change == "nan":
        native[3] = np.nan
    elif change == "negative":
        native[0] = -1
    elif change == "shape":
        pilot.sequence["timestamps"] = native[:, None]
    else:
        pilot.sequence["timestamps"] = np.arange(6)
    save_sequence(pilot)
    assert "staging_invalid" in codes(audit(pilot))


@pytest.mark.parametrize("key", ["hand_valid", "left_hand_world_raw", "left_hand_world", "frame_index", "head_left_camera_rotation"])
def test_shape_errors(pilot, key):
    pilot.sequence[key] = pilot.sequence[key][:-1]
    save_sequence(pilot)
    assert not audit(pilot)["summary"]["passed"]


@pytest.mark.parametrize("change,code", [("mask", "hand_mask"), ("raw", "masked_hand"), ("invalid", "masked_hand")])
def test_mask_consistency(pilot, change, code):
    if change == "mask":
        pilot.sequence["hand_valid"][2, 0] = True
    elif change == "raw":
        pilot.sequence["left_hand_world_raw"][0, 0, 0] += 1
    else:
        pilot.sequence["left_hand_world"][2] = 0
    save_sequence(pilot)
    assert code in codes(audit(pilot))


def test_nonfinite_raw_is_allowed_only_when_masked(pilot):
    pilot.sequence["right_hand_world_raw"][0, 4, 1] = np.inf
    pilot.sequence["right_hand_world"][0] = np.nan
    pilot.sequence["hand_valid"][0, 1] = False
    save_sequence(pilot)
    assert audit(pilot)["summary"]["passed"]


@pytest.mark.parametrize("change", ["null", "manifest_null", "source_missing", "source_hash", "manifest_hash"])
def test_staging_provenance_is_not_optional(pilot, change):
    provenance = pilot.meta["provenance"]
    if change == "null":
        pilot.meta["provenance"] = None
    elif change == "manifest_null":
        provenance["manifest"] = None
    elif change == "source_missing":
        provenance["verified_source_files"] = []
    elif change == "source_hash":
        provenance["verified_source_files"][0]["sha256"] = "0" * 64
    else:
        provenance["manifest"]["sha256"] = "0" * 64
    write_json(pilot.directory / "meta.json", pilot.meta)
    assert not audit(pilot)["summary"]["passed"]


@pytest.mark.parametrize("content", [b"garbage", b"PK\x03\x04truncated"])
def test_corrupt_npz_produces_report_instead_of_crash(pilot, content):
    (pilot.directory / "sequence.npz").write_bytes(content)
    assert "staging_invalid" in codes(audit(pilot))


@pytest.mark.parametrize("value,code,level", [(0, "quaternion_zero", "errors"), (np.nan, "spatial_nonfinite", "errors"),
                                           (np.inf, "spatial_nonfinite", "errors"), (1.2, "quaternion_norm", "warnings")])
def test_quaternion_diagnostics(pilot, value, code, level):
    pilot.sequence["head_left_camera_rotation"][1, 0] = value
    save_sequence(pilot)
    report = audit(pilot)
    assert code in codes(report, level)
    assert report["summary"]["passed"] == (level == "warnings")
    json.dumps(report, allow_nan=False)


def test_large_wrist_jump_is_warning_not_unit_assumption(pilot):
    for suffix in ("right_hand_world", "right_hand_world_raw"):
        pilot.sequence[suffix][-1, 0] = 1000
    save_sequence(pilot)
    report = audit(pilot)
    assert report["summary"]["passed"]
    assert "wrist_jump" in codes(report, "warnings")
    assert report["episodes"][0]["sequence"]["wrists"]["right"]["high_jump_frame_indices"] == [5]


@pytest.mark.parametrize("mode", ["overlap", "gap", "outside", "empty"])
def test_event_coverage_diagnostics_and_half_open_boundaries(pilot, mode):
    if mode == "overlap":
        pilot.events["subtask_annotations"][0]["end_time"] = .4
    elif mode == "gap":
        pilot.events["subtask_annotations"][0]["end_time"] = .2
    elif mode == "outside":
        pilot.events["subtask_annotations"][-1]["end_time"] = 2
    else:
        pilot.events["subtask_annotations"] = []
    replace_source(pilot, "annotation", pilot.events)
    write_json(pilot.directory / "events.json", pilot.events)
    write_timeline(pilot.directory / "event_timeline.csv", pilot.sequence, pilot.events)
    report = audit(pilot)
    assert report["summary"]["passed"]
    event_summary = report["episodes"][0]["events"]
    if mode == "overlap":
        assert event_summary["overlap_frames"] == 1
    elif mode == "gap":
        assert event_summary["unlabelled_frames"] == 1
    elif mode == "outside":
        assert event_summary["out_of_range_events"] == 1
    else:
        assert event_summary["unlabelled_frames"] == 6


@pytest.mark.parametrize("field,value", [("end_time", 0), ("start_time", -1), ("end_time", np.inf),
                                       ("subtask_id", "second"), ("description", None)])
def test_invalid_events(pilot, field, value):
    pilot.events["subtask_annotations"][0][field] = value
    write_json(pilot.directory / "events.json", pilot.events)
    assert "staging_invalid" in codes(audit(pilot))


def test_timeline_cannot_map_boundary_to_previous_event(pilot):
    path = pilot.directory / "event_timeline.csv"
    content = path.read_text()
    path.write_text(content.replace('"[""second""]"', '"[""first""]"', 1))
    assert "staging_invalid" in codes(audit(pilot))


@pytest.mark.parametrize("mode", ["count", "drift", "jitter", "offset", "float32"])
def test_video_timeline_checks(pilot, monkeypatch, mode):
    def probe(path):
        result = pilot.probe(path)
        if mode == "count":
            result["decoded_frame_count"] = 5
            result["pts"] = result["pts"][:-1]
        elif mode == "drift":
            result["pts"] *= 1.01
        elif mode == "jitter":
            result["pts"][2] += .01
        elif mode == "offset":
            result["pts"] += 10
        else:
            result["pts"] = result["pts"].astype(np.float32).astype(np.float64)
        return result

    monkeypatch.setattr(auditor, "probe_video", probe)
    report = audit(pilot)
    assert report["summary"]["passed"] == (mode in ("offset", "float32"))
    if mode == "offset":
        assert "video_start_offset" in codes(report, "warnings")


@pytest.mark.parametrize("returncode,stderr", [(0, b"decoder warning password=secret"), (1, b""), (1, b"bad frames")])
def test_ffprobe_stderr_and_nonzero_rejected_without_leaking_output(pilot, monkeypatch, returncode, stderr):
    monkeypatch.setattr(auditor.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, returncode, b"{}", stderr))
    with pytest.raises(auditor.AuditError, match="decoding failed") as error:
        pilot.original_probe(pilot.sources["head_left_video"])
    assert "password" not in str(error.value) and "secret" not in str(error.value)


@pytest.mark.parametrize("mode", ["valid", "old_pts", "count", "no_pts", "duplicate", "nan", "invalid_json"])
def test_ffprobe_decoded_records_and_legacy_pts(pilot, monkeypatch, mode):
    frames = [{"pkt_pts_time" if mode == "old_pts" else "pts_time": str(index / 10)} for index in range(6)]
    count = "5" if mode == "count" else "6"
    if mode == "no_pts":
        frames[2] = {}
    elif mode == "duplicate":
        frames[2]["pts_time"] = ".1"
    elif mode == "nan":
        frames[2]["pts_time"] = "nan"
    output = b"invalid" if mode == "invalid_json" else json.dumps({"frames": frames, "streams": [{"nb_read_frames": count}]}).encode()

    def run(command, **kwargs):
        assert "-count_frames" in command and "-show_frames" in command
        assert "-read_intervals" not in command and "-skip_frame" not in command
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(auditor.subprocess, "run", run)
    if mode in ("valid", "old_pts"):
        assert pilot.original_probe(pilot.sources["head_left_video"])["decoded_frame_count"] == 6
    else:
        with pytest.raises(auditor.AuditError):
            pilot.original_probe(pilot.sources["head_left_video"])


def test_cli_creates_report_and_refuses_overwrite(pilot, tmp_path, capsys):
    output = tmp_path / "report.json"
    arguments = ["--manifest", str(pilot.manifest), "--dataset-root", str(pilot.root),
                 "--staging-root", str(pilot.staging), "--output", str(output)]
    assert auditor.main(arguments) == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        auditor.main(arguments)
    assert error.value.code == 2 and output.read_bytes() == before
    pilot.sources["data"].unlink()
    arguments[-1] = str(tmp_path / "failure.json")
    assert auditor.main(arguments) == 1
    assert not json.loads(Path(arguments[-1]).read_text())["summary"]["passed"]
    assert "NEW report" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["duplicate", "no_episodes", "bad_uuid", "unsafe_path", "symlink"])
def test_invalid_manifest_paths(pilot, change, tmp_path):
    episodes = [pilot.episode]
    if change == "duplicate":
        episodes *= 2
    elif change == "no_episodes":
        episodes = []
    elif change == "bad_uuid":
        pilot.episode["source_episode_uuid"] = "../bad"
    elif change == "unsafe_path":
        pilot.episode["files"][0]["path"] = "../private-file"
    else:
        path = pilot.sources["info"]
        outside = tmp_path / "outside.json"
        path.rename(outside)
        path.symlink_to(outside)
    write_json(pilot.manifest, {"selected_episodes": episodes})
    assert not audit(pilot)["summary"]["passed"]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe unavailable")
def test_real_tiny_video_full_decode_and_corruption(pilot, monkeypatch, tmp_path):
    video = tmp_path / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=10",
                    "-frames:v", "6", "-c:v", "mpeg4", "-threads", "1", str(video)], check=True, capture_output=True)
    replace_source(pilot, "head_left_video", video.read_bytes())
    monkeypatch.setattr(auditor, "probe_video", pilot.original_probe)
    report = audit(pilot)
    assert report["summary"]["passed"], report["errors"]
    assert report["summary"]["decoded_frames"] == 6
    damaged = video.read_bytes()[:100]
    replace_source(pilot, "head_left_video", damaged)
    report = audit(pilot)
    assert "video_decode" in codes(report)
    assert not report["summary"]["passed"]
