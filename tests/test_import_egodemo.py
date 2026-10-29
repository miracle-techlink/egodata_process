import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/dataset/import_egodemo.py"
SPEC = importlib.util.spec_from_file_location("import_egodemo", MODULE_PATH)
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)
EPISODE_UUID = "00000000-0000-0000-0000-000000000001"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 123\n"


@pytest.fixture
def source_sequence():
    features = {key: {"shape": [1]} for key in ("timestamp", *importer.INDEX_COLUMNS)}
    for suffix, (_, shape) in importer.SPATIAL_FEATURES.items():
        features[f"observation.state.{suffix}"] = {"shape": list(shape), "dtype": "float32"}
    info = {"features": features, "fps": 2.0, "total_frames": 4, "total_episodes": 1}
    rows = []
    for frame in range(4):
        row = {"timestamp": frame / 2, "frame_index": frame + 10, "episode_index": 7,
               "index": frame + 100, "task_index": 2}
        for suffix, (_, shape) in importer.SPATIAL_FEATURES.items():
            values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + frame
            if frame % 2:
                values = values.flatten()
            row[f"observation.state.{suffix}"] = values.tolist()
        rows.append(row)
    quality = {"hand_bad_frame_details": [
        {"frame": 11, "left_bad_reason": "occluded", "right_bad_reason": None},
        {"frame": 12, "left_bad_reason": "", "right_bad_reason": ["low confidence"]},
    ], "original_extra": {"ratio": 0.5}}
    return rows, info, quality


@pytest.fixture
def source_events():
    return {
        "episode_description": "原文任务描述", "unknown_field": [1, "preserved"],
        "subtask_annotations": [
            {"description": "Reach a shelf", "start_time": 0.0, "end_time": 0.5,
             "subtask_id": "source-1", "confidence": 0.9},
            {"description": "Adjust a label", "start_time": 1.0, "end_time": 1.5,
             "subtask_id": 2},
        ],
    }


def test_normalization_preserves_shapes_coordinates_and_raw_hands(source_sequence):
    rows, info, quality = source_sequence
    original = copy.deepcopy((rows, info, quality))
    sequence = importer.normalize_sequence(rows, info, quality)
    assert (rows, info, quality) == original
    assert sequence["timestamps"].tolist() == [0.0, 0.5, 1.0, 1.5]
    assert sequence["frame_index"].tolist() == [10, 11, 12, 13]
    assert sequence["head_world"].shape == (4, 1, 3)
    assert sequence["head_world_rotation"].shape == (4, 1, 4)
    for suffix, (name, shape) in importer.SPATIAL_FEATURES.items():
        output = sequence[f"{name}_raw"] if "hand_" in suffix else sequence[name]
        assert output.shape == (4, *shape)
        assert output.dtype == np.float32
        for position, row in enumerate(rows):
            np.testing.assert_array_equal(output[position], np.asarray(row[f"observation.state.{suffix}"]).reshape(shape))
    assert sequence["hand_valid"].dtype == np.bool_
    assert sequence["hand_valid"].tolist() == [[True, True], [False, True], [True, False], [True, True]]
    assert np.isnan(sequence["left_hand_world"][1]).all()
    assert np.isfinite(sequence["left_hand_world_raw"][1]).all()
    assert np.isnan(sequence["right_hand_world"][2]).all()
    assert not any(key.startswith("label_") or "obj" in key for key in sequence)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), None])
def test_nonfinite_hand_coordinates_mask_entire_hand(source_sequence, bad_value):
    rows, info, quality = source_sequence
    rows[0]["observation.state.hand_left_world"][0][1] = bad_value
    sequence = importer.normalize_sequence(rows, info, quality)
    assert not sequence["hand_valid"][0, 0]
    assert np.isnan(sequence["left_hand_world"][0]).all()
    assert sequence["hand_valid"][0, 1]


@pytest.mark.parametrize("key,value,match", [
    ("timestamp", float("nan"), "finite"),
    ("timestamp", float("inf"), "finite"),
    ("timestamp", 0.0, "strictly increasing"),
    ("timestamp", -1.0, "nonnegative"),
    ("frame_index", 10, "strictly increasing"),
    ("frame_index", 10.5, "integer"),
    ("frame_index", float("inf"), "finite"),
    ("frame_index", True, "finite"),
    ("index", 100, "strictly increasing"),
    ("episode_index", 8, "Mixed episode_index"),
    ("task_index", -1, "integer"),
])
def test_invalid_timeline_or_episode_fails(source_sequence, key, value, match):
    rows, info, quality = source_sequence
    rows[1][key] = value
    with pytest.raises(importer.ImportStageError, match=match):
        importer.normalize_sequence(rows, info, quality)


def test_declared_shape_does_not_allow_transposed_hand(source_sequence):
    rows, info, quality = source_sequence
    rows[0]["observation.state.hand_left_world"] = np.zeros((3, 21)).tolist()
    with pytest.raises(importer.ImportStageError, match="expected shape"):
        importer.normalize_sequence(rows, info, quality)


@pytest.mark.parametrize("value", ["not a coordinate", True, 1e100])
def test_invalid_coordinate_values_fail_without_silent_conversion(source_sequence, value):
    rows, info, quality = source_sequence
    rows[0]["observation.state.hand_left_world"][0][0] = value
    if value is True:
        rows[0]["observation.state.hand_left_world"] = [[True] * 3 for _ in range(21)]
    with pytest.raises(importer.ImportStageError, match="coordinates|coordinate array"):
        importer.normalize_sequence(rows, info, quality)


def test_missing_required_feature_and_row_value_fail(source_sequence):
    rows, info, quality = source_sequence
    key = "observation.state.head_left_camera_rotation"
    del rows[0][key]
    with pytest.raises(importer.ImportStageError, match=key):
        importer.normalize_sequence(rows, info, quality)
    del info["features"][key]
    with pytest.raises(importer.ImportStageError, match=key):
        importer.normalize_sequence(rows, info, quality)


@pytest.mark.parametrize("field,value,match", [
    ("fps", [2.0], "finite"),
    ("fps", 0, "positive"),
    ("fps", -1, "positive"),
    ("fps", float("nan"), "finite"),
    ("fps", float("inf"), "finite"),
    ("total_frames", [4], "finite"),
    ("total_frames", 10 ** 400, "finite"),
    ("total_episodes", 2, "do not mix episodes"),
])
def test_malformed_info_metadata_fails(source_sequence, field, value, match):
    rows, info, quality = source_sequence
    info[field] = value
    with pytest.raises(importer.ImportStageError, match=match):
        importer.normalize_sequence(rows, info, quality)


@pytest.mark.parametrize("field", ["total_frames", "fps", "features"])
def test_missing_info_metadata_fails(source_sequence, field):
    rows, info, quality = source_sequence
    del info[field]
    with pytest.raises(importer.ImportStageError, match=field):
        importer.normalize_sequence(rows, info, quality)


def test_counts_and_feature_declarations_are_checked(source_sequence):
    rows, info, quality = source_sequence
    with pytest.raises(importer.ImportStageError, match="Frame count mismatch"):
        importer.normalize_sequence(rows[:-1], info, quality)
    info["features"]["observation.state.hand_left_world"]["shape"] = [63]
    with pytest.raises(importer.ImportStageError, match="shape must be"):
        importer.normalize_sequence(rows, info, quality)


@pytest.mark.parametrize("change", ["unknown_frame", "duplicate_frame", "missing_reason", "missing_details"])
def test_invalid_bad_frame_metadata_fails(source_sequence, change):
    rows, info, quality = source_sequence
    if change == "unknown_frame":
        quality["hand_bad_frame_details"][0]["frame"] = 0
    elif change == "duplicate_frame":
        quality["hand_bad_frame_details"].append(quality["hand_bad_frame_details"][0].copy())
    elif change == "missing_reason":
        del quality["hand_bad_frame_details"][0]["left_bad_reason"]
    else:
        del quality["hand_bad_frame_details"]
    with pytest.raises(importer.ImportStageError, match="hand_bad_frame_details"):
        importer.normalize_sequence(rows, info, quality)


def test_events_preserved_and_gaps_are_unlabelled(source_sequence, source_events):
    original = copy.deepcopy(source_events)
    events = importer.validate_events(source_events)
    sequence = importer.normalize_sequence(*source_sequence)
    timeline = list(importer.event_timeline(sequence, events))
    assert source_events == original
    assert events is source_events["subtask_annotations"]
    assert [row["subtask_ids"] for row in timeline] == [["source-1"], [], [2], []]
    assert [row["timestamp"] for row in timeline] == sequence["timestamps"].tolist()


@pytest.mark.parametrize("first_end,expected_at_boundary", [(1.0, [2]), (1.25, ["source-1", 2])])
def test_adjacent_or_overlapping_events_keep_original_ids(source_sequence, source_events,
                                                         first_end, expected_at_boundary):
    source_events["subtask_annotations"][0]["end_time"] = first_end
    original = copy.deepcopy(source_events)
    events = importer.validate_events(source_events)
    timeline = list(importer.event_timeline(importer.normalize_sequence(*source_sequence), events))
    assert timeline[2]["timestamp"] == 1.0
    assert timeline[2]["subtask_ids"] == expected_at_boundary
    assert source_events == original


@pytest.mark.parametrize("field,value,match", [
    ("start_time", float("nan"), "finite"),
    ("end_time", float("inf"), "finite"),
    ("start_time", -1, "require 0"),
    ("end_time", 1.0, "start_time < end_time"),
    ("start_time", "1.0", "finite"),
    ("start_time", [1.0], "finite"),
    ("subtask_id", "source-1", "duplicate subtask_id"),
])
def test_invalid_event_ranges_fail(source_events, field, value, match):
    source_events["subtask_annotations"][1][field] = value
    with pytest.raises(importer.ImportStageError, match=match):
        importer.validate_events(source_events)


def test_reversed_events_and_missing_annotations_fail(source_events):
    source_events["subtask_annotations"].reverse()
    with pytest.raises(importer.ImportStageError, match="monotone"):
        importer.validate_events(source_events)
    with pytest.raises(importer.ImportStageError, match="episode_description"):
        importer.validate_events({})


def test_empty_events_leave_every_frame_unlabelled(source_sequence):
    sequence = importer.normalize_sequence(*source_sequence)
    events = importer.validate_events({"episode_description": "No semantic events", "subtask_annotations": []})
    assert all(row["subtask_ids"] == [] for row in importer.event_timeline(sequence, events))


@pytest.fixture
def payload(tmp_path, source_sequence, source_events, monkeypatch):
    rows, info, quality = source_sequence
    root, output = tmp_path / "payload", tmp_path / "staged"
    base = f"EgoProStandard/lerobot/test-task/{EPISODE_UUID}"
    files = []

    def write(relative, role, contents):
        path = root / base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        files.append({"path": f"{base}/{relative}", "role": role,
                      "expected_size": len(contents), "expected_sha256": hashlib.sha256(contents).hexdigest(),
                      "ready": False, "downloaded": False})
        return path

    write("meta/info.json", "info", json.dumps(info).encode())
    annotation_path = write("annotation.json", "annotation", json.dumps(source_events, ensure_ascii=False, indent=4).encode())
    quality_path = write("bad_frame_ratio.json", "bad_frame_ratio", json.dumps(quality, indent=4).encode())
    second = write("data/chunk-001/file-000.parquet", "data", b"PAR1secondPAR1")
    first = write("data/chunk-000/file-000.parquet", "data", b"PAR1firstPAR1")
    video = write("videos/observation.images.head_left/chunk-000/file-000.mp4", "head_left_video", b"not decoded")
    episode = {"source_episode_uuid": EPISODE_UUID, "variant": "EgoProStandard", "task_name": "test-task",
               "lerobot_path": base, "frame_count": 4, "fps": 2.0, "files": files}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"selected_episodes": [episode]}))

    def read_rows(paths):
        assert paths == [first, second]
        return rows

    monkeypatch.setattr(importer, "read_parquet_rows", read_rows)
    return SimpleNamespace(root=root, output=output, manifest=manifest, episode=episode,
                           annotation=annotation_path, quality=quality_path, video=video,
                           rows=rows, info=info, first=first)


def test_offline_staging_preserves_sources_and_reports_limits(payload):
    result = importer.import_manifest(payload.manifest, payload.root, payload.output)
    destination = payload.output / EPISODE_UUID
    assert result == [{"source_episode_uuid": EPISODE_UUID, "output_dir": str(destination), "frame_count": 4}]
    assert (destination / "events.json").read_bytes() == payload.annotation.read_bytes()
    assert (destination / "bad_frame_ratio.json").read_bytes() == payload.quality.read_bytes()
    with np.load(destination / "sequence.npz", allow_pickle=False) as sequence:
        assert sequence["left_hand_world"].shape == (4, 21, 3)
        assert sequence["frame_index"].tolist() == [10, 11, 12, 13]
        assert sequence["head_left_camera_rotation"].dtype == np.float32
        assert not any(key.startswith("label_") for key in sequence.files)
    meta = json.loads((destination / "meta.json").read_text())
    assert meta["coordinate_system"] == "world"
    assert meta["quaternion_order"] == ["qw", "qx", "qy", "qz"]
    assert meta["timeline"] == "native_annotated"
    assert meta["nominal_fps"] == meta["manifest_fps"] == meta["measured_fps"] == 2.0
    assert meta["fps_discrepancies"] == meta["quality_warnings"] == []
    assert not meta["world_to_camera_transform_applied"]
    assert not meta["rotation_direction_verified"]
    assert not meta["training_ready"]
    assert not meta["events"]["automatic_label_mapping"]
    assert meta["provenance"]["selected_episode"] == payload.episode
    assert len(meta["provenance"]["verified_source_files"]) == 5
    assert meta["video_verification"]["files"][0]["status"] == "payload_only_not_decoded"
    assert not meta["video_verification"]["files"][0]["decoded"]
    with (destination / "event_timeline.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [json.loads(row["subtask_ids"]) for row in rows] == [["source-1"], [], [2], []]


def test_new_payload_root_overrides_stale_manifest_root_and_flags(payload):
    for record in payload.episode["files"]:
        record.update(status="lfs_pointer", ready=False, downloaded=False)
    payload.manifest.write_text(json.dumps({
        "dataset_root": "/old/metadata-only/checkout",
        "selected_episodes": [payload.episode],
    }))
    originals = {path: path.read_bytes() for path in (payload.manifest, payload.annotation, payload.quality)}

    importer.import_manifest(payload.manifest, payload.root, payload.output)

    meta = json.loads((payload.output / EPISODE_UUID / "meta.json").read_text())
    for record in meta["provenance"]["verified_source_files"]:
        assert record["payload_verified"]
        assert not record["manifest_ready"] and not record["manifest_downloaded"]
    for path, contents in originals.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("role", ["info", "annotation", "bad_frame_ratio", "data"])
def test_lfs_pointers_are_rejected_before_parquet_loading(payload, role, monkeypatch):
    record = next(record for record in payload.episode["files"] if record["role"] == role)
    (payload.root / record["path"]).write_bytes(LFS_POINTER)
    monkeypatch.setattr(importer, "read_parquet_rows", lambda paths: pytest.fail("Must reject pointer before parsing"))
    with pytest.raises(importer.ImportStageError, match="Git LFS pointer"):
        importer.import_manifest(payload.manifest, payload.root, payload.output)
    assert not (payload.output / EPISODE_UUID).exists()


@pytest.mark.parametrize("failure", ["missing", "size", "checksum"])
def test_invalid_payload_integrity_fails(payload, failure):
    if failure == "missing":
        payload.first.unlink()
        match = "Missing payload"
    else:
        original = payload.first.read_bytes()
        payload.first.write_bytes(original + b"extra" if failure == "size" else b"X" + original[1:])
        match = "Size mismatch" if failure == "size" else "SHA256 mismatch"
    with pytest.raises(importer.ImportStageError, match=match):
        importer.import_manifest(payload.manifest, payload.root, payload.output)


@pytest.mark.parametrize("failure", ["missing", "pointer"])
def test_unavailable_video_does_not_claim_verification_or_block_staging(payload, failure):
    if failure == "missing":
        payload.video.unlink()
    else:
        payload.video.write_bytes(LFS_POINTER)
    importer.import_manifest(payload.manifest, payload.root, payload.output)
    meta = json.loads((payload.output / EPISODE_UUID / "meta.json").read_text())
    video = meta["video_verification"]["files"][0]
    assert video["status"] == "unavailable_or_invalid"
    assert not video["payload_verified"] and not video["decoded"]


@pytest.mark.parametrize("change,match", [
    ("duplicate_uuid", "Duplicate selected UUID"),
    ("missing_info", "require exactly one info"),
    ("escape", "Unsafe repository-relative path"),
    ("wrong_uuid", "variant/UUID disagrees"),
    ("wrong_parquet", "Not a LeRobot frame parquet"),
])
def test_manifest_identity_and_paths_are_validated(payload, change, match):
    document = {"selected_episodes": [payload.episode]}
    if change == "duplicate_uuid":
        document["selected_episodes"].append(copy.deepcopy(payload.episode))
    elif change == "missing_info":
        payload.episode["files"] = [record for record in payload.episode["files"] if record["role"] != "info"]
    elif change == "escape":
        payload.episode["files"][0]["path"] = "../outside.json"
    elif change == "wrong_uuid":
        payload.episode["source_episode_uuid"] = "00000000-0000-0000-0000-000000000002"
    else:
        next(record for record in payload.episode["files"] if record["role"] == "data")["path"] = (
            payload.episode["lerobot_path"] + "/meta/episodes.parquet"
        )
    payload.manifest.write_text(json.dumps(document))
    with pytest.raises(importer.ImportStageError, match=match):
        importer.import_manifest(payload.manifest, payload.root, payload.output)


def test_symlink_escape_is_rejected(payload, tmp_path):
    outside = tmp_path / "external.parquet"
    outside.write_bytes(payload.first.read_bytes())
    payload.first.unlink()
    payload.first.symlink_to(outside)
    with pytest.raises(importer.ImportStageError, match="escapes dataset root"):
        importer.import_manifest(payload.manifest, payload.root, payload.output)


def test_bad_rows_leave_no_partial_uuid_directory(payload):
    payload.rows[1]["episode_index"] = 55
    with pytest.raises(importer.ImportStageError, match="Mixed episode_index"):
        importer.import_manifest(payload.manifest, payload.root, payload.output)
    assert not (payload.output / EPISODE_UUID).exists()


def test_manifest_count_conflict_fails(payload):
    payload.episode["frame_count"] = 5
    payload.manifest.write_text(json.dumps({"selected_episodes": [payload.episode]}))
    with pytest.raises(importer.ImportStageError, match="manifest frame_count disagrees"):
        importer.import_manifest(payload.manifest, payload.root, payload.output)
    assert not (payload.output / EPISODE_UUID).exists()


@pytest.mark.parametrize("count,nominal_fps,manifest_fps", [
    (1194, 30.0, 29.974874371859297),
    (379, 30.002, 29.922839),
])
def test_biased_manifest_fps_warns_without_changing_native_timestamps(payload, count, nominal_fps, manifest_fps):
    template = copy.deepcopy(payload.rows[0])
    timestamps = (np.arange(count, dtype=np.float64) / nominal_fps).astype(np.float32).astype(np.float64)
    payload.rows[:] = [dict(copy.deepcopy(template), timestamp=float(timestamp), frame_index=frame, index=frame)
                       for frame, timestamp in enumerate(timestamps)]
    payload.info.update(fps=nominal_fps, total_frames=count)
    info_record = next(record for record in payload.episode["files"] if record["role"] == "info")
    contents = json.dumps(payload.info).encode()
    (payload.root / info_record["path"]).write_bytes(contents)
    info_record.update(expected_size=len(contents), expected_sha256=hashlib.sha256(contents).hexdigest())
    payload.episode.update(fps=manifest_fps, frame_count=count, duration_s=count / nominal_fps)
    payload.manifest.write_text(json.dumps({"selected_episodes": [payload.episode]}))

    importer.import_manifest(payload.manifest, payload.root, payload.output)

    destination = payload.output / EPISODE_UUID
    with np.load(destination / "sequence.npz") as sequence:
        np.testing.assert_array_equal(sequence["timestamps"], timestamps)
        np.testing.assert_array_equal(sequence["frame_index"], np.arange(count))
    meta = json.loads((destination / "meta.json").read_text())
    assert meta["fps"] == meta["nominal_fps"] == nominal_fps
    assert meta["manifest_fps"] == manifest_fps
    assert meta["measured_fps"] == pytest.approx(1.0 / np.diff(timestamps).mean())
    assert meta["measured_fps"] == pytest.approx(nominal_fps)
    assert "manifest_fps_differs_from_nominal_fps" in meta["quality_warnings"]
    assert "manifest_fps_differs_from_measured_fps" in meta["quality_warnings"]
    assert meta["fps_discrepancies"][0]["difference"] == pytest.approx(manifest_fps - nominal_fps)
    assert meta["timing_source_of_truth"].startswith("native parquet timestamps")


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "30", [30]])
def test_invalid_manifest_fps_still_fails(payload, fps):
    payload.episode["fps"] = fps
    payload.manifest.write_text(json.dumps({"selected_episodes": [payload.episode]}))
    with pytest.raises(importer.ImportStageError, match="manifest fps.*finite|manifest fps.*positive"):
        importer.import_manifest(payload.manifest, payload.root, payload.output)


def test_existing_stage_is_not_overwritten(payload):
    importer.import_manifest(payload.manifest, payload.root, payload.output)
    with pytest.raises(importer.ImportStageError, match="Output already exists"):
        importer.import_manifest(payload.manifest, payload.root, payload.output)


def test_parquet_dependency_is_optional_and_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(importer.ImportStageError, match="optional pyarrow"):
        importer.read_parquet_rows([Path("unused.parquet")])


def test_lazy_parquet_reader_uses_exact_paths(monkeypatch):
    package, parquet = ModuleType("pyarrow"), ModuleType("pyarrow.parquet")
    calls = []

    def read_table(path):
        calls.append(path)
        return SimpleNamespace(to_pylist=lambda: [{"from_file": path}])

    parquet.read_table = read_table
    package.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", package)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    paths = [Path("data/chunk-000/file-000.parquet"), Path("data/chunk-001/file-000.parquet")]
    assert importer.read_parquet_rows(paths) == [{"from_file": str(path)} for path in paths]
    assert calls == [str(path) for path in paths]


def test_cli_and_help_work_without_parquet_installed(payload, capsys):
    assert importer.main(["--manifest", str(payload.manifest), "--dataset-root", str(payload.root),
                          "--output-dir", str(payload.output)]) == 0
    assert len(json.loads(capsys.readouterr().out)["staged_episodes"]) == 1
    result = subprocess.run([sys.executable, str(MODULE_PATH), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--manifest" in result.stdout
