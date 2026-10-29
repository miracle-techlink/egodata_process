import cv2
import numpy as np
import pytest

from src.dataset.render_fusion_acceptance import (
    calibrated_intrinsics,
    draw_skeleton,
    validate_video_alignment,
    validated_episode_payload,
    verified_source_file,
)


def row():
    return {"episode_uuid": "demo", "length": 10, "camera_intrinsics/head_left": [800, 800, 960, 728, 0, 0, 0, 0]}


def test_uses_annotated_calibration_not_fallback():
    assert calibrated_intrinsics(row(), "demo", 10) == {"fx": 800, "fy": 800, "cx": 960, "cy": 728}


@pytest.mark.parametrize("case", ["uuid", "count", "missing", "nonzero_distortion", "nan", "focal", "shape"])
def test_rejects_untrusted_calibration(case):
    data = row()
    if case == "uuid":
        data["episode_uuid"] = "different"
    elif case == "count":
        data["length"] = 9
    elif case == "missing":
        del data["camera_intrinsics/head_left"]
    elif case == "nonzero_distortion":
        data["camera_intrinsics/head_left"][4] = .1
    elif case == "nan":
        data["camera_intrinsics/head_left"][0] = float("nan")
    elif case == "focal":
        data["camera_intrinsics/head_left"][0] = -1
    elif case == "shape":
        data["camera_intrinsics/head_left"] = [800, 800, 960, 728]
    with pytest.raises(ValueError):
        calibrated_intrinsics(data, "demo", 10)


def test_nan_and_huge_points_do_not_draw_or_overflow():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for value in (float("nan"), float("inf"), 1e99):
        draw_skeleton(image, np.full((21, 2), value), (255, 0, 0))
    assert np.count_nonzero(image) == 0


def test_visible_points_render_real_geometry():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    points = np.column_stack((np.arange(21) + 30, np.arange(21) + 20))
    draw_skeleton(image, points, (255, 0, 0))
    assert cv2.countNonZero(image[:, :, 0]) > 0


def test_payload_provenance_does_not_replace_resolved_read_path(tmp_path):
    payload = tmp_path / "episode.parquet"
    payload.write_bytes(b"small-test-payload")
    source = verified_source_file(payload, {"path": "other/root/episode.parquet", "role": "episodes"})
    assert source["path"] == str(payload.resolve())
    assert source["relative_path"] == "other/root/episode.parquet"
    assert source["payload_verified"] is True


EPISODE_UUID = "d057beac-dab1-43a1-9192-6b8b06abddaf"
OTHER_UUID = "ce22bfbe-3daa-4cac-b43a-7c202f6c87ee"
PAYLOAD_SUFFIXES = {
    "episodes": "meta/episodes/chunk-000/file-000.parquet",
    "head_left_video": "videos/observation.images.head_left/chunk-000/file-000.mp4",
}


def episode_payload_fixture(tmp_path, role="head_left_video", variant="EgoProStandard", task="Adjust Shelf Labels"):
    root = tmp_path / "dataset"
    base = f"{variant}/lerobot/{task}/{EPISODE_UUID}"
    episode = {
        "source_episode_uuid": EPISODE_UUID,
        "task_name": task,
        "variant": variant,
        "lerobot_path": base,
    }
    record = {"role": role, "path": f"{base}/{PAYLOAD_SUFFIXES[role]}"}
    payload = root / record["path"]
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"synthetic path-validation fixture, not a real video")
    return episode, record, root


@pytest.mark.parametrize("variant", ["EgoProStandard", "EgoProStandard-body"])
@pytest.mark.parametrize("role", ["episodes", "head_left_video"])
def test_payload_binding_accepts_exact_annotated_episode_paths(tmp_path, role, variant):
    episode, record, root = episode_payload_fixture(tmp_path, role=role, variant=variant)
    assert validated_episode_payload(episode, record, root) == (root / record["path"]).resolve()


@pytest.mark.parametrize("role", ["episodes", "head_left_video"])
def test_payload_binding_accepts_real_task_name_with_slash(tmp_path, role):
    episode, record, root = episode_payload_fixture(
        tmp_path, role=role, task="Roller Blackout Blind Raising/Lowering",
    )
    assert validated_episode_payload(episode, record, root) == (root / record["path"]).resolve()


@pytest.mark.parametrize("case", [
    "head_right", "other_uuid", "other_task", "other_variant", "wrong_chunk",
    "wrong_file", "wrong_role", "unsupported_role", "absolute_path", "root_escape",
])
def test_payload_binding_rejects_wrong_video_identity_or_suffix(tmp_path, case):
    episode, record, root = episode_payload_fixture(tmp_path)
    if case == "head_right":
        record["path"] = record["path"].replace("observation.images.head_left", "observation.images.head_right")
    elif case == "other_uuid":
        record["path"] = record["path"].replace(EPISODE_UUID, OTHER_UUID)
    elif case == "other_task":
        record["path"] = record["path"].replace("Adjust Shelf Labels", "Different Task")
    elif case == "other_variant":
        record["path"] = record["path"].replace("EgoProStandard/", "EgoProStandard-body/")
    elif case == "wrong_chunk":
        record["path"] = record["path"].replace("chunk-000", "chunk-001")
    elif case == "wrong_file":
        record["path"] = record["path"].replace("file-000.mp4", "file-001.mp4")
    elif case == "wrong_role":
        record["role"] = "episodes"
    elif case == "unsupported_role":
        record["role"] = "head_right_video"
    elif case == "absolute_path":
        record["path"] = str(root / record["path"])
    else:
        record["path"] = "../" + record["path"]
    with pytest.raises(ValueError):
        validated_episode_payload(episode, record, root)


@pytest.mark.parametrize("key,value", [
    ("source_episode_uuid", OTHER_UUID),
    ("task_name", "Different Task"),
    ("variant", "EgoProStandard-body"),
    ("variant", "EgoProRaw"),
    ("lerobot_path", f"EgoProStandard/lerobot/Adjust Shelf Labels/{OTHER_UUID}"),
])
def test_payload_binding_rejects_manifest_identity_disagreement(tmp_path, key, value):
    episode, record, root = episode_payload_fixture(tmp_path)
    episode[key] = value
    with pytest.raises(ValueError):
        validated_episode_payload(episode, record, root)


def test_payload_binding_rejects_calibration_from_another_uuid(tmp_path):
    episode, record, root = episode_payload_fixture(tmp_path, role="episodes")
    record["path"] = record["path"].replace(EPISODE_UUID, OTHER_UUID)
    with pytest.raises(ValueError):
        validated_episode_payload(episode, record, root)


@pytest.mark.parametrize("task", ["../other", "/absolute/task"])
def test_payload_binding_rejects_unsafe_task_even_with_matching_record(tmp_path, task):
    episode, record, root = episode_payload_fixture(tmp_path)
    episode["task_name"] = task
    episode["lerobot_path"] = f"EgoProStandard/lerobot/{task}/{EPISODE_UUID}"
    record["path"] = f"{episode['lerobot_path']}/{PAYLOAD_SUFFIXES['head_left_video']}"
    with pytest.raises(ValueError):
        validated_episode_payload(episode, record, root)


def test_payload_binding_rejects_symlink_escape(tmp_path):
    episode, record, root = episode_payload_fixture(tmp_path)
    payload = root / record["path"]
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside dataset root")
    payload.unlink()
    payload.symlink_to(outside)
    with pytest.raises(ValueError):
        validated_episode_payload(episode, record, root)


@pytest.fixture
def aligned_video_fixture():
    timestamps = np.arange(6, dtype=np.float64) / 30
    sequence = {"timestamps": timestamps.copy(), "frame_index": np.arange(6, dtype=np.int64)}
    video = {"pts": timestamps.copy(), "decoded_frame_count": 6, "stream": {}, "method": "synthetic ffprobe result"}
    return sequence, video


def test_alignment_accepts_exact_zero_origin_pts(aligned_video_fixture):
    sequence, video = aligned_video_fixture
    report = validate_video_alignment(sequence, video)
    assert report["decoded_frame_count"] == 6
    assert report["max_absolute_pts_error_seconds"] == 0
    assert report["frame_index_zero_based_sequential"] is True


def test_alignment_accepts_pts_rounding_below_tolerance(aligned_video_fixture):
    sequence, video = aligned_video_fixture
    video["pts"] += np.array([0, 5e-5, -5e-5, 2e-5, -2e-5, 0])
    report = validate_video_alignment(sequence, video)
    assert report["max_absolute_pts_error_seconds"] == pytest.approx(5e-5)
    assert report["tolerance_seconds"] == 1e-4


def test_alignment_accepts_absolute_tolerance_boundary(aligned_video_fixture):
    sequence, video = aligned_video_fixture
    sequence["timestamps"][0] = 1e-4
    report = validate_video_alignment(sequence, video)
    assert report["max_absolute_pts_error_seconds"] == 1e-4


@pytest.mark.parametrize("case", [
    "timestamps_plus_four", "pts_plus_four", "both_plus_four", "index_starts_100",
    "dropped_index", "duplicate_index", "float_index", "index_shape", "index_length",
    "native_nan", "native_inf", "native_nonmonotonic", "native_shape", "empty",
    "pts_nan", "pts_inf", "pts_nonmonotonic", "pts_shape", "pts_count",
    "decoded_count", "timestamp_drift", "above_tolerance",
])
def test_alignment_rejects_frame_or_absolute_time_mismatch(aligned_video_fixture, case):
    sequence, video = aligned_video_fixture
    if case == "timestamps_plus_four":
        sequence["timestamps"] += 4
    elif case == "pts_plus_four":
        video["pts"] += 4
    elif case == "both_plus_four":
        sequence["timestamps"] += 4
        video["pts"] += 4
    elif case == "index_starts_100":
        sequence["frame_index"] += 100
    elif case == "dropped_index":
        sequence["frame_index"][3:] += 1
    elif case == "duplicate_index":
        sequence["frame_index"][3] = 2
    elif case == "float_index":
        sequence["frame_index"] = sequence["frame_index"].astype(np.float64)
    elif case == "index_shape":
        sequence["frame_index"] = sequence["frame_index"][:, None]
    elif case == "index_length":
        sequence["frame_index"] = sequence["frame_index"][:-1]
    elif case == "native_nan":
        sequence["timestamps"][2] = np.nan
    elif case == "native_inf":
        sequence["timestamps"][2] = np.inf
    elif case == "native_nonmonotonic":
        sequence["timestamps"][2] = sequence["timestamps"][1]
    elif case == "native_shape":
        sequence["timestamps"] = sequence["timestamps"][:, None]
    elif case == "empty":
        sequence["timestamps"] = np.array([])
        sequence["frame_index"] = np.array([], dtype=np.int64)
        video["pts"] = np.array([])
        video["decoded_frame_count"] = 0
    elif case == "pts_nan":
        video["pts"][2] = np.nan
    elif case == "pts_inf":
        video["pts"][2] = np.inf
    elif case == "pts_nonmonotonic":
        video["pts"][2] = video["pts"][1]
    elif case == "pts_shape":
        video["pts"] = video["pts"][:, None]
    elif case == "pts_count":
        video["pts"] = video["pts"][:-1]
    elif case == "decoded_count":
        video["decoded_frame_count"] -= 1
    elif case == "timestamp_drift":
        sequence["timestamps"] *= 1.01
    else:
        sequence["timestamps"][0] = 1e-4 + 1e-8
    with pytest.raises(ValueError):
        validate_video_alignment(sequence, video)


def test_alignment_validation_does_not_change_inputs(aligned_video_fixture):
    sequence, video = aligned_video_fixture
    before = {key: values.copy() for key, values in sequence.items()}
    pts_before = video["pts"].copy()
    validate_video_alignment(sequence, video)
    for key, values in before.items():
        np.testing.assert_array_equal(sequence[key], values)
    np.testing.assert_array_equal(video["pts"], pts_before)
