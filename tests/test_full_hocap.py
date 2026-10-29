"""Synthetic CPU contract tests; no downloads, model inference or frozen-data writes."""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

from src.pose6d import full_hocap


def write_label(path, intrinsics, segmentation, object_names):
    with path.open("wb") as stream:
        np.savez(
            stream,
            cam_K=intrinsics,
            seg_mask=segmentation,
            obj_class_names=np.asarray(object_names),
            obj_class_inds=np.asarray([41, 73], dtype=np.int64),
            obj_poses=np.full((2, 4, 4), np.nan, dtype=np.float32),
        )


@pytest.fixture
def acquisition(tmp_path):
    root = tmp_path / "acquisition"
    subject_id = "subject_1"
    acquisition_id = "20261022_123456"
    camera_id = "105322251564"
    object_ids = ["003_cracker_box", "025_mug"]
    captures = root / "captures_primary" / subject_id / acquisition_id / camera_id
    captures.mkdir(parents=True)
    metadata = root / "labels_selected" / f"{subject_id}-{acquisition_id}.meta.yaml"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(yaml.safe_dump({"num_frames": 3, "object_ids": object_ids}), encoding="utf-8")
    intrinsics = np.array([[20.0, 0.0, 4.0], [0.0, 25.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    segmentation = np.zeros((12, 16), dtype=np.uint8)
    segmentation[2:7, 2:5] = 1
    segmentation[5:8, 3:7] = 1
    segmentation[3:9, 10:14] = 2
    segmentation[5:7, 11:13] = 0
    for frame in range(3):
        prefix = captures / f"{camera_id}-{frame:06d}"
        image = np.full((12, 16, 3), [10 + frame, 20, 30], dtype=np.uint8)
        depth = np.full((12, 16), 1000 + frame, dtype=np.uint16)
        assert cv2.imwrite(str(prefix) + ".color.jpg", image)
        assert cv2.imwrite(str(prefix) + ".depth.png", depth)
        write_label(Path(str(prefix) + ".label.npy"), intrinsics, segmentation, object_ids)
    models = root / "models"
    models.mkdir()
    mesh = "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    for object_id in object_ids:
        (models / f"{object_id}.textured_mesh.obj").write_text(mesh, encoding="utf-8")
    return {
        "acquisition_root": root,
        "subject_id": subject_id,
        "acquisition_id": acquisition_id,
        "camera_id": camera_id,
        "output_dir": tmp_path / "prepared",
        "captures": captures,
        "metadata": metadata,
        "intrinsics": intrinsics,
        "segmentation": segmentation,
        "object_ids": object_ids,
    }


def prepare(acquisition, **overrides):
    arguments = {key: acquisition[key] for key in
                 ("acquisition_root", "subject_id", "acquisition_id", "camera_id", "output_dir")}
    arguments.update(overrides)
    return full_hocap.prepare_hocap(**arguments)


def capture_path(acquisition, frame, suffix):
    return acquisition["captures"] / f"{acquisition['camera_id']}-{frame:06d}.{suffix}"


def source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_prepare_full_three_frames_and_preserve_unknown_source_time(acquisition):
    before = source_hashes(acquisition["acquisition_root"])
    manifest = prepare(acquisition)
    assert isinstance(manifest, dict)
    assert manifest["source_frame_count"] == 3
    assert manifest["frame_ids"] == [0, 1, 2]
    assert [frame["frame_id"] for frame in manifest["frames"]] == [0, 1, 2]
    assert [item["object_id"] for item in manifest["objects"]] == acquisition["object_ids"]
    for frame in manifest["frames"]:
        for role, suffix in (("rgb", "color.jpg"), ("depth", "depth.png")):
            asset = frame[role]
            expected = capture_path(acquisition, frame["frame_id"], suffix)
            assert Path(asset["path"]).resolve() == expected.resolve()
            assert asset["size_bytes"] == expected.stat().st_size
            assert asset["sha256"] == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert manifest["source_timestamp_s"] == [None, None, None]
    assert manifest["source_timestamp_valid"] is False
    saved = json.loads((acquisition["output_dir"] / "scene_manifest.json").read_text(encoding="utf-8"))
    assert saved["source_frame_count"] == 3
    assert saved["frame_ids"] == [0, 1, 2]
    assert saved["frames"] == manifest["frames"]
    assert saved["source_timestamp_s"] == [None, None, None]
    assert saved["source_timestamp_valid"] is False
    assert source_hashes(acquisition["acquisition_root"]) == before


def test_masks_follow_local_class_index_plus_one_not_global_class_indices(acquisition):
    prepare(acquisition)
    for index, object_id in enumerate(acquisition["object_ids"]):
        mask_path = acquisition["output_dir"] / "masks" / f"{object_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        assert mask is not None
        assert mask.shape == acquisition["segmentation"].shape
        np.testing.assert_array_equal(mask > 0, acquisition["segmentation"] == index + 1)


def test_first_label_only_accesses_intrinsics_segmentation_and_class_names(acquisition, monkeypatch):
    first_label = capture_path(acquisition, 0, "label.npy")
    with np.load(first_label, allow_pickle=False) as archive:
        archive_type = type(archive)
    original_getitem = archive_type.__getitem__
    accessed = []

    def checked_getitem(archive, key):
        assert key in {"cam_K", "seg_mask", "obj_class_names"}, f"Forbidden reference field accessed: {key}"
        accessed.append(key)
        return original_getitem(archive, key)

    monkeypatch.setattr(archive_type, "__getitem__", checked_getitem)
    prepare(acquisition)
    assert set(accessed) == {"cam_K", "seg_mask", "obj_class_names"}


def test_preparation_does_not_require_global_class_indices_or_reference_poses(acquisition):
    with capture_path(acquisition, 0, "label.npy").open("wb") as stream:
        np.savez(stream, cam_K=acquisition["intrinsics"], seg_mask=acquisition["segmentation"],
                 obj_class_names=np.asarray(acquisition["object_ids"]))
    manifest = prepare(acquisition)
    assert manifest["source_timestamp_s"] == [None, None, None]
    assert (acquisition["output_dir"] / "scene_manifest.json").is_file()


@pytest.mark.parametrize("suffix", ["color.jpg", "depth.png", "label.npy"])
@pytest.mark.parametrize("frame", [0, 1, 2])
def test_missing_any_frame_modality_is_rejected_without_truncation(acquisition, suffix, frame):
    capture_path(acquisition, frame, suffix).unlink()
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        prepare(acquisition)


@pytest.mark.parametrize("suffix", ["color.jpg", "depth.png", "label.npy"])
def test_extra_frame_modality_beyond_metadata_is_rejected(acquisition, suffix):
    capture_path(acquisition, 3, suffix).write_bytes(capture_path(acquisition, 2, suffix).read_bytes())
    with pytest.raises((ValueError, RuntimeError)):
        prepare(acquisition)


def test_later_label_contents_are_not_read_for_preparation(acquisition):
    for frame in (1, 2):
        path = capture_path(acquisition, frame, "label.npy")
        path.write_bytes(b"Later reference data must not enter preparation or inference")
    manifest = prepare(acquisition)
    assert manifest["source_frame_count"] == 3
    assert manifest["frame_ids"] == [0, 1, 2]


def test_matching_source_streams_still_reject_interior_gap_and_extra_tail(acquisition):
    for suffix in ("color.jpg", "depth.png", "label.npy"):
        capture_path(acquisition, 1, suffix).rename(capture_path(acquisition, 3, suffix))
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        prepare(acquisition)


@pytest.mark.parametrize("depth_scale", [0.0, -0.001, float("nan"), float("inf")])
def test_depth_scale_must_be_finite_and_positive(acquisition, depth_scale):
    with pytest.raises(ValueError):
        prepare(acquisition, depth_scale=depth_scale)


@pytest.mark.parametrize("missing_class", [1, 2])
def test_empty_initial_object_mask_is_rejected(acquisition, missing_class):
    segmentation = acquisition["segmentation"].copy()
    segmentation[segmentation == missing_class] = 0
    write_label(capture_path(acquisition, 0, "label.npy"), acquisition["intrinsics"],
                segmentation, acquisition["object_ids"])
    with pytest.raises((ValueError, RuntimeError)):
        prepare(acquisition)


@pytest.mark.parametrize("kind", ["empty_directory", "nonempty_directory", "file"])
def test_existing_output_is_rejected_and_never_overwritten(acquisition, kind):
    output = acquisition["output_dir"]
    if kind == "file":
        output.write_bytes(b"keep this existing file")
        before = output.read_bytes()
    else:
        output.mkdir()
        if kind == "nonempty_directory":
            (output / "scene_manifest.json").write_bytes(b"preserve previous result")
        before = source_hashes(output)
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        prepare(acquisition)
    assert (output.read_bytes() if output.is_file() else source_hashes(output)) == before


def test_nonuniform_source_timestamps_are_kept_exactly(acquisition, tmp_path):
    values = [15.0, 15.041, 15.173]
    timestamps = tmp_path / "source_timestamps.json"
    timestamps.write_text(json.dumps(values), encoding="utf-8")
    before = timestamps.read_bytes()
    manifest = prepare(acquisition, source_timestamps=timestamps)
    assert manifest["source_timestamp_valid"] is True
    assert manifest["source_timestamp_s"] == values
    saved = json.loads((acquisition["output_dir"] / "scene_manifest.json").read_text(encoding="utf-8"))
    assert saved["source_timestamp_s"] == values
    assert saved["source_timestamp_valid"] is True
    assert timestamps.read_bytes() == before


@pytest.mark.parametrize("values", [
    [], [0.0, 0.1], [0.0, 0.1, 0.2, 0.3], [0.0, 0.0, 0.1], [0.2, 0.1, 0.0],
    [0.0, float("nan"), 0.2], [0.0, 0.1, float("inf")], [float("-inf"), 0.1, 0.2],
])
def test_invalid_source_timestamps_are_rejected_not_replaced_by_default_fps(acquisition, tmp_path, values):
    timestamps = tmp_path / "invalid_timestamps.json"
    timestamps.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare(acquisition, source_timestamps=timestamps)


@pytest.fixture
def surface_inputs():
    landmarks = np.full((2, 21, 3), np.nan, dtype=np.float32)
    landmarks[0] = [0.5, 0.5, 0.75]
    depth = np.ones((12, 16), dtype=np.float64)
    intrinsics = np.array([[20.0, 0.0, 4.0], [0.0, 25.0, 3.0], [0.0, 0.0, 1.0]])
    return landmarks, depth, intrinsics


def test_surface_unprojects_metric_median_depth_and_leaves_missing_hand_nan(surface_inputs):
    landmarks, depth, intrinsics = surface_inputs
    originals = [value.copy() for value in surface_inputs]
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert points.shape == (2, 21, 3)
    assert valid.shape == (2, 21)
    assert valid.dtype == np.bool_
    assert valid[0].all() and not valid[1].any()
    np.testing.assert_allclose(points[0], np.tile([0.2, 0.12, 1.0], (21, 1)), atol=1e-6)
    assert np.isnan(points[1]).all()
    for original, actual in zip(originals, surface_inputs):
        np.testing.assert_array_equal(actual, original)


def test_surface_uses_width_height_not_width_minus_one_and_preserves_subpixel_ray(surface_inputs):
    landmarks, depth, intrinsics = surface_inputs
    landmarks[0] = [0.53125, 0.375, -999.0]
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert valid[0].all()
    np.testing.assert_allclose(points[0], np.tile([0.225, 0.06, 1.0], (21, 1)), atol=1e-6)


@pytest.mark.parametrize("coordinate,value", [
    (0, -0.0001), (0, 1.0), (0, 1.1), (1, -0.0001), (1, 1.0), (1, 1.1),
    (0, float("nan")), (1, float("inf")), (2, float("nan")),
])
def test_invalid_landmark_is_nan_without_invalidating_other_joints(surface_inputs, coordinate, value):
    landmarks, depth, intrinsics = surface_inputs
    landmarks[0, 0, coordinate] = value
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert not valid[0, 0]
    assert np.isnan(points[0, 0]).all()
    assert valid[0, 1:].all()


def test_zero_coordinate_uses_clipped_window_without_wrapping(surface_inputs):
    landmarks, depth, intrinsics = surface_inputs
    landmarks[0] = [0.0, 0.0, 0.0]
    depth[-2:, :] = 2.0
    depth[:, -2:] = 2.0
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert valid[0].all()
    np.testing.assert_allclose(points[0], np.tile([-0.2, -0.12, 1.0], (21, 1)), atol=1e-6)


@pytest.mark.parametrize("value", [0.0, -1.0, 0.05, 3.01, float("nan"), float("inf")])
def test_invalid_or_out_of_range_depth_produces_only_nan(surface_inputs, value):
    landmarks, depth, intrinsics = surface_inputs
    depth.fill(value)
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert not valid.any()
    assert np.isnan(points).all()


@pytest.mark.parametrize("value", [0.1, 3.0])
def test_depth_range_endpoints_are_inclusive(surface_inputs, value):
    landmarks, depth, intrinsics = surface_inputs
    depth.fill(value)
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert valid[0].all()
    np.testing.assert_allclose(points[0, :, 2], value)


@pytest.mark.parametrize("positive_count", [0, 1, 2, 3])
def test_window_needs_at_least_three_positive_depth_samples(surface_inputs, positive_count):
    landmarks, depth, intrinsics = surface_inputs
    depth.fill(0)
    for row, column in [(6, 8), (6, 7), (6, 9)][:positive_count]:
        depth[row, column] = 1.0
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert bool(valid[0].all()) is (positive_count >= 3)
    if positive_count >= 3:
        np.testing.assert_allclose(points[0, :, 2], 1.0)
    else:
        assert np.isnan(points).all()


def test_radius_controls_depth_neighborhood(surface_inputs):
    landmarks, depth, intrinsics = surface_inputs
    depth.fill(0)
    depth[6, 8] = depth[4, 8] = depth[8, 8] = 1.0
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics, radius=1)
    assert not valid.any()
    assert np.isnan(points).all()
    _, wider_valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics, radius=2)
    assert wider_valid[0].all()


def test_median_and_percentile_spread_ignore_single_extreme_positive_outlier(surface_inputs):
    landmarks, depth, intrinsics = surface_inputs
    depth[6, 8] = 2.9
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert valid[0].all()
    np.testing.assert_allclose(points[0, :, 2], 1.0)


def test_spread_guard_rejects_mixed_depth_surfaces_and_obeys_threshold(surface_inputs):
    landmarks, depth, intrinsics = surface_inputs
    depth[4:9, 6:11] = np.array([1.0] * 12 + [1.2] * 13).reshape(5, 5)
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics)
    assert not valid.any()
    assert np.isnan(points).all()
    points, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics, max_spread_m=0.3)
    assert valid[0].all()
    np.testing.assert_allclose(points[0, :, 2], 1.2)


def test_custom_depth_bounds_are_applied(surface_inputs):
    landmarks, depth, intrinsics = surface_inputs
    depth.fill(0.075)
    _, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics, min_depth_m=0.05)
    assert valid[0].all()
    depth.fill(3.5)
    _, valid = full_hocap.depth_surface_points(landmarks, depth, intrinsics, max_depth_m=4.0)
    assert valid[0].all()


def proper_pose():
    angle = 0.4
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    pose[:3, 3] = [0.2, -0.1, 1.25]
    return pose


def test_validate_pose_accepts_so3_and_returns_independent_copy():
    pose = proper_pose()
    original = pose.copy()
    checked = full_hocap.validate_pose(pose)
    assert isinstance(checked, np.ndarray)
    assert checked.shape == (4, 4)
    np.testing.assert_allclose(checked, original, atol=1e-6)
    assert not np.shares_memory(checked, pose)
    checked[0, 3] = 100
    np.testing.assert_array_equal(pose, original)


@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (4, 3), (5, 5), (16,)])
def test_validate_pose_rejects_wrong_shape(shape):
    with pytest.raises(ValueError):
        full_hocap.validate_pose(np.ones(shape, dtype=np.float64))


@pytest.mark.parametrize("depth", [0.0, -0.001])
def test_validate_pose_rejects_nonpositive_camera_z(depth):
    pose = proper_pose()
    pose[2, 3] = depth
    with pytest.raises(ValueError):
        full_hocap.validate_pose(pose)


@pytest.mark.parametrize("rotation", [
    np.diag([-1.0, 1.0, 1.0]),
    np.diag([2.0, 1.0, 1.0]),
    np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.zeros((3, 3)),
])
def test_validate_pose_rejects_reflection_scale_shear_and_singular_rotation(rotation):
    pose = proper_pose()
    pose[:3, :3] = rotation
    with pytest.raises(ValueError):
        full_hocap.validate_pose(pose)


@pytest.mark.parametrize("row", [[0, 0, 0, 0], [0, 0, 0, 2], [0.1, 0, 0, 1]])
def test_validate_pose_rejects_nonhomogeneous_last_row(row):
    pose = proper_pose()
    pose[3] = row
    with pytest.raises(ValueError):
        full_hocap.validate_pose(pose)


@pytest.mark.parametrize("index,value", [
    ((0, 0), float("nan")), ((0, 3), float("nan")),
    ((1, 3), float("inf")), ((2, 3), float("inf")),
])
def test_validate_pose_rejects_nonfinite_rotation_or_translation(index, value):
    pose = proper_pose()
    pose[index] = value
    with pytest.raises(ValueError):
        full_hocap.validate_pose(pose)


@pytest.mark.parametrize("source_time_known", [False, True])
@pytest.mark.parametrize("source_time_consistent", [False, True])
@pytest.mark.parametrize("test_only", [False, True])
def test_audit_cli_serializes_numpy_check_and_preserves_gates(
    tmp_path, monkeypatch, capsys, source_time_known, source_time_consistent, test_only
):
    """Synthetic artifacts exercise the report/NPZ/CSV/video audit, never inference."""
    count = 3
    root = tmp_path / "synthetic_audit"
    root.mkdir()
    manifest_path = root / "scene_manifest.json"
    full_hocap.write_json(manifest_path, {
        "source_frame_count": count,
        "objects": [{"object_id": "synthetic_object"}],
        "source_timestamp_valid": source_time_known,
        "image_shape": [24, 32],
    })
    pose = proper_pose().astype(np.float32)
    poses = np.broadcast_to(pose, (1, count, 4, 4)).copy()
    left = np.full((count, 21, 3), 0.5, dtype=np.float32)
    right = np.full_like(left, np.nan)
    timestamps = np.arange(count, dtype=np.float64) / 30
    if not source_time_known:
        timestamps[:] = np.nan
    if not source_time_consistent:
        timestamps[0] = np.nan if source_time_known else 0.0
    np.savez(root / "sequence.npz",
             frame_ids=np.arange(count), object_poses=poses,
             object_valid=np.ones((1, count), dtype=bool),
             object_projection_visible=np.ones((1, count), dtype=bool),
             left_hand=left, right_hand=right,
             hand_valid=np.tile([True, False], (count, 1)),
             hand_camera_surface_m=np.full((count, 2, 21, 3), np.nan),
             hand_camera_surface_valid=np.zeros((count, 2, 21), dtype=bool),
             source_timestamp_s=timestamps,
             source_timestamp_valid=np.full(count, source_time_known, dtype=bool),
             processing_elapsed_s=np.arange(count, dtype=np.float64) + 1)
    (root / "timing.csv").write_text("frame_id\n0\n1\n2\n", encoding="utf-8")
    writer = cv2.VideoWriter(str(root / "overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    assert writer.isOpened(), "Synthetic audit fixture needs the mp4v encoder"
    try:
        for index in range(count):
            writer.write(np.full((24, 32, 3), 40 + index * 30, dtype=np.uint8))
    finally:
        writer.release()
    full_hocap.write_json(root / "report.json", {
        "fixture": "SYNTHETIC TEST ONLY; no model inference",
        "scene_manifest": full_hocap.file_record(manifest_path),
        "status": "complete", "frames_completed": count,
        "test_only": test_only,
        "object_backend": {"real_model": not test_only},
        "hand_backend": {"test_only": test_only},
        "outputs": [full_hocap.file_record(root / name)
                    for name in ("sequence.npz", "timing.csv", "overlay.mp4")],
    })
    before = source_hashes(root)
    monkeypatch.setattr(sys, "argv", ["full_hocap", "audit", "--output-dir", str(root)])
    expected_pass = source_time_consistent and not test_only
    if expected_pass:
        full_hocap.main()
    else:
        with pytest.raises(SystemExit) as error:
            full_hocap.main()
        assert error.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
    assert saved == printed
    assert saved["engineering_pass"] is expected_pass
    assert saved["checks"]["unknown_source_time_not_fabricated"] is source_time_consistent
    assert all(type(value) is bool for value in saved["checks"].values())
    assert saved["decoded_video_frames"] == count
    assert saved["coverage"]["hand_any_frames"] == count
    assert saved["formal_training_allowed"] is False
    assert saved["accuracy"] is None
    assert saved["independent_accuracy_pass"] is None
    after = source_hashes(root)
    assert {name: after[name] for name in before} == before
    acceptance_bytes = (root / "acceptance.json").read_bytes()
    with pytest.raises(FileExistsError, match="do not overwrite"):
        full_hocap.audit_output(root)
    assert (root / "acceptance.json").read_bytes() == acceptance_bytes
