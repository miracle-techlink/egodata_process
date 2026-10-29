"""Synthetic fixtures test fail-closed gate logic, never real pose accuracy."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from src.pose6d.acceptance import audit_object_pose


EPISODE_UUID = "d057beac-dab1-43a1-9192-6b8b06abddaf"
OTHER_UUID = "ce22bfbe-3daa-4cac-b43a-7c202f6c87ee"
FRAME_COUNT = 4


@pytest.fixture
def synthetic_gate_fixture(tmp_path):
    poses = np.tile(np.eye(4, dtype=np.float32), (2, FRAME_COUNT, 1, 1))
    poses[..., 2, 3] = 1.5
    arrays = {
        "poses": poses,
        "object_ids": np.array(["obj_000", "obj_001"]),
        "bboxes": np.tile(np.array([10, 20, 110, 120], dtype=np.int32), (2, FRAME_COUNT, 1)),
    }
    identity = {"source_episode_uuid": EPISODE_UUID, "camera_id": "head_left"}
    meta = {
        "source_clip": f"/fixtures/{EPISODE_UUID}/head_left.mp4",
        "num_frames": FRAME_COUNT, "num_objects": 2, "object_ids": arrays["object_ids"].tolist(),
        "width": 640, "height": 480, "fps": 30,
        "pose_estimator": "foundationpose", "depth_source": "depth_anything_v2",
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "source": "dataset_calibration.json"},
        "bbox_undetected_value": -1,
        "acceptance_evidence": {
            **identity, "pose_convention": "object_to_camera",
            "depth": {"kind": "estimated_metric", "units": "m", "reference": "metric_checkpoint_manifest.json"},
            "calibration": {**identity, "confirmed": True, "reference": "calibration_review.json", "image_size": [640, 480]},
            "time_alignment": {**identity, "confirmed": True, "reference": "frame_alignment_review.json", "frame_count": FRAME_COUNT},
        },
    }
    return tmp_path, arrays, meta


def run_gate(fixture, **kwargs):
    directory, arrays, meta = fixture
    npz_path, meta_path = directory / "pose.npz", directory / "pose.meta.json"
    np.savez(npz_path, **arrays)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = audit_object_pose(npz_path, meta_path, EPISODE_UUID, FRAME_COUNT, **kwargs)
    assert result["accuracy_success_rate"] is None
    assert result["joint_success_rate"] is None
    assert result["accuracy_evaluated"] is False
    json.dumps(result, allow_nan=False)
    return result


def issue_codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_synthetic_fixture_only_certifies_gate_coverage(synthetic_gate_fixture):
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "coverage_only"
    assert result["issues"] == []
    assert result["frames_total"] == FRAME_COUNT
    assert result["frames_with_real_pose"] == FRAME_COUNT
    assert result["coverage"] == 1.0
    assert result["depth_kind"] == "estimated_metric"


def test_coverage_counts_frames_not_object_detections(synthetic_gate_fixture):
    _, arrays, _ = synthetic_gate_fixture
    arrays["poses"][:, 0] = np.nan
    arrays["bboxes"][:, 0] = -1
    arrays["poses"][0, 1] = np.nan
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "coverage_only"
    assert result["frames_with_real_pose"] == 3
    assert result["coverage"] == 0.75


@pytest.mark.parametrize("empty_objects", [False, True])
def test_no_poses_is_blocked_not_success(synthetic_gate_fixture, empty_objects):
    _, arrays, meta = synthetic_gate_fixture
    arrays["poses"][:] = np.nan
    if empty_objects:
        for key in arrays:
            arrays[key] = arrays[key][:0]
        meta["object_ids"] = []
        meta["num_objects"] = 0
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "blocked"
    assert result["coverage"] == 0
    assert "no_usable_pose" in issue_codes(result)


@pytest.mark.parametrize("missing", ["npz", "meta", "both"])
def test_missing_artifacts_block_without_exception(synthetic_gate_fixture, missing):
    directory, _, _ = synthetic_gate_fixture
    run_gate(synthetic_gate_fixture)
    for name in ("pose.npz", "pose.meta.json"):
        if missing == "both" or (name.endswith("npz") == (missing == "npz")):
            (directory / name).unlink()
    result = audit_object_pose(directory / "pose.npz", directory / "pose.meta.json", EPISODE_UUID, FRAME_COUNT)
    assert result["status"] == "blocked"
    assert result["frames_total"] == FRAME_COUNT
    assert result["frames_with_real_pose"] == 0
    assert result["accuracy_success_rate"] is None


@pytest.mark.parametrize("estimator", ["mock", "MoCk", "foundationpose_mock", "synthetic", "none", "unknown", "", None, 42])
def test_rejects_mock_even_without_quality_flags(synthetic_gate_fixture, estimator):
    _, _, meta = synthetic_gate_fixture
    assert "quality_flags" not in meta
    meta["pose_estimator"] = estimator
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "not_accepted"
    assert result["frames_with_valid_matrix"] == FRAME_COUNT
    assert result["frames_with_real_pose"] == 0
    assert "invalid_pose_estimator" in issue_codes(result)


@pytest.mark.parametrize("source", [None, "none", "unknown", "relative_depth", "depth_anything_v2_relative", "mock_depth"])
def test_depth_source_must_be_actual_and_metric(synthetic_gate_fixture, source):
    _, _, meta = synthetic_gate_fixture
    meta["depth_source"] = source
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "not_accepted"
    assert result["coverage"] == 0


@pytest.mark.parametrize("source", [None, "heuristic", "heuristic_from_image", "unknown", ""])
def test_intrinsics_without_real_provenance_block(synthetic_gate_fixture, source):
    _, _, meta = synthetic_gate_fixture
    meta["intrinsics"]["source"] = source
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] != "coverage_only"
    assert result["coverage"] == 0


@pytest.mark.parametrize("location", ["evidence", "calibration", "time_alignment", "top_level", "source_clip", "reference"])
def test_mixed_acquisition_rejected(synthetic_gate_fixture, location):
    _, _, meta = synthetic_gate_fixture
    evidence = meta["acceptance_evidence"]
    if location == "evidence":
        evidence["source_episode_uuid"] = OTHER_UUID
    elif location in ("calibration", "time_alignment"):
        evidence[location]["source_episode_uuid"] = OTHER_UUID
    elif location == "top_level":
        meta["source_episode_uuid"] = OTHER_UUID
    elif location == "source_clip":
        meta["source_clip"] = meta["source_clip"].replace(EPISODE_UUID, OTHER_UUID)
    else:
        evidence["calibration"]["reference"] = f"calibration/{OTHER_UUID}/review.json"
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "not_accepted"
    assert result["coverage"] == 0


@pytest.mark.parametrize("location", ["evidence", "calibration", "time_alignment", "top_level", "expected"])
def test_camera_mismatch_rejected(synthetic_gate_fixture, location):
    _, _, meta = synthetic_gate_fixture
    evidence = meta["acceptance_evidence"]
    kwargs = {}
    if location == "expected":
        kwargs["expected_camera_id"] = "head_right"
    elif location == "top_level":
        meta["camera_id"] = "head_right"
    elif location == "evidence":
        evidence["camera_id"] = "head_right"
    else:
        evidence[location]["camera_id"] = "head_right"
    result = run_gate(synthetic_gate_fixture, **kwargs)
    assert result["status"] == "not_accepted"
    assert result["coverage"] == 0


@pytest.mark.parametrize("case", ["missing", "calibration", "time_alignment", "depth", "confirmation", "string_confirmation", "reference", "metric_kind", "units", "convention"])
def test_evidence_is_not_inferred_from_backend_name(synthetic_gate_fixture, case):
    _, _, meta = synthetic_gate_fixture
    evidence = meta["acceptance_evidence"]
    if case == "missing":
        del meta["acceptance_evidence"]
    elif case in ("calibration", "time_alignment", "depth"):
        del evidence[case]
    elif case in ("confirmation", "string_confirmation"):
        evidence["time_alignment"]["confirmed"] = False if case == "confirmation" else "true"
    elif case == "reference":
        del evidence["time_alignment"]["reference"]
    elif case == "metric_kind":
        evidence["depth"]["kind"] = "relative"
    elif case == "units":
        evidence["depth"]["units"] = "unknown"
    else:
        evidence["pose_convention"] = "camera_to_object"
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "blocked"
    assert result["coverage"] == 0


@pytest.mark.parametrize("case", ["scale", "shear", "reflection", "homogeneous", "negative_depth", "zero_depth", "partial_nan", "inf", "all_inf"])
def test_invalid_matrices_reject_even_if_other_objects_are_valid(synthetic_gate_fixture, case):
    _, arrays, _ = synthetic_gate_fixture
    pose = arrays["poses"][0, 0]
    if case == "scale":
        pose[0, 0] = 2
    elif case == "shear":
        pose[0, 1] = 0.2
    elif case == "reflection":
        pose[0, 0] = -1
    elif case == "homogeneous":
        pose[3, 2] = 0.1
    elif case == "negative_depth":
        pose[2, 3] = -1
    elif case == "zero_depth":
        pose[2, 3] = 0
    elif case == "partial_nan":
        pose[0, 0] = np.nan
    elif case == "inf":
        pose[0, 0] = np.inf
    else:
        pose[:] = np.inf
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "not_accepted"
    assert result["frames_with_real_pose"] == FRAME_COUNT


def test_nontrivial_rotation_and_roundoff_are_allowed(synthetic_gate_fixture):
    _, arrays, _ = synthetic_gate_fixture
    angle = 0.4
    arrays["poses"][..., :3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    arrays["poses"][0, 0, 3, 0] = 1e-7
    assert run_gate(synthetic_gate_fixture)["status"] == "coverage_only"


@pytest.mark.parametrize("case", ["short", "pose_shape", "pose_count", "pose_int", "bbox_shape", "bbox_float", "ids_shape", "ids_duplicate", "ids_empty", "ids_numeric", "missing"])
def test_npz_schema_mismatches_rejected(synthetic_gate_fixture, case):
    _, arrays, _ = synthetic_gate_fixture
    if case == "short":
        arrays["poses"] = arrays["poses"][:, :-1]
    elif case == "pose_shape":
        arrays["poses"] = arrays["poses"][..., :3]
    elif case == "pose_count":
        arrays["poses"] = arrays["poses"][:1]
    elif case == "pose_int":
        arrays["poses"] = arrays["poses"].astype(np.int32)
    elif case == "bbox_shape":
        arrays["bboxes"] = arrays["bboxes"][:, :-1]
    elif case == "bbox_float":
        arrays["bboxes"] = arrays["bboxes"].astype(np.float32)
    elif case == "ids_shape":
        arrays["object_ids"] = arrays["object_ids"][None]
    elif case == "ids_duplicate":
        arrays["object_ids"][1] = arrays["object_ids"][0]
    elif case == "ids_empty":
        arrays["object_ids"][0] = ""
    elif case == "ids_numeric":
        arrays["object_ids"] = np.array([1, 2])
    else:
        del arrays["poses"]
    assert run_gate(synthetic_gate_fixture)["status"] == "not_accepted"


@pytest.mark.parametrize("key,value", [("num_frames", 5), ("num_objects", 1), ("object_ids", ["obj_001", "obj_000"]), ("width", 0), ("height", True), ("fps", 0), ("bbox_undetected_value", 0)])
def test_writer_metadata_must_match_arrays(synthetic_gate_fixture, key, value):
    _, _, meta = synthetic_gate_fixture
    meta[key] = value
    assert run_gate(synthetic_gate_fixture)["status"] == "not_accepted"


@pytest.mark.parametrize("case", ["image_size", "frame_count", "bool_frame_count", "bad_fx", "missing_intrinsics", "bad_flags", "mock_flag", "heuristic_flag"])
def test_metadata_contradictions_cannot_authorize_coverage(synthetic_gate_fixture, case):
    _, _, meta = synthetic_gate_fixture
    if case == "image_size":
        meta["acceptance_evidence"]["calibration"]["image_size"] = [1280, 960]
    elif case in ("frame_count", "bool_frame_count"):
        meta["acceptance_evidence"]["time_alignment"]["frame_count"] = 5 if case == "frame_count" else True
    elif case == "bad_fx":
        meta["intrinsics"]["fx"] = -1
    elif case == "missing_intrinsics":
        del meta["intrinsics"]
    elif case == "bad_flags":
        meta["quality_flags"] = "ok"
    else:
        meta["quality_flags"] = ["mock_pose" if case == "mock_flag" else "heuristic_intrinsics"]
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] != "coverage_only"
    assert result["coverage"] == 0


@pytest.mark.parametrize("bbox", [[-1, 20, 110, 120], [-2, -2, -2, -2], [110, 20, 10, 120], [10, 20, 10, 120], [10, 20, 640, 120], [10, 20, 110, 480], [-1, -1, -1, -1]])
def test_invalid_bbox_or_finite_pose_with_missing_bbox_rejected(synthetic_gate_fixture, bbox):
    _, arrays, _ = synthetic_gate_fixture
    arrays["bboxes"][0, 0] = bbox
    assert run_gate(synthetic_gate_fixture)["status"] == "not_accepted"


@pytest.mark.parametrize("field", ["object_ids", "poses", "bboxes", "extra"])
def test_pickle_arrays_rejected_even_in_optional_fields(synthetic_gate_fixture, field):
    _, arrays, _ = synthetic_gate_fixture
    arrays[field] = np.array([{"payload": "must never unpickle"}], dtype=object)
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "not_accepted"
    assert "invalid_or_unsafe_npz" in issue_codes(result)


@pytest.mark.parametrize("bad_json", ["{", "[]", '{"fps": NaN}', '{"pose_estimator":"mock","pose_estimator":"foundationpose"}'])
def test_invalid_or_ambiguous_json_rejected(synthetic_gate_fixture, bad_json):
    directory, _, _ = synthetic_gate_fixture
    run_gate(synthetic_gate_fixture)
    (directory / "pose.meta.json").write_text(bad_json, encoding="utf-8")
    result = audit_object_pose(directory / "pose.npz", directory / "pose.meta.json", EPISODE_UUID, FRAME_COUNT)
    assert result["status"] == "not_accepted"
    assert "invalid_meta" in issue_codes(result)


@pytest.mark.parametrize("raw", [b"not an archive", b"PK\x03\x04bad", b"\x80\x04N."])
def test_corrupt_or_pickled_files_rejected(synthetic_gate_fixture, raw):
    directory, _, _ = synthetic_gate_fixture
    run_gate(synthetic_gate_fixture)
    (directory / "pose.npz").write_bytes(raw)
    result = audit_object_pose(directory / "pose.npz", directory / "pose.meta.json", EPISODE_UUID, FRAME_COUNT)
    assert result["status"] == "not_accepted"


@pytest.mark.parametrize("frame_count", [0, -1, 1.5, True, None])
def test_invalid_expected_count_blocks(synthetic_gate_fixture, frame_count):
    directory, _, _ = synthetic_gate_fixture
    result = audit_object_pose(directory / "pose.npz", directory / "pose.meta.json", EPISODE_UUID, frame_count)
    assert result["status"] == "blocked"
    assert "invalid_request" in issue_codes(result)


def test_accuracy_claims_in_meta_never_become_accuracy_results(synthetic_gate_fixture):
    _, _, meta = synthetic_gate_fixture
    meta["accuracy_evaluated"] = True
    meta["accuracy_success_rate"] = 1.0
    meta["joint_success_rate"] = 1.0
    meta["acceptance_evidence"]["depth"]["kind"] = "measured_metric"
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "coverage_only"
    assert result["depth_kind"] == "measured_metric"


def _unpickle_marker(path):
    Path(path).write_text("UNSAFE UNPICKLE", encoding="utf-8")


class _PickleProbe:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return _unpickle_marker, (str(self.marker),)


def test_pickle_payload_is_never_executed(synthetic_gate_fixture):
    directory, arrays, _ = synthetic_gate_fixture
    marker = directory / "unpickle_should_never_run.txt"
    arrays["extra"] = np.array([_PickleProbe(marker)], dtype=object)
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "not_accepted"
    assert not marker.exists()


def test_non_array_zip_member_rejected_without_exception(synthetic_gate_fixture):
    directory, _, _ = synthetic_gate_fixture
    run_gate(synthetic_gate_fixture)
    with ZipFile(directory / "pose.npz", "a") as archive:
        archive.writestr("unexpected.txt", "not an array")
    result = audit_object_pose(directory / "pose.npz", directory / "pose.meta.json", EPISODE_UUID, FRAME_COUNT)
    assert result["status"] == "not_accepted"
    assert "invalid_or_unsafe_npz" in issue_codes(result)


def test_duplicate_npz_members_rejected(synthetic_gate_fixture):
    directory, _, _ = synthetic_gate_fixture
    run_gate(synthetic_gate_fixture)
    with ZipFile(directory / "pose.npz", "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("poses.npy", archive.read("poses.npy"))
    result = audit_object_pose(directory / "pose.npz", directory / "pose.meta.json", EPISODE_UUID, FRAME_COUNT)
    assert result["status"] == "not_accepted"


def test_npy_file_does_not_impersonate_npz(synthetic_gate_fixture):
    directory, arrays, _ = synthetic_gate_fixture
    run_gate(synthetic_gate_fixture)
    with (directory / "pose.npz").open("wb") as stream:
        np.save(stream, arrays["poses"])
    result = audit_object_pose(directory / "pose.npz", directory / "pose.meta.json", EPISODE_UUID, FRAME_COUNT)
    assert result["status"] == "not_accepted"


def test_extreme_json_integer_is_rejected_without_exception(synthetic_gate_fixture):
    _, _, meta = synthetic_gate_fixture
    meta["intrinsics"]["fx"] = 10 ** 400
    assert run_gate(synthetic_gate_fixture)["status"] == "not_accepted"


def test_float64_overflow_cannot_create_valid_pose(synthetic_gate_fixture):
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("Platform has no extended floating type.")
    _, arrays, _ = synthetic_gate_fixture
    arrays["poses"] = arrays["poses"].astype(np.longdouble)
    arrays["poses"][..., 0, 3] = np.finfo(np.longdouble).max
    result = run_gate(synthetic_gate_fixture)
    assert result["status"] == "not_accepted"
    assert result["coverage"] == 0


def test_gate_never_modifies_artifacts(synthetic_gate_fixture):
    directory, _, _ = synthetic_gate_fixture
    run_gate(synthetic_gate_fixture)
    paths = [directory / "pose.npz", directory / "pose.meta.json"]
    before = [path.read_bytes() for path in paths]
    audit_object_pose(*paths, EPISODE_UUID, FRAME_COUNT)
    assert [path.read_bytes() for path in paths] == before
