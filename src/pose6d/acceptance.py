"""Fail-closed numerical and provenance gate, not a pose accuracy evaluator."""

import json
import re
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
_UNAVAILABLE = {"", "none", "null", "unknown", "unspecified", "auto", "base", "n/a"}
_SIMULATED = ("mock", "synthetic", "dummy", "placeholder", "simulated", "fake")


def _issue(report, code, message, blocked=False):
    report["issues"].append({"code": code, "severity": "blocked" if blocked else "error", "message": message})
    if not blocked or report["status"] != "not_accepted":
        report["status"] = "blocked" if blocked else "not_accepted"


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except OverflowError:
        return False


def _unusable_source(value):
    return not _text(value) or value.strip().lower() in _UNAVAILABLE or any(
        marker in value.lower() for marker in (*_SIMULATED, "heuristic")
    )


def _identity(report, evidence, label, episode_uuid, camera_id):
    if evidence.get("source_episode_uuid") != episode_uuid:
        _issue(report, f"{label}_episode_mismatch", f"{label}.source_episode_uuid must match the requested acquisition.")
    if evidence.get("camera_id") != camera_id:
        _issue(report, f"{label}_camera_mismatch", f"{label}.camera_id must match the requested camera.")


def _reference(report, value, label, episode_uuid):
    if _unusable_source(value):
        _issue(report, f"{label}_reference_missing", f"{label} needs an explicit, non-simulated evidence reference.", blocked=True)
    elif any(found.lower() != episode_uuid.lower() for found in _UUID_PATTERN.findall(value)):
        _issue(report, f"{label}_reference_episode_mismatch", f"{label} reference names another acquisition.")


def _metadata(report, meta, episode_uuid, camera_id, frame_count, object_ids):
    previous_issues = len(report["issues"])
    for key, expected in (("num_frames", frame_count), ("num_objects", len(object_ids))):
        if not _integer(meta.get(key)) or meta[key] != expected:
            _issue(report, f"meta_{key}_mismatch", f"meta.{key} must equal {expected}.")
    if meta.get("object_ids") != object_ids:
        _issue(report, "meta_object_ids_mismatch", "Metadata object_ids must match NPZ row order exactly.")
    for key in ("width", "height"):
        if not _integer(meta.get(key)) or meta[key] <= 0:
            _issue(report, f"invalid_{key}", f"meta.{key} must be a positive integer.")
    if not _number(meta.get("fps")) or meta["fps"] <= 0:
        _issue(report, "invalid_fps", "meta.fps must be finite and positive.")
    if "bbox_undetected_value" in meta and (
        not _integer(meta["bbox_undetected_value"]) or meta["bbox_undetected_value"] != -1
    ):
        _issue(report, "invalid_bbox_sentinel", "Only the -1 undetected bbox sentinel is supported.")

    for key in ("pose_estimator", "depth_source"):
        value = meta.get(key)
        report[key] = value if isinstance(value, str) else None
        if _unusable_source(value):
            _issue(report, f"invalid_{key}", f"meta.{key} must name an actual non-mock source.")
    if isinstance(meta.get("depth_source"), str) and "relative" in meta["depth_source"].lower():
        _issue(report, "relative_depth", "Relative depth does not establish a shared metric coordinate system.")

    flags = meta.get("quality_flags", [])
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        _issue(report, "invalid_quality_flags", "quality_flags must be a list of strings when present.")
    elif any(any(marker in flag.lower() for marker in (*_SIMULATED, "heuristic_intrinsics", "relative_depth", "no_depth")) for flag in flags):
        _issue(report, "disqualifying_quality_flags", "quality_flags report simulated poses, unsupported depth, or heuristic intrinsics.")

    intrinsics = meta.get("intrinsics")
    if not isinstance(intrinsics, dict):
        _issue(report, "missing_intrinsics", "Explicit intrinsics with provenance are required.", blocked=True)
    else:
        _reference(report, intrinsics.get("source"), "intrinsics", episode_uuid)
        if not all(_number(intrinsics.get(key)) for key in ("fx", "fy", "cx", "cy")) or any(
            _number(intrinsics.get(key)) and intrinsics[key] <= 0 for key in ("fx", "fy")
        ):
            _issue(report, "invalid_intrinsics", "fx/fy must be positive; all four intrinsics must be finite numbers.")

    _reference(report, meta.get("source_clip"), "source_clip", episode_uuid)
    for key, expected in (("source_episode_uuid", episode_uuid), ("camera_id", camera_id)):
        if key in meta and meta[key] != expected:
            _issue(report, f"meta_{key}_mismatch", f"meta.{key} contradicts the requested acquisition/camera.")
    if "pose_convention" in meta and meta["pose_convention"] != "object_to_camera":
        _issue(report, "meta_pose_convention_mismatch", "Poses must map object coordinates into the camera frame.")

    evidence = meta.get("acceptance_evidence")
    if not isinstance(evidence, dict):
        _issue(report, "missing_acceptance_evidence", "Reviewed acquisition, metric-depth, calibration and time-alignment evidence is required.", blocked=True)
        return False
    _identity(report, evidence, "evidence", episode_uuid, camera_id)
    if evidence.get("pose_convention") != "object_to_camera":
        _issue(report, "pose_convention_unconfirmed", "Evidence must explicitly confirm object_to_camera matrices.", blocked=True)

    depth = evidence.get("depth")
    if not isinstance(depth, dict):
        _issue(report, "missing_depth_evidence", "Depth backend name alone cannot establish metric depth.", blocked=True)
    else:
        report["depth_kind"] = depth.get("kind") if isinstance(depth.get("kind"), str) else None
        if depth.get("kind") not in ("estimated_metric", "measured_metric") or depth.get("units") != "m":
            _issue(report, "unconfirmed_metric_depth", "Depth evidence must distinguish estimated_metric/measured_metric in meters.", blocked=True)
        _reference(report, depth.get("reference"), "depth", episode_uuid)

    for label in ("calibration", "time_alignment"):
        item = evidence.get(label)
        if not isinstance(item, dict):
            _issue(report, f"missing_{label}_evidence", f"{label} evidence is required.", blocked=True)
            continue
        if item.get("confirmed") is not True:
            _issue(report, f"{label}_unconfirmed", f"{label}.confirmed must be true after external review.", blocked=True)
        _reference(report, item.get("reference"), label, episode_uuid)
        _identity(report, item, label, episode_uuid, camera_id)
        if label == "calibration":
            size = item.get("image_size")
            if not isinstance(size, list) or len(size) != 2 or not all(_integer(value) for value in size) or size != [meta.get("width"), meta.get("height")]:
                _issue(report, "calibration_image_size_mismatch", "Calibration image_size [width,height] must match the pose image dimensions.")
        elif not _integer(item.get("frame_count")) or item["frame_count"] != frame_count:
            _issue(report, "time_alignment_frame_count_mismatch", "Time-alignment evidence must cover the complete requested frame sequence.")
    return len(report["issues"]) == previous_issues


def _matrices(report, poses):
    missing = np.isnan(poses).all(axis=(-2, -1))
    finite = np.isfinite(poses).all(axis=(-2, -1))
    malformed = ~(missing | finite)
    if malformed.any():
        _issue(report, "nonfinite_or_partial_nan_pose", f"{int(malformed.sum())} poses contain partial NaNs or infinities; only all-NaN missing poses are allowed.")
    valid = np.zeros(poses.shape[:2], dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        matrices = poses[finite].astype(np.float64)
        representable = np.isfinite(matrices).all(axis=(-2, -1))
        rotations = matrices[:, :3, :3]
        orthogonal = np.isclose(rotations.swapaxes(-2, -1) @ rotations, np.eye(3), atol=1e-4, rtol=0).all(axis=(-2, -1))
        proper = np.isclose(np.linalg.det(rotations), 1, atol=1e-4, rtol=0)
    homogeneous = np.isclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-5, rtol=0).all(axis=-1)
    positive_depth = matrices[:, 2, 3] > 0
    for code, mask in (("unrepresentable_pose", representable), ("nonorthogonal_rotation", orthogonal), ("improper_rotation", proper), ("invalid_homogeneous_row", homogeneous), ("nonpositive_depth", positive_depth)):
        if not mask.all():
            _issue(report, code, f"{int((~mask).sum())} finite poses fail {code}.")
    valid[finite] = representable & orthogonal & proper & homogeneous & positive_depth
    report["frames_with_valid_matrix"] = int(valid.any(axis=0).sum())
    return valid, finite


def _bboxes(report, bboxes, finite_pose, meta):
    missing = (bboxes == -1).all(axis=-1)
    valid = (bboxes >= 0).all(axis=-1) & (bboxes[..., 2] > bboxes[..., 0]) & (bboxes[..., 3] > bboxes[..., 1])
    if all(_integer(meta.get(key)) and meta[key] > 0 for key in ("width", "height")):
        valid &= (bboxes[..., (0, 2)] < meta["width"]).all(axis=-1) & (bboxes[..., (1, 3)] < meta["height"]).all(axis=-1)
    if (~(missing | valid)).any():
        _issue(report, "invalid_bboxes", "Bboxes must be nonempty in-image xyxy boxes or entirely -1.")
    if (finite_pose & missing).any():
        _issue(report, "pose_without_bbox", "A finite pose cannot have an undetected bbox.")
    return valid


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def audit_object_pose(npz_path, meta_path, expected_episode_uuid, expected_frame_count, expected_camera_id="head_left"):
    """Audit existing poses/object_ids/bboxes without loading pickle or changing files.

    The writer's existing metadata is retained. A separate, caller-reviewed
    ``meta.acceptance_evidence`` object is required by this audit only::

        {
          "source_episode_uuid": "<expected UUID>", "camera_id": "head_left",
          "pose_convention": "object_to_camera",
          "depth": {"kind": "estimated_metric", "units": "m", "reference": "<report/checkpoint>"},
          "calibration": {"confirmed": true, "reference": "<calibration report>",
            "source_episode_uuid": "<expected UUID>", "camera_id": "head_left",
            "image_size": [640, 480]},
          "time_alignment": {"confirmed": true, "reference": "<alignment report>",
            "source_episode_uuid": "<expected UUID>", "camera_id": "head_left", "frame_count": 100}
        }

    Depth kind is estimated_metric or measured_metric; neither is independent
    pose ground truth. Calibration must describe the actual undistorted image
    geometry; alignment must establish every pose frame's correspondence to the
    target video, not assume Raw and annotated frame indices match. References
    identify externally reviewed evidence: this function checks declarations and
    consistency, not the referenced files or whether the estimator really ran.

    Status is blocked (missing prerequisites/no usable poses), not_accepted
    (invalid artifacts/contradictions), or coverage_only. There is no accuracy
    acceptance status: accuracy_evaluated is always false and both success rates
    remain null, even if metadata claims ground truth. Missing files are blocked.
    frames_total is the requested sequence length; frames_with_real_pose counts
    frames with at least one finite, rigid, positive-depth pose and valid bbox
    only after all provenance checks pass. Local invalid matrices still reject
    the artifact while other valid frames can contribute diagnostic coverage.
    All-NaN poses are missing, not errors; detections may exist without poses.
    SO(3) absolute tolerance is 1e-4; homogeneous-row tolerance is 1e-5.
    Synthetic fixtures can test this gate but do not establish real-world results.
    """
    report = {
        "status": "coverage_only", "issues": [],
        "frames_total": int(expected_frame_count) if _integer(expected_frame_count) and expected_frame_count > 0 else 0,
        "frames_with_real_pose": 0, "frames_with_valid_matrix": 0, "coverage": 0.0,
        "accuracy_success_rate": None, "joint_success_rate": None, "accuracy_evaluated": False,
        "pose_estimator": None, "depth_source": None, "depth_kind": None,
    }
    if not _text(expected_episode_uuid) or not _text(expected_camera_id) or report["frames_total"] == 0:
        _issue(report, "invalid_request", "Expected acquisition/camera and a positive integer frame count are required.", blocked=True)
        return report
    try:
        paths = (Path(npz_path), Path(meta_path))
        for label, path in zip(("npz", "meta"), paths):
            if not path.is_file():
                _issue(report, f"missing_{label}", f"{label} file is missing or is not a regular file: {path}", blocked=True)
    except (OSError, TypeError, ValueError) as error:
        _issue(report, "invalid_path", str(error), blocked=True)
    if report["issues"]:
        return report
    try:
        with paths[1].open(encoding="utf-8") as stream:
            meta = json.load(stream, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
        if not isinstance(meta, dict):
            raise ValueError("Metadata must be a JSON object.")
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        _issue(report, "invalid_meta", str(error))
        return report
    try:
        loaded = np.load(paths[0], allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("Expected a named NPZ archive, not an NPY array.")
        with loaded as archive:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError("Duplicate NPZ keys.")
            arrays = {key: archive[key] for key in archive.files}
        if not all(isinstance(array, np.ndarray) for array in arrays.values()):
            raise ValueError("NPZ members must all be NumPy arrays.")
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("Object arrays/pickle are forbidden, including optional fields.")
    except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
        _issue(report, "invalid_or_unsafe_npz", str(error))
        return report
    if not all(key in arrays for key in ("poses", "object_ids", "bboxes")):
        _issue(report, "missing_arrays", "NPZ must contain poses, object_ids and bboxes.")
        return report
    poses, object_ids, bboxes = (arrays[key] for key in ("poses", "object_ids", "bboxes"))
    if object_ids.ndim != 1 or object_ids.dtype.kind not in "US":
        _issue(report, "invalid_object_ids", "object_ids must be a one-dimensional non-pickle string array.")
        return report
    try:
        ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in object_ids.tolist()]
    except UnicodeError as error:
        _issue(report, "invalid_object_ids", str(error))
        return report
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        _issue(report, "invalid_object_ids", "Object IDs must be nonempty and unique.")
        return report
    if poses.ndim != 4 or poses.shape[0] != len(ids) or poses.shape[2:] != (4, 4) or poses.dtype.kind != "f":
        _issue(report, "invalid_poses_schema", "poses must be a floating (N,T,4,4) array matching object_ids.")
        return report
    report["observed_frame_count"] = int(poses.shape[1])
    if poses.shape[1] != expected_frame_count:
        _issue(report, "frame_count_mismatch", "NPZ sequence length does not match the requested sequence.")
        return report
    if bboxes.shape != poses.shape[:2] + (4,) or bboxes.dtype.kind not in "iu":
        _issue(report, "invalid_bboxes_schema", "bboxes must be an integer (N,T,4) array matching poses.")
        return report
    metadata_ok = _metadata(report, meta, expected_episode_uuid, expected_camera_id, expected_frame_count, ids)
    valid_pose, finite_pose = _matrices(report, poses)
    valid_bbox = _bboxes(report, bboxes, finite_pose, meta)
    if metadata_ok:
        report["frames_with_real_pose"] = int((valid_pose & valid_bbox).any(axis=0).sum())
        report["coverage"] = report["frames_with_real_pose"] / report["frames_total"]
    if report["status"] == "coverage_only" and not report["frames_with_real_pose"]:
        _issue(report, "no_usable_pose", "No usable object pose is present; all-NaN arrays are not a successful estimate.", blocked=True)
    return report
