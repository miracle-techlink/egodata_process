"""Forecast errors against future estimator outputs, never GT accuracy."""

from __future__ import annotations

import numpy as np

from .data import target_indices, validate_horizons, validate_transforms


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None}


def evaluate_sequence(sequence, predictions, horizons=(0.5, 1.0), window_size=32, image_size=(640, 480)):
    horizons = validate_horizons(horizons)
    hands = predictions["hand_future"]
    objects = predictions["object_future"]
    count = len(sequence.timestamps)
    if not np.array_equal(predictions.get("timestamps"), sequence.timestamps):
        raise ValueError("prediction timestamps do not align with source observations")
    if not np.array_equal(predictions.get("horizons_seconds"), horizons):
        raise ValueError("prediction horizons do not match evaluation horizons")
    ready = np.asarray(predictions.get("ready"))
    if ready.shape != (count,) or ready.dtype != np.bool_:
        raise ValueError("ready must be an explicit boolean frame mask")
    if hands.shape != (count, len(horizons), 2, 21, 3):
        raise ValueError("hand prediction shape mismatch")
    if objects.shape != (count, len(horizons), sequence.objects.shape[1], 4, 4):
        raise ValueError("object prediction shape mismatch")
    validate_transforms(objects)
    results = []
    for horizon_index, horizon in enumerate(horizons):
        values = {"hand_xy_pixels": [], "object_translation_mm": [], "object_rotation_degrees": []}
        eligible_hands = eligible_objects = eligible_frames = joint_frames = 0
        for observation_index in range(window_size - 1, count):
            start = observation_index - window_size + 1
            if np.any(np.diff(sequence.timestamps[start:observation_index + 1]) > 1.5 / sequence.fps + 1e-9):
                continue
            targets = target_indices(sequence.timestamps, observation_index, [horizon])
            if targets is None:
                continue
            target = int(targets[0])
            reference_hands = sequence.hands[target]
            reference_objects = sequence.objects[target]
            current_hands = np.isfinite(sequence.hands[observation_index]).all(axis=(-2, -1))
            current_objects = np.isfinite(sequence.objects[observation_index]).all(axis=(-2, -1))
            hand_reference_valid = np.isfinite(reference_hands).all(axis=(-2, -1)) & current_hands
            object_reference_valid = np.isfinite(reference_objects).all(axis=(-2, -1)) & current_objects
            hand_prediction = hands[observation_index, horizon_index]
            object_prediction = objects[observation_index, horizon_index]
            hand_valid = hand_reference_valid & np.isfinite(hand_prediction).all(axis=(-2, -1)) & ready[observation_index]
            object_valid = object_reference_valid & np.isfinite(object_prediction).all(axis=(-2, -1)) & ready[observation_index]
            eligible_hands += int(hand_reference_valid.sum())
            eligible_objects += int(object_reference_valid.sum())
            eligible_frames += int(hand_reference_valid.any() and object_reference_valid.any())
            joint_frames += int(hand_valid.any() and object_valid.any())
            if hand_valid.any():
                errors = (hand_prediction[hand_valid, :, :2] - reference_hands[hand_valid, :, :2]) * image_size
                values["hand_xy_pixels"].extend(np.linalg.norm(errors, axis=-1).mean(axis=-1).tolist())
            if object_valid.any():
                predicted = object_prediction[object_valid]
                actual = reference_objects[object_valid]
                values["object_translation_mm"].extend((np.linalg.norm(predicted[:, :3, 3] - actual[:, :3, 3], axis=-1) * 1000).tolist())
                relative = predicted[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)
                cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
                values["object_rotation_degrees"].extend(np.degrees(np.arccos(cosine)).tolist())
        results.append({"horizon_seconds": float(horizon), "eligible_hand_targets": eligible_hands,
                        "eligible_object_targets": eligible_objects, "eligible_joint_frames": eligible_frames,
                        "predicted_joint_frames": joint_frames,
                        "joint_coverage": joint_frames / eligible_frames if eligible_frames else None,
                        **{name: _summary(entries) for name, entries in values.items()}})
    return {"sequence_id": sequence.sequence_id, "group_id": sequence.group_id,
            "frame_count": count, "horizons": results, "independent_accuracy": None,
            "reference": "future MediaPipe/FoundationPose estimates; pseudo-label consistency only",
            "rotation_caveat": "raw SO(3) error; symmetry not resolved, not absolute object orientation accuracy",
            "hand_depth_error": None, "metric_hand_camera_fusion": False}


def aggregate_reports(reports):
    if not reports:
        return {"sequences": 0, "groups": 0, "horizons": []}
    groups = [report["group_id"] for report in reports]
    if len(set(groups)) != len(groups):
        raise ValueError("episode-equal aggregation requires one camera/sequence per acquisition group")
    output = []
    for horizon_index, reference in enumerate(reports[0]["horizons"]):
        entries = [report["horizons"][horizon_index] for report in reports]
        if any(entry["horizon_seconds"] != reference["horizon_seconds"] for entry in entries):
            raise ValueError("inconsistent prediction horizons")
        combined = {"horizon_seconds": reference["horizon_seconds"]}
        for metric in ("hand_xy_pixels", "object_translation_mm", "object_rotation_degrees"):
            valid = [entry[metric] for entry in entries if entry[metric]["count"]]
            count = sum(entry["count"] for entry in valid)
            combined[metric] = {"count": count, "episodes_with_targets": len(valid),
                                "episode_equal_mean": float(np.mean([entry["mean"] for entry in valid])) if valid else None,
                                "sample_weighted_mean": sum(entry["mean"] * entry["count"] for entry in valid) / count if count else None}
        eligible = sum(entry["eligible_joint_frames"] for entry in entries)
        predicted = sum(entry["predicted_joint_frames"] for entry in entries)
        combined.update(eligible_joint_frames=eligible, predicted_joint_frames=predicted,
                        joint_coverage=predicted / eligible if eligible else None)
        output.append(combined)
    return {"sequences": len(reports), "groups": len(set(groups)), "horizons": output,
            "independent_accuracy": None}
