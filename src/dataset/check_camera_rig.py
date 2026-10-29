"""Compare quaternion conventions using a rigid stereo baseline, not image calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def quaternion_matrices(quaternions):
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Expected quaternion shape (frames, 4) in wxyz order")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms < 1e-8):
        raise ValueError("Quaternions must be finite and nonzero")
    scalar, xaxis, yaxis, zaxis = (values / norms[:, None]).T
    entries = [
        1 - 2 * (yaxis ** 2 + zaxis ** 2), 2 * (xaxis * yaxis - zaxis * scalar),
        2 * (xaxis * zaxis + yaxis * scalar), 2 * (xaxis * yaxis + zaxis * scalar),
        1 - 2 * (xaxis ** 2 + zaxis ** 2), 2 * (yaxis * zaxis - xaxis * scalar),
        2 * (xaxis * zaxis - yaxis * scalar), 2 * (yaxis * zaxis + xaxis * scalar),
        1 - 2 * (xaxis ** 2 + yaxis ** 2),
    ]
    return np.stack(entries, axis=1).reshape(-1, 3, 3)


def vector_summary(vectors):
    mean = vectors.mean(axis=0)
    residuals = np.linalg.norm(vectors - mean, axis=1)
    return {
        "mean_vector": mean.tolist(),
        "rms_from_mean": float(np.sqrt(np.mean(residuals ** 2))),
        "max_from_mean": float(residuals.max()),
    }


def compare_conventions(left_positions, right_positions, left_rotations):
    left_positions = np.asarray(left_positions, dtype=np.float64)
    right_positions = np.asarray(right_positions, dtype=np.float64)
    if left_positions.ndim != 2 or left_positions.shape[1] != 3:
        raise ValueError("Expected left positions shape (frames, 3)")
    if right_positions.shape != left_positions.shape or len(left_positions) < 3:
        raise ValueError("Need at least three paired camera positions")
    if not np.isfinite(left_positions).all() or not np.isfinite(right_positions).all():
        raise ValueError("Camera positions must be finite")
    matrices = quaternion_matrices(left_rotations)
    if len(matrices) != len(left_positions):
        raise ValueError("Quaternion and position frame counts differ")
    displacement = right_positions - left_positions
    baseline = np.linalg.norm(displacement, axis=1)
    if np.any(baseline < 1e-8):
        raise ValueError("Stereo cameras must have nonzero baseline")
    camera_to_world = vector_summary(np.einsum("nji,nj->ni", matrices, displacement))
    world_to_camera = vector_summary(np.einsum("nij,nj->ni", matrices, displacement))
    relative_rotation = np.einsum("ji,njk->nik", matrices[0], matrices)
    rotation_angle = np.arccos(np.clip((np.trace(relative_rotation, axis1=1, axis2=2) - 1) / 2, -1, 1))
    baseline_mean = float(baseline.mean())
    lower = min(camera_to_world["rms_from_mean"], world_to_camera["rms_from_mean"])
    higher = max(camera_to_world["rms_from_mean"], world_to_camera["rms_from_mean"])
    ratio = higher / max(lower, baseline_mean * 1e-12)
    informative = bool(rotation_angle.max() >= 0.02 and ratio >= 100 and lower / baseline_mean <= 0.001)
    favored = None
    if informative:
        favored = "camera_to_world" if camera_to_world["rms_from_mean"] < world_to_camera["rms_from_mean"] else "world_to_camera"
    return {
        "frame_count": len(left_positions),
        "coordinate_units": "native dataset units; physical scale not independently verified",
        "quaternion_norm_max_deviation": float(np.max(np.abs(np.linalg.norm(left_rotations, axis=1) - 1))),
        "baseline_mean": baseline_mean,
        "baseline_std": float(baseline.std()),
        "rotation_max_from_first_radians": float(rotation_angle.max()),
        "camera_to_world_hypothesis": camera_to_world,
        "world_to_camera_hypothesis": world_to_camera,
        "rigidity_rms_ratio": ratio,
        "favored_convention": favored,
        "heuristic_evidence_thresholds": {
            "minimum_rotation_radians": 0.02,
            "minimum_rms_ratio": 100,
            "maximum_relative_rigidity_rms": 0.001,
        },
        "projection_calibrated": False,
        "automatic_coordinate_conversion_authorized": False,
    }


def audit_rig(staging_root):
    episodes = []
    for directory in sorted(Path(staging_root).iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        meta_path, sequence_path = directory / "meta.json", directory / "sequence.npz"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("coordinate_system") != "world" or meta.get("quaternion_order") != ["qw", "qx", "qy", "qz"]:
            raise ValueError(f"Unknown coordinate convention: {directory.name}")
        with np.load(sequence_path, allow_pickle=False) as sequence:
            result = compare_conventions(
                sequence["head_left_camera_position"], sequence["head_right_camera_position"],
                sequence["head_left_camera_rotation"],
            )
        result.update(
            source_episode_uuid=meta["source_episode_uuid"],
            task_name=meta["task_name"],
            sequence_sha256=hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
            meta_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        )
        episodes.append(result)
    if not episodes:
        raise ValueError("No staged episodes found")
    return {
        "experiment": "stereo_rig_quaternion_convention_diagnostic",
        "staging_root": str(Path(staging_root).resolve()),
        "episode_count": len(episodes),
        "frame_count": sum(episode["frame_count"] for episode in episodes),
        "camera_to_world_favored_count": sum(episode["favored_convention"] == "camera_to_world" for episode in episodes),
        "world_to_camera_favored_count": sum(episode["favored_convention"] == "world_to_camera" for episode in episodes),
        "inconclusive_count": sum(episode["favored_convention"] is None for episode in episodes),
        "limitation": "Rigid baseline evidence only; intrinsics, distortion, optical axes and pixel reprojection remain unverified. No source arrays transformed.",
        "episodes": episodes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; select a new report path")
    report = audit_rig(args.staging_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
