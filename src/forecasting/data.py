"""Validated observation loading and acquisition-isolated forecast samples."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.dataset.build_samples import object_features


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_horizons(horizons):
    result = np.asarray(horizons, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("horizons must be a nonempty finite vector")
    if np.any(result <= 0) or np.any(np.diff(result) <= 0):
        raise ValueError("horizons must be positive and strictly increasing")
    return result


def validate_transforms(poses):
    poses = np.asarray(poses)
    if poses.shape[-2:] != (4, 4):
        raise ValueError("poses must end in (4,4)")
    finite = np.isfinite(poses).all(axis=(-2, -1))
    missing = np.isnan(poses).all(axis=(-2, -1))
    if not (finite | missing).all():
        raise ValueError("a pose must be entirely finite or entirely NaN")
    valid = poses[finite]
    if len(valid):
        rotations = valid[:, :3, :3]
        if not np.allclose(valid[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("invalid homogeneous matrix last row")
        if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-3):
            raise ValueError("non-orthonormal rotation")
        if not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
            raise ValueError("rotation determinant is not +1")
    return finite


def history_features(objects, bboxes, timestamps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) != len(objects) or np.any(np.diff(timestamps) <= 0) or not np.isfinite(timestamps).all():
        raise ValueError("feature timestamps must align and be strictly increasing")
    features = object_features(objects.transpose(1, 0, 2, 3), bboxes.transpose(1, 0, 2), 1.0)[0]
    valid = np.isfinite(objects).all(axis=(-2, -1))
    if len(timestamps) > 1:
        both = valid[1:] & valid[:-1]
        velocity = np.diff(objects[:, :, :3, 3], axis=0) / np.diff(timestamps)[:, None, None]
        features[1:, :, 10:13] = np.where(both[..., None], velocity, 0)
        features[~valid] = np.nan
    return features


@dataclass
class PoseSequence:
    sequence_id: str
    group_id: str
    hands: np.ndarray
    objects: np.ndarray
    bboxes: np.ndarray
    timestamps: np.ndarray
    metadata: dict
    records: list
    observation_dir: Path
    scene_dir: Path | None = None

    def __post_init__(self):
        count = len(self.timestamps)
        if not self.sequence_id or not self.group_id:
            raise ValueError("sequence_id and acquisition group_id are required")
        if self.hands.shape != (count, 2, 21, 3):
            raise ValueError("hands must be (T,2,21,3)")
        if self.objects.ndim != 4 or self.objects.shape[0] != count:
            raise ValueError("objects must be (T,N,4,4)")
        if self.bboxes.shape != self.objects.shape[:2] + (4,):
            raise ValueError("bboxes must be (T,N,4)")
        if count < 2 or not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("at least two strictly increasing finite timestamps required")
        hand_finite = np.isfinite(self.hands).all(axis=(-2, -1))
        if not (hand_finite | np.isnan(self.hands).all(axis=(-2, -1))).all():
            raise ValueError("each hand must be all finite or all NaN")
        validate_transforms(self.objects)
        if len(self.records) != count:
            raise ValueError("frame records and observation lengths differ")

    @property
    def fps(self):
        return 1.0 / float(np.median(np.diff(self.timestamps)))

    def features(self):
        return history_features(self.objects, self.bboxes, self.timestamps)


def load_sequence(observation_dir, sequence_id, group_id, scene_dir=None):
    root = Path(observation_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("hand_coordinate_space") != "image_normalized_xy_relative_z":
        raise ValueError("this prototype requires explicit mixed-coordinate observations")
    if metadata.get("viewpoint") != "third_person_fixed_camera":
        raise ValueError("moving-camera sequences require an explicit stabilization adapter")
    if metadata.get("pose_convention") != "original_mesh_object_to_camera":
        raise ValueError("object coordinate convention not supported")
    hand_backend = metadata.get("hand_backend", {})
    if (not metadata.get("object_backend", {}).get("real_model") or hand_backend.get("test_only")
            or hand_backend.get("backend") != "mediapipe.solutions.hands.Hands"
            or not hand_backend.get("backend_version") or not hand_backend.get("model_files")):
        raise ValueError("real estimator provenance required; injected fixtures are not accepted")
    if metadata.get("mesh_units") != "m_source_declared":
        raise ValueError("source-declared metric mesh units required")
    with np.load(root / "hand_poses.npz", allow_pickle=False) as hands:
        hand_values = np.stack([hands["left_hand"], hands["right_hand"]], axis=1)
    with np.load(root / "object_poses.npz", allow_pickle=False) as objects:
        poses = objects["poses"].transpose(1, 0, 2, 3)
        bboxes = objects["bboxes"].transpose(1, 0, 2)
    with (root / "frames.csv").open() as source:
        rows = list(csv.DictReader(source))
    timestamps = np.asarray([float(row["timestamp_ms"]) / 1000 for row in rows])
    if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("observation frame indices are not contiguous")
    if metadata.get("num_frames") != len(rows):
        raise ValueError("metadata frame count mismatch")
    records = metadata.get("frames", [])
    if len(records) != len(rows):
        raise ValueError("source frame records do not match observations")
    for record, row in zip(records, rows):
        if record["stem"] != row["stem"]:
            raise ValueError("source frame ID mismatch")
    metadata["observation_hashes"] = {name: file_hash(root / name) for name in
                                       ("hand_poses.npz", "object_poses.npz", "frames.csv", "metadata.json")}
    return PoseSequence(sequence_id, group_id, hand_values, poses, bboxes, timestamps,
                        metadata, records, root, Path(scene_dir) if scene_dir else None)


def target_indices(timestamps, observation_index, horizons, gap_factor=1.5):
    horizons = validate_horizons(horizons)
    times = np.asarray(timestamps, dtype=np.float64)
    if observation_index < 0 or observation_index >= len(times):
        raise ValueError("observation index outside timestamps")
    cadence = np.median(np.diff(times))
    requested = times[observation_index] + horizons
    indices = np.searchsorted(times, requested - 1e-9, side="left")
    if np.any(indices >= len(times)):
        return None
    if np.any(times[indices] - requested > cadence * gap_factor + 1e-9):
        return None
    if np.any(np.diff(times[observation_index:indices[-1] + 1]) > cadence * gap_factor + 1e-9):
        return None
    return indices


def build_samples(sequence, window_size=32, horizons=(0.5, 1.0), stride=1):
    if window_size < 2 or stride < 1:
        raise ValueError("window_size >= 2 and stride >= 1 required")
    horizons = validate_horizons(horizons)
    samples = []
    rejected = {"no_future_or_gap": 0, "history_gap": 0, "no_joint_target": 0}
    for end in range(window_size - 1, len(sequence.timestamps), stride):
        indices = target_indices(sequence.timestamps, end, horizons)
        if indices is None:
            rejected["no_future_or_gap"] += 1
            continue
        start = end - window_size + 1
        history_times = sequence.timestamps[start:end + 1]
        history_fps = 1.0 / float(np.median(np.diff(history_times)))
        if np.any(np.diff(history_times) > 1.5 / history_fps + 1e-9):
            rejected["history_gap"] += 1
            continue
        hand_target = sequence.hands[indices].copy()
        object_target = sequence.objects[indices].copy()
        hand_current = np.isfinite(sequence.hands[end]).all(axis=(-2, -1))
        object_current = np.isfinite(sequence.objects[end]).all(axis=(-2, -1))
        hand_target[:, ~hand_current] = np.nan
        object_target[:, ~object_current] = np.nan
        if (not np.isfinite(hand_target).all(axis=(-2, -1)).any()
                or not np.isfinite(object_target).all(axis=(-2, -1)).any()):
            rejected["no_joint_target"] += 1
            continue
        samples.append({
            "sequence_id": sequence.sequence_id, "group_id": sequence.group_id,
            "sample_id": f"{sequence.sequence_id}:{end}", "observation_index": end,
            "observation_time": float(sequence.timestamps[end]), "target_indices": indices,
            "actual_horizons": sequence.timestamps[indices] - sequence.timestamps[end],
            "hand_seq": sequence.hands[start:end + 1].copy(),
            "obj_seq": history_features(sequence.objects[start:end + 1], sequence.bboxes[start:end + 1], history_times),
            "obj_mask": np.ones(sequence.objects.shape[1], dtype=bool),
            "hand_future": hand_target, "object_future": object_target,
        })
    return samples, rejected


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("contract_version") != "pose_forecast_v1":
        raise ValueError("unsupported forecast contract")
    entries = manifest.get("sequences", [])
    if not entries:
        raise ValueError("empty sequence manifest")
    groups = {}
    observation_splits = {}
    source_splits = {}
    seen = set()
    splits = {name: [] for name in ("train", "val", "test", "demo")}
    for entry in entries:
        split = entry["split"]
        if split not in splits or entry["sequence_id"] in seen:
            raise ValueError("unknown split or repeated sequence")
        seen.add(entry["sequence_id"])
        group = entry["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError("acquisition group leaks across splits")
        groups[group] = split
        observation_dir = Path(entry["observation_dir"])
        if not observation_dir.is_absolute():
            observation_dir = path.parent / observation_dir
        observation_key = str(observation_dir.resolve())
        if observation_key in observation_splits and observation_splits[observation_key] != split:
            raise ValueError("same observation directory leaks across splits under aliases")
        observation_splits[observation_key] = split
        sequence = load_sequence(observation_dir, entry["sequence_id"], group, entry.get("scene_dir"))
        frame_hashes = [record.get("rgb", {}).get("sha256") for record in sequence.records]
        if all(frame_hashes):
            fingerprint = hashlib.sha256(json.dumps(frame_hashes).encode()).hexdigest()
            if fingerprint in source_splits and source_splits[fingerprint] != split:
                raise ValueError("identical source frame sequence leaks across splits")
            source_splits[fingerprint] = split
        splits[split].append(sequence)
    return manifest, splits
