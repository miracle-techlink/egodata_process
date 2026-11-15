"""Stage annotated EgoDemo timelines without generating HOI labels or transforms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
from uuid import UUID

import numpy as np


SPATIAL_FEATURES = {
    "hand_left_world": ("left_hand_world", (21, 3)),
    "hand_right_world": ("right_hand_world", (21, 3)),
    "head_world": ("head_world", (1, 3)),
    "head_world_rotation": ("head_world_rotation", (1, 4)),
    "head_left_camera_position": ("head_left_camera_position", (3,)),
    "head_left_camera_rotation": ("head_left_camera_rotation", (4,)),
    "head_right_camera_position": ("head_right_camera_position", (3,)),
    "head_right_camera_rotation": ("head_right_camera_rotation", (4,)),
}
REQUIRED_SPATIAL = (
    "hand_left_world", "hand_right_world",
    "head_left_camera_position", "head_left_camera_rotation",
)
INDEX_COLUMNS = ("frame_index", "episode_index", "index", "task_index")
REQUIRED_FILES = {
    "info": "meta/info.json",
    "annotation": "annotation.json",
    "bad_frame_ratio": "bad_frame_ratio.json",
}


class ImportStageError(ValueError):
    """A source cannot safely be interpreted as one annotated episode."""


def _number(value, name):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ImportStageError(f"{name}: expected a finite number")
    return value


def _scalar(value, name):
    array = np.asarray(value)
    if array.shape not in ((), (1,)):
        raise ImportStageError(f"{name}: expected a scalar or shape [1]")
    return _number(array.item(), name)


def _integer(value, name, minimum=0):
    value = _scalar(value, name)
    if int(value) != value or not minimum <= value <= np.iinfo(np.int64).max:
        raise ImportStageError(f"{name}: expected an integer >= {minimum} within int64 range")
    return int(value)


def _declared_shape(features, key, expected):
    feature = features.get(key)
    if not isinstance(feature, dict) or feature.get("shape") != list(expected):
        raise ImportStageError(f"meta/info.json features[{key!r}].shape must be {list(expected)}")


def _spatial_array(value, shape, name):
    try:
        array = np.asarray(value)
        if array.shape not in (shape, (math.prod(shape),)):
            raise ImportStageError(f"{name}: expected shape {shape} or a flattened row, got {array.shape}")
        if array.dtype.kind not in "fiu":
            if array.dtype.kind != "O" or any(
                item is not None and (
                    isinstance(item, (bool, np.bool_))
                    or not isinstance(item, (int, float, np.integer, np.floating))
                ) for item in array.flat
            ):
                raise ImportStageError(f"{name}: expected numeric coordinates (nulls may represent NaN)")
        with np.errstate(over="raise", invalid="raise"):
            return array.astype(np.float32).reshape(shape)
    except (TypeError, ValueError, FloatingPointError) as error:
        if isinstance(error, ImportStageError):
            raise
        raise ImportStageError(f"{name}: invalid coordinate array: {error}") from error


def normalize_sequence(rows, info, bad_frames):
    """Validate list-like rows; retain raw worlds and mask only staged hand arrays."""
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        raise ImportStageError("meta/info.json must contain a features object")
    count = _integer(_number(info.get("total_frames"), "meta/info.json total_frames"),
                     "meta/info.json total_frames", minimum=1)
    fps = _number(info.get("fps"), "meta/info.json fps")
    if fps <= 0:
        raise ImportStageError("meta/info.json fps must be positive")
    if "total_episodes" in info and _integer(info["total_episodes"], "total_episodes") != 1:
        raise ImportStageError("meta/info.json total_episodes must be 1; do not mix episodes")
    rows = list(rows)
    if len(rows) != count:
        raise ImportStageError(f"Frame count mismatch: meta/info.json total_frames={count}, parquet rows={len(rows)}")
    if any(not isinstance(row, dict) for row in rows):
        raise ImportStageError("Parquet rows must be mappings of declared feature names to values")
    features = info["features"]
    for suffix in REQUIRED_SPATIAL:
        key = f"observation.state.{suffix}"
        _declared_shape(features, key, SPATIAL_FEATURES[suffix][1])
    sequence = {}
    for key in ("timestamp", *INDEX_COLUMNS):
        _declared_shape(features, key, (1,))
        values = []
        for row_number, row in enumerate(rows):
            name = f"row {row_number} {key}"
            values.append(_scalar(row.get(key), name) if key == "timestamp" else _integer(row.get(key), name))
        sequence["timestamps" if key == "timestamp" else key] = np.asarray(
            values, dtype=np.float64 if key == "timestamp" else np.int64,
        )
    for key in ("timestamps", "frame_index", "index"):
        values = sequence[key]
        if (values < 0).any() or not (values[1:] > values[:-1]).all():
            raise ImportStageError(f"{key}: values must be nonnegative and strictly increasing in source order")
    if len(np.unique(sequence["episode_index"])) != 1:
        raise ImportStageError("Mixed episode_index values in one UUID; refusing to concatenate episodes")
    for suffix, (output_name, shape) in SPATIAL_FEATURES.items():
        key = f"observation.state.{suffix}"
        if key not in features:
            continue
        _declared_shape(features, key, shape)
        sequence[output_name] = np.stack([
            _spatial_array(row.get(key), shape, f"row {row_number} {key}")
            for row_number, row in enumerate(rows)
        ])
    if not isinstance(bad_frames, dict) or not isinstance(bad_frames.get("hand_bad_frame_details"), list):
        raise ImportStageError("bad_frame_ratio.json must contain a hand_bad_frame_details list")
    hand_valid = np.stack([
        np.isfinite(sequence[name]).all(axis=(1, 2))
        for name in ("left_hand_world", "right_hand_world")
    ], axis=1)
    positions = {int(frame): position for position, frame in enumerate(sequence["frame_index"])}
    seen = set()
    for detail in bad_frames["hand_bad_frame_details"]:
        if not isinstance(detail, dict) or not {"frame", "left_bad_reason", "right_bad_reason"} <= detail.keys():
            raise ImportStageError("hand_bad_frame_details entries require frame, left_bad_reason and right_bad_reason")
        frame = _integer(detail["frame"], "hand_bad_frame_details frame")
        if frame not in positions or frame in seen:
            raise ImportStageError(f"hand_bad_frame_details frame {frame} is absent from frame_index or duplicated")
        seen.add(frame)
        for hand_index, side in enumerate(("left", "right")):
            if detail[f"{side}_bad_reason"]:
                hand_valid[positions[frame], hand_index] = False
    for hand_index, name in enumerate(("left_hand_world", "right_hand_world")):
        sequence[f"{name}_raw"] = sequence[name].copy()
        sequence[name][~hand_valid[:, hand_index]] = np.nan
    sequence["hand_valid"] = hand_valid
    return sequence


def validate_events(annotation):
    """Validate source intervals without rewriting descriptions, IDs or extra fields."""
    if not isinstance(annotation, dict) or not isinstance(annotation.get("episode_description"), str):
        raise ImportStageError("annotation.json requires an episode_description string")
    events = annotation.get("subtask_annotations")
    if not isinstance(events, list):
        raise ImportStageError("annotation.json requires a subtask_annotations list")
    previous_start, previous_end = -math.inf, -math.inf
    identifiers = set()
    for position, event in enumerate(events):
        prefix = f"subtask_annotations[{position}]"
        if not isinstance(event, dict) or not isinstance(event.get("description"), str):
            raise ImportStageError(f"{prefix}: description must be a string")
        identifier = event.get("subtask_id")
        if isinstance(identifier, bool) or not isinstance(identifier, (int, str)) or identifier == "":
            raise ImportStageError(f"{prefix}: subtask_id must be an integer or nonempty string")
        if identifier in identifiers:
            raise ImportStageError(f"{prefix}: duplicate subtask_id {identifier!r}")
        identifiers.add(identifier)
        start = _number(event.get("start_time"), f"{prefix} start_time")
        end = _number(event.get("end_time"), f"{prefix} end_time")
        if start < 0 or end <= start:
            raise ImportStageError(f"{prefix}: require 0 <= start_time < end_time")
        if start < previous_start or end < previous_end:
            raise ImportStageError(f"{prefix}: event start/end times must be monotone in source order")
        previous_start, previous_end = start, end
    return events


def event_timeline(sequence, events):
    """Map half-open semantic intervals to native timestamps; [] means unlabelled."""
    for frame, timestamp in zip(sequence["frame_index"], sequence["timestamps"]):
        yield {
            "frame_index": int(frame), "timestamp": float(timestamp),
            "subtask_ids": [event["subtask_id"] for event in events
                            if event["start_time"] <= timestamp < event["end_time"]],
        }


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ImportStageError(f"Expected a repository-relative path, got {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")) or ":" in value:
        raise ImportStageError(f"Unsafe repository-relative path: {value!r}")
    return path


def _local_path(root, relative):
    path = root / _relative_path(relative)
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as error:
        raise ImportStageError(f"Cannot resolve dataset path {relative}: {error}") from error
    if not resolved.is_relative_to(root):
        raise ImportStageError(f"Path escapes dataset root: {relative}")
    return path


def verify_payload(path, record):
    """Verify current bytes, never trust a manifest's possibly stale ready flags."""
    if not path.is_file():
        raise ImportStageError(f"Missing payload: {path}; download this exact manifest path first")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        prefix = stream.read(8192)
        if not prefix:
            raise ImportStageError(f"Empty payload: {path}")
        if prefix.lstrip().startswith((b"version https://git-lfs", b"oid sha256:")):
            raise ImportStageError(f"Git LFS pointer, not a downloaded payload: {path}")
        digest.update(prefix)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    size, checksum = path.stat().st_size, digest.hexdigest()
    expected_size, expected_sha = record.get("expected_size"), record.get("expected_sha256")
    if expected_size is not None and size != _integer(expected_size, f"{path} expected_size", minimum=1):
        raise ImportStageError(f"Size mismatch for {path}: expected {expected_size}, found {size}")
    if expected_sha is not None and (
        not isinstance(expected_sha, str) or checksum != expected_sha.lower()
    ):
        raise ImportStageError(f"SHA256 mismatch for {path}: verify or download the payload again")
    return {
        "path": record["path"], "role": record["role"], "size": size, "sha256": checksum,
        "expected_size": expected_size, "expected_sha256": expected_sha,
        "manifest_ready": record.get("ready"), "manifest_downloaded": record.get("downloaded"),
        "payload_verified": True,
    }


def _read_json(path):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ImportStageError(f"Invalid JSON metadata at {path}: {error}") from error


def read_parquet_rows(paths):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportStageError(
            "Parquet staging requires optional pyarrow; run in an environment providing pyarrow.parquet. "
            "Pure normalization and metadata tests do not require it."
        ) from error
    rows = []
    for path in paths:
        try:
            rows.extend(parquet.read_table(str(path)).to_pylist())
        except Exception as error:
            raise ImportStageError(f"Cannot read LeRobot parquet {path}: {error}") from error
    return rows


def _episode_files(episode, root):
    if not isinstance(episode, dict):
        raise ImportStageError("selected_episodes entries must be objects")
    source_uuid = episode.get("source_episode_uuid")
    try:
        if not isinstance(source_uuid, str) or str(UUID(source_uuid)) != source_uuid.lower():
            raise ValueError("not a canonical UUID")
    except ValueError as error:
        raise ImportStageError(f"Invalid source_episode_uuid: {source_uuid!r}") from error
    variant, task = episode.get("variant"), episode.get("task_name")
    if variant not in ("EgoProStandard", "EgoProStandard-body") or not isinstance(task, str) or not task.strip():
        raise ImportStageError(f"{source_uuid}: require an annotated variant and nonempty task_name")
    base = _relative_path(episode.get("lerobot_path"))
    if base.parts[0] != variant or base.parts[-1].lower() != source_uuid.lower():
        raise ImportStageError(f"{source_uuid}: variant/UUID disagrees with lerobot_path")
    files = episode.get("files")
    if not isinstance(files, list):
        raise ImportStageError(f"{source_uuid}: files must be a list of exact repository-relative paths")
    grouped, seen = {}, set()
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("role"), str):
            raise ImportStageError(f"{source_uuid}: malformed file record")
        relative = _relative_path(record.get("path"))
        if not relative.is_relative_to(base) or relative == base or str(relative) in seen:
            raise ImportStageError(f"{source_uuid}: duplicate file or file outside lerobot_path: {relative}")
        seen.add(str(relative))
        path = _local_path(root, str(relative))
        grouped.setdefault(record["role"], []).append((path, record))
    for role, suffix in REQUIRED_FILES.items():
        matches = grouped.get(role, [])
        if len(matches) != 1 or matches[0][1]["path"] != str(base / suffix):
            raise ImportStageError(f"{source_uuid}: require exactly one {role} entry at {base / suffix}")
    data = grouped.get("data", [])
    if not data:
        raise ImportStageError(f"{source_uuid}: no data/chunk-*/*.parquet entries in manifest")
    for _, record in data:
        parts = PurePosixPath(record["path"]).relative_to(base).parts
        if len(parts) != 3 or parts[0] != "data" or not parts[1].startswith("chunk-") or not parts[2].endswith(".parquet"):
            raise ImportStageError(f"Not a LeRobot frame parquet data/chunk-*/*.parquet: {record['path']}")
    grouped["data"] = sorted(data, key=lambda entry: entry[1]["path"])
    return source_uuid.lower(), grouped


def stage_episode(episode, dataset_root, output_dir, manifest_provenance):
    source_uuid, grouped = _episode_files(episode, dataset_root)
    destination = output_dir / source_uuid
    if destination.exists() or destination.is_symlink():
        raise ImportStageError(f"Output already exists: {destination}; choose a fresh output directory")
    verified = []
    for role in (*REQUIRED_FILES, "data"):
        verified.extend(verify_payload(path, record) for path, record in grouped[role])
    info = _read_json(grouped["info"][0][0])
    annotation_path, quality_path = grouped["annotation"][0][0], grouped["bad_frame_ratio"][0][0]
    annotation, bad_frames = _read_json(annotation_path), _read_json(quality_path)
    events = validate_events(annotation)
    sequence = normalize_sequence(read_parquet_rows([path for path, _ in grouped["data"]]), info, bad_frames)
    count = len(sequence["timestamps"])
    if episode.get("frame_count") is not None and _integer(episode["frame_count"], "manifest frame_count") != count:
        raise ImportStageError(f"{source_uuid}: manifest frame_count disagrees with meta/info.json and parquet")
    nominal_fps = float(info["fps"])
    manifest_fps = episode.get("fps")
    if manifest_fps is not None:
        manifest_fps = _number(manifest_fps, "manifest fps")
        if manifest_fps <= 0:
            raise ImportStageError("manifest fps must be positive")
    intervals = np.diff(sequence["timestamps"])
    measured_fps = 1.0 / float(intervals.mean()) if len(intervals) else None
    fps_discrepancies, quality_warnings = [], []
    for source, value, reference, reference_value in (
        ("manifest_fps", manifest_fps, "nominal_fps", nominal_fps),
        ("measured_fps", measured_fps, "nominal_fps", nominal_fps),
        ("manifest_fps", manifest_fps, "measured_fps", measured_fps),
    ):
        if value is not None and reference_value is not None and not math.isclose(value, reference_value, rel_tol=1e-6):
            quality_warnings.append(f"{source}_differs_from_{reference}")
            fps_discrepancies.append({
                "source": source, "value": value, "reference": reference, "reference_value": reference_value,
                "difference": value - reference_value, "relative_difference": (value - reference_value) / reference_value,
            })
    video_status = []
    for role, entries in grouped.items():
        if role.endswith("_video"):
            for path, record in entries:
                status = {"path": record["path"], "decoded": False, "timeline_alignment_verified": False}
                try:
                    status.update(verify_payload(path, record), status="payload_only_not_decoded")
                except ImportStageError as error:
                    status.update(status="unavailable_or_invalid", payload_verified=False, reason=str(error))
                video_status.append(status)
    meta = {
        "stage": "annotated_world_sequence", "training_ready": False,
        "source_episode_uuid": source_uuid, "variant": episode["variant"], "task_name": episode["task_name"],
        "coordinate_system": "world", "quaternion_order": ["qw", "qx", "qy", "qz"],
        "world_to_camera_transform_applied": False, "rotation_direction_verified": False,
        "timeline": "native_annotated", "raw_video_alignment_verified": False,
        "frame_count": count, "fps": nominal_fps, "episode_index": int(sequence["episode_index"][0]),
        "nominal_fps": nominal_fps, "manifest_fps": manifest_fps, "measured_fps": measured_fps,
        "timing_source_of_truth": "native parquet timestamps; no resampling or timestamp changes",
        "measured_fps_method": "1 / mean(diff(timestamps)); null for a single frame",
        "fps_discrepancy_relative_tolerance": 1e-6,
        "fps_discrepancies": fps_discrepancies, "quality_warnings": quality_warnings,
        "timestamp_range_seconds": [float(sequence["timestamps"][0]), float(sequence["timestamps"][-1])],
        "hand_valid_order": ["left", "right"],
        "hand_valid_counts": sequence["hand_valid"].sum(axis=0).tolist(),
        "hand_mask_policy": "finite raw coordinates and no recorded bad-hand reason; invalid staged rows are NaN",
        "raw_hand_arrays": ["left_hand_world_raw", "right_hand_world_raw"],
        "events": {
            "count": len(events), "hoi_classes": False, "automatic_label_mapping": False,
            "intervals": "[start_time, end_time)", "unlabelled_frames": "subtask_ids=[]",
        },
        "video_verification": {"decoding_performed": False, "files": video_status},
        "provenance": {
            "manifest": manifest_provenance, "selected_episode": episode,
            "source_info": info, "verified_source_files": verified,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{source_uuid}.", dir=output_dir) as temporary:
        staged = Path(temporary) / source_uuid
        staged.mkdir()
        np.savez_compressed(staged / "sequence.npz", **sequence)
        (staged / "events.json").write_bytes(annotation_path.read_bytes())
        (staged / "bad_frame_ratio.json").write_bytes(quality_path.read_bytes())
        (staged / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (staged / "event_timeline.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["frame_index", "timestamp", "subtask_ids"])
            writer.writeheader()
            for row in event_timeline(sequence, events):
                row["subtask_ids"] = json.dumps(row["subtask_ids"], ensure_ascii=False)
                writer.writerow(row)
        staged.rename(destination)
    return {"source_episode_uuid": source_uuid, "output_dir": str(destination), "frame_count": count}


def import_manifest(manifest_path, dataset_root, output_dir):
    manifest_path, dataset_root, output_dir = map(lambda value: Path(value).expanduser().resolve(), (
        manifest_path, dataset_root, output_dir,
    ))
    if not dataset_root.is_dir():
        raise ImportStageError(f"Dataset root is not a directory: {dataset_root}")
    manifest_record = {"path": str(manifest_path), "role": "selected_manifest"}
    provenance = verify_payload(manifest_path, manifest_record)
    manifest = _read_json(manifest_path)
    episodes = manifest.get("selected_episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise ImportStageError("Manifest must contain a nonempty selected_episodes list")
    seen = set()
    for episode in episodes:
        source_uuid, _ = _episode_files(episode, dataset_root)
        if source_uuid in seen:
            raise ImportStageError(f"Duplicate selected UUID {source_uuid}; do not mix variants or episodes")
        seen.add(source_uuid)
    return [stage_episode(episode, dataset_root, output_dir, provenance) for episode in episodes]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = import_manifest(args.manifest, args.dataset_root, args.output_dir)
    except (ImportStageError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"staged_episodes": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
