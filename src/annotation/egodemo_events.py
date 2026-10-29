"""Convert EgoDemo VLM responses into reviewable native-timeline HOI candidates."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from src.annotation.schema import (
    project_frame_label, validate_event, validate_none_interval, write_jsonl,
)
from src.dataset.import_egodemo import validate_events as validate_source_events


INTERACTION_TYPES = {"reach", "grasp", "hold", "move", "place", "release"}
NATIVE_TIMELINES = {"native", "native_annotated", "annotated_native"}
RUN_FIELDS = (
    "run_id", "model", "revision", "prompt_template_sha256", "decoding_parameters",
    "environment", "sequence_id", "timeline", "frame_count",
)


class AnnotationError(ValueError):
    """Input cannot be converted without guessing labels, tracks, or frames."""


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise AnnotationError(f"Invalid JSON at {path}: {error}") from error


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path, role):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise AnnotationError(f"Missing {role} input: {path}")
    return {"role": role, "path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AnnotationError(f"{name} must be a finite number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise AnnotationError(f"{name} must be a positive integer")
    return int(value)


def _native_domain_end(timestamps, fps, meta):
    if len(timestamps) > 1:
        interval = float(np.median(np.diff(timestamps)))
        return float(timestamps[-1] + interval), "last_timestamp_plus_median_native_interval"
    candidates = []
    if fps is not None:
        candidates.append(("cli_fps", _finite_number(fps, "fps")))
    if isinstance(meta.get("fps"), (int, float)) and not isinstance(meta.get("fps"), bool):
        candidates.append(("meta_fps", _finite_number(meta["fps"], "meta fps")))
    if not candidates:
        raise AnnotationError("A single-frame native timeline requires a positive fps; refusing to guess its end")
    for name, value in candidates:
        if value <= 0:
            raise AnnotationError(f"{name} must be positive")
    if len(candidates) == 2 and not math.isclose(candidates[0][1], candidates[1][1], rel_tol=1e-9):
        raise AnnotationError("CLI fps and meta fps disagree for a single-frame native timeline")
    return float(timestamps[-1] + 1.0 / candidates[0][1]), f"last_timestamp_plus_1/{candidates[0][0]}"


def load_native_timeline(sequence_dir, sequence_id=None, fps=None):
    """Load one zero-based annotated timeline without resampling or raw-frame inference."""
    sequence_dir = Path(sequence_dir).expanduser().resolve()
    sequence_path = sequence_dir / "sequence.npz"
    events_path = sequence_dir / "events.json"
    meta_path = sequence_dir / "meta.json"
    if not sequence_path.is_file() or not events_path.is_file():
        raise AnnotationError(f"{sequence_dir}: require sequence.npz and events.json")
    meta = _read_json(meta_path) if meta_path.is_file() else {}
    if not isinstance(meta, dict):
        raise AnnotationError("meta.json must contain an object")
    with np.load(sequence_path, allow_pickle=False) as archive:
        if "timestamps" not in archive.files or "frame_index" not in archive.files:
            raise AnnotationError("sequence.npz requires native timestamps and frame_index")
        timestamps = np.asarray(archive["timestamps"])
        frame_indices = np.asarray(archive["frame_index"])
    if timestamps.ndim != 1 or not len(timestamps) or timestamps.dtype.kind not in "fiu":
        raise AnnotationError("timestamps must be a nonempty numeric vector")
    if frame_indices.shape != timestamps.shape or frame_indices.dtype.kind not in "iu":
        raise AnnotationError("frame_index must be an integer vector matching timestamps")
    timestamps = timestamps.astype(np.float64)
    frame_indices = frame_indices.astype(np.int64)
    if not np.isfinite(timestamps).all() or (timestamps < 0).any() or (np.diff(timestamps) <= 0).any():
        raise AnnotationError("native timestamps must be finite, nonnegative and strictly increasing")
    if not np.array_equal(frame_indices, np.arange(len(timestamps), dtype=np.int64)):
        raise AnnotationError(
            "native frame_index must be exactly 0..T-1; nonzero/sparse/raw timelines need an explicit mapping"
        )
    resolved_id = sequence_id or meta.get("source_episode_uuid") or sequence_dir.name
    if not isinstance(resolved_id, str) or not resolved_id.strip():
        raise AnnotationError("sequence_id must be a nonempty string")
    annotation = _read_json(events_path)
    try:
        source_events = validate_source_events(annotation)
    except ValueError as error:
        raise AnnotationError(f"Invalid source events: {error}") from error
    source_maps = []
    seen_ids = set()
    for original in source_events:
        identifier = str(original["subtask_id"])
        if identifier in seen_ids:
            raise AnnotationError(f"Source IDs collide after string conversion: {identifier!r}")
        seen_ids.add(identifier)
        selected = np.flatnonzero(
            (timestamps >= float(original["start_time"])) & (timestamps < float(original["end_time"]))
        )
        source_maps.append({
            "source_event_id": identifier,
            "start_time": float(original["start_time"]),
            "end_time": float(original["end_time"]),
            "description": original["description"],
            "start_frame": int(selected[0]) if selected.size else None,
            "end_frame": int(selected[-1] + 1) if selected.size else None,
            "mapped_frame_count": int(selected.size),
            "original_event": copy.deepcopy(original),
        })
    domain_end, domain_method = _native_domain_end(timestamps, fps, meta)
    return {
        "sequence_dir": sequence_dir,
        "sequence_path": sequence_path,
        "events_path": events_path,
        "meta_path": meta_path if meta_path.is_file() else None,
        "meta": meta,
        "sequence_id": resolved_id,
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "frame_count": len(timestamps),
        "domain_end": domain_end,
        "domain_method": domain_method,
        "source_annotation": annotation,
        "source_maps": source_maps,
    }


def timeline_document(native):
    return {
        "schema_version": "hoi_annotation_v1",
        "sequence_id": native["sequence_id"],
        "timeline": "native_annotated",
        "raw_frame_alignment": False,
        "frame_count": native["frame_count"],
        "frame_index": native["frame_indices"].tolist(),
        "timestamps": native["timestamps"].tolist(),
        "domain_end_seconds": native["domain_end"],
        "domain_end_method": native["domain_method"],
        "interval_convention": "[start_time,end_time); [start_frame,end_frame)",
        "source_events": [
            {key: value for key, value in source.items() if key != "original_event"}
            for source in native["source_maps"]
        ],
    }


def _decode_response(value, location):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise AnnotationError(f"Invalid structured VLM response at {location}: {error}") from error


def _walk_vlm_payload(payload, context, records, location="root", container_keys=("events", "responses")):
    payload = _decode_response(payload, location)
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            _walk_vlm_payload(item, context, records, f"{location}[{index}]", container_keys)
        return
    if not isinstance(payload, dict):
        raise AnnotationError(f"VLM payload at {location} must be an object, list, or JSON string")
    nested_context = {**context, **{key: payload[key] for key in RUN_FIELDS if key in payload}}
    if any(key in payload for key in ("start_time", "start_sec", "start_seconds", "start_frame")):
        records.append((payload, nested_context, location))
        return
    for key in container_keys:
        if key in payload:
            if not isinstance(payload[key], list):
                raise AnnotationError(f"VLM {key} at {location} must be a list")
            for index, item in enumerate(payload[key]):
                _walk_vlm_payload(item, nested_context, records, f"{location}.{key}[{index}]", container_keys)
            return
    for key in ("response", "raw_response", "content"):
        if key in payload:
            _walk_vlm_payload(payload[key], nested_context, records, f"{location}.{key}", container_keys)
            return
    raise AnnotationError(f"No VLM records found at {location}; expected {container_keys}")


def _load_vlm_records(path, container_keys):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise AnnotationError(f"Missing VLM input: {path}")
    records = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        _walk_vlm_payload(
                            json.loads(line), {}, records, f"{path.name}:{line_number}", container_keys
                        )
        else:
            _walk_vlm_payload(_read_json(path), {}, records, path.name, container_keys)
    except json.JSONDecodeError as error:
        raise AnnotationError(f"Invalid VLM JSON/JSONL at {path}: {error}") from error
    contexts = []
    for _, context, _ in records:
        if context not in contexts:
            contexts.append(context)
    return records, contexts


def load_vlm_events(path):
    return _load_vlm_records(path, ("events", "responses"))


def load_vlm_none_intervals(path):
    return _load_vlm_records(path, ("none_intervals",))


def _event_seconds(record, names):
    present = [name for name in names if name in record]
    if not present:
        raise AnnotationError(f"VLM record requires {names[0]} in seconds")
    value = _finite_number(record[present[0]], present[0])
    for name in present[1:]:
        other = _finite_number(record[name], name)
        if not math.isclose(value, other, rel_tol=0, abs_tol=1e-9):
            raise AnnotationError(f"Conflicting VLM time fields: {present}")
    return value


def _overlap_seconds(start, end, source):
    return max(0.0, min(end, source["end_time"]) - max(start, source["start_time"]))


def _check_vlm_context(record, context, native, location):
    sequence_id = record.get("sequence_id", context.get("sequence_id"))
    if sequence_id is not None and sequence_id != native["sequence_id"]:
        raise AnnotationError(
            f"{location}: sequence_id {sequence_id!r} does not match {native['sequence_id']!r}"
        )
    timeline = record.get("timeline", context.get("timeline"))
    if timeline is not None and timeline not in NATIVE_TIMELINES:
        raise AnnotationError(
            f"{location}: timeline {timeline!r} is not the annotated native timeline; raw frames are forbidden"
        )
    frame_count = record.get("frame_count", context.get("frame_count"))
    if frame_count is not None and frame_count != native["frame_count"]:
        raise AnnotationError(
            f"{location}: VLM frame_count={frame_count} disagrees with native {native['frame_count']}"
        )
    if record.get("review_status", "pending") != "pending":
        raise AnnotationError(f"{location}: reviewed VLM input must use apply, not prepare")
    if record.get("geometry_status", "not_checked") != "not_checked":
        raise AnnotationError(f"{location}: VLM input geometry_status must be not_checked")
    if record.get("annotation_mode", "offline_noncausal") != "offline_noncausal":
        raise AnnotationError(f"{location}: VLM input annotation_mode must be offline_noncausal")
    if record.get("allowed_as_model_input", False) is not False:
        raise AnnotationError(f"{location}: VLM input allowed_as_model_input must be false")


def _seconds_interval(native, start, end, location):
    timestamps = native["timestamps"]
    if start < 0 or end <= start:
        raise AnnotationError(f"{location}: require 0 <= start_time < end_time")
    tolerance = 1e-9
    if start < timestamps[0] - tolerance or end > native["domain_end"] + tolerance:
        raise AnnotationError(
            f"{location}: seconds [{start},{end}) exceed native domain "
            f"[{timestamps[0]},{native['domain_end']})"
        )
    selected = np.flatnonzero((timestamps >= start) & (timestamps < end))
    if not selected.size:
        raise AnnotationError(f"{location}: interval contains no native timestamp; refusing nearest-frame guessing")
    return int(selected[0]), int(selected[-1] + 1), selected


def _record_interval(native, record, location):
    start_names = ("start_time", "start_sec", "start_seconds")
    end_names = ("end_time", "end_sec", "end_seconds")
    has_start_seconds = any(name in record for name in start_names)
    has_end_seconds = any(name in record for name in end_names)
    has_seconds = has_start_seconds or has_end_seconds
    if has_seconds and not (has_start_seconds and has_end_seconds):
        raise AnnotationError(f"{location}: seconds boundaries must include both start and end")
    has_start_frame = "start_frame" in record
    has_end_frame = "end_frame" in record
    if has_start_frame != has_end_frame:
        raise AnnotationError(f"{location}: native boundaries require both start_frame and end_frame")
    if has_start_frame:
        start = record["start_frame"]
        end = record["end_frame"]
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (start, end)):
            raise AnnotationError(f"{location}: start_frame/end_frame must be native integers")
        start, end = int(start), int(end)
        if not 0 <= start < end <= native["frame_count"]:
            raise AnnotationError(
                f"{location}: require 0 <= start_frame < end_frame <= {native['frame_count']}"
            )
        if has_seconds:
            second_start = _event_seconds(record, start_names)
            second_end = _event_seconds(record, end_names)
            second_frames = _seconds_interval(native, second_start, second_end, location)
            if second_frames[:2] != (start, end):
                raise AnnotationError(
                    f"{location}: native frame and seconds boundaries disagree on the native timestamp timeline"
                )
        timestamps = native["timestamps"]
        start_time = float(timestamps[start])
        end_time = float(timestamps[end]) if end < native["frame_count"] else native["domain_end"]
        return start, end, start_time, end_time, np.arange(start, end, dtype=np.int64)
    if not has_seconds:
        raise AnnotationError(f"{location}: require native start_frame/end_frame or seconds boundaries")
    start_time = _event_seconds(record, start_names)
    end_time = _event_seconds(record, end_names)
    start, end, selected = _seconds_interval(native, start_time, end_time, location)
    return start, end, start_time, end_time, selected


def _record_evidence(record, selected, location):
    evidence = record.get("evidence_frames")
    if (not isinstance(evidence, list) or not evidence
            or any(isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) for frame in evidence)
            or len(set(evidence)) != len(evidence)):
        raise AnnotationError(f"{location}: evidence_frames must be a nonempty list of unique native integers")
    evidence = [int(frame) for frame in evidence]
    selected_set = set(selected.tolist())
    if any(frame not in selected_set for frame in evidence):
        raise AnnotationError(
            f"{location}: every evidence frame must exist inside the native interval"
        )
    return evidence


def _record_sources(native, record, start_time, end_time, location, *, infer_missing):
    source_by_id = {source["source_event_id"]: source for source in native["source_maps"]}
    if "source_event_ids" not in record:
        if not infer_missing:
            return [], []
        sources = [
            source for source in native["source_maps"]
            if _overlap_seconds(start_time, end_time, source) > 0
        ]
        if not sources:
            raise AnnotationError(f"{location}: no overlapping original semantic event")
        return [source["source_event_id"] for source in sources], sources
    raw_ids = record["source_event_ids"]
    if not isinstance(raw_ids, list):
        raise AnnotationError(f"{location}: source_event_ids must be a list when supplied")
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in raw_ids):
        raise AnnotationError(f"{location}: source_event_ids entries must be strings or integers")
    source_ids = [str(value) for value in raw_ids]
    if len(set(source_ids)) != len(source_ids):
        raise AnnotationError(f"{location}: duplicate source_event_ids")
    missing = [value for value in source_ids if value not in source_by_id]
    if missing:
        raise AnnotationError(f"{location}: unknown source_event_ids {missing}")
    sources = [source_by_id[value] for value in source_ids]
    if any(_overlap_seconds(start_time, end_time, source) <= 0 for source in sources):
        raise AnnotationError(
            f"{location}: declared source_event_ids do not temporally overlap the VLM interval; "
            "possible raw/annotated timeline conflict"
        )
    return source_ids, sources


def _unpack_record(item, position):
    if isinstance(item, tuple) and len(item) == 3 and isinstance(item[1], dict):
        return item
    return item, {}, f"records[{position}]"


def convert_vlm_events(native, records):
    """Convert each VLM phase independently and retain all overlapping candidates."""
    converted = []
    identifiers = set()
    for position, item in enumerate(records):
        event, context, location = _unpack_record(item, position)
        if not isinstance(event, dict):
            raise AnnotationError(f"{location}: VLM event must be an object")
        _check_vlm_context(event, context, native, location)
        start, end, start_time, end_time, selected = _record_interval(native, event, location)
        source_ids, sources = _record_sources(
            native, event, start_time, end_time, location, infer_missing="source_event_ids" not in event
        )
        evidence = _record_evidence(event, selected, location)
        event_id = event.get("event_id", f"{native['sequence_id']}_vlm_{position:04d}")
        if not isinstance(event_id, str) or not event_id.strip() or event_id in identifiers:
            raise AnnotationError(f"{location}: missing or duplicate VLM event_id {event_id!r}")
        identifiers.add(event_id)
        proposed_object_id = event.get("object_id")
        if proposed_object_id is not None and (
            not isinstance(proposed_object_id, str) or not proposed_object_id.strip()
        ):
            raise AnnotationError(f"{location}: object_id must be null or a nonempty proposed track ID")
        candidate = {
            "event_id": event_id,
            "sequence_id": native["sequence_id"],
            "subject": event.get("subject"),
            "object_id": None,
            "object_hint": event.get("object_hint"),
            "interaction_type": event.get("interaction_type"),
            "start_frame": start,
            "end_frame": end,
            "start_time": start_time,
            "end_time": end_time,
            "evidence_frames": evidence,
            "confidence": event.get("confidence"),
            "label_source": "source_text+vlm" if source_ids else "vlm",
            "review_status": "pending",
            "geometry_status": "not_checked",
            "source_event_ids": source_ids,
            "source_events": [copy.deepcopy(source["original_event"]) for source in sources],
            "notes": event.get("notes", ""),
            "proposed_object_id": proposed_object_id,
            "track_link_status": "not_provided" if proposed_object_id is None else "unverified",
            "geometry_evidence": {
                "method": "not_evaluated",
                "scope": "none",
                "3d_verified": False,
                "automatic_accept": False,
            },
            "annotation_mode": "offline_noncausal",
            "allowed_as_model_input": False,
            "vlm_run": {key: copy.deepcopy(value) for key, value in context.items()},
            "vlm_response": copy.deepcopy(event),
            "revision_history": [],
        }
        try:
            validate_event(candidate, native["frame_count"])
        except ValueError as error:
            raise AnnotationError(f"{location}: invalid VLM candidate: {error}") from error
        converted.append(candidate)
    return converted


def convert_vlm_none_intervals(native, records):
    """Convert only explicit VLM none proposals; never infer none from timeline gaps."""
    converted = []
    identifiers = set()
    for position, item in enumerate(records):
        interval, context, location = _unpack_record(item, position)
        if not isinstance(interval, dict):
            raise AnnotationError(f"{location}: VLM none interval must be an object")
        _check_vlm_context(interval, context, native, location)
        start, end, start_time, end_time, selected = _record_interval(native, interval, location)
        source_ids, sources = _record_sources(
            native, interval, start_time, end_time, location, infer_missing=False
        )
        evidence = _record_evidence(interval, selected, location)
        interval_id = interval.get("interval_id", f"{native['sequence_id']}_vlm_none_{position:04d}")
        if not isinstance(interval_id, str) or not interval_id.strip() or interval_id in identifiers:
            raise AnnotationError(f"{location}: missing or duplicate VLM interval_id {interval_id!r}")
        identifiers.add(interval_id)
        candidate = {
            "interval_id": interval_id,
            "sequence_id": native["sequence_id"],
            "start_frame": start,
            "end_frame": end,
            "start_time": start_time,
            "end_time": end_time,
            "evidence_frames": evidence,
            "confidence": interval.get("confidence"),
            "label_source": "vlm",
            "review_status": "pending",
            "geometry_status": "not_checked",
            "source_event_ids": source_ids,
            "source_events": [copy.deepcopy(source["original_event"]) for source in sources],
            "notes": interval.get("notes", ""),
            "annotation_mode": "offline_noncausal",
            "allowed_as_model_input": False,
            "vlm_run": {key: copy.deepcopy(value) for key, value in context.items()},
            "vlm_response": copy.deepcopy(interval),
            "revision_history": [],
        }
        try:
            validate_none_interval(candidate, native["frame_count"])
        except ValueError as error:
            raise AnnotationError(f"{location}: invalid VLM none candidate: {error}") from error
        converted.append(candidate)
    return converted


def _npz_dimension(archive, names, label):
    found = []
    for name in names:
        if name not in archive.files:
            continue
        value = np.asarray(archive[name])
        if value.shape not in ((), (1,)) or value.dtype.kind not in "iu":
            raise AnnotationError(f"{label} {name} must be an integer scalar")
        found.append((name, int(value.reshape(-1)[0])))
    if not found:
        return None
    if len({value for _, value in found}) != 1:
        raise AnnotationError(f"{label} image dimensions disagree: {found}")
    return found[0][1]


def _merge_dimension(provided, discovered, name):
    if provided is None:
        return discovered
    provided = _positive_integer(provided, name)
    if discovered is not None and provided != discovered:
        raise AnnotationError(f"CLI {name}={provided} disagrees with NPZ value {discovered}")
    return provided


def _check_optional_native_axes(archive, native, label):
    checks = []
    if "frame_index" in archive.files:
        values = np.asarray(archive["frame_index"])
        if values.shape != (native["frame_count"],) or values.dtype.kind not in "iu":
            raise AnnotationError(f"{label} frame_index must match the native frame vector shape")
        if not np.array_equal(values, native["frame_indices"]):
            raise AnnotationError(f"{label} frame_index conflicts with the annotated native timeline")
        checks.append("frame_index_exact")
    if "timestamps" in archive.files:
        values = np.asarray(archive["timestamps"])
        if values.shape != (native["frame_count"],) or values.dtype.kind not in "fiu":
            raise AnnotationError(f"{label} timestamps must match the native timestamp vector shape")
        if not np.isfinite(values).all() or not np.allclose(
            values, native["timestamps"], rtol=0, atol=1e-9
        ):
            raise AnnotationError(f"{label} timestamps conflict with the annotated native timeline")
        checks.append("timestamps_exact_within_1e-9_seconds")
    return checks or ["frame_length_only"]


def load_track_inputs(native, hand_pose_path=None, pose6d_path=None, image_width=None, image_height=None):
    tracks = {
        "object_ids": [], "bboxes": None, "hands": None, "hand_valid": None,
        "image_width": None, "image_height": None, "pose_path": None, "hand_path": None,
        "native_axis_checks": {},
    }
    width, height = image_width, image_height
    if pose6d_path is not None:
        path = Path(pose6d_path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as archive:
            if "object_ids" not in archive.files or "bboxes" not in archive.files:
                raise AnnotationError("pose6d NPZ requires object_ids and bboxes for track linking")
            tracks["native_axis_checks"]["pose6d"] = _check_optional_native_axes(archive, native, "pose6d")
            ids = np.asarray(archive["object_ids"])
            bboxes = np.asarray(archive["bboxes"])
            if ids.ndim != 1 or ids.dtype.kind not in "US":
                raise AnnotationError("pose6d object_ids must be a non-pickled Unicode string vector")
            object_ids = ids.tolist()
            if any(not value for value in object_ids) or len(set(object_ids)) != len(object_ids):
                raise AnnotationError("pose6d object_ids must be unique and nonempty")
            if bboxes.shape != (len(object_ids), native["frame_count"], 4) or bboxes.dtype.kind not in "fiu":
                raise AnnotationError(
                    f"pose6d bboxes must have shape (N,{native['frame_count']},4) on the same native timeline"
                )
            width = _merge_dimension(width, _npz_dimension(archive, ("image_width", "width"), "pose6d"), "image_width")
            height = _merge_dimension(height, _npz_dimension(archive, ("image_height", "height"), "pose6d"), "image_height")
        tracks.update(
            object_ids=object_ids, bboxes=bboxes.astype(np.float64), pose_path=path,
        )
    if hand_pose_path is not None:
        path = Path(hand_pose_path).expanduser().resolve()
        hands = {}
        with np.load(path, allow_pickle=False) as archive:
            tracks["native_axis_checks"]["hand_pose"] = _check_optional_native_axes(archive, native, "hand pose")
            for side in ("left", "right"):
                key = f"{side}_hand"
                if key not in archive.files:
                    raise AnnotationError(
                        f"hand pose NPZ requires normalized {key}; world/camera hand arrays are not 2D evidence"
                    )
                value = np.asarray(archive[key])
                if value.shape != (native["frame_count"], 21, 3) or value.dtype.kind not in "fiu":
                    raise AnnotationError(
                        f"{key} must have shape ({native['frame_count']},21,3) on the native timeline"
                    )
                hands[side] = value.astype(np.float64)
            hand_valid = None
            if "hand_valid" in archive.files:
                hand_valid = np.asarray(archive["hand_valid"])
                if hand_valid.shape != (native["frame_count"], 2):
                    raise AnnotationError("hand_valid must have shape (T,2)")
                hand_valid = hand_valid.astype(bool)
            width = _merge_dimension(width, _npz_dimension(archive, ("image_width", "width"), "hand pose"), "image_width")
            height = _merge_dimension(height, _npz_dimension(archive, ("image_height", "height"), "hand pose"), "image_height")
        tracks.update(hands=hands, hand_valid=hand_valid, hand_path=path)
    tracks["image_width"] = _positive_integer(width, "image_width") if width is not None else None
    tracks["image_height"] = _positive_integer(height, "image_height") if height is not None else None
    return tracks


def link_event_tracks(event, tracks, expand_ratio=0.25):
    """Link an existing proposed track ID and attach non-authoritative 2D evidence."""
    expand_ratio = _finite_number(expand_ratio, "expand_ratio")
    if expand_ratio < 0:
        raise AnnotationError("expand_ratio must be nonnegative")
    proposed = event.get("proposed_object_id")
    event["object_id"] = None
    event["track_link_status"] = "not_provided" if proposed is None else "unverified"
    event["track_link_evidence"] = None
    event["geometry_status"] = "not_checked"
    event["geometry_evidence"] = {
        "method": "not_evaluated", "scope": "none", "3d_verified": False,
        "automatic_accept": False,
    }
    if proposed is None:
        return event
    if tracks is None or tracks.get("bboxes") is None:
        return event
    if proposed not in tracks["object_ids"]:
        event["track_link_status"] = "not_found"
        event["geometry_evidence"]["reason"] = "proposed_object_id_not_in_tracker"
        return event
    track_index = tracks["object_ids"].index(proposed)
    event["object_id"] = proposed
    event["track_link_status"] = "linked"
    event["track_link_evidence"] = {
        "method": "existing_candidate_object_id_matched_to_pose6d_object_ids",
        "track_index_at_link": track_index,
        "native_axis_checks": copy.deepcopy(tracks["native_axis_checks"]),
        "object_hint_used_as_track_id": False,
    }
    if tracks.get("hands") is None:
        event["geometry_evidence"]["reason"] = "hand_pose_not_supplied"
        return event
    width, height = tracks.get("image_width"), tracks.get("image_height")
    if width is None or height is None:
        event["geometry_status"] = "inconclusive"
        event["geometry_evidence"] = {
            "method": "2d_wrist_to_expanded_bbox",
            "scope": "normalized_hand_image_xy_and_pixel_bbox",
            "3d_verified": False,
            "automatic_accept": False,
            "reason": "image_width_and_image_height_required",
        }
        return event
    hand = tracks["hands"][event["subject"]]
    hand_index = 0 if event["subject"] == "left" else 1
    details = []
    for frame in event["evidence_frames"]:
        detail = {"frame": frame, "usable": False, "inside_expanded_bbox": None,
                  "normalized_distance": None}
        valid_hand = tracks["hand_valid"] is None or bool(tracks["hand_valid"][frame, hand_index])
        wrist = hand[frame, 0]
        bbox = tracks["bboxes"][track_index, frame]
        if valid_hand and np.isfinite(wrist[:2]).all() and np.isfinite(bbox).all():
            x1, y1, x2, y2 = bbox.tolist()
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
                x, y = float(wrist[0] * width), float(wrist[1] * height)
                dx = max(x1 - x, 0.0, x - x2)
                dy = max(y1 - y, 0.0, y - y2)
                diagonal = math.hypot(x2 - x1, y2 - y1)
                expanded = (
                    x1 - (x2 - x1) * expand_ratio, y1 - (y2 - y1) * expand_ratio,
                    x2 + (x2 - x1) * expand_ratio, y2 + (y2 - y1) * expand_ratio,
                )
                detail.update(
                    usable=True,
                    wrist_normalized_xy=[float(wrist[0]), float(wrist[1])],
                    wrist_pixel_xy=[x, y],
                    bbox_xyxy=[x1, y1, x2, y2],
                    expanded_bbox_xyxy=list(expanded),
                    normalized_distance=float(math.hypot(dx, dy) / diagonal),
                    inside_expanded_bbox=bool(
                        expanded[0] <= x <= expanded[2] and expanded[1] <= y <= expanded[3]
                    ),
                )
            else:
                detail["reason"] = "invalid_or_out_of_image_bbox"
        else:
            detail["reason"] = "invalid_hand_or_bbox"
        details.append(detail)
    usable = [item for item in details if item["usable"]]
    hits = [item for item in details if item["inside_expanded_bbox"]]
    if hits:
        status = "passed"
    elif usable and len(usable) == len(details):
        status = "failed"
    else:
        status = "inconclusive"
    distances = [item["normalized_distance"] for item in usable]
    event["geometry_status"] = status
    event["geometry_evidence"] = {
        "method": "2d_wrist_to_expanded_bbox",
        "scope": "normalized_hand_image_xy_and_pixel_bbox",
        "3d_verified": False,
        "depth_used": False,
        "camera_intrinsics_used": False,
        "coordinate_transform_verified": False,
        "automatic_accept": False,
        "expand_ratio": expand_ratio,
        "evidence_frame_count": len(details),
        "usable_frame_count": len(usable),
        "inside_expanded_bbox_count": len(hits),
        "minimum_normalized_distance": min(distances) if distances else None,
        "median_normalized_distance": float(np.median(distances)) if distances else None,
        "frames": details,
        "interpretation": "review evidence only; not contact, 3D geometry, or automatic acceptance",
    }
    return event


def _coverage(events, frame_count):
    coverage = np.zeros(frame_count, dtype=np.int64)
    for event in events:
        coverage[event["start_frame"]:event["end_frame"]] += 1
    return coverage


def _counts(records, key, values):
    return {value: sum(record.get(key) == value for record in records) for value in values}


def build_summary(
    native, events, none_intervals, raw_vlm_count=None, object_ids=None, render=None,
    raw_vlm_none_count=None,
):
    frame_count = native["frame_count"]
    source_coverage = np.zeros(frame_count, dtype=np.int64)
    for source in native["source_maps"]:
        if source["start_frame"] is not None:
            source_coverage[source["start_frame"]:source["end_frame"]] += 1
    candidate_coverage = _coverage(events, frame_count)
    accepted = [event for event in events if event["review_status"] == "accepted"]
    accepted_coverage = _coverage(accepted, frame_count)
    accepted_none = [interval for interval in none_intervals if interval["review_status"] == "accepted"]
    none_coverage = _coverage(accepted_none, frame_count)
    if object_ids is None:
        object_ids = sorted({event["object_id"] for event in events if event.get("object_id")})
    reasons = {}
    valid_frames = 0
    for frame in range(frame_count):
        label = project_frame_label(events, none_intervals, frame, object_ids)
        valid_frames += int(label["valid"])
        reasons[label["reason"]] = reasons.get(label["reason"], 0) + 1
    return {
        "schema_version": "hoi_annotation_v1",
        "sequence_id": native["sequence_id"],
        "frame_count": frame_count,
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
        "training_ready": False,
        "semantic_projection_only": True,
        "3d_geometry_verified": False,
        "raw_vlm_candidates": len(events) if raw_vlm_count is None else raw_vlm_count,
        "raw_vlm_none_candidates": (
            len(none_intervals) if raw_vlm_none_count is None else raw_vlm_none_count
        ),
        "structurally_valid_candidates": len(events),
        "structurally_valid_none_candidates": len(none_intervals),
        "track_linked_candidates": sum(event.get("track_link_status") == "linked" for event in events),
        "review_status_counts": _counts(
            events + none_intervals, "review_status", ["pending", "accepted", "rejected", "needs_revision"]
        ),
        "geometry_status_counts": _counts(
            events, "geometry_status", ["not_checked", "passed", "failed", "inconclusive"]
        ),
        "source_events": {
            "count": len(native["source_maps"]),
            "covered_frames": int((source_coverage > 0).sum()),
            "unlabelled_frames": int((source_coverage == 0).sum()),
            "overlap_frames": int((source_coverage > 1).sum()),
            "unlabelled_meaning": "unknown/unlabelled, never none",
        },
        "candidate_coverage": {
            "covered_frames": int((candidate_coverage > 0).sum()),
            "overlap_frames": int((candidate_coverage > 1).sum()),
            "max_simultaneous": int(candidate_coverage.max()) if frame_count else 0,
        },
        "accepted_event_coverage_frames": int((accepted_coverage > 0).sum()),
        "accepted_event_overlap_frames": int((accepted_coverage > 1).sum()),
        "accepted_none_intervals": len(accepted_none),
        "accepted_none_coverage_frames": int((none_coverage > 0).sum()),
        "semantic_projection": {
            "valid_frames": valid_frames,
            "label_valid_zero_frames": frame_count - valid_frames,
            "reason_counts": reasons,
        },
        "negative_sample_policy": "only accepted explicit none_intervals; gaps and pending/rejected events are not none",
        "geometry_policy": "2D wrist-bbox results are review evidence, never automatic acceptance or 3D verification",
        "render": render or {"status": "not_requested"},
    }


def _inspect_source_review(path):
    if path is None:
        return None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames:
            raise AnnotationError("annotation_review.csv requires a header")
        rows = list(reader)
    statuses = {}
    for row in rows:
        status = row.get("review_status") or row.get("status") or "unspecified"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "row_count": len(rows),
        "status_counts": statuses,
        "semantic_use": "provenance only; existing source-level review rows do not accept VLM candidates",
    }


def _check_output(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir


def _provenance(stage, native, inputs, extra=None):
    outputs = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            outputs.append({
                "path": path.relative_to(stage).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            })
    value = {
        "schema_version": "hoi_annotation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sequence_id": native["sequence_id"],
        "timeline": "native_annotated",
        "raw_frame_alignment": False,
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
        "training_ready": False,
        "inputs": inputs,
        "outputs": outputs,
        "implementation": {
            "egodemo_events_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    if extra:
        value.update(extra)
    _write_json(stage / "provenance.json", value)


def prepare_annotations(
    sequence_dir, vlm_events, output_dir, *, vlm_none_intervals=None, source_review=None,
    hand_pose=None, pose6d=None, image_width=None, image_height=None, sequence_id=None,
    fps=None, video=None, render_format="mp4", expand_ratio=0.25,
):
    """Create a fresh pending-review directory; never generate none from source gaps."""
    output = _check_output(output_dir)
    native = load_native_timeline(sequence_dir, sequence_id=sequence_id, fps=fps)
    records, run_contexts = load_vlm_events(vlm_events)
    events = convert_vlm_events(native, records)
    none_records, none_run_contexts = ([], [])
    if vlm_none_intervals is not None:
        none_records, none_run_contexts = load_vlm_none_intervals(vlm_none_intervals)
    none_intervals = convert_vlm_none_intervals(native, none_records)
    collisions = sorted(
        {event["event_id"] for event in events}
        & {interval["interval_id"] for interval in none_intervals}
    )
    if collisions:
        raise AnnotationError(f"VLM event/none identifiers collide across inputs: {collisions}")
    tracks = load_track_inputs(
        native, hand_pose_path=hand_pose, pose6d_path=pose6d,
        image_width=image_width, image_height=image_height,
    ) if hand_pose is not None or pose6d is not None else None
    for event in events:
        link_event_tracks(event, tracks, expand_ratio=expand_ratio)
        validate_event(event, native["frame_count"])
    inputs = [
        _input_record(native["sequence_path"], "sequence_npz"),
        _input_record(native["events_path"], "source_events_json"),
        _input_record(vlm_events, "vlm_events"),
    ]
    if vlm_none_intervals is not None:
        inputs.append(_input_record(vlm_none_intervals, "vlm_none_intervals"))
    if native["meta_path"] is not None:
        inputs.append(_input_record(native["meta_path"], "sequence_meta"))
    for path, role in (
        (source_review, "source_annotation_review_csv"), (hand_pose, "hand_pose_npz"),
        (pose6d, "pose6d_npz"), (video, "annotated_video"),
    ):
        if path is not None:
            inputs.append(_input_record(path, role))
    source_review_info = _inspect_source_review(source_review)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        shutil.copyfile(native["events_path"], stage / "source_events.json")
        suffix = ".jsonl" if Path(vlm_events).suffix.lower() == ".jsonl" else ".json"
        shutil.copyfile(vlm_events, stage / f"vlm_input{suffix}")
        if vlm_none_intervals is not None:
            none_suffix = ".jsonl" if Path(vlm_none_intervals).suffix.lower() == ".jsonl" else ".json"
            shutil.copyfile(vlm_none_intervals, stage / f"vlm_none_input{none_suffix}")
        if source_review is not None:
            shutil.copyfile(source_review, stage / "source_annotation_review.csv")
        _write_json(stage / "timeline.json", timeline_document(native))
        write_jsonl(stage / "events.jsonl", events, native["frame_count"])
        write_jsonl(stage / "none_intervals.jsonl", none_intervals, native["frame_count"])
        from src.annotation.review import write_review_csv
        write_review_csv(stage / "review.csv", events, none_intervals)
        render_info = {"status": "not_requested", "reason": "provide --video to render annotated evidence"}
        if video is not None:
            from src.annotation.render_review import render_review
            render_path = stage / ("review.mp4" if render_format == "mp4" else "contact_sheet.jpg")
            render_info = render_review(
                stage, video, render_path, output_format=render_format,
            )
            render_info["path"] = str(output / render_path.name)
        summary = build_summary(
            native, events, none_intervals, raw_vlm_count=len(records),
            object_ids=tracks["object_ids"] if tracks else [], render=render_info,
            raw_vlm_none_count=len(none_records),
        )
        _write_json(stage / "summary.json", summary)
        _provenance(stage, native, inputs, {
            "operation": "prepare",
            "vlm_run_contexts": run_contexts,
            "vlm_none_run_contexts": none_run_contexts,
            "source_review": source_review_info,
            "render": render_info,
        })
        stage.rename(output)
    return {"output_dir": str(output), **summary}


def _add_prepare_parser(subparsers):
    parser = subparsers.add_parser("prepare", help="convert VLM candidates onto the native timeline")
    parser.add_argument("--sequence-dir", required=True, type=Path)
    parser.add_argument("--vlm-events", required=True, type=Path, help="JSON or JSONL structured VLM events")
    parser.add_argument(
        "--vlm-none-intervals", type=Path,
        help="optional explicit VLM none JSON/JSONL; omissions never become none",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-review", type=Path, help="optional existing annotation_review.csv; provenance only")
    parser.add_argument("--hand-pose", type=Path, help="optional normalized image-space hands for 2D evidence")
    parser.add_argument("--pose6d", type=Path, help="tracker object_ids and native pixel bboxes")
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--sequence-id")
    parser.add_argument("--fps", type=float, help="required only for a single-frame timeline without meta fps")
    parser.add_argument("--video", type=Path, help="matching annotated video, never a raw video")
    parser.add_argument("--render-format", choices=("mp4", "contact-sheet"), default="mp4")
    parser.add_argument("--expand-ratio", type=float, default=0.25)
    return parser


def _add_apply_parser(subparsers):
    parser = subparsers.add_parser("apply", help="apply human review into a fresh directory")
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--hand-pose", type=Path, help="optional normalized hands for recomputing 2D evidence")
    parser.add_argument("--pose6d", type=Path, help="required when adding/changing a track link")
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--video", type=Path, help="matching annotated video, never a raw video")
    parser.add_argument("--render-format", choices=("mp4", "contact-sheet"), default="mp4")
    parser.add_argument("--expand-ratio", type=float, default=0.25)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_parser(subparsers)
    _add_apply_parser(subparsers)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_annotations(
                args.sequence_dir, args.vlm_events, args.output_dir,
                vlm_none_intervals=args.vlm_none_intervals,
                source_review=args.source_review, hand_pose=args.hand_pose, pose6d=args.pose6d,
                image_width=args.image_width, image_height=args.image_height,
                sequence_id=args.sequence_id, fps=args.fps, video=args.video,
                render_format=args.render_format, expand_ratio=args.expand_ratio,
            )
        else:
            from src.annotation.review import apply_review
            result = apply_review(
                args.annotation_dir, args.output_dir, review_csv=args.review_csv,
                hand_pose=args.hand_pose, pose6d=args.pose6d,
                image_width=args.image_width, image_height=args.image_height,
                video=args.video, render_format=args.render_format, expand_ratio=args.expand_ratio,
            )
    except (AnnotationError, ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
