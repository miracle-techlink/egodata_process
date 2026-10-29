"""Validation and single-head projection for reviewed human-object events."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

SUBJECT_TO_ID = {"left": 0, "right": 1, "none": 2}
INTERACTION_TYPES = ("reach", "grasp", "hold", "move", "place", "release")
INTERACTION_TO_ID = {"none": 0, **{name: index + 1 for index, name in enumerate(INTERACTION_TYPES)}}
REVIEW_STATUSES = {"pending", "accepted", "rejected", "needs_revision"}
GEOMETRY_STATUSES = {"not_checked", "passed", "failed", "inconclusive"}
LABEL_SOURCES = {"vlm", "source_text", "source_text+vlm", "geometry_assisted", "human"}
ANNOTATION_MODE = "offline_noncausal"
_INTERACTION_FIELDS = ("subject", "object_id", "object_hint", "interaction_type")
_OCCLUDED_STATUSES = {"occluded", "unknown", "pending"}


class _InvalidAnnotationProvenance(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(record, key, *, allow_empty=False):
    value = record.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{key} must be a {'string' if allow_empty else 'nonempty string'}")
    return value


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _common(record, frame_count):
    if not isinstance(record, dict):
        raise ValueError("annotation record must be an object")
    _text(record, "sequence_id")
    start, end = record.get("start_frame"), record.get("end_frame")
    if not _integer(start) or not _integer(end) or start < 0 or end <= start:
        raise ValueError("start_frame/end_frame must be integers with 0 <= start < end")
    if frame_count is not None:
        if not _integer(frame_count) or frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        if end > frame_count:
            raise ValueError(f"event [{start}, {end}) exceeds frame_count={frame_count}")
    evidence = record.get("evidence_frames")
    if (not isinstance(evidence, list) or not evidence
            or any(not _integer(frame) for frame in evidence)
            or len(set(evidence)) != len(evidence)):
        raise ValueError("evidence_frames must be a nonempty list of unique integers")
    if any(frame < start or frame >= end for frame in evidence):
        raise ValueError("every evidence frame must lie inside [start_frame, end_frame)")
    confidence = record.get("confidence")
    if confidence is not None and (isinstance(confidence, bool)
                                   or not isinstance(confidence, (int, float))
                                   or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("confidence must be null or a finite number in [0, 1]")
    if record.get("label_source") not in LABEL_SOURCES:
        raise ValueError(f"label_source must be one of {sorted(LABEL_SOURCES)}")
    if record.get("review_status") not in REVIEW_STATUSES:
        raise ValueError(f"review_status must be one of {sorted(REVIEW_STATUSES)}")
    if record.get("review_status") == "accepted":
        _text(record, "reviewer")
        _text(record, "reviewed_at")
    if not isinstance(record.get("notes", ""), str):
        raise ValueError("notes must be a string")
    if "revision_history" in record and not isinstance(record["revision_history"], list):
        raise ValueError("revision_history must be a list")
    return start, end


def _valid_provenance(record):
    return (record.get("annotation_mode", ANNOTATION_MODE) == ANNOTATION_MODE
            and record.get("allowed_as_model_input", False) is False)


def _validate_provenance(record):
    if not _valid_provenance(record):
        raise _InvalidAnnotationProvenance(
            "annotation_mode must be offline_noncausal and allowed_as_model_input must be false"
        )


def _occluded(record):
    return record.get("occluded") is True or record.get("occlusion_status") in _OCCLUDED_STATUSES


def validate_event(event, frame_count=None):
    """Validate one actual interaction; rejected/pending events are not labels."""
    _common(event, frame_count)
    _text(event, "event_id")
    if event.get("subject") not in ("left", "right"):
        raise ValueError("an interaction event subject must be left or right")
    object_id = event.get("object_id")
    if object_id is not None and (not isinstance(object_id, str) or not object_id.strip()):
        raise ValueError("object_id must be null or a nonempty stable track identifier")
    _text(event, "object_hint")
    if event.get("interaction_type") not in INTERACTION_TYPES:
        raise ValueError(f"interaction_type must be one of {INTERACTION_TYPES}; none is separate")
    if event.get("geometry_status") not in GEOMETRY_STATUSES:
        raise ValueError(f"geometry_status must be one of {sorted(GEOMETRY_STATUSES)}")
    source_ids = event.get("source_event_ids")
    if not isinstance(source_ids, list) or any(not isinstance(item, str) for item in source_ids):
        raise ValueError("source_event_ids must be a list of source identifiers")
    if event.get("review_status") == "accepted" and object_id is None:
        raise ValueError("accepted interaction events require a linked stable object_id")
    _validate_provenance(event)


def validate_none_interval(interval, frame_count=None):
    """Validate an explicit no-interaction interval, never an unlabelled gap."""
    _common(interval, frame_count)
    interaction_fields = [field for field in _INTERACTION_FIELDS if field in interval]
    if interaction_fields:
        raise ValueError(
            "none intervals must not contain interaction fields: " + ", ".join(interaction_fields)
        )
    identifier = interval.get("interval_id", interval.get("event_id"))
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("none interval requires a nonempty interval_id")
    _validate_provenance(interval)


def _validator(record):
    if isinstance(record, dict) and any(field in record for field in _INTERACTION_FIELDS):
        return validate_event
    return validate_none_interval


def read_jsonl(path, frame_count=None):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                _validator(record)(record, frame_count)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    return records


def write_jsonl(path, records, frame_count=None, overwrite=False):
    """Validate all records before creating a UTF-8 JSONL file."""
    path = Path(path)
    records = list(records)
    identifiers = []
    for record in records:
        _validator(record)(record, frame_count)
        identifiers.append(record.get("event_id", record.get("interval_id")))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("annotation identifiers must be unique within a JSONL file")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    if not overwrite:
        try:
            with path.open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _invalid(frame_index, reason, active):
    return {
        "subject": SUBJECT_TO_ID["none"], "subject_name": "none",
        "object_slot": -1, "object_id": "", "interaction": 0,
        "interaction_name": "none", "valid": False, "reason": reason,
        "event_ids": [item.get("event_id", item.get("interval_id")) for item in active],
        "source": "unreviewed", "label_source": "unreviewed",
        "geometry_status": "not_checked", "annotation_mode": ANNOTATION_MODE,
        "allowed_as_model_input": False, "target_frame": frame_index,
    }


def project_frame_label(events, none_intervals, frame_index, object_ids):
    """Project overlapping annotations onto the legacy three-head target.

    This is a semantic projection only. Formal release additionally checks geometry,
    timestamps, frozen assets and their quality evidence.
    """
    if not _integer(frame_index) or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    object_ids = list(object_ids)
    if len(set(object_ids)) != len(object_ids) or any(not isinstance(item, str) or not item for item in object_ids):
        raise ValueError("object_ids must be unique nonempty stable identifiers")
    active_events = [event for event in events
                     if event["start_frame"] <= frame_index < event["end_frame"]]
    active_none = [interval for interval in none_intervals
                   if interval["start_frame"] <= frame_index < interval["end_frame"]]
    for record in active_events:
        try:
            validate_event(record)
        except _InvalidAnnotationProvenance:
            pass
    for record in active_none:
        try:
            validate_none_interval(record)
        except _InvalidAnnotationProvenance:
            pass
    active = active_events + active_none
    unresolved = [item for item in active if item["review_status"] in ("pending", "needs_revision")]
    if unresolved:
        return _invalid(frame_index, "pending_or_needs_revision", unresolved)
    accepted_events = [item for item in active_events if item["review_status"] == "accepted"]
    accepted_none = [item for item in active_none if item["review_status"] == "accepted"]
    if len(accepted_events) > 1 or (accepted_events and accepted_none) or len(accepted_none) > 1:
        return _invalid(frame_index, "overlapping_accepted_labels", accepted_events + accepted_none)
    if not accepted_events and not accepted_none:
        return _invalid(frame_index, "unlabelled_or_rejected", active)
    accepted = accepted_events + accepted_none
    record = accepted[0]
    if not _valid_provenance(record):
        return _invalid(frame_index, "invalid_annotation_provenance", [record])
    if _occluded(record):
        return _invalid(frame_index, "occluded", [record])
    if accepted_none:
        result = _invalid(frame_index, "reviewed_none", [record])
        result.update(valid=True, reason="reviewed_none", source=record["label_source"],
                      label_source=record["label_source"])
        return result
    event = record
    if event["object_id"] not in object_ids:
        return _invalid(frame_index, "accepted_event_object_not_in_tracks", [event])
    interaction = event["interaction_type"]
    return {
        "subject": SUBJECT_TO_ID[event["subject"]], "subject_name": event["subject"],
        "object_slot": object_ids.index(event["object_id"]), "object_id": event["object_id"],
        "interaction": INTERACTION_TO_ID[interaction], "interaction_name": interaction,
        "valid": True, "reason": "reviewed_event", "event_ids": [event["event_id"]],
        "source": event["label_source"], "label_source": event["label_source"],
        "geometry_status": event["geometry_status"], "annotation_mode": ANNOTATION_MODE,
        "allowed_as_model_input": False, "target_frame": frame_index,
    }
