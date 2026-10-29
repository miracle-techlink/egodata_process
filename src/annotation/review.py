"""Human review CSV and non-destructive application for EgoDemo HOI candidates."""

from __future__ import annotations

import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from src.annotation.egodemo_events import (
    AnnotationError,
    _check_output,
    _input_record,
    _provenance,
    _read_json,
    _sha256,
    _write_json,
    build_summary,
    link_event_tracks,
    load_track_inputs,
)
from src.annotation.schema import validate_event, validate_none_interval, write_jsonl


REVIEW_COLUMNS = [
    "record_type", "record_id", "sequence_id", "review_status", "reviewer", "reviewed_at",
    "subject", "object_id", "proposed_object_id", "object_hint", "interaction_type",
    "start_frame", "end_frame", "evidence_frames", "source_event_ids", "notes",
    "original_start_frame", "original_end_frame", "label_source", "geometry_status",
    "track_link_status",
]
REVIEW_STATUSES = {"pending", "accepted", "rejected", "needs_revision"}
EDITABLE_FIELDS = (
    "subject", "object_id", "object_hint", "interaction_type", "start_frame", "end_frame",
    "evidence_frames", "source_event_ids", "notes",
)


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _row(record, record_type):
    none = record_type == "none"
    identifier = record["interval_id"] if none else record["event_id"]
    original = record.get("original_candidate", record)
    return {
        "record_type": record_type,
        "record_id": identifier,
        "sequence_id": record["sequence_id"],
        "review_status": record["review_status"],
        "reviewer": record.get("reviewer", ""),
        "reviewed_at": record.get("reviewed_at", ""),
        "subject": "" if none else record["subject"],
        "object_id": "" if none else record.get("object_id") or "",
        "proposed_object_id": "" if none else record.get("proposed_object_id") or "",
        "object_hint": "" if none else record["object_hint"],
        "interaction_type": "" if none else record["interaction_type"],
        "start_frame": record["start_frame"],
        "end_frame": record["end_frame"],
        "evidence_frames": _json_text(record["evidence_frames"]),
        "source_event_ids": _json_text(record.get("source_event_ids", [])),
        "notes": record.get("notes", ""),
        "original_start_frame": original["start_frame"],
        "original_end_frame": original["end_frame"],
        "label_source": record["label_source"],
        "geometry_status": record.get("geometry_status", "not_checked"),
        "track_link_status": "" if none else record.get("track_link_status", ""),
    }


def write_review_csv(path, events, none_intervals):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite review CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for event in events:
            writer.writerow(_row(event, "event"))
        for interval in none_intervals:
            writer.writerow(_row(interval, "none"))
    return path


def _read_jsonl(path, validator, frame_count):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validator(record, frame_count)
            except (json.JSONDecodeError, ValueError) as error:
                raise AnnotationError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    identifiers = [record.get("event_id", record.get("interval_id")) for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise AnnotationError(f"Duplicate annotation identifiers in {path}")
    return records


def _load_native(annotation_dir):
    document = _read_json(Path(annotation_dir) / "timeline.json")
    if not isinstance(document, dict) or document.get("timeline") not in {
        "native", "native_annotated", "annotated_native"
    }:
        raise AnnotationError("timeline.json must declare the annotated native timeline")
    frame_count = document.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise AnnotationError("timeline.json frame_count must be a positive integer")
    timestamps = np.asarray(document.get("timestamps"), dtype=np.float64)
    frame_indices = document.get("frame_index")
    if timestamps.shape != (frame_count,) or not np.isfinite(timestamps).all():
        raise AnnotationError("timeline.json timestamps must be finite and match frame_count")
    if (np.diff(timestamps) <= 0).any() or frame_indices != list(range(frame_count)):
        raise AnnotationError("timeline.json must preserve strictly increasing native 0..T-1 frames")
    sequence_id = document.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise AnnotationError("timeline.json sequence_id must be nonempty")
    source_maps = document.get("source_events")
    if not isinstance(source_maps, list):
        raise AnnotationError("timeline.json source_events must be a list")
    for source in source_maps:
        start, end = source.get("start_frame"), source.get("end_frame")
        if start is None or end is None:
            if start is not None or end is not None:
                raise AnnotationError("Source native boundaries must both be null or both be integers")
            continue
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                or not isinstance(end, int) or not 0 <= start < end <= frame_count):
            raise AnnotationError("Invalid source native boundaries in timeline.json")
    return {
        "sequence_id": sequence_id,
        "frame_count": frame_count,
        "timestamps": timestamps,
        "frame_indices": np.arange(frame_count, dtype=np.int64),
        "domain_end": document.get("domain_end_seconds"),
        "domain_method": document.get("domain_end_method"),
        "source_maps": source_maps,
    }


def _read_review(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(fields) != set(REVIEW_COLUMNS):
            missing = sorted(set(REVIEW_COLUMNS) - set(fields))
            extra = sorted(set(fields) - set(REVIEW_COLUMNS))
            raise AnnotationError(f"review CSV columns differ; missing={missing}, extra={extra}")
        rows = []
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise AnnotationError(f"Malformed review CSV row at line {line_number}")
            row = {key: value.strip() for key, value in row.items()}
            row["_line"] = line_number
            rows.append(row)
    seen = set()
    for row in rows:
        key = row["record_id"]
        if not key or key in seen:
            raise AnnotationError(f"Missing or duplicate review record_id at line {row['_line']}: {key!r}")
        seen.add(key)
        if row["record_type"] not in {"event", "none"}:
            raise AnnotationError(f"Line {row['_line']}: record_type must be event or none")
        if row["review_status"] not in REVIEW_STATUSES:
            raise AnnotationError(f"Line {row['_line']}: unsupported review_status")
    return rows


def _identity(row):
    if row["review_status"] == "pending":
        return
    if not row["reviewer"].strip():
        raise AnnotationError(f"{row['record_id']}: {row['review_status']} requires reviewer")
    value = row["reviewed_at"]
    if not value:
        raise AnnotationError(f"{row['record_id']}: {row['review_status']} requires reviewed_at")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise AnnotationError(f"{row['record_id']}: reviewed_at must be ISO 8601 with a timezone") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AnnotationError(f"{row['record_id']}: reviewed_at must include a timezone")


def _integer(value, row, field, minimum=0):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"{row['record_id']}: {field} must be an integer") from error
    if str(result) != value.strip() or result < minimum:
        raise AnnotationError(f"{row['record_id']}: invalid {field} {value!r}")
    return result


def _json_list(value, row, field):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise AnnotationError(f"{row['record_id']}: {field} must be a JSON list") from error
    if not isinstance(result, list):
        raise AnnotationError(f"{row['record_id']}: {field} must be a JSON list")
    return result


def _common_from_row(row, native):
    if row["sequence_id"] != native["sequence_id"]:
        raise AnnotationError(f"{row['record_id']}: review sequence_id disagrees with native timeline")
    start = _integer(row["start_frame"], row, "start_frame")
    end = _integer(row["end_frame"], row, "end_frame")
    if end <= start or end > native["frame_count"]:
        raise AnnotationError(
            f"{row['record_id']}: require 0 <= start_frame < end_frame <= {native['frame_count']}"
        )
    evidence = _json_list(row["evidence_frames"], row, "evidence_frames")
    if (not evidence or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in evidence)
            or len(set(evidence)) != len(evidence)
            or any(frame < start or frame >= end for frame in evidence)):
        raise AnnotationError(f"{row['record_id']}: evidence frames must be unique and inside reviewed bounds")
    source_ids = _json_list(row["source_event_ids"], row, "source_event_ids")
    if any(not isinstance(value, str) for value in source_ids):
        raise AnnotationError(f"{row['record_id']}: source_event_ids must contain strings")
    return {
        "start_frame": start,
        "end_frame": end,
        "evidence_frames": evidence,
        "source_event_ids": source_ids,
        "notes": row["notes"],
    }


def _snapshot(record):
    result = {field: copy.deepcopy(record.get(field)) for field in EDITABLE_FIELDS}
    result.update(
        review_status=record["review_status"],
        reviewer=record.get("reviewer"),
        reviewed_at=record.get("reviewed_at"),
        track_link_status=record.get("track_link_status"),
        geometry_status=record.get("geometry_status", "not_checked"),
        label_source=record["label_source"],
    )
    return result


def _row_snapshot(row, common):
    result = {field: copy.deepcopy(common.get(field)) for field in EDITABLE_FIELDS}
    result.update(
        subject=row["subject"] or None,
        object_id=row["object_id"] or None,
        object_hint=row["object_hint"],
        interaction_type=row["interaction_type"] or None,
        review_status=row["review_status"],
        reviewer=row["reviewer"] or None,
        reviewed_at=row["reviewed_at"] or None,
        label_source=row["label_source"],
        track_link_status=row["track_link_status"] or None,
        geometry_status=row["geometry_status"],
    )
    return result


def _append_revision(record, before, original_record):
    if "original_candidate" not in record:
        original = copy.deepcopy(original_record)
        original.pop("revision_history", None)
        original.pop("original_candidate", None)
        record["original_candidate"] = original
    history = copy.deepcopy(record.get("revision_history", []))
    history.append({
        "revision": len(history) + 1,
        "reviewer": record.get("reviewer"),
        "reviewed_at": record.get("reviewed_at"),
        "review_status": record["review_status"],
        "before": before,
        "after": _snapshot(record),
    })
    record["revision_history"] = history


def _apply_existing_identity(record, row):
    if row["review_status"] == "pending":
        return
    record["reviewer"] = row["reviewer"]
    record["reviewed_at"] = row["reviewed_at"]


def _apply_event(record, row, native, tracks, expand_ratio):
    common = _common_from_row(row, native)
    current = _snapshot(record)
    proposed_snapshot = _row_snapshot(row, common)
    unchanged = all(
        proposed_snapshot.get(field) == current.get(field)
        for field in (*EDITABLE_FIELDS, "review_status", "reviewer", "reviewed_at")
    )
    if row["review_status"] == "pending":
        changed_semantics = any(proposed_snapshot.get(field) != current.get(field) for field in EDITABLE_FIELDS)
        if changed_semantics:
            raise AnnotationError(f"{record['event_id']}: pending rows cannot silently modify candidates")
        return copy.deepcopy(record)
    _identity(row)
    if row["label_source"] != record["label_source"] or row["geometry_status"] != record["geometry_status"]:
        raise AnnotationError(f"{record['event_id']}: label_source/geometry_status are not manually editable")
    if common["source_event_ids"] != record["source_event_ids"]:
        raise AnnotationError(f"{record['event_id']}: source_event_ids and their order must be preserved")
    updated = copy.deepcopy(record)
    updated.update(common)
    updated.update(
        subject=row["subject"],
        object_hint=row["object_hint"],
        interaction_type=row["interaction_type"],
        review_status=row["review_status"],
        label_source="human",
    )
    _apply_existing_identity(updated, row)
    reviewed_object_id = row["object_id"] or None
    current_object_id = record.get("object_id")
    link_changed = reviewed_object_id != current_object_id or record.get("track_link_status") != "linked"
    if reviewed_object_id is not None and (link_changed or tracks is not None):
        if tracks is None:
            raise AnnotationError(
                f"{record['event_id']}: changing or establishing object_id requires matching --pose6d"
            )
        updated["proposed_object_id"] = reviewed_object_id
        link_event_tracks(updated, tracks, expand_ratio=expand_ratio)
        if updated["track_link_status"] != "linked":
            raise AnnotationError(f"{record['event_id']}: reviewed object_id is not a real tracker ID")
    elif reviewed_object_id is None:
        if row["review_status"] == "accepted":
            raise AnnotationError(f"{record['event_id']}: accepted interaction requires a linked stable object_id")
        updated["proposed_object_id"] = None
        updated["object_id"] = None
        updated["track_link_status"] = "not_provided"
        updated["geometry_status"] = "not_checked"
        updated["geometry_evidence"] = {
            "method": "not_evaluated", "scope": "none", "3d_verified": False,
            "automatic_accept": False,
        }
    elif tracks is not None:
        updated["proposed_object_id"] = reviewed_object_id
        link_event_tracks(updated, tracks, expand_ratio=expand_ratio)
    else:
        if record.get("track_link_status") != "linked":
            raise AnnotationError(f"{record['event_id']}: existing object_id lacks a verified track link")
        updated["object_id"] = reviewed_object_id
        updated["proposed_object_id"] = reviewed_object_id
    if row["review_status"] == "accepted" and (
        updated.get("track_link_status") != "linked" or updated.get("object_id") is None
    ):
        raise AnnotationError(f"{record['event_id']}: accepted interaction requires a verified tracker link")
    try:
        validate_event(updated, native["frame_count"])
    except ValueError as error:
        raise AnnotationError(f"{record['event_id']}: invalid reviewed event: {error}") from error
    if not unchanged:
        _append_revision(updated, current, record)
    return updated


def _new_none(row, native):
    common = _common_from_row(row, native)
    if common["source_event_ids"]:
        raise AnnotationError(f"{row['record_id']}: a human none interval cannot inherit semantic event IDs")
    if row["subject"] or row["object_id"] or row["object_hint"] or row["interaction_type"]:
        raise AnnotationError(f"{row['record_id']}: none rows must leave subject/object/interaction empty")
    interval = {
        "interval_id": row["record_id"],
        "sequence_id": native["sequence_id"],
        **common,
        "confidence": None,
        "label_source": "human",
        "review_status": "pending",
        "geometry_status": "not_checked",
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
        "revision_history": [],
    }
    return interval


def _apply_none(record, row, native, is_new):
    common = _common_from_row(row, native)
    if row["subject"] or row["object_id"] or row["object_hint"] or row["interaction_type"]:
        raise AnnotationError(f"{row['record_id']}: none rows must leave subject/object/interaction empty")
    if is_new and common["source_event_ids"]:
        raise AnnotationError(f"{row['record_id']}: a human none interval cannot inherit semantic source event IDs")
    if not is_new and common["source_event_ids"] != record.get("source_event_ids", []):
        raise AnnotationError(f"{row['record_id']}: source_event_ids and their order must be preserved")
    if row["review_status"] == "pending":
        if not is_new and any(common[field] != record.get(field) for field in common):
            raise AnnotationError(f"{row['record_id']}: pending rows cannot silently modify none intervals")
        return copy.deepcopy(record)
    _identity(row)
    current = _snapshot(record)
    updated = copy.deepcopy(record)
    updated.update(common)
    updated["review_status"] = row["review_status"]
    updated["label_source"] = "human"
    _apply_existing_identity(updated, row)
    try:
        validate_none_interval(updated, native["frame_count"])
    except ValueError as error:
        raise AnnotationError(f"{row['record_id']}: invalid none interval: {error}") from error
    after = _snapshot(updated)
    changed = any(after.get(field) != current.get(field) for field in (
        *EDITABLE_FIELDS, "review_status", "reviewer", "reviewed_at", "label_source"
    ))
    if changed or is_new:
        _append_revision(updated, current, record)
    return updated


def apply_review(
    annotation_dir, output_dir, *, review_csv=None, hand_pose=None, pose6d=None,
    image_width=None, image_height=None, video=None, render_format="mp4", expand_ratio=0.25,
):
    """Apply a complete review CSV into a fresh directory and retain all prior candidates."""
    annotation_dir = Path(annotation_dir).expanduser().resolve()
    output = _check_output(output_dir)
    native = _load_native(annotation_dir)
    events = _read_jsonl(annotation_dir / "events.jsonl", validate_event, native["frame_count"])
    none_intervals = _read_jsonl(
        annotation_dir / "none_intervals.jsonl", validate_none_interval, native["frame_count"]
    )
    collisions = sorted(
        {event["event_id"] for event in events}
        & {interval["interval_id"] for interval in none_intervals}
    )
    if collisions:
        raise AnnotationError(f"Event/none identifiers collide across annotation files: {collisions}")
    review_path = Path(review_csv).expanduser().resolve() if review_csv else annotation_dir / "review.csv"
    rows = _read_review(review_path)
    row_by_id = {row["record_id"]: row for row in rows}
    existing_ids = {
        *(event["event_id"] for event in events),
        *(interval["interval_id"] for interval in none_intervals),
    }
    missing = sorted(existing_ids - set(row_by_id))
    if missing:
        raise AnnotationError(f"Review CSV is missing existing records: {missing}")
    for event in events:
        if row_by_id[event["event_id"]]["record_type"] != "event":
            raise AnnotationError(f"{event['event_id']}: record_type must be event")
    for interval in none_intervals:
        if row_by_id[interval["interval_id"]]["record_type"] != "none":
            raise AnnotationError(f"{interval['interval_id']}: record_type must be none")
    tracks = load_track_inputs(
        native, hand_pose_path=hand_pose, pose6d_path=pose6d,
        image_width=image_width, image_height=image_height,
    ) if hand_pose is not None or pose6d is not None else None
    reviewed_events = [
        _apply_event(event, row_by_id[event["event_id"]], native, tracks, expand_ratio)
        for event in events
    ]
    reviewed_none = [
        _apply_none(interval, row_by_id[interval["interval_id"]], native, False)
        for interval in none_intervals
    ]
    for interval in none_intervals:
        row_by_id.pop(interval["interval_id"])
    for event in events:
        row_by_id.pop(event["event_id"])
    for row in rows:
        if row["record_id"] not in row_by_id:
            continue
        if row["record_type"] != "none":
            raise AnnotationError(f"{row['record_id']}: unknown event ID; refusing to invent a human HOI event")
        interval = _new_none(row, native)
        reviewed_none.append(_apply_none(interval, row, native, True))
    for event in reviewed_events:
        validate_event(event, native["frame_count"])
    for interval in reviewed_none:
        validate_none_interval(interval, native["frame_count"])
    prior_summary_path = annotation_dir / "summary.json"
    prior_summary = _read_json(prior_summary_path) if prior_summary_path.is_file() else {}
    raw_vlm_count = prior_summary.get("raw_vlm_candidates", len(events))
    raw_vlm_none_count = prior_summary.get("raw_vlm_none_candidates", len(none_intervals))
    inputs = []
    for name, role in (
        ("events.jsonl", "candidate_events"), ("none_intervals.jsonl", "candidate_none_intervals"),
        ("review.csv", "prepared_review_csv"), ("timeline.json", "native_timeline"),
        ("provenance.json", "parent_provenance"),
    ):
        path = annotation_dir / name
        if path.is_file():
            inputs.append(_input_record(path, role))
    inputs.append(_input_record(review_path, "applied_review_csv"))
    for path, role in ((hand_pose, "hand_pose_npz"), (pose6d, "pose6d_npz"), (video, "annotated_video")):
        if path is not None:
            inputs.append(_input_record(path, role))
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        for path in annotation_dir.iterdir():
            if path.is_file() and (
                path.name in {"source_events.json", "source_annotation_review.csv"}
                or path.name.startswith(("vlm_input.", "vlm_none_input."))
            ):
                shutil.copyfile(path, stage / path.name)
        parent_provenance = annotation_dir / "provenance.json"
        if parent_provenance.is_file():
            shutil.copyfile(parent_provenance, stage / "parent_provenance.json")
        shutil.copyfile(annotation_dir / "timeline.json", stage / "timeline.json")
        write_jsonl(stage / "events.jsonl", reviewed_events, native["frame_count"])
        write_jsonl(stage / "none_intervals.jsonl", reviewed_none, native["frame_count"])
        write_review_csv(stage / "review.csv", reviewed_events, reviewed_none)
        render_info = {"status": "not_requested", "reason": "provide --video to render reviewed evidence"}
        if video is not None:
            from src.annotation.render_review import render_review
            render_path = stage / ("review.mp4" if render_format == "mp4" else "contact_sheet.jpg")
            render_info = render_review(stage, video, render_path, output_format=render_format)
            render_info["path"] = str(output / render_path.name)
        object_ids = tracks["object_ids"] if tracks else sorted({
            event["object_id"] for event in reviewed_events if event.get("object_id")
        })
        summary = build_summary(
            native, reviewed_events, reviewed_none, raw_vlm_count=raw_vlm_count,
            object_ids=object_ids, render=render_info, raw_vlm_none_count=raw_vlm_none_count,
        )
        _write_json(stage / "summary.json", summary)
        _provenance(stage, native, inputs, {
            "operation": "apply",
            "parent_annotation_dir": str(annotation_dir),
            "review_implementation_sha256": _sha256(Path(__file__).resolve()),
            "render": render_info,
        })
        stage.rename(output)
    return {"output_dir": str(output), **summary}
