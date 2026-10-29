"""Apply exported review-decision CSVs back into per-sequence review dirs (hoi_annotation_v1).

This is the fast-review-kit counterpart of src/annotation/review.py. The kit
exports ``review_decisions_20261023.csv`` with one row per *decided* candidate:

    sequence_id,record_id,review_status,reviewer,object_id,notes

Undecided records are simply absent and stay ``pending``. All statuses are
human decisions (gold); the video kit carries no tracker/geometry evidence, so
accepted object_ids are recorded as ``proposed_object_id`` with
``track_link_status="human_pending_geometry"`` and ``geometry_status``
unchanged (``not_checked``). Per contract the accepted object identity is
exactly the one the reviewer typed -- tracker reconciliation is a later step.

Writes (the only files this script touches, and only without --dry-run):

* ``<root>/<seq8>_qwen25vl7b_20261023_review/review.csv``   (updated in place, .bak backup)
* ``<root>/<seq8>_qwen25vl7b_20261023_review/summary.json`` (regenerated counters + projection)
* ``<root>/<seq8>_qwen25vl7b_20261023_review/labels_projected.jsonl`` (new, one line per native frame)

Usage:

    python -m src.annotation.apply_review --decisions review_decisions_20261023.csv \
        --root /root/gpufree-data/datasets/ego_hoi/vlm_annotation_20261023 [--dry-run] \
        [--seq8 0458190e ...]

The VLM input directory ``<root>/<seq8>_qwen25vl7b_20261023/`` is never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.annotation.egodemo_events import AnnotationError, _read_json
from src.annotation.schema import (
    project_frame_label,
    validate_event,
    validate_none_interval,
    write_jsonl,
)

DECISION_COLUMNS = ("sequence_id", "record_id", "review_status", "reviewer", "object_id", "notes")
DECISION_STATUSES = {"accepted", "rejected", "needs_revision"}
DIRECTORY_SUFFIX = "_qwen25vl7b_20261023"


def _seq8(sequence_id):
    if not isinstance(sequence_id, str) or len(sequence_id) < 8:
        raise AnnotationError(f"sequence_id must be a UUID-like string: {sequence_id!r}")
    return sequence_id[:8]


def load_decisions(path):
    """Read and structurally validate the exported decision CSV.

    Returns {record_id: {"sequence_id", "status", "reviewer", "object_id", "notes", "line"}}.
    Duplicate record_ids and semantic violations are rejected here, per-sequence
    cross-checks happen in validate_decisions().
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise AnnotationError(f"decision CSV not found: {path}")
    decisions = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        missing = [col for col in DECISION_COLUMNS if col not in fields]
        if missing:
            raise AnnotationError(f"decision CSV is missing columns: {missing} (found {fields})")
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise AnnotationError(f"malformed decision CSV row at line {line_number}")
            row = {key: (value or "").strip() for key, value in row.items()}
            record_id = row["record_id"]
            if not record_id:
                raise AnnotationError(f"line {line_number}: record_id is empty")
            if record_id in decisions:
                raise AnnotationError(f"line {line_number}: duplicate decision for {record_id}")
            if row["review_status"] not in DECISION_STATUSES:
                raise AnnotationError(
                    f"{record_id}: review_status must be one of {sorted(DECISION_STATUSES)}, "
                    f"got {row['review_status']!r}"
                )
            if not row["reviewer"]:
                raise AnnotationError(f"{record_id}: reviewer is empty")
            if not row["sequence_id"]:
                raise AnnotationError(f"{record_id}: sequence_id is empty")
            if row["review_status"] == "accepted" and not row["object_id"]:
                raise AnnotationError(f"{record_id}: accepted decisions require a nonempty object_id")
            decisions[record_id] = {
                "sequence_id": row["sequence_id"],
                "status": row["review_status"],
                "reviewer": row["reviewer"],
                "object_id": row["object_id"],
                "notes": row["notes"],
                "line": line_number,
            }
    if not decisions:
        raise AnnotationError("decision CSV contains no rows; refusing an empty apply")
    return decisions


def _load_sequence(review_dir):
    native = _read_json(review_dir / "timeline.json")
    frame_count = native.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise AnnotationError(f"{review_dir}/timeline.json: invalid frame_count")
    events, none_intervals = [], []
    for name, validator, bucket in (
        ("events.jsonl", validate_event, events),
        ("none_intervals.jsonl", validate_none_interval, none_intervals),
    ):
        with (review_dir / name).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    validator(record, frame_count)
                except ValueError as error:
                    raise AnnotationError(f"{review_dir}/{name}:{line_number}: {error}") from error
                bucket.append(record)
    summary = _read_json(review_dir / "summary.json") if (review_dir / "summary.json").is_file() else {}
    return {"native": native, "events": events, "none_intervals": none_intervals, "summary": summary}


def validate_decisions(decisions, sequences):
    """Cross-check every decision against the review dirs. Returns (matched, unknown)."""
    known_ids, sequence_by_record = set(), {}
    for review_dir, seq in sequences.items():
        sequence_id = seq["native"]["sequence_id"]
        for record in seq["events"]:
            known_ids.add(record["event_id"])
            sequence_by_record[record["event_id"]] = sequence_id
        for interval in seq["none_intervals"]:
            known_ids.add(interval["interval_id"])
            sequence_by_record[interval["interval_id"]] = sequence_id
    unknown = [record_id for record_id in decisions if record_id not in known_ids]
    if unknown:
        raise AnnotationError(
            "decision CSV references record_ids not present in the collection:\n  " + "\n  ".join(unknown)
        )
    mismatched = [
        record_id for record_id, decision in decisions.items()
        if decision["sequence_id"] != sequence_by_record[record_id]
    ]
    if mismatched:
        detail = "; ".join(
            f"{rid}: csv={decisions[rid]['sequence_id']} actual={sequence_by_record[rid]}" for rid in mismatched
        )
        raise AnnotationError(f"decision CSV sequence_id disagrees with the collection: {detail}")
    return True


def _apply_to_event(event, decision, reviewed_at):
    event["review_status"] = decision["status"]
    event["reviewer"] = decision["reviewer"]
    event["reviewed_at"] = reviewed_at
    event["label_source"] = "human"
    if decision["notes"]:
        event["notes"] = decision["notes"]
    if decision["status"] == "accepted":
        event["object_id"] = decision["object_id"]
        event["proposed_object_id"] = decision["object_id"]
        event["track_link_status"] = "human_pending_geometry"
    elif decision["status"] == "rejected":
        event["object_id"] = None
        event["proposed_object_id"] = None
        event["track_link_status"] = "not_provided"
    else:  # needs_revision: keep semantics pending, record the proposed identity if given
        event["proposed_object_id"] = decision["object_id"] or None
    return event


def apply_to_sequence(seq, decisions, reviewed_at):
    """Mutate seq['events']/['none_intervals'] in place; returns (updated_ids, per-frame labels)."""
    events, none_intervals, native = seq["events"], seq["none_intervals"], seq["native"]
    frame_count = native["frame_count"]
    updated = []
    for record in events:
        decision = decisions.get(record["event_id"])
        if decision is None:
            continue
        _apply_to_event(record, decision, reviewed_at)
        updated.append(record["event_id"])
    for interval in none_intervals:
        decision = decisions.get(interval["interval_id"])
        if decision is None:
            continue
        interval["review_status"] = decision["status"]
        interval["reviewer"] = decision["reviewer"]
        interval["reviewed_at"] = reviewed_at
        interval["label_source"] = "human"
        if decision["notes"]:
            interval["notes"] = decision["notes"]
        updated.append(interval["interval_id"])
    try:
        for record in events:
            validate_event(record, frame_count)
        for interval in none_intervals:
            validate_none_interval(interval, frame_count)
    except ValueError as error:
        raise AnnotationError(f"{native['sequence_id']}: post-apply validation failed: {error}") from error
    object_ids = sorted({event["object_id"] for event in events if event.get("object_id")})
    labels = [
        project_frame_label(events, none_intervals, frame, object_ids) for frame in range(frame_count)
    ]
    return updated, labels, object_ids


def _refresh_summary(summary, seq, labels, object_ids):
    frame_count = seq["native"]["frame_count"]
    events, none_intervals = seq["events"], seq["none_intervals"]
    statuses = Counter(record["review_status"] for record in events + none_intervals)
    geometry = Counter(event["geometry_status"] for event in events)
    reasons = Counter(label["reason"] for label in labels)
    valid_frames = sum(1 for label in labels if label["valid"])

    def coverage(records):
        mask = [0] * frame_count
        for record in records:
            for frame in range(record["start_frame"], min(record["end_frame"], frame_count)):
                mask[frame] += 1
        return mask

    accepted = [event for event in events if event["review_status"] == "accepted"]
    accepted_none = [interval for interval in none_intervals if interval["review_status"] == "accepted"]
    acc_cov, none_cov = coverage(accepted), coverage(accepted_none)
    refreshed = dict(summary)
    refreshed.update({
        "review_status_counts": {
            key: statuses.get(key, 0) for key in ("pending", "accepted", "rejected", "needs_revision")
        },
        "geometry_status_counts": {
            key: geometry.get(key, 0) for key in ("not_checked", "passed", "failed", "inconclusive")
        },
        "track_linked_candidates": sum(
            event.get("track_link_status") == "linked" for event in events
        ),
        "accepted_event_coverage_frames": sum(1 for value in acc_cov if value > 0),
        "accepted_event_overlap_frames": sum(1 for value in acc_cov if value > 1),
        "accepted_none_intervals": len(accepted_none),
        "accepted_none_coverage_frames": sum(1 for value in none_cov if value > 0),
        "accepted_object_ids": object_ids,
        "semantic_projection": {
            "valid_frames": valid_frames,
            "label_valid_zero_frames": frame_count - valid_frames,
            "reason_counts": dict(sorted(reasons.items())),
        },
        "human_review_completed": statuses.get("pending", 0) == 0,
        "training_ready": False,
        "semantic_projection_only": True,
    })
    return refreshed


def _refresh_review_csv(path, decisions, reviewed_at):
    """Rewrite the per-sequence review.csv in place (decided rows only). Keeps column order."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fieldnames = reader.fieldnames
        rows = [row for row in reader]
    touched = 0
    for row in rows:
        decision = decisions.get(row["record_id"])
        if decision is None:
            continue  # undecided rows stay pending
        row["review_status"] = decision["status"]
        row["reviewer"] = decision["reviewer"]
        row["reviewed_at"] = reviewed_at
        row["label_source"] = "human"
        if decision["notes"]:
            row["notes"] = decision["notes"]
        if row["record_type"] == "event":
            if decision["status"] == "accepted":
                row["object_id"] = decision["object_id"]
                row["proposed_object_id"] = decision["object_id"]
                row["track_link_status"] = "human_pending_geometry"
            elif decision["status"] == "rejected":
                row["object_id"] = ""
                row["proposed_object_id"] = ""
                row["track_link_status"] = "not_provided"
            else:
                row["proposed_object_id"] = decision["object_id"]
        # none rows carry no object fields; status/reviewer/notes already updated above
        touched += 1
    return fieldnames, rows, touched


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json_overwrite(path, value):
    """Same serialization style as egodemo_events._write_json, but replaces an existing file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp_apply")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _persist_sequence(review_dir, seq):
    """Rewrite events.jsonl / none_intervals.jsonl so review.csv stays consistent."""
    write_jsonl(review_dir / "events.jsonl", seq["events"], seq["native"]["frame_count"], overwrite=True)
    write_jsonl(
        review_dir / "none_intervals.jsonl", seq["none_intervals"], seq["native"]["frame_count"],
        overwrite=True,
    )


def run(decisions_path, root, *, dry_run=False, seq8_filter=None):
    decisions = load_decisions(decisions_path)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise AnnotationError(f"collection root not found: {root}")

    seq8s = sorted({_seq8(decision["sequence_id"]) for decision in decisions.values()})
    if seq8_filter:
        seq8s = [item for item in seq8s if item in seq8_filter]
    sequences = {}
    for seq8 in seq8s:
        review_dir = root / f"{seq8}{DIRECTORY_SUFFIX}_review"
        if not review_dir.is_dir():
            raise AnnotationError(f"missing review directory for seq8 {seq8}: {review_dir}")
        sequences[review_dir] = _load_sequence(review_dir)
    validate_decisions(decisions, sequences)

    decisions_by_sequence = {}
    for record_id, decision in decisions.items():
        decisions_by_sequence.setdefault(_seq8(decision["sequence_id"]), {})[record_id] = decision

    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    report = []
    for review_dir, seq in sequences.items():
        seq8 = review_dir.name[:8]
        scoped = decisions_by_sequence.get(seq8, {})
        if not scoped:
            continue
        # Deep-ish copy semantics: apply on the loaded records, but only write on real run.
        updated, labels, object_ids = apply_to_sequence(seq, scoped, reviewed_at)
        refreshed = _refresh_summary(seq["summary"], seq, labels, object_ids)
        fieldnames, csv_rows, csv_touched = _refresh_review_csv(
            review_dir / "review.csv", scoped, reviewed_at
        )
        if csv_touched != len(scoped):
            raise AnnotationError(
                f"{review_dir.name}: review.csv update touched {csv_touched} rows but "
                f"{len(scoped)} decisions were scoped here (record_type mismatch?)"
            )
        labels_path = review_dir / "labels_projected.jsonl"
        if labels_path.exists():
            raise AnnotationError(
                f"refusing to overwrite existing labels projection: {labels_path} (delete it first to re-apply)"
            )
        report.append({
            "review_dir": str(review_dir),
            "updated": len(updated),
            "statuses": dict(Counter(decision["status"] for decision in scoped.values())),
            "projection": refreshed["semantic_projection"],
        })
        if dry_run:
            continue
        csv_path = review_dir / "review.csv"
        shutil.copyfile(csv_path, review_dir / "review.csv.bak_apply")
        _write_csv(csv_path, fieldnames, csv_rows)
        _persist_sequence(review_dir, seq)
        _write_json_overwrite(review_dir / "summary.json", refreshed)
        _write_jsonl(labels_path, labels)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True, help="exported review_decisions_20261023.csv")
    parser.add_argument("--root", required=True, help="vlm_annotation_20261023 root (read-only except *_review)")
    parser.add_argument("--seq8", nargs="*", default=None, help="restrict apply to these seq8 prefixes")
    parser.add_argument("--dry-run", action="store_true", help="validate only; write nothing")
    args = parser.parse_args(argv)
    report = run(args.decisions, args.root, dry_run=args.dry_run, seq8_filter=set(args.seq8 or []))
    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(f"[{mode}] {len(report)} sequence(s) processed")
    for item in report:
        proj = item["projection"]
        print(
            f"  {Path(item['review_dir']).name}: updated={item['updated']} "
            f"statuses={item['statuses']} valid_frames={proj['valid_frames']} "
            f"label_valid_zero={proj['label_valid_zero_frames']}"
        )
    if args.dry_run:
        print("dry-run only: review.csv / summary.json / labels_projected.jsonl untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
