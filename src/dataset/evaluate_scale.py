"""Audit real dataset scale without counting views or windows as acquisitions.

Example:
    python -m src.dataset.evaluate_scale \
      --progress /data/target_120_progress.json \
      --manifest /data/candidate_manifest.server.json \
      --freeze /data/frozen_server_v2_complete \
      --annotations /data/annotations \
      --sample-index /data/samples/dataset_index.json \
      --json-output scale.json --markdown-output scale.md

Only actual target-frame inventories determine valid/invalid target frames. Missing
inventories stay unknown; accepted events alone do not create projected labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys


SPLITS = ("train", "val", "test")
ALL_SPLITS = (*SPLITS, "candidate")
INTERACTION_TYPES = ("reach", "grasp", "hold", "move", "place", "release")
REVIEW_STATUSES = ("accepted", "rejected", "pending", "needs_revision")
SUBJECT_TO_ID = {"left": 0, "right": 1, "none": 2}
INTERACTION_TO_ID = {"none": 0, **{name: index + 1 for index, name in enumerate(INTERACTION_TYPES)}}
LABEL_SOURCES = {"vlm", "source_text", "source_text+vlm", "geometry_assisted", "human"}
GEOMETRY_STATUSES = {"not_checked", "passed", "failed", "inconclusive"}
UNKNOWN = {"", "unknown", "pending", "none", "null", "n/a", "na", "unverified", "?"}
SEMANTIC_FIELDS = ("object_semantic_labels", "object_categories", "object_category_ids")
DEFAULT_BENCHMARK = Path(__file__).resolve().parents[2] / "configs/scale_benchmark_v1.json"
DEFAULT_RELEASE_CONFIG = Path(__file__).resolve().parents[2] / "configs/human_dataset_v1.json"


class ScaleEvaluationError(ValueError):
    pass


def _known(value):
    return isinstance(value, str) and value.strip().lower() not in UNKNOWN


def _read_json(path):
    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ScaleEvaluationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScaleEvaluationError(f"Expected a JSON object: {path}")
    return value, raw


def _provenance(path, raw=None):
    path = Path(path)
    if raw is None:
        raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _values(value):
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        value = [value]
    return [item.strip() for item in value if _known(item)]


def _namespace(record, value):
    dataset = record.get("dataset") if _known(record.get("dataset")) else "unknown_dataset"
    return f"{dataset}:{value}"


def _dimension(groups, records, field, *, namespace=True, semantic=False):
    namespaced_values = set()
    display_values = set()
    unknown = 0
    for group in groups:
        group_values = set()
        for index in group["record_indices"]:
            record = records[index]
            fields = SEMANTIC_FIELDS if semantic else (field,)
            for candidate in fields:
                for value in _values(record.get(candidate)):
                    group_values.add(value)
                    namespaced_values.add(_namespace(record, value) if namespace else value)
                    display_values.add(value)
        if not group_values:
            unknown += 1
    if not groups:
        status = "no_acquisitions"
    elif unknown == len(groups):
        status = "unknown"
    elif unknown:
        status = "partially_known"
    else:
        status = "known"
    return {
        "count": len(namespaced_values),
        "values": sorted(display_values),
        "namespaced_values": sorted(namespaced_values),
        "unknown_acquisitions": unknown,
        "status": status,
    }


def _summarize(groups, records):
    groups = list(groups)
    indices = {index for group in groups for index in group["record_indices"]}
    return {
        "records": len(indices),
        "independent_acquisitions": len(groups),
        "duplicate_view_or_record_excess": len(indices) - len(groups),
        "dimensions": {
            "subjects": _dimension(groups, records, "subject_id"),
            "scenes": _dimension(groups, records, "scene_id"),
            "object_instances": _dimension(groups, records, "object_ids"),
            "object_semantic_categories": _dimension(groups, records, None, semantic=True),
            "actions": _dimension(groups, records, "action_labels", namespace=False),
        },
    }


def _load_freeze(freeze):
    freeze = Path(freeze)
    plan, plan_raw = _read_json(freeze / "candidate_split.json")
    audit, audit_raw = _read_json(freeze / "audit_report.json")
    seal, seal_raw = _read_json(freeze / "freeze.json")
    training_path = freeze / "training_manifest.json"
    training = None
    training_raw = None
    if training_path.is_file():
        training, training_raw = _read_json(training_path)
    provenance = {
        "path": str(freeze.resolve()),
        "candidate_split_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "audit_report_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "freeze_sha256": hashlib.sha256(seal_raw).hexdigest(),
        "training_manifest": _provenance(training_path, training_raw) if training is not None else None,
    }
    return plan, audit, seal, training, provenance


def _build_acquisitions(records, plan, audit, training):
    entries = plan.get("records")
    if not isinstance(entries, list):
        raise ScaleEvaluationError("candidate_split.json needs a records list")
    by_sequence = {}
    for entry in entries:
        if not isinstance(entry, dict) or not _known(entry.get("sequence_id")):
            raise ScaleEvaluationError("Every candidate split entry needs a sequence_id")
        if entry["sequence_id"] in by_sequence:
            raise ScaleEvaluationError(f"Duplicate candidate split entry: {entry['sequence_id']}")
        by_sequence[entry["sequence_id"]] = entry
    audit_records = audit.get("records") if isinstance(audit.get("records"), list) else []
    formal_sequences = set()
    if training is not None:
        split_entries = training.get("splits")
        if not isinstance(split_entries, dict):
            raise ScaleEvaluationError("training_manifest.json needs splits")
        for split in SPLITS:
            for entry in split_entries.get(split, []):
                if isinstance(entry, dict) and _known(entry.get("sequence_id")):
                    formal_sequences.add(entry["sequence_id"])
    groups = {}
    issues = []
    aliases = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not _known(record.get("sequence_id")):
            raise ScaleEvaluationError(f"Manifest record {index} needs a sequence_id")
        sequence_id = record["sequence_id"]
        entry = by_sequence.get(sequence_id)
        if entry is None:
            issues.append({"code": "split_entry_missing", "sequence_id": sequence_id})
            entry = {}
        acquisition_id = record.get("acquisition_id") if _known(record.get("acquisition_id")) else sequence_id
        group_id = entry.get("acquisition_group_id") if _known(entry.get("acquisition_group_id")) else None
        group_id = group_id or f"{record.get('dataset', 'unknown_dataset')}:{acquisition_id}"
        group = groups.setdefault(group_id, {
            "group_id": group_id, "record_indices": [], "sequence_ids": set(), "splits": set(),
            "pools": set(), "qualified_records": [], "formal_records": [], "aliases": {group_id},
            "split_conflicts": set(),
        })
        group["record_indices"].append(index)
        group["sequence_ids"].add(sequence_id)
        frozen_split = entry.get("split") if entry.get("split") in ALL_SPLITS else None
        declared_split = record.get("split") if record.get("split") in ALL_SPLITS else None
        if frozen_split is not None and declared_split is not None and frozen_split != declared_split:
            group["split_conflicts"].add(sequence_id)
            issues.append({"code": "acquisition_record_split_conflict", "sequence_id": sequence_id,
                           "frozen_split": frozen_split, "record_split": declared_split})
        split = frozen_split or declared_split
        group["splits"].add(split if split in ALL_SPLITS else "candidate")
        group["pools"].add("auxiliary" if record.get("pool") == "auxiliary" else "joint")
        audited = audit_records[index] if index < len(audit_records) and isinstance(audit_records[index], dict) else {}
        downloaded = audited.get("downloaded")
        if downloaded is None:
            downloaded = record.get("download_status") == "downloaded"
        group.setdefault("downloaded_records", []).append(bool(downloaded))
        ready = bool(entry.get("joint_pose_ready") and audited.get("joint_pose_ready"))
        if ready:
            group["qualified_records"].append(index)
        if sequence_id in formal_sequences:
            group["formal_records"].append(index)
        values = [sequence_id, acquisition_id, record.get("source_episode_uuid"), record.get("source_video")]
        source_ids = record.get("source_ids")
        if isinstance(source_ids, dict):
            values.extend(source_ids.get(key) for key in ("source_episode_uuid", "demo_episode_id"))
        group["aliases"].update(value for value in values if _known(value))
    result = []
    for group in groups.values():
        group["split"] = next(iter(group["splits"])) if len(group["splits"]) == 1 else "leakage"
        group["pool"] = next(iter(group["pools"])) if len(group["pools"]) == 1 else "mixed"
        group["downloaded"] = any(group["downloaded_records"])
        group["qualified"] = bool(group["qualified_records"]) and group["split"] in SPLITS and group["pool"] == "joint"
        group["formal"] = bool(group["formal_records"]) and group["split"] in SPLITS and group["pool"] == "joint"
        if training is not None:
            group["qualified"] = group["formal"] and bool(group["qualified_records"])
        if group["split"] == "leakage" or group["pool"] == "mixed" or group["split_conflicts"]:
            group["qualified"] = False
            group["formal"] = False
        if group["split"] == "leakage" or group["pool"] == "mixed":
            issues.append({"code": "acquisition_group_conflict", "group_id": group["group_id"]})
        for alias in group["aliases"]:
            if alias in aliases and aliases[alias] != group["group_id"]:
                issues.append({"code": "alias_group_conflict", "alias": alias})
            aliases[alias] = group["group_id"]
        result.append(group)
    if len(by_sequence) != len(records):
        issues.append({"code": "manifest_split_record_count_mismatch"})
    return sorted(result, key=lambda group: group["group_id"]), aliases, issues


def _by_split(groups, records, selected):
    return {
        split: _summarize([group for group in groups if selected(group) and group["split"] == split], records)
        for split in ALL_SPLITS
    }


def _annotation_group(record, groups, aliases):
    sequence_id = record.get("sequence_id")
    group_id = aliases.get(sequence_id) if _known(sequence_id) else None
    if group_id is None:
        return None
    return next((group for group in groups if group["group_id"] == group_id), None)


def _annotation_identifier(record, *, none=False):
    return record.get("interval_id", record.get("event_id")) if none else record.get("event_id")


def _annotation_status(record, *, none=False):
    if not isinstance(record, dict):
        return "invalid"
    if record.get("review_status") not in REVIEW_STATUSES:
        return "invalid"
    start, end = record.get("start_frame"), record.get("end_frame")
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
            or not isinstance(end, int) or start < 0 or end <= start):
        return "invalid"
    evidence = record.get("evidence_frames")
    if (not isinstance(evidence, list) or not evidence
            or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in evidence)
            or len(set(evidence)) != len(evidence)
            or any(frame < start or frame >= end for frame in evidence)):
        return "invalid"
    confidence = record.get("confidence")
    if (confidence is not None and (isinstance(confidence, bool)
            or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1)):
        return "invalid"
    if record.get("label_source") not in LABEL_SOURCES:
        return "invalid"
    if not isinstance(record.get("notes", ""), str):
        return "invalid"
    if "revision_history" in record and not isinstance(record["revision_history"], list):
        return "invalid"
    if none:
        if any(field in record for field in ("subject", "object_id", "object_hint", "interaction_type")):
            return "invalid"
        if not _known(_annotation_identifier(record, none=True)):
            return "invalid"
    else:
        if not _known(record.get("event_id")) or record.get("subject") not in ("left", "right"):
            return "invalid"
        object_id = record.get("object_id")
        if object_id is not None and not _known(object_id):
            return "invalid"
        if not _known(record.get("object_hint")) or record.get("interaction_type") not in INTERACTION_TYPES:
            return "invalid"
        if record.get("geometry_status") not in GEOMETRY_STATUSES:
            return "invalid"
        source_ids = record.get("source_event_ids")
        if not isinstance(source_ids, list) or any(not isinstance(item, str) for item in source_ids):
            return "invalid"
    if record.get("review_status") == "accepted":
        if not _known(record.get("reviewer")) or not _known(record.get("reviewed_at")):
            return "invalid"
        if not none and not _known(record.get("object_id")):
            return "invalid"
    return record["review_status"]


def _read_annotation_file(path, *, none=False):
    records = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    record = None
                records.append((record, _annotation_status(record, none=none), line_number))
    except OSError as exc:
        raise ScaleEvaluationError(f"Cannot read annotations {path}: {exc}") from exc
    return records


def _empty_annotation_counts():
    return {
        "raw_records": 0,
        "structurally_valid": 0,
        "invalid": 0,
        "by_status": {status: 0 for status in REVIEW_STATUSES},
        "by_type": {
            name: {status: 0 for status in REVIEW_STATUSES} for name in INTERACTION_TYPES
        },
    }


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _audit_annotations(annotation_dir, groups, aliases):
    events = _empty_annotation_counts()
    none_counts = {status: 0 for status in REVIEW_STATUSES}
    none_invalid = 0
    none_raw = 0
    accepted_none_frames = 0
    accepted_none_overlap_frames = 0
    observed_joint_accepted_events = 0
    observed_joint_accepted_none = 0
    per_split = {
        split: {
            "events": {status: 0 for status in REVIEW_STATUSES},
            "accepted_by_type": {name: 0 for name in INTERACTION_TYPES},
            "none_intervals": {status: 0 for status in REVIEW_STATUSES},
            "invalid": 0,
        } for split in (*ALL_SPLITS, "unmatched", "leakage")
    }
    annotation_records = defaultdict(lambda: {"events": [], "none_intervals": []})
    accepted_inventory = []
    none_intervals_by_group = defaultdict(list)
    unmatched = 0
    files = 0
    provided_files = defaultdict(set)
    expected_sequences = sorted({
        sequence_id for group in groups
        if group["pool"] == "joint" and group["split"] in SPLITS
        for sequence_id in group["sequence_ids"]
    })
    if annotation_dir is not None:
        annotation_dir = Path(annotation_dir)
        if not annotation_dir.is_dir():
            raise ScaleEvaluationError(f"Annotations directory does not exist: {annotation_dir}")
        for sequence_dir in sorted(path for path in annotation_dir.iterdir() if path.is_dir()):
            for filename, is_none in (("events.jsonl", False), ("none_intervals.jsonl", True)):
                path = sequence_dir / filename
                if not path.is_file():
                    continue
                files += 1
                provided_files[sequence_dir.name].add("none_intervals" if is_none else "events")
                for record, status, line_number in _read_annotation_file(path, none=is_none):
                    if is_none:
                        none_raw += 1
                    else:
                        events["raw_records"] += 1
                    if status == "invalid" or not _known(record.get("sequence_id")):
                        if is_none:
                            none_invalid += 1
                        else:
                            events["invalid"] += 1
                        continue
                    group = _annotation_group(record, groups, aliases)
                    split = group["split"] if group is not None else "unmatched"
                    if group is None:
                        unmatched += 1
                    else:
                        key = "none_intervals" if is_none else "events"
                        annotation_records[group["group_id"]][key].append(record)
                    formal_joint = bool(group is not None and group["pool"] == "joint" and split in SPLITS)
                    if status == "accepted" and formal_joint:
                        if is_none:
                            observed_joint_accepted_none += 1
                        else:
                            observed_joint_accepted_events += 1
                    if status == "accepted" and group is not None:
                        accepted_inventory.append({
                            "group_id": group["group_id"], "sequence_id": record["sequence_id"],
                            "id": _annotation_identifier(record, none=is_none),
                            "record_type": "none_interval" if is_none else "event",
                            "start_frame": record["start_frame"], "end_frame": record["end_frame"],
                            "split": split,
                        })
                    if is_none:
                        none_counts[status] += 1
                        per_split[split]["none_intervals"][status] += 1
                        if status == "accepted" and group is not None and group["pool"] == "joint":
                            none_intervals_by_group[group["group_id"]].append(
                                (record["start_frame"], record["end_frame"])
                            )
                    else:
                        events["structurally_valid"] += 1
                        events["by_status"][status] += 1
                        events["by_type"][record["interaction_type"]][status] += 1
                        per_split[split]["events"][status] += 1
                        if status == "accepted":
                            per_split[split]["accepted_by_type"][record["interaction_type"]] += 1
        for group_id, intervals in none_intervals_by_group.items():
            frame_sum = sum(end - start for start, end in intervals)
            merged = _merge_intervals(intervals)
            union = sum(end - start for start, end in merged)
            accepted_none_frames += union
            accepted_none_overlap_frames += frame_sum - union
    sequence_coverage = {}
    missing_sequences = []
    for sequence_id in expected_sequences:
        present = provided_files.get(sequence_id, set())
        missing = [kind for kind in ("events", "none_intervals") if kind not in present]
        status = "not_provided" if annotation_dir is None else "complete" if not missing else "incomplete"
        sequence_coverage[sequence_id] = {"status": status, "missing_files": missing}
        if missing or annotation_dir is None:
            missing_sequences.append(sequence_id)
    if annotation_dir is None:
        coverage_status = "not_provided"
    elif missing_sequences:
        coverage_status = "incomplete"
    elif observed_joint_accepted_events == 0 and observed_joint_accepted_none == 0:
        coverage_status = "provided_complete_zero"
    else:
        coverage_status = "provided_complete"
    complete = coverage_status.startswith("provided_complete")
    legacy = None
    if annotation_dir is not None:
        review_path = annotation_dir / "annotation_review.csv"
        if review_path.is_file():
            with review_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            statuses = Counter(row.get("review_status", "") for row in rows)
            legacy = {
                "rows": len(rows),
                "by_status": {status: statuses.get(status, 0) for status in REVIEW_STATUSES},
                "blank_three_head_labels": sum(
                    all(not _known(row.get(key)) for key in ("subject", "object_track_id", "interaction_class"))
                    for row in rows
                ),
                "counts_as_six_class_hoi_events": False,
                "reason": "Source semantic subtasks are not projected HOI labels; pending rows are not none.",
            }
    return {
        "directory_present": annotation_dir is not None,
        "coverage_status": coverage_status,
        "coverage": {
            "status": coverage_status,
            "scope": "joint_acquisition_sequences_with_frozen_train_val_test_split",
            "expected_sequences": len(expected_sequences),
            "complete_sequences": sum(item["status"] == "complete" for item in sequence_coverage.values()),
            "missing_or_incomplete_sequences": missing_sequences,
            "by_sequence": sequence_coverage,
        },
        "jsonl_files_read": files,
        "interaction_events": events,
        "none_intervals": {
            "raw_records": none_raw,
            "invalid": none_invalid,
            "by_status": none_counts,
            "accepted_coverage_frames_union": accepted_none_frames,
            "accepted_overlap_frames": accepted_none_overlap_frames,
        },
        "by_split": per_split,
        "joint_accepted_interaction_events": observed_joint_accepted_events if complete else None,
        "joint_accepted_none_intervals": observed_joint_accepted_none if complete else None,
        "observed_joint_accepted_interaction_events": observed_joint_accepted_events,
        "observed_joint_accepted_none_intervals": observed_joint_accepted_none,
        "accepted_records": accepted_inventory,
        "unmatched_records": unmatched,
        "legacy_source_semantic_review": legacy,
    }, annotation_records


def _resolve_sample_group(entry, groups, aliases):
    for key in ("split_group", "sequence_id", "acquisition_id", "source_episode_uuid"):
        value = entry.get(key)
        if _known(value) and value in aliases:
            group_id = aliases[value]
            return next(group for group in groups if group["group_id"] == group_id)
    return None


def _annotation_provenance_valid(record):
    return (record.get("annotation_mode", "offline_noncausal") == "offline_noncausal"
            and record.get("allowed_as_model_input", False) is False)


def _annotation_occluded(record):
    return (record.get("occluded") is True
            or record.get("occlusion_status") in {"occluded", "unknown", "pending"})


def _project_annotation(records, frame_index):
    events = [item for item in records["events"]
              if item["start_frame"] <= frame_index < item["end_frame"]]
    none_intervals = [item for item in records["none_intervals"]
                      if item["start_frame"] <= frame_index < item["end_frame"]]
    active = events + none_intervals
    unresolved = [item for item in active if item["review_status"] in ("pending", "needs_revision")]
    if unresolved:
        return {"valid": False, "reason": "pending_or_needs_revision", "event_ids": [
            _annotation_identifier(item, none=item in none_intervals) for item in unresolved
        ]}
    accepted_events = [item for item in events if item["review_status"] == "accepted"]
    accepted_none = [item for item in none_intervals if item["review_status"] == "accepted"]
    if len(accepted_events) > 1 or (accepted_events and accepted_none) or len(accepted_none) > 1:
        return {"valid": False, "reason": "overlapping_accepted_labels", "event_ids": [
            _annotation_identifier(item, none=item in none_intervals)
            for item in accepted_events + accepted_none
        ]}
    if not accepted_events and not accepted_none:
        return {"valid": False, "reason": "unlabelled_or_rejected", "event_ids": [
            _annotation_identifier(item, none=item in none_intervals) for item in active
        ]}
    record = (accepted_events + accepted_none)[0]
    identifier = _annotation_identifier(record, none=bool(accepted_none))
    if not _annotation_provenance_valid(record):
        return {"valid": False, "reason": "invalid_annotation_provenance", "event_ids": [identifier]}
    if _annotation_occluded(record):
        return {"valid": False, "reason": "occluded", "event_ids": [identifier]}
    if accepted_none:
        return {
            "valid": True, "kind": "none", "event_ids": [identifier], "record": record,
            "subject": SUBJECT_TO_ID["none"], "object_id": "", "interaction": INTERACTION_TO_ID["none"],
            "interaction_name": "none", "label_source": record["label_source"],
        }
    event = record
    return {
        "valid": True, "kind": "event", "event_ids": [identifier], "record": event,
        "subject": SUBJECT_TO_ID[event["subject"]], "object_id": event["object_id"],
        "interaction": INTERACTION_TO_ID[event["interaction_type"]],
        "interaction_name": event["interaction_type"], "label_source": event["label_source"],
    }


def _integer_field(entry, key):
    value = entry.get(key)
    return value if type(value) is int else None


def _projection_issue(entry, expected):
    required = ("label_subject", "label_object_slot", "label_object_id", "label_interaction",
                "label_interaction_name", "label_event_ids", "label_source", "annotation_mode")
    missing = [key for key in required if key not in entry]
    if missing:
        return {"reason": "missing_projection_fields", "fields": missing}
    event_ids = entry["label_event_ids"]
    if (not isinstance(event_ids, list) or any(not isinstance(item, str) for item in event_ids)
            or len(event_ids) != len(set(event_ids)) or set(event_ids) != set(expected["event_ids"])):
        return {"reason": "label_event_ids_mismatch", "expected": expected["event_ids"],
                "actual": event_ids if isinstance(event_ids, list) else None}
    checks = (
        ("label_subject", expected["subject"]),
        ("label_object_id", expected["object_id"]),
        ("label_interaction", expected["interaction"]),
        ("label_interaction_name", expected["interaction_name"]),
        ("label_source", expected["label_source"]),
        ("annotation_mode", "offline_noncausal"),
    )
    for key, expected_value in checks:
        if entry.get(key) != expected_value:
            return {"reason": f"{key}_mismatch", "expected": expected_value, "actual": entry.get(key)}
    slot = _integer_field(entry, "label_object_slot")
    expected_slot = -1 if expected["kind"] == "none" else 0
    if slot is None or (expected["kind"] == "none" and slot != -1) or (expected["kind"] == "event" and slot < 0):
        return {"reason": "label_object_slot_mismatch", "expected": expected_slot, "actual": entry.get("label_object_slot")}
    if expected["kind"] == "event" and "object_id" in entry and entry["object_id"] != expected["object_id"]:
        return {"reason": "object_id_mismatch", "expected": expected["object_id"], "actual": entry["object_id"]}
    return None


def _empty_frame_counts():
    return {"generated_windows": 0, "valid_target_frames": 0, "invalid_target_frames": 0,
            "unknown_validity_target_frames": 0, "usable_valid_target_frames": 0,
            "valid_but_unverified_target_frames": 0, "usable_windows": 0,
            "linked_but_not_fully_verified": 0, "rejected_usable": 0}


def _missing_sample_inventory(annotations, groups):
    no_qualified = not any(group["qualified"] for group in groups)
    no_accepted_proven = annotations["coverage_status"] == "provided_complete_zero"
    known_zero = no_qualified or no_accepted_proven
    frames = {
        "inventory_present": False,
        "inventory_status": "not_provided",
        "valid": None,
        "invalid": None,
        "unknown_validity": None,
        "usable_valid": 0 if known_zero else None,
        "verified_usable_valid": 0,
        "usable_independent_acquisitions": 0 if known_zero else None,
        "verified_usable_independent_acquisitions": 0,
        "valid_but_unverified": None,
        "linked_but_not_fully_verified": None,
        "rejected_usable": None,
        "projection_diagnostics": {"by_reason": {}},
        "note": "No target-frame inventory; raw video frames and pending intervals are not converted into labels.",
    }
    windows = {
        "inventory_present": False,
        "inventory_status": "not_provided",
        "generated": None,
        "formal_usable": 0 if known_zero else None,
        "verified_formal_usable": 0,
        "counted_as_independent_acquisitions": False,
        "issues": [],
    }
    return frames, windows


def _audit_samples(sample_index, groups, aliases, annotation_records, annotations):
    if sample_index is None:
        return _missing_sample_inventory(annotations, groups)
    payload, _ = _read_json(sample_index)
    no_qualified = not any(group["qualified"] for group in groups)
    no_accepted_proven = annotations["coverage_status"] == "provided_complete_zero"
    known_zero = no_qualified or no_accepted_proven
    missing_splits = [split for split in SPLITS if split not in payload]
    if missing_splits:
        issue = {"code": "sample_index_split_missing", "splits": missing_splits}
        frames = {
            "inventory_present": True, "inventory_status": "incomplete", "valid": None,
            "invalid": None, "unknown_validity": None,
            "usable_valid": 0 if known_zero else None, "verified_usable_valid": 0,
            "usable_independent_acquisitions": 0 if known_zero else None,
            "verified_usable_independent_acquisitions": 0, "valid_but_unverified": None,
            "linked_but_not_fully_verified": None, "rejected_usable": None,
            "projection_diagnostics": {"by_reason": {}},
        }
        windows = {
            "inventory_present": True, "inventory_status": "incomplete", "generated": None,
            "formal_usable": 0 if known_zero else None, "verified_formal_usable": 0,
            "counted_as_independent_acquisitions": False, "issues": [issue],
        }
        return frames, windows
    counts = _empty_frame_counts()
    per_split = {split: _empty_frame_counts() for split in SPLITS}
    sample_ids = set()
    usable_groups = set()
    issues = []
    reasons = Counter()
    usable_unknown = False
    for split in SPLITS:
        entries = payload[split]
        if not isinstance(entries, list):
            raise ScaleEvaluationError(f"Sample index {split} must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                issues.append({"code": "sample_not_object", "split": split})
                continue
            sample_id = entry.get("sample_id")
            if not _known(sample_id) or sample_id in sample_ids:
                issues.append({"code": "duplicate_or_missing_sample_id", "split": split})
                continue
            sample_ids.add(sample_id)
            counts["generated_windows"] += 1
            per_split[split]["generated_windows"] += 1
            group = _resolve_sample_group(entry, groups, aliases)
            split_conflict = False
            entry_split = entry.get("split")
            if _known(entry_split) and entry_split in ALL_SPLITS and entry_split != split:
                split_conflict = True
                issues.append({"code": "sample_split_field_conflict", "sample_id": sample_id,
                               "index_split": split, "entry_split": entry_split})
            if group is None:
                issues.append({"code": "sample_group_unresolved", "sample_id": sample_id, "split": split})
            elif group["split"] != split:
                split_conflict = True
                issues.append({"code": "sample_split_mismatch", "sample_id": sample_id,
                               "index_split": split, "frozen_group_split": group["split"],
                               "group_id": group["group_id"]})
            valid = entry.get("label_valid")
            if type(valid) is int:
                valid = bool(valid) if valid in (0, 1) else None
            elif type(valid) is not bool:
                valid = None
            if valid is True:
                counts["valid_target_frames"] += 1
                per_split[split]["valid_target_frames"] += 1
            elif valid is False:
                counts["invalid_target_frames"] += 1
                per_split[split]["invalid_target_frames"] += 1
                continue
            else:
                counts["unknown_validity_target_frames"] += 1
                per_split[split]["unknown_validity_target_frames"] += 1
                issues.append({"code": "sample_label_valid_unknown", "sample_id": sample_id, "split": split})
                if not known_zero and group is not None and group["qualified"] and not split_conflict:
                    usable_unknown = True
                continue
            reason = None
            incomplete = False
            expected = None
            if group is None:
                reason = "sample_group_unresolved"
            elif split_conflict:
                reason = "split_mismatch"
            elif not group["qualified"]:
                reason = "group_not_qualified"
            else:
                sequence_id = entry.get("sequence_id")
                if not _known(sequence_id) and len(group["sequence_ids"]) == 1:
                    sequence_id = next(iter(group["sequence_ids"]))
                coverage = annotations["coverage"].get("by_sequence", {}).get(sequence_id or "")
                if not _known(sequence_id):
                    reason = "sample_sequence_missing"
                    incomplete = True
                elif coverage is None or coverage["status"] != "complete":
                    reason = "annotations_" + (coverage["status"] if coverage is not None else "not_provided")
                    incomplete = True
                else:
                    target = _integer_field(entry, "target_frame")
                    if target is None or target < 0:
                        reason = "missing_target_frame"
                        incomplete = True
                    else:
                        expected = _project_annotation(annotation_records[group["group_id"]], target)
                        if not expected["valid"]:
                            reason = expected["reason"]
                        else:
                            mismatch = _projection_issue(entry, expected)
                            if mismatch is not None:
                                reason = mismatch["reason"]
                                incomplete = reason == "missing_projection_fields"
            if reason is None:
                counts["usable_valid_target_frames"] += 1
                counts["usable_windows"] += 1
                per_split[split]["usable_valid_target_frames"] += 1
                per_split[split]["usable_windows"] += 1
                usable_groups.add(group["group_id"])
                continue
            counts["valid_but_unverified_target_frames"] += 1
            per_split[split]["valid_but_unverified_target_frames"] += 1
            reasons[reason] += 1
            if incomplete:
                counts["linked_but_not_fully_verified"] += 1
                per_split[split]["linked_but_not_fully_verified"] += 1
                usable_unknown = True
                code = "sample_projection_incomplete"
            else:
                counts["rejected_usable"] += 1
                per_split[split]["rejected_usable"] += 1
                code = "sample_projection_mismatch"
            issues.append({"code": code, "sample_id": sample_id, "split": split,
                           "group_id": group["group_id"] if group is not None else None,
                           "reason": reason})
    inventory_status = "provided_complete_zero" if counts["generated_windows"] == 0 else "provided"
    usable_valid = None if usable_unknown else counts["usable_valid_target_frames"]
    usable_acquisitions = None if usable_unknown else len(usable_groups)
    frames = {
        "inventory_present": True,
        "inventory_status": inventory_status,
        "valid": counts["valid_target_frames"],
        "invalid": counts["invalid_target_frames"],
        "unknown_validity": counts["unknown_validity_target_frames"],
        "usable_valid": usable_valid,
        "verified_usable_valid": counts["usable_valid_target_frames"],
        "usable_independent_acquisitions": usable_acquisitions,
        "verified_usable_independent_acquisitions": len(usable_groups),
        "valid_but_unverified": counts["valid_but_unverified_target_frames"],
        "linked_but_not_fully_verified": counts["linked_but_not_fully_verified"],
        "rejected_usable": counts["rejected_usable"],
        "projection_diagnostics": {
            "by_reason": dict(sorted(reasons.items())),
            "object_slot_verification": "event slots require nonnegative index; exact track order is not present in the index",
        },
        "by_split": {
            split: {key: values[key] for key in ("valid_target_frames", "invalid_target_frames",
                    "unknown_validity_target_frames", "usable_valid_target_frames",
                    "valid_but_unverified_target_frames", "linked_but_not_fully_verified",
                    "rejected_usable")}
            for split, values in per_split.items()
        },
    }
    windows = {
        "inventory_present": True,
        "inventory_status": inventory_status,
        "generated": counts["generated_windows"],
        "formal_usable": usable_valid,
        "verified_formal_usable": counts["usable_windows"],
        "by_split": {split: {"generated": values["generated_windows"],
                             "formal_usable": (None if usable_unknown else values["usable_windows"]),
                             "verified_formal_usable": values["usable_windows"]}
                     for split, values in per_split.items()},
        "counted_as_independent_acquisitions": False,
        "issues": issues,
    }
    return frames, windows


def benchmark_design_summary(config):
    if config.get("schema_version") != "scale_benchmark_v1":
        raise ScaleEvaluationError("Benchmark schema_version must be scale_benchmark_v1")
    designs = config.get("designs")
    if not isinstance(designs, list) or not designs:
        raise ScaleEvaluationError("Benchmark needs a nonempty designs list")
    summaries = []
    for design in designs:
        budgets = design.get("train_acquisition_budgets")
        if (not isinstance(budgets, list) or not budgets
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in budgets)
                or budgets != sorted(set(budgets))):
            raise ScaleEvaluationError(f"Invalid training budgets in {design.get('design_id')}")
        holdout = design.get("holdout_independent_acquisitions")
        if not isinstance(holdout, dict) or any(type(holdout.get(split)) is not int or holdout[split] < 1
                                               for split in ("val", "test")):
            raise ScaleEvaluationError("Benchmark holdout needs positive integer val/test acquisitions")
        included = design.get("holdout_included_in_fixed_total") is True
        fixed_total = design.get("fixed_total_independent_acquisitions")
        if included:
            if type(fixed_total) is not int or fixed_total <= 0:
                raise ScaleEvaluationError("Included-holdout design needs a positive fixed total")
            train_pool = fixed_total - holdout["val"] - holdout["test"]
            minimum_total = fixed_total
            additional_holdout = 0
        else:
            train_pool = max(budgets)
            minimum_total = train_pool + holdout["val"] + holdout["test"]
            additional_holdout = holdout["val"] + holdout["test"]
            if fixed_total is not None:
                raise ScaleEvaluationError("Additional-holdout design must not set a fixed total")
        if max(budgets) != train_pool or min(budgets) > train_pool:
            raise ScaleEvaluationError(f"Budgets do not fill the declared train pool in {design.get('design_id')}")
        summaries.append({
            "design_id": design.get("design_id"),
            "budget_unit": "independent_training_acquisitions_not_windows_or_total_segments",
            "train_acquisition_budgets": budgets,
            "train_pool": train_pool,
            "val": holdout["val"],
            "test": holdout["test"],
            "holdout_included_in_fixed_total": included,
            "additional_independent_holdout": additional_holdout,
            "minimum_total_independent_acquisitions": minimum_total,
            "actual_total_may_exceed_minimum_because_groups_are_indivisible": True,
        })
    return summaries


def cluster_bootstrap(rows, *, cluster_key="split_group", value_key="value", weight_key="weight",
                      resamples=1000, seed=20261022, confidence=0.95, include_replicates=False):
    """Bootstrap whole acquisition clusters, preserving every correlated row in a cluster."""
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    clusters = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or not _known(row.get(cluster_key)):
            raise ValueError("Every row needs a known cluster identifier")
        value = row.get(value_key)
        weight = row.get(weight_key, 1.0)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError("Values must be finite and weights finite/nonnegative")
        clusters[row[cluster_key]].append((float(value), float(weight)))
    if not clusters:
        raise ValueError("At least one cluster is required")
    identifiers = sorted(clusters)

    def estimate(sampled):
        numerator = denominator = 0.0
        multiplicity = Counter(sampled)
        for identifier, count in multiplicity.items():
            for value, weight in clusters[identifier]:
                numerator += count * weight * value
                denominator += count * weight
        if denominator == 0:
            raise ValueError("Bootstrap total weight must be positive")
        return numerator / denominator

    observed = estimate(identifiers)
    rng = random.Random(seed)
    replicates = [estimate([identifiers[rng.randrange(len(identifiers))] for _ in identifiers])
                  for _ in range(resamples)]
    ordered = sorted(replicates)

    def percentile(probability):
        position = (len(ordered) - 1) * probability
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    result = {
        "observed": observed,
        "ci_low": percentile((1 - confidence) / 2),
        "ci_high": percentile(1 - (1 - confidence) / 2),
        "confidence": confidence,
        "cluster_count": len(identifiers),
        "row_count": sum(len(values) for values in clusters.values()),
        "resamples": resamples,
        "seed": seed,
        "resampling_unit": "independent_acquisition_cluster_not_windows",
    }
    if include_replicates:
        result["replicates"] = replicates
    return result


def _release_minimums():
    config, raw = _read_json(DEFAULT_RELEASE_CONFIG)
    minimums = config.get("minimums")
    required = ("acquisitions", *SPLITS)
    if (not isinstance(minimums, dict)
            or any(type(minimums.get(key)) is not int or minimums[key] < 1 for key in required)):
        raise ScaleEvaluationError("Release config needs positive acquisition and split minimums")
    return {key: minimums[key] for key in required}, raw


def evaluate_scale(progress_path, manifest_path, freeze_path, annotations_path=None,
                   sample_index_path=None, benchmark_path=None):
    progress, progress_raw = _read_json(progress_path)
    manifest, manifest_raw = _read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ScaleEvaluationError("Candidate manifest needs a records list")
    plan, audit, seal, training, freeze_provenance = _load_freeze(freeze_path)
    groups, aliases, issues = _build_acquisitions(records, plan, audit, training)
    joint = [group for group in groups if group["pool"] == "joint" and group["downloaded"]]
    auxiliary = [group for group in groups if group["pool"] == "auxiliary" and group["downloaded"]]
    qualified = [group for group in groups if group["qualified"]]
    layers = {
        "downloaded_joint_candidates": {
            **_summarize(joint, records),
            "by_split": _by_split(groups, records, lambda group: group in joint),
            "counts_toward_120": False,
        },
        "downloaded_auxiliary": {
            **_summarize(auxiliary, records),
            "counts_toward_120": False,
        },
        "qualified_joint": {
            **_summarize(qualified, records),
            "by_split": _by_split(groups, records, lambda group: group in qualified),
            "counts_toward_120": True,
        },
    }
    annotations, annotation_records = _audit_annotations(annotations_path, groups, aliases)
    frames, windows = _audit_samples(sample_index_path, groups, aliases, annotation_records, annotations)
    benchmark_path = Path(benchmark_path) if benchmark_path is not None else DEFAULT_BENCHMARK
    benchmark, benchmark_raw = _read_json(benchmark_path)
    designs = benchmark_design_summary(benchmark)
    release_minimums, release_raw = _release_minimums()
    target = progress.get("target_qualified_independent_acquisitions")
    if type(target) is not int or target < 1:
        raise ScaleEvaluationError("Progress needs a positive target_qualified_independent_acquisitions")
    progress_qualified = progress.get("accepted_qualified_joint_acquisitions")
    if type(progress_qualified) is not int or progress_qualified < 0:
        raise ScaleEvaluationError("Progress needs a nonnegative accepted qualified count")
    actual_qualified = layers["qualified_joint"]["independent_acquisitions"]
    if actual_qualified != progress_qualified:
        issues.append({"code": "qualified_progress_freeze_mismatch", "progress": progress_qualified,
                       "freeze": actual_qualified})
    if layers["downloaded_joint_candidates"]["independent_acquisitions"] != progress.get(
            "downloaded_joint_source_candidates"):
        issues.append({"code": "downloaded_progress_freeze_mismatch"})
    for split in SPLITS:
        actual = layers["qualified_joint"]["by_split"][split]["independent_acquisitions"]
        declared = progress.get("current_qualified_split", {}).get(split)
        if actual != declared:
            issues.append({"code": "qualified_split_progress_mismatch", "split": split,
                           "progress": declared, "freeze": actual})
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    expected_sources = (
        (audit.get("source_ledger"), "manifest_freeze_source_sha_mismatch"),
        (seal.get("source_ledger"), "freeze_source_sha_mismatch"),
        (training.get("source_ledger_sha256") if training is not None else None,
         "training_manifest_source_sha_mismatch"),
    )
    for value, code in expected_sources:
        expected_sha = value.get("sha256") if isinstance(value, dict) else value
        if expected_sha and expected_sha != manifest_sha:
            issues.append({"code": code})
    if annotations["unmatched_records"]:
        issues.append({"code": "annotation_sequence_not_in_manifest",
                       "count": annotations["unmatched_records"]})
    invalid_annotations = (annotations["interaction_events"]["invalid"]
                           + annotations["none_intervals"]["invalid"])
    if invalid_annotations:
        issues.append({"code": "invalid_annotation_records", "count": invalid_annotations})
    issues.extend(windows.get("issues", []))
    supervised_acquisitions = frames["usable_independent_acquisitions"]
    supervised_frames = frames["usable_valid"]
    source_training_allowed = bool(progress.get("training_allowed") is True
                                   and audit.get("training_allowed") is True
                                   and seal.get("training_allowed") is True)
    split_checks = {}
    for split in SPLITS:
        actual = layers["qualified_joint"]["by_split"][split]["independent_acquisitions"]
        split_checks[split] = {"actual": actual, "minimum": release_minimums[split],
                               "pass": actual >= release_minimums[split]}
    gate_checks = {
        "source_flags": {"pass": source_training_allowed},
        "training_manifest_accepted": {
            "pass": bool(training is not None and training.get("status") == "accepted"
                         and training.get("training_allowed") is True)
        },
        "annotation_coverage_complete": {
            "pass": annotations["coverage_status"] in ("provided_complete", "provided_complete_zero"),
            "status": annotations["coverage_status"],
        },
        "accepted_interaction_events": {
            "actual": annotations["joint_accepted_interaction_events"],
            "pass": isinstance(annotations["joint_accepted_interaction_events"], int)
                    and annotations["joint_accepted_interaction_events"] > 0,
        },
        "usable_target_frames": {
            "actual": supervised_frames,
            "pass": isinstance(supervised_frames, int) and supervised_frames > 0,
        },
        "qualified_total": {
            "actual": actual_qualified,
            "progress_target": target,
            "release_minimum": release_minimums["acquisitions"],
            "pass": actual_qualified >= max(target, release_minimums["acquisitions"]),
        },
        "qualified_splits": split_checks,
        "inputs_consistent": {"pass": not issues, "issue_count": len(issues)},
    }
    training_allowed = all(
        check["pass"] if isinstance(check, dict) and "pass" in check
        else all(item["pass"] for item in check.values())
        for check in gate_checks.values()
    )
    report = {
        "schema_version": "dataset_scale_evaluation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observed_at": progress.get("observed_at"),
        "inputs": {
            "progress": _provenance(progress_path, progress_raw),
            "manifest": _provenance(manifest_path, manifest_raw),
            "freeze": freeze_provenance,
            "annotations": str(Path(annotations_path).resolve()) if annotations_path is not None else None,
            "sample_index": str(Path(sample_index_path).resolve()) if sample_index_path is not None else None,
            "benchmark": _provenance(benchmark_path, benchmark_raw),
            "release_config": _provenance(DEFAULT_RELEASE_CONFIG, release_raw),
        },
        "input_consistent": not issues,
        "issues": issues,
        "target": {
            "qualified_independent_acquisitions": target,
            "qualified_actual": actual_qualified,
            "qualified_gap": max(target - actual_qualified, 0),
            "threshold_source": "project_internal_gate_not_official_benchmark",
            "progress_declared_qualified": progress_qualified,
            "progress_declared_gap": progress.get("qualified_gap_to_120"),
        },
        "training_manifest_present": training is not None,
        "source_training_allowed": source_training_allowed,
        "training_allowed": training_allowed,
        "formal_training_gate": {
            "scope": "formal_release_gate_not_manual_smoke_authorization",
            "checks": gate_checks,
            "manual_smoke_requires_explicit_separate_authorization": True,
        },
        "layers": layers,
        "annotations": annotations,
        "target_frames": frames,
        "windows": windows,
        "current_supervision": {
            "accepted_joint_interaction_events": annotations["joint_accepted_interaction_events"],
            "accepted_joint_none_intervals": annotations["joint_accepted_none_intervals"],
            "observed_accepted_joint_interaction_events": annotations["observed_joint_accepted_interaction_events"],
            "observed_accepted_joint_none_intervals": annotations["observed_joint_accepted_none_intervals"],
            "usable_independent_acquisitions": supervised_acquisitions,
            "usable_target_frames": supervised_frames,
            "verified_usable_independent_acquisitions": frames["verified_usable_independent_acquisitions"],
            "verified_usable_target_frames": frames["verified_usable_valid"],
            "is_120_when_accepted_events_are_zero": False,
        },
        "sufficiency": {
            "engineering_closed_loop": {
                "status": ("blocked_now" if supervised_frames == 0 else
                           "pending_target_frame_inventory" if supervised_frames is None else
                           "small_real_pilot_possible_with_explicit_manual_authorization"),
                "requires_120": False,
            },
            "small_model_baseline": {
                "status": "not_assessable_until_accepted_labels_and_predictions",
                "120_interpretation": "possible_pilot_scale_not_an_accuracy_or_power_guarantee",
            },
            "fixed_scene_transfer": {
                "status": "limited_evidence",
                "120_interpretation": "may_support_a_pilot_only_with_representative_grouped_holdout",
            },
            "open_scene_generalization": {
                "status": "not_established_by_120",
                "120_interpretation": "single_internal_count_cannot_establish_open_scene_generalization",
            },
            "statistical_power": None,
            "final_accuracy": None,
            "learning_curve_plateau": None,
            "per_class_scarcity_or_failure": {
                name: {"accepted_events": annotations["interaction_events"]["by_type"][name]["accepted"],
                       "failure_assessment": None} for name in INTERACTION_TYPES
            },
            "not_computable_until": ["accepted events and explicit none intervals", "valid target-frame projections",
                                     "frozen model predictions with acquisition group IDs"],
        },
        "benchmark": {
            "schema_version": benchmark["schema_version"],
            "designs": designs,
            "seeds": benchmark.get("seeds"),
            "models": benchmark.get("models"),
            "test_policy": benchmark.get("evaluation_protocol", {}).get("test_policy"),
            "plateau": None,
            "status": ("not_runnable_with_zero_usable_supervision" if supervised_frames == 0 else
                       "not_assessable_with_unknown_usable_supervision" if supervised_frames is None else
                       "pending_results"),
        },
    }
    return report


def _count(summary, split=None, dimension=None):
    selected = summary if split is None else summary["by_split"][split]
    return selected["independent_acquisitions"] if dimension is None else selected["dimensions"][dimension]["count"]


def _display(value):
    return "未知（未完整提供）" if value is None else str(value)


def render_markdown(report):
    joint = report["layers"]["downloaded_joint_candidates"]
    qualified = report["layers"]["qualified_joint"]
    auxiliary = report["layers"]["downloaded_auxiliary"]
    events = report["annotations"]["interaction_events"]
    none = report["annotations"]["none_intervals"]
    frames = report["target_frames"]
    windows = report["windows"]
    target = report["target"]["qualified_independent_acquisitions"]
    supervision = report["current_supervision"]
    accepted_events = supervision["accepted_joint_interaction_events"]
    accepted_none = supervision["accepted_joint_none_intervals"]
    if accepted_events is None or accepted_none is None:
        supervision_text = (
            f"accepted 总量未知（annotations coverage=`{report['annotations']['coverage_status']}`）；"
            f"已观测 accepted 交互事件={supervision['observed_accepted_joint_interaction_events']}，"
            f"accepted none={supervision['observed_accepted_joint_none_intervals']}"
        )
    else:
        supervision_text = f"accepted 交互事件={accepted_events}，accepted none={accepted_none}"
    usable = frames["usable_valid"]
    if usable == 0:
        engineering_text = f"工程闭环不需要 {target}，但当前正式可用监督为 0，正式闭环仍被阻塞。"
    elif usable is None:
        engineering_text = f"工程闭环不需要 {target}；正式可用监督总量未知，不能用未知计数放行。"
    else:
        engineering_text = (f"工程闭环不需要 {target}；当前已验证 {usable} 个正式可用目标帧，"
                            "小样本调试仍需另行明确人工授权。")
    lines = [
        "# 数据规模充足性评估",
        "",
        f"- 数据观测时间：`{report.get('observed_at')}`；{target} 是项目内部门槛，不是官方标准。",
        f"- 联合下载候选：**{joint['independent_acquisitions']}** 个独立采集；正式 qualified：**{qualified['independent_acquisitions']}**；距 {target} 缺口：**{report['target']['qualified_gap']}**。",
        f"- 辅助池：**{auxiliary['independent_acquisitions']}** 个，另有 {joint['duplicate_view_or_record_excess']} 条重复机位/记录差额；均不抵扣 {target}。",
        f"- `training_manifest.json`：{'存在' if report['training_manifest_present'] else '不存在'}；正式 `training_allowed={str(report['training_allowed']).lower()}`，该开关不授权小样本 smoke。",
        "",
        "## 独立采集与覆盖",
        "",
        "| 层级 | train | val | test | 合计 | 被试 | 场景 | 物体实例 | 语义类别 | 动作 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (("联合下载候选", joint), ("正式 qualified", qualified)):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, *[_count(summary, split) for split in SPLITS], _count(summary),
            *[_count(summary, dimension=key) for key in ("subjects", "scenes", "object_instances",
                                                         "object_semantic_categories", "actions")]))
    lines.extend([
        "",
        f"未知维度不补猜：联合候选场景状态为 `{joint['dimensions']['scenes']['status']}`，物体语义类别状态为 `{joint['dimensions']['object_semantic_categories']['status']}`；{joint['dimensions']['object_instances']['count']} 个实例 ID 不等于语义类别数。",
        "",
        "## 审核监督与窗口",
        "",
        f"- annotations coverage=`{report['annotations']['coverage_status']}`；sample inventory=`{frames['inventory_status']}`。{supervision_text}。",
        "- 六类事件按 accepted/rejected/pending/needs_revision 统计；pending 不是监督，也不是 none。",
    ])
    for name in INTERACTION_TYPES:
        counts = events["by_type"][name]
        lines.append(f"  - `{name}`: " + "/".join(str(counts[status]) for status in REVIEW_STATUSES))
    lines.extend([
        f"- 显式 none 区间：accepted={none['by_status']['accepted']}，rejected={none['by_status']['rejected']}，pending={none['by_status']['pending']}；accepted 覆盖帧（按采集取并集）={none['accepted_coverage_frames_union']}。",
        f"- 当前可用于正式监督的独立采集={_display(supervision['usable_independent_acquisitions'])}，有效目标帧={_display(usable)}；已完整验证目标帧={frames['verified_usable_valid']}，链接但未完全验证={_display(frames['linked_but_not_fully_verified'])}，拒绝 usable={_display(frames['rejected_usable'])}。",
        f"- 声明有效/无效目标帧：{_display(frames['valid'])}/{_display(frames['invalid'])}；已登记窗口={_display(windows['generated'])}，正式可用窗口={_display(windows['formal_usable'])}。窗口不作为独立样本量。",
    ])
    legacy = report["annotations"].get("legacy_source_semantic_review")
    if legacy:
        lines.append(f"- 旧原文语义审核表另列 {legacy['rows']} 条，pending={legacy['by_status']['pending']}；未折算成六类 HOI。")
    lines.extend([
        "",
        f"## {target} 是否够",
        "",
        f"- {engineering_text}",
        f"- 小模型基线：{target} 只能在质量、类别覆盖和隔离划分合格后作为试点规模，不保证准确率或统计功效。",
        "- 固定场景迁移：可作有条件的小规模验证；当前被试、来源和类别覆盖限制不能由窗口数掩盖。",
        f"- 开放场景泛化：{target} 这个数量本身不能建立结论，需要更多被试、场景、来源和语义物体覆盖。",
        "",
        "## 学习曲线预算（均为独立采集，不是窗口）",
        "",
    ])
    for design in report["benchmark"]["designs"]:
        lines.append(
            f"- `{design['design_id']}`：训练预算 {design['train_acquisition_budgets']}，val/test={design['val']}/{design['test']}，"
            f"最低总采集={design['minimum_total_independent_acquisitions']}；"
            + ("val/test 已含在固定总量内。" if design["holdout_included_in_fixed_total"]
               else f"val/test 是训练 {target} 之外的额外独立采集。")
            + "整组划分可能使实际总量更高。"
        )
    lines.extend([
        "",
        f"至少 {report['benchmark'].get('minimum_seeds', 3)} seeds，比较 LSTM/Mamba 与多数类、持续交互基线；只用 train 拟合归一化，只用 val 选模和判断 plateau，test 在协议冻结后最终只评一次。置信区间必须按独立采集 cluster bootstrap。{supervision_text}；尚无冻结模型预测，不能给出 plateau、统计功效、每类失败率或最终准确率。",
        "",
    ])
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--sample-index", type=Path)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json",
                        help="stdout format when the corresponding output path is omitted")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_scale(args.progress, args.manifest, args.freeze, args.annotations,
                                args.sample_index, args.benchmark)
        json_output = _json_text(report)
        markdown_output = render_markdown(report)
        if args.json_output is not None:
            args.json_output.write_text(json_output, encoding="utf-8")
        elif args.format == "json":
            sys.stdout.write(json_output)
        if args.markdown_output is not None:
            args.markdown_output.write_text(markdown_output, encoding="utf-8")
        elif args.format == "markdown":
            sys.stdout.write(markdown_output)
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
