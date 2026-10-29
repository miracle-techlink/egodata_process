#!/usr/bin/env python3
"""Build reviewed three-head HOI samples from an accepted immutable freeze.

Usage:
    python -m src.dataset.build_formal_samples --freeze-dir FREEZE \
        --annotations-dir ANNOTATIONS --out-dir NEW_SAMPLES

The formal input contract extends human_pose_dataset_v1 with verified
hand/object NPZ sidecars, native frame_ids and timestamps, and hoi_annotation_v1
JSONL assets. Candidate freezes and fixture-mode releases are never accepted.
"""

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

try:
    from .build_samples import make_sample, object_features, parse_source_video
    from .human_dataset import verify_freeze
    from .label_rules import INTERACTION_VOCAB
except ImportError:
    from build_samples import make_sample, object_features, parse_source_video
    from human_dataset import verify_freeze
    from label_rules import INTERACTION_VOCAB


SCHEMA_VERSION = "hoi_formal_samples_v1"
ANNOTATION_SCHEMA = "hoi_annotation_v1"
CONTRACT_VERSION = "2026-10-22"
SPLITS = ("train", "val", "test")
SUBJECT_IDS = {"left": 0, "right": 1, "none": 2}
INTERACTION_IDS = {name: index for index, name in enumerate(INTERACTION_VOCAB)}
REVIEW_STATUSES = {"accepted", "rejected", "pending", "needs_revision"}
GEOMETRY_STATUSES = {"not_checked", "passed", "failed", "inconclusive"}
LABEL_SOURCES = {"vlm", "source_text", "source_text+vlm", "geometry_assisted", "human"}
ANNOTATION_MODE = "offline_noncausal"
HAND_ROLES = ("hand_pose", "hand_observations")
POSE_ROLES = ("pose6d", "object_observations")
HAND_META_ROLES = ("hand_pose_meta", "hand_observations_meta")
POSE_META_ROLES = ("pose6d_meta", "object_observations_meta")
EVENT_ROLES = ("hoi_labels", "hoi_events")
NONE_ROLES = ("hoi_none_intervals", "none_intervals")


class FormalSampleError(ValueError):
    def __init__(self, code, message, report=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.report = report


class FreezeRejected(FormalSampleError):
    pass


def _require(condition, code, message):
    if not condition:
        raise FormalSampleError(code, message)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise FormalSampleError("json_invalid", f"{path}: {exc}") from exc
    return value


def _write_new(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _known_string(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value):
    return type(value) is int


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _load_freeze(freeze_dir):
    root = Path(freeze_dir).resolve()
    report = verify_freeze(root)
    manifest_path = root / "training_manifest.json"
    problems = []
    if not manifest_path.is_file():
        problems.append("training_manifest.json is missing; candidate-only data cannot produce formal samples")
    if not report.get("integrity_verified"):
        problems.append("freeze integrity or asset verification failed")
    if not report.get("training_allowed"):
        problems.append("the frozen release does not grant training permission")
    if problems:
        raise FreezeRejected("freeze_gate_closed", "; ".join(problems), report=report)
    try:
        training = _read_json(manifest_path)
        seal = _read_json(root / "freeze.json")
        ledger = _read_json(root / "source_ledger.json")
    except FormalSampleError as exc:
        raise FreezeRejected("freeze_gate_closed", exc.message, report=report) from exc
    _require(training.get("schema_version") == "human_pose_dataset_v1" and
             training.get("status") == "accepted" and training.get("training_allowed") is True and
             training.get("split_before_windows") is True,
             "training_manifest_invalid", "accepted training permission and split-before-windows evidence are required")
    records = ledger.get("records") if isinstance(ledger, dict) else None
    _require(isinstance(records, list), "source_ledger_invalid", "source ledger records must be a list")
    assignments = []
    seen = set()
    for split in SPLITS:
        entries = training.get("splits", {}).get(split)
        _require(isinstance(entries, list), "training_manifest_invalid", f"missing split list: {split}")
        for entry in entries:
            _require(isinstance(entry, dict), "training_manifest_invalid", "split entries must be objects")
            index = entry.get("record_index")
            sequence_id = entry.get("sequence_id")
            _require(_integer(index) and 0 <= index < len(records) and _known_string(sequence_id),
                     "training_manifest_invalid", "invalid record_index or sequence_id")
            _require(sequence_id not in seen, "duplicate_sequence", f"sequence occurs more than once: {sequence_id}")
            seen.add(sequence_id)
            record = records[index]
            _require(isinstance(record, dict) and record.get("sequence_id") == sequence_id and
                     entry.get("joint_pose_ready") is True,
                     "training_manifest_invalid", f"manifest/source mismatch for {sequence_id}")
            assignments.append({"split": split, "entry": entry, "record": record})
    _require(assignments, "training_release_empty", "the accepted training manifest contains no sequences")
    _validate_split_isolation(assignments)
    return {
        "root": root,
        "report": report,
        "training": training,
        "seal": seal,
        "ledger": ledger,
        "assignments": assignments,
        "asset_base": Path(seal["asset_base"]).resolve(),
        "freeze_sha256": report["freeze_sha256"],
        "config_sha256": _sha256_file(root / "config.json"),
        "effective_config_sha256": seal["config"]["effective_sha256"],
        "source_ledger_sha256": seal["source_ledger"]["sha256"],
        "training_manifest_sha256": _sha256_file(manifest_path),
    }


def _identity_tokens(assignment):
    entry, record = assignment["entry"], assignment["record"]
    tokens = set()
    for key in ("group_id", "acquisition_group_id", "subject_group_id"):
        if _known_string(entry.get(key)):
            tokens.add(f"manifest:{key}:{entry[key]}")
    for key in ("acquisition_id", "sequence_id", "source_content_id", "source_sequence_id",
                "source_episode_uuid", "source_video", "parent_sequence_id", "camera_sequence_id"):
        if _known_string(record.get(key)):
            tokens.add(f"source:{record[key]}")
    if _known_string(record.get("subject_id")):
        tokens.add(f"subject:{record.get('dataset')}:{record['subject_id']}")
    for key in ("aliases", "source_aliases", "acquisition_aliases"):
        values = record.get(key)
        if isinstance(values, list):
            tokens.update(f"source:{value}" for value in values if _known_string(value))
    values = record.get("subject_aliases")
    if isinstance(values, list):
        tokens.update(f"subject_alias:{value}" for value in values if _known_string(value))
    assets = record.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict) and asset.get("role") in ("rgb", "depth") and _known_string(asset.get("sha256")):
                tokens.add(f"content:{asset['sha256'].lower()}")
    return tokens


def _validate_split_isolation(assignments):
    owners = {}
    for assignment in assignments:
        split = assignment["split"]
        sequence_id = assignment["record"]["sequence_id"]
        for token in _identity_tokens(assignment):
            previous = owners.setdefault(token, (split, sequence_id))
            _require(previous[0] == split, "split_leakage",
                     f"{sequence_id} and {previous[1]} share {token} across {previous[0]} and {split}")


def _resolve_asset(record, roles, asset_base):
    assets = record.get("assets")
    matches = [asset for asset in assets if isinstance(asset, dict) and asset.get("role") in roles] \
        if isinstance(assets, list) else []
    sequence_id = record.get("sequence_id")
    _require(len(matches) == 1, "asset_role_missing_or_ambiguous",
             f"{sequence_id}: exactly one verified asset with role {'/'.join(roles)} is required")
    asset = matches[0]
    name = asset.get("path")
    declared = asset.get("sha256")
    _require(_known_string(name) and _known_string(declared), "asset_hash_missing",
             f"{sequence_id}: asset {roles[0]} needs a local path and SHA-256")
    path = Path(name)
    path = (asset_base / path).resolve() if not path.is_absolute() else path.resolve()
    _require(path.is_file() and not path.is_symlink(), "asset_missing", f"{sequence_id}: missing regular asset {path}")
    actual = _sha256_file(path)
    _require(actual == declared.lower(), "asset_sha256_mismatch", f"{sequence_id}: content changed for {path}")
    if "size_bytes" in asset:
        _require(_integer(asset["size_bytes"]) and asset["size_bytes"] == path.stat().st_size,
                 "asset_size_mismatch", f"{sequence_id}: size changed for {path}")
    return {"role": asset["role"], "path": path, "sha256": actual, "declared_path": name}


def _resolve_sequence_assets(record, asset_base, annotations_dir):
    sequence_id = record["sequence_id"]
    assets = {
        "hand": _resolve_asset(record, HAND_ROLES, asset_base),
        "pose": _resolve_asset(record, POSE_ROLES, asset_base),
        "hand_meta": _resolve_asset(record, HAND_META_ROLES, asset_base),
        "pose_meta": _resolve_asset(record, POSE_META_ROLES, asset_base),
        "timestamps": _resolve_asset(record, ("timestamps",), asset_base),
        "frame_ids": _resolve_asset(record, ("frame_ids",), asset_base),
        "events": _resolve_asset(record, EVENT_ROLES, asset_base),
        "none": _resolve_asset(record, NONE_ROLES, asset_base),
    }
    _require(assets["hand_meta"]["path"] == assets["hand"]["path"].with_suffix(".meta.json"),
             "hand_meta_mismatch", f"{sequence_id}: hand sidecar must be the NPZ .meta.json companion")
    _require(assets["pose_meta"]["path"] == assets["pose"]["path"].with_suffix(".meta.json"),
             "pose_meta_mismatch", f"{sequence_id}: pose sidecar must be the NPZ .meta.json companion")
    annotation_root = Path(annotations_dir).resolve() / sequence_id
    _require(assets["events"]["path"] == (annotation_root / "events.jsonl").resolve(),
             "annotation_binding_mismatch", f"{sequence_id}: events.jsonl is not the frozen hoi_labels asset")
    _require(assets["none"]["path"] == (annotation_root / "none_intervals.jsonl").resolve(),
             "annotation_binding_mismatch", f"{sequence_id}: none_intervals.jsonl is not the frozen asset")
    hashes = {key: value["sha256"] for key, value in assets.items()}
    assets["source_content_sha256"] = _sha256_bytes(_json_bytes(hashes))
    assets["annotation_sha256"] = _sha256_bytes(_json_bytes({
        "events": assets["events"]["sha256"], "none_intervals": assets["none"]["sha256"]}))
    return assets


def _read_table(path, key, csv_key, integer=False):
    path = Path(path)
    try:
        if path.suffix.lower() == ".json":
            value = _read_json(path)
            values = value.get(key) if isinstance(value, dict) else value
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                values = [row[csv_key] for row in csv.DictReader(stream)]
        else:
            raise FormalSampleError("time_asset_invalid", f"{path}: expected JSON or CSV")
        _require(isinstance(values, list), "time_asset_invalid", f"{path}: {key} must be an array")
        converted = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError("boolean table values are forbidden")
            if integer:
                if not isinstance(value, int):
                    if not isinstance(value, str) or value.strip() != str(int(value.strip())):
                        raise ValueError("frame_ids must contain literal integers")
                    value = int(value.strip())
                converted.append(value)
            else:
                converted.append(float(value))
        values = converted
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise FormalSampleError("time_asset_invalid", f"{path}: invalid {key}: {exc}") from exc
    return values


def _validate_meta(meta, record, assets, kind):
    sequence_id = record["sequence_id"]
    _require(isinstance(meta, dict), "asset_meta_invalid", f"{sequence_id}: {kind} metadata must be an object")
    _require(meta.get("sequence_id") == sequence_id and meta.get("frame_count") == record.get("frame_count"),
             "asset_meta_mismatch", f"{sequence_id}: {kind} metadata sequence/frame_count mismatch")
    for key in ("fps", "width", "height"):
        _require(_finite_number(meta.get(key)) and meta[key] > 0, "asset_meta_missing",
                 f"{sequence_id}: {kind} metadata needs measured positive {key}")
    _require(_integer(meta["width"]) and _integer(meta["height"]), "image_size_invalid",
             f"{sequence_id}: image width/height must be integers")
    _require(math.isclose(float(meta["fps"]), float(record["fps"]), rel_tol=1e-6, abs_tol=1e-9),
             "fps_mismatch", f"{sequence_id}: {kind} FPS differs from the frozen timebase")
    _require(meta.get("timestamps_sha256") == assets["timestamps"]["sha256"] and
             meta.get("frame_ids_sha256") == assets["frame_ids"]["sha256"],
             "native_timeline_binding_missing", f"{sequence_id}: {kind} metadata must bind both native table SHA-256 values")
    source = record.get("timestamp_source")
    source = source.get("kind") if isinstance(source, dict) else source
    _require(meta.get("timestamp_source") == source, "timestamp_source_mismatch",
             f"{sequence_id}: {kind} metadata timestamp_source mismatch")


def _validate_pose_geometry(poses, bboxes, image_wh, sequence_id):
    nan = np.isnan(poses).any(axis=(2, 3))
    missing = nan
    valid = ~missing
    _require(not np.isinf(poses).any(), "pose_geometry_invalid", f"{sequence_id}: non-finite object pose")
    if missing.any():
        _require(bool(np.isnan(poses[missing]).all()), "pose_geometry_invalid",
                 f"{sequence_id}: missing object poses must be entirely NaN")
    if valid.any():
        selected = poses[valid].astype(np.float64)
        rotation = selected[:, :3, :3]
        orthogonality = np.matmul(rotation.transpose(0, 2, 1), rotation)
        error = np.max(np.abs(orthogonality - np.eye(3)[None]), axis=(1, 2))
        determinant = np.linalg.det(rotation)
        _require(bool((error <= 1e-4).all() and (np.abs(determinant - 1.0) <= 1e-4).all() and
                      np.allclose(selected[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6)),
                 "pose_geometry_invalid", f"{sequence_id}: object poses must be rigid object-to-camera matrices")
    if missing.any():
        _require(bool((bboxes[missing] == -1).all()), "bbox_geometry_invalid",
                 f"{sequence_id}: missing pose frames need bbox=-1")
    if valid.any():
        boxes = bboxes[valid]
        _require(bool((boxes[..., 2] > boxes[..., 0]).all() and (boxes[..., 3] > boxes[..., 1]).all()),
                 "bbox_geometry_invalid", f"{sequence_id}: valid poses need positive-area xyxy boxes")
    return valid


def _load_sequence(record, assets):
    sequence_id = record["sequence_id"]
    frame_count = record.get("frame_count")
    fps = record.get("fps")
    _require(_integer(frame_count) and frame_count >= 2 and _finite_number(fps) and fps > 0,
             "timebase_missing", f"{sequence_id}: real frame_count and FPS are required")
    timestamps = _read_table(assets["timestamps"]["path"], "timestamps", "timestamp_seconds")
    frame_ids = _read_table(assets["frame_ids"]["path"], "frame_ids", "frame_id", integer=True)
    _require(len(timestamps) == frame_count and len(frame_ids) == frame_count,
             "native_timeline_mismatch", f"{sequence_id}: native tables must cover every frame")
    _require(all(_finite_number(value) for value in timestamps) and
             all(right > left for left, right in zip(timestamps, timestamps[1:])),
             "native_timestamps_invalid", f"{sequence_id}: timestamps must be finite and strictly increasing")
    _require(all(_integer(value) and value >= 0 for value in frame_ids) and
             all(right - left == 1 for left, right in zip(frame_ids, frame_ids[1:])),
             "native_frame_ids_invalid", f"{sequence_id}: frame_ids must be continuous and nonnegative")
    span = timestamps[-1] - timestamps[0]
    _require(span > 0 and math.isclose((frame_count - 1) / span, float(fps), rel_tol=0.01, abs_tol=1e-9),
             "fps_mismatch", f"{sequence_id}: native timestamps disagree with frozen FPS")
    hand_meta = _read_json(assets["hand_meta"]["path"])
    pose_meta = _read_json(assets["pose_meta"]["path"])
    _validate_meta(hand_meta, record, assets, "hand")
    _validate_meta(pose_meta, record, assets, "pose")
    _require((hand_meta["width"], hand_meta["height"]) == (pose_meta["width"], pose_meta["height"]),
             "image_size_mismatch", f"{sequence_id}: hand/object image dimensions differ")
    try:
        with np.load(assets["hand"]["path"], allow_pickle=False) as hand:
            left = np.asarray(hand["left_hand"], np.float32)
            right = np.asarray(hand["right_hand"], np.float32)
            conf = np.asarray(hand["conf"], np.float32) if "conf" in hand else None
        with np.load(assets["pose"]["path"], allow_pickle=False) as pose:
            object_ids = [str(value) for value in pose["object_ids"]]
            raw_poses = np.asarray(pose["poses"])
            raw_bboxes = np.asarray(pose["bboxes"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FormalSampleError("trajectory_asset_invalid", f"{sequence_id}: {exc}") from exc
    _require(left.shape == (frame_count, 21, 3) and right.shape == (frame_count, 21, 3),
             "hand_shape_mismatch", f"{sequence_id}: hand arrays must be (T,21,3)")
    _require(conf is None or conf.shape == (frame_count, 2), "hand_shape_mismatch",
             f"{sequence_id}: conf must be (T,2)")
    _require(not np.isinf(left).any() and not np.isinf(right).any(), "hand_geometry_invalid",
             f"{sequence_id}: hand observations contain infinity")
    count = len(object_ids)
    _require(count > 0, "object_trajectory_missing", f"{sequence_id}: formal samples require object tracks")
    _require(all(_known_string(value) for value in object_ids) and len(set(object_ids)) == count,
             "object_id_invalid", f"{sequence_id}: stable unique object_ids are required")
    _require(set(object_ids) == set(record.get("object_ids", [])),
             "object_track_mismatch", f"{sequence_id}: pose object_ids differ from the frozen ledger")
    _require(raw_poses.shape == (count, frame_count, 4, 4) and raw_bboxes.shape == (count, frame_count, 4),
             "object_shape_mismatch", f"{sequence_id}: poses/bboxes must be (N,T,4,4)/(N,T,4)")
    _require(np.issubdtype(raw_bboxes.dtype, np.integer), "bbox_geometry_invalid",
             f"{sequence_id}: bboxes must have an integer dtype")
    poses = np.asarray(raw_poses, np.float32)
    bboxes = np.asarray(raw_bboxes, np.int32)
    pose_valid = _validate_pose_geometry(poses, bboxes,
                                         (hand_meta["width"], hand_meta["height"]), sequence_id)
    _require(bool(pose_valid.any()), "object_trajectory_missing", f"{sequence_id}: all object poses are missing")
    return {
        "clip_id": sequence_id,
        "source_video": record.get("source_video") or parse_source_video(sequence_id),
        "T": frame_count,
        "fps": float(fps),
        "image_wh": (hand_meta["width"], hand_meta["height"]),
        "left": left,
        "right": right,
        "object_ids": object_ids,
        "poses": poses,
        "bboxes": bboxes,
        "pose_valid": pose_valid,
        "timestamps": np.asarray(timestamps, np.float64),
        "frame_ids": np.asarray(frame_ids, np.int64),
    }


def _reviewed_at(value):
    if not _known_string(value):
        return False
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _read_annotations(path, sequence_id, frame_ids, none=False):
    frame_set = set(int(value) for value in frame_ids)
    first, last = min(frame_set), max(frame_set)
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FormalSampleError("annotation_unreadable", f"{sequence_id}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise FormalSampleError("annotation_json_invalid", f"{path}:{line_number}: {exc}") from exc
        location = f"{path}:{line_number}"
        _require(isinstance(item, dict), "annotation_invalid", f"{location}: expected an object")
        if "schema_version" in item:
            _require(item["schema_version"] == ANNOTATION_SCHEMA, "annotation_schema_invalid",
                     f"{location}: unsupported schema_version")
        identifier = item.get("interval_id", item.get("event_id")) if none else item.get("event_id")
        _require(_known_string(identifier) and item.get("sequence_id") == sequence_id,
                 "annotation_identity_invalid", f"{location}: event/interval ID or sequence_id mismatch")
        item["event_id"] = identifier
        start, end = item.get("start_frame"), item.get("end_frame")
        _require(_integer(start) and _integer(end) and first <= start < end <= last + 1,
                 "annotation_bounds_invalid", f"{location}: native half-open bounds are invalid")
        evidence = item.get("evidence_frames")
        _require(isinstance(evidence, list) and evidence and
                 all(_integer(frame) and frame in frame_set and start <= frame < end for frame in evidence) and
                 len(set(evidence)) == len(evidence),
                 "annotation_evidence_invalid", f"{location}: unique in-event native evidence_frames are required")
        geometry = item.get("geometry_status", "not_checked") if none else item.get("geometry_status")
        _require(item.get("review_status") in REVIEW_STATUSES and geometry in GEOMETRY_STATUSES,
                 "annotation_status_invalid", f"{location}: unsupported review/geometry status")
        item["geometry_status"] = geometry
        _require(item.get("label_source") in LABEL_SOURCES and item.get("annotation_mode") == ANNOTATION_MODE and
                 item.get("allowed_as_model_input") is False,
                 "annotation_provenance_invalid", f"{location}: label_source/offline mode/input isolation are required")
        source_ids = item.get("source_event_ids")
        _require(isinstance(source_ids, list) and all(isinstance(value, str) for value in source_ids),
                 "annotation_provenance_invalid", f"{location}: source_event_ids must be a string list")
        if item.get("confidence") is not None:
            _require(_finite_number(item["confidence"]) and 0 <= item["confidence"] <= 1,
                     "annotation_invalid", f"{location}: confidence must be finite and in [0,1]")
        if none:
            _require(item.get("subject") in (None, "none") and item.get("object_id") in (None, "") and
                     item.get("interaction_type") in (None, "none"),
                     "none_interval_invalid", f"{location}: none interval has interaction fields")
        else:
            _require(item.get("subject") in ("left", "right") and
                     item.get("interaction_type") in INTERACTION_IDS and item.get("interaction_type") != "none" and
                     _known_string(item.get("object_hint")),
                     "event_class_invalid", f"{location}: unsupported subject/interaction/object_hint")
            object_id = item.get("object_id")
            _require(object_id is None or _known_string(object_id), "event_object_invalid",
                     f"{location}: object_id must be null or a stable ID")
        if item["review_status"] == "accepted":
            _require(_known_string(item.get("reviewer")) and _reviewed_at(item.get("reviewed_at")),
                     "accepted_review_evidence_missing",
                     f"{location}: accepted labels need reviewer and timezone-aware reviewed_at")
        records.append(item)
    return records


def _load_annotation_set(assets, sequence_id, frame_ids, object_ids):
    events = _read_annotations(assets["events"]["path"], sequence_id, frame_ids)
    none_intervals = _read_annotations(assets["none"]["path"], sequence_id, frame_ids, none=True)
    identifiers = [item["event_id"] for item in events + none_intervals]
    _require(len(identifiers) == len(set(identifiers)), "duplicate_event_id",
             f"{sequence_id}: event_ids must be unique across events and none_intervals")
    _require(any(item["review_status"] == "accepted" for item in events + none_intervals),
             "accepted_annotations_missing", f"{sequence_id}: pending/rejected-only annotations cannot be released")
    slots = {object_id: index for index, object_id in enumerate(object_ids)}
    for event in events:
        if event["review_status"] == "accepted" and event.get("object_id") is not None:
            _require(event["object_id"] in slots, "object_track_mismatch",
                     f"{sequence_id}: accepted event {event['event_id']} references an unknown object track")
    return events, none_intervals


def _occluded(item):
    return item.get("occluded") is True or item.get("occlusion_status") in {"occluded", "unknown", "pending"}


def _invalid_label(reason, items=()):
    return {"label_valid": 0, "reason": reason,
            "evidence_event_ids": sorted({item["event_id"] for item in items})}


def _project_label(frame, events, none_intervals, object_ids):
    active_events = [item for item in events if item["start_frame"] <= frame < item["end_frame"]]
    active_none = [item for item in none_intervals if item["start_frame"] <= frame < item["end_frame"]]
    active = active_events + active_none
    unresolved = [item for item in active if item["review_status"] in ("pending", "needs_revision")]
    if unresolved:
        reason = "pending_annotation" if any(item["review_status"] == "pending" for item in unresolved) \
            else "annotation_needs_revision"
        return _invalid_label(reason, active)
    accepted_events = [item for item in active_events if item["review_status"] == "accepted"]
    accepted_none = [item for item in active_none if item["review_status"] == "accepted"]
    if accepted_events and accepted_none:
        return _invalid_label("accepted_event_none_conflict", active)
    if len(accepted_events) > 1:
        return _invalid_label("multiple_accepted_events", accepted_events)
    if len(accepted_none) > 1:
        return _invalid_label("multiple_accepted_none_intervals", accepted_none)
    if not accepted_events and not accepted_none:
        return _invalid_label("unannotated_or_rejected", active)
    item = (accepted_events + accepted_none)[0]
    if _occluded(item):
        return _invalid_label("occluded", [item])
    if item["geometry_status"] != "passed" and not (accepted_none and item["geometry_status"] == "not_checked"):
        return _invalid_label("geometry_not_passed", [item])
    common = {"label_valid": 1, "reason": None, "label_event_ids": [item["event_id"]],
              "label_source": item["label_source"], "annotation_mode": item["annotation_mode"],
              "reviewer": item["reviewer"], "reviewed_at": item["reviewed_at"],
              "source_event_ids": list(item.get("source_event_ids") or [])}
    if accepted_none:
        return {**common, "label_subject": 2, "label_object_slot": -1, "label_object_id": "",
                "label_interaction": 0, "label_interaction_name": "none"}
    if item.get("object_id") is None:
        return _invalid_label("unknown_object", [item])
    _require(item["object_id"] in object_ids, "object_track_mismatch",
             f"accepted event {item['event_id']} has no object slot")
    slot = object_ids.index(item["object_id"])
    interaction = item["interaction_type"]
    return {**common, "label_subject": SUBJECT_IDS[item["subject"]], "label_object_slot": slot,
            "label_object_id": item["object_id"], "label_interaction": INTERACTION_IDS[interaction],
            "label_interaction_name": interaction}


def _target_index(timestamps, observation_index, horizon_seconds):
    if horizon_seconds is None:
        return observation_index + 1, None
    requested = float(timestamps[observation_index]) + horizon_seconds
    _require(math.isfinite(requested), "horizon_invalid", "requested target timestamp is not finite")
    target = int(np.searchsorted(timestamps, requested, side="left"))
    return target, requested


def _event_statistics(events, none_intervals, frame_ids):
    stats = Counter()
    for item in events:
        stats[f"events.review.{item['review_status']}"] += 1
        stats[f"events.geometry.{item['geometry_status']}"] += 1
        if item.get("object_id") is not None:
            stats["events.track_linked"] += 1
    for item in none_intervals:
        stats[f"none.review.{item['review_status']}"] += 1
        stats[f"none.geometry.{item['geometry_status']}"] += 1
    covered = {frame for frame in frame_ids for item in none_intervals
               if item["review_status"] == "accepted" and item["start_frame"] <= frame < item["end_frame"]}
    stats["none.accepted_coverage_frames"] = len(covered)
    return stats


def _code_provenance():
    package = __package__ or ""
    module_names = {__name__, make_sample.__module__, verify_freeze.__module__}
    module_names.update({f"{package}.label_rules" if package else "label_rules",
                         f"{package}.rot6d" if package else "rot6d"})
    files = {}
    for name in sorted(module_names):
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        if path:
            files[str(Path(path).resolve())] = _sha256_file(path)
    own = str(Path(__file__).resolve())
    _require(own in files and len(files) == 5, "code_provenance_missing",
             "builder, legacy tensors, freeze, label and rotation code hashes are required")
    return {"sha256": files[own], "files": files,
            "bundle_sha256": _sha256_bytes(_json_bytes(files))}


def _sample_content_sha256(sample):
    digest = hashlib.sha256()
    for key in sorted(sample):
        if key == "content_sha256":
            continue
        value = np.asarray(sample[key])
        _require(not value.dtype.hasobject, "sample_serialization_invalid", f"object dtype is forbidden: {key}")
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(_json_bytes(list(value.shape)))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _annotation_provenance(info):
    return {key: {"role": info["assets"][key]["role"], "sha256": info["assets"][key]["sha256"],
                  "declared_path": info["assets"][key]["declared_path"]}
            for key in ("events", "none")}


def _make_output_sample(info, data, feats, valid, plan, n_max, context, include_invalid):
    projection = plan["projection"]
    target_index = plan["target_index"]
    observation_index = plan["observation_index"]
    start_index = plan["start_index"]
    labels = (np.full(data["T"], 2, np.int64), np.full(data["T"], -1, np.int64),
              np.zeros(data["T"], np.int64))
    if projection["label_valid"]:
        for label, key in zip(labels, ("label_subject", "label_object_slot", "label_interaction")):
            label[target_index] = projection[key]
    sample = make_sample(data, feats, valid, labels, start_index, info["window"], n_max,
                         horizon_frames=target_index - observation_index)
    sample["source_video"] = data["source_video"]
    sample["split_group"] = info["entry"].get("group_id") or info["entry"]["acquisition_group_id"]
    sample["start_index"] = np.int64(start_index)
    sample["target_index"] = np.int64(target_index)
    sample["observation_end_index"] = np.int64(observation_index)
    sample["start_frame"] = np.int64(data["frame_ids"][start_index])
    sample["target_frame"] = np.int64(data["frame_ids"][target_index])
    sample["observation_end_frame"] = np.int64(data["frame_ids"][observation_index])
    sample["observation_end_timestamp"] = np.float64(data["timestamps"][observation_index])
    sample["target_timestamp"] = np.float64(data["timestamps"][target_index])
    requested = plan["requested_target_timestamp"]
    sample["requested_target_timestamp"] = np.float64(requested if requested is not None else
                                                      data["timestamps"][target_index])
    sample["horizon_seconds"] = np.float64(data["timestamps"][target_index] -
                                           data["timestamps"][observation_index])
    sample["sequence_id"] = info["sequence_id"]
    sample["split"] = info["split"]
    sample["acquisition_group_id"] = info["entry"].get("acquisition_group_id", "")
    sample["subject_group_id"] = info["entry"].get("subject_group_id", "") or ""
    sample["label_valid"] = np.int64(projection["label_valid"])
    sample["annotation_mode"] = projection.get("annotation_mode", ANNOTATION_MODE)
    sample["training_allowed"] = np.bool_(not include_invalid)
    if projection["label_valid"]:
        sample["label_event_ids"] = np.asarray(projection["label_event_ids"], dtype=str)
        sample["label_source"] = projection["label_source"]
        sample["reviewer"] = projection["reviewer"]
        sample["reviewed_at"] = projection["reviewed_at"]
        sample["source_event_ids"] = np.asarray(projection["source_event_ids"], dtype=str)
        sample["invalid_reason"] = ""
    else:
        sample["label_subject"] = np.int64(-1)
        sample["label_object_slot"] = np.int64(-1)
        sample["label_object_id"] = ""
        sample["label_interaction"] = np.int64(-1)
        sample["label_interaction_name"] = ""
        sample["label_event_ids"] = np.asarray([], dtype=str)
        sample["label_source"] = ""
        sample["reviewer"] = ""
        sample["reviewed_at"] = ""
        sample["source_event_ids"] = np.asarray([], dtype=str)
        sample["invalid_reason"] = projection["reason"]
    sample["invalid_evidence_event_ids"] = np.asarray(projection.get("evidence_event_ids", []), dtype=str)
    sample["freeze_sha256"] = context["freeze_sha256"]
    sample["config_sha256"] = context["config_sha256"]
    sample["effective_config_sha256"] = context["effective_config_sha256"]
    sample["source_ledger_sha256"] = context["source_ledger_sha256"]
    sample["training_manifest_sha256"] = context["training_manifest_sha256"]
    sample["code_sha256"] = context["code"]["sha256"]
    sample["code_bundle_sha256"] = context["code"]["bundle_sha256"]
    sample["source_content_sha256"] = info["assets"]["source_content_sha256"]
    sample["annotation_sha256"] = info["assets"]["annotation_sha256"]
    sample["annotation_events_sha256"] = info["assets"]["events"]["sha256"]
    sample["annotation_none_intervals_sha256"] = info["assets"]["none"]["sha256"]
    sample["hand_pose_sha256"] = info["assets"]["hand"]["sha256"]
    sample["pose6d_sha256"] = info["assets"]["pose"]["sha256"]
    sample["native_timestamps_sha256"] = info["assets"]["timestamps"]["sha256"]
    sample["native_frame_ids_sha256"] = info["assets"]["frame_ids"]["sha256"]
    sample["content_sha256"] = _sample_content_sha256(sample)
    _require(int(sample["target_index"]) >= int(sample["observation_end_index"]) + 1 and
             int(sample["target_index"]) > int(sample["observation_end_index"]),
             "future_isolation_failed", "target must follow the observation window")
    return sample


def _index_entry(sample, sample_sha256):
    keys = ("sample_id", "sequence_id", "clip_id", "source_video", "split", "split_group",
            "acquisition_group_id", "subject_group_id", "start_index", "target_index",
            "observation_end_index", "start_frame", "target_frame", "observation_end_frame",
            "observation_end_timestamp", "target_timestamp", "requested_target_timestamp",
            "horizon_frames", "horizon_seconds", "fps", "label_valid", "label_subject",
            "label_object_slot", "label_object_id", "label_interaction", "label_interaction_name",
            "label_source", "annotation_mode", "reviewer", "reviewed_at", "invalid_reason",
            "training_allowed", "content_sha256", "source_content_sha256", "annotation_sha256", "freeze_sha256",
            "config_sha256", "source_ledger_sha256", "code_sha256")
    entry = {key: _scalar(sample[key]) for key in keys}
    entry["label_event_ids"] = [str(value) for value in sample["label_event_ids"]]
    entry["sample_sha256"] = sample_sha256
    return entry


def _stats_json(stats):
    return {key: value for key, value in sorted(stats.items())}


def build_formal_samples(freeze_dir, annotations_dir, out_dir, *, window=32, stride=16,
                         horizon_seconds=None, n_max=0, include_invalid=False, audit_only=False):
    """Preflight all release inputs, then write a new formal sample directory."""
    _require(_integer(window) and window > 0 and _integer(stride) and stride > 0,
             "window_invalid", "window and stride must be positive integers")
    _require(_integer(n_max) and n_max >= 0, "n_max_invalid", "n_max must be a nonnegative integer")
    if horizon_seconds is not None:
        _require(_finite_number(horizon_seconds) and horizon_seconds > 0,
                 "horizon_invalid", "horizon_seconds must be finite and positive")
        horizon_seconds = float(horizon_seconds)
    destination = Path(out_dir).absolute()
    if not audit_only and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(f"Output destination already exists: {destination}")
    try:
        context = _load_freeze(freeze_dir)
    except FreezeRejected as exc:
        if not audit_only:
            raise
        return {"schema_version": SCHEMA_VERSION, "audit_only": True, "training_allowed": False,
                "release_status": "blocked", "error": str(exc), "freeze_audit": exc.report}
    context["code"] = _code_provenance()
    infos = []
    valid_counts = Counter()
    invalid_counts = Counter()
    invalid_by_split = {split: Counter() for split in SPLITS}
    target_out_counts = Counter()
    class_counts = {split: Counter() for split in SPLITS}
    subject_counts = {split: Counter() for split in SPLITS}
    raw_event_stats = Counter()
    skipped = []
    for assignment in context["assignments"]:
        record, split, entry = assignment["record"], assignment["split"], assignment["entry"]
        sequence_id = record["sequence_id"]
        assets = _resolve_sequence_assets(record, context["asset_base"], annotations_dir)
        data = _load_sequence(record, assets)
        events, none_intervals = _load_annotation_set(assets, sequence_id, data["frame_ids"], data["object_ids"])
        raw_event_stats.update(_event_statistics(events, none_intervals, data["frame_ids"]))
        plans = []
        for start_index in range(0, data["T"] - window + 1, stride):
            observation_index = start_index + window - 1
            target_index, requested = _target_index(data["timestamps"], observation_index, horizon_seconds)
            native_start = int(data["frame_ids"][start_index])
            if target_index >= data["T"]:
                target_out_counts[split] += 1
                skipped.append({"sequence_id": sequence_id, "split": split, "start_frame": native_start,
                                "reason": "target_out_of_range"})
                continue
            native_target = int(data["frame_ids"][target_index])
            projection = _project_label(native_target, events, none_intervals, data["object_ids"])
            if projection["label_valid"] and projection["label_object_slot"] >= 0 and \
                    not data["pose_valid"][projection["label_object_slot"], target_index]:
                projection = _invalid_label("object_pose_missing_at_target",
                                            [item for item in events if item["event_id"] in
                                             projection.get("label_event_ids", [])])
            plan = {"start_index": start_index, "observation_index": observation_index,
                    "target_index": target_index, "requested_target_timestamp": requested,
                    "projection": projection, "sample_id": f"{sequence_id}_{native_start:06d}"}
            plans.append(plan)
            if projection["label_valid"]:
                valid_counts[split] += 1
                class_counts[split][projection["label_interaction_name"]] += 1
                subject = next(name for name, value in SUBJECT_IDS.items() if value == projection["label_subject"])
                subject_counts[split][subject] += 1
            else:
                reason = projection["reason"]
                invalid_counts[reason] += 1
                invalid_by_split[split][reason] += 1
                if not include_invalid:
                    skipped.append({"sequence_id": sequence_id, "split": split, "start_frame": native_start,
                                    "target_frame": native_target, "reason": reason,
                                    "evidence_event_ids": projection.get("evidence_event_ids", [])})
        infos.append({**assignment, "sequence_id": sequence_id, "assets": assets, "events": events,
                      "none_intervals": none_intervals, "object_ids": data["object_ids"],
                      "object_count": len(data["object_ids"]), "frame_count": data["T"], "plans": plans,
                      "window": window})
    effective_n_max = max((info["object_count"] for info in infos if info["split"] == "train"), default=0) \
        if n_max == 0 else n_max
    _require(effective_n_max > 0, "n_max_unavailable", "no train object statistics are available")
    for info in infos:
        _require(info["object_count"] > 0, "object_trajectory_missing",
                 f"{info['sequence_id']}: no object tracks")
        _require(info["object_count"] <= effective_n_max, "n_max_capacity_exceeded",
                 f"{info['sequence_id']}: {info['object_count']} tracks exceed N_max={effective_n_max}; "
                 "tracks are never silently dropped")
    _require(sum(valid_counts.values()) > 0, "valid_labels_missing",
             "no reviewed, unambiguous, geometrically valid labels project onto requested targets")
    for split in SPLITS:
        _require(valid_counts[split] > 0, "valid_split_empty", f"{split}: no valid formal samples")
    stats = {
        "sequences": len(infos),
        "valid_counts": {"total": sum(valid_counts.values()), **{split: valid_counts[split] for split in SPLITS}},
        "invalid_counts": {"total": sum(invalid_counts.values()), "by_reason": _stats_json(invalid_counts),
                           "by_split": {split: {"total": sum(invalid_by_split[split].values()),
                                                "by_reason": _stats_json(invalid_by_split[split])}
                                        for split in SPLITS}},
        "target_out_of_range": _stats_json(target_out_counts),
        "event_statistics": _stats_json(raw_event_stats),
        "class_counts": {split: _stats_json(class_counts[split]) for split in SPLITS},
        "subject_counts": {split: _stats_json(subject_counts[split]) for split in SPLITS},
        "n_max": effective_n_max,
        "n_max_source": "explicit" if n_max else "train_max_object_tracks",
        "train_object_track_counts": {info["sequence_id"]: info["object_count"] for info in infos
                                      if info["split"] == "train"},
        "dropped_objects": 0,
    }
    common = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "annotation_schema": ANNOTATION_SCHEMA,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "generator": "src/dataset/build_formal_samples.py",
        "training_allowed": not include_invalid,
        "debug_only": include_invalid,
        "include_invalid": include_invalid,
        "config": {"window": window, "stride": stride, "horizon_seconds": horizon_seconds,
                   "n_max": effective_n_max, "n_max_argument": n_max},
        "provenance": {"freeze_sha256": context["freeze_sha256"], "config_sha256": context["config_sha256"],
                       "effective_config_sha256": context["effective_config_sha256"],
                       "source_ledger_sha256": context["source_ledger_sha256"],
                       "training_manifest_sha256": context["training_manifest_sha256"],
                       "code": context["code"],
                       "annotations": {info["sequence_id"]: _annotation_provenance(info) for info in infos}},
        "splits": {split: [{"sequence_id": info["sequence_id"], "group_id": info["entry"].get("group_id"),
                            "acquisition_group_id": info["entry"].get("acquisition_group_id"),
                            "subject_group_id": info["entry"].get("subject_group_id")}
                           for info in infos if info["split"] == split] for split in SPLITS},
        "stats": stats,
    }
    if audit_only:
        return {**common, "audit_only": True, "release_status": "audited_not_written",
                "skipped_invalid": skipped}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    index = {split: [] for split in SPLITS}
    written = Counter()
    for info in infos:
        record = info["record"]
        assets = _resolve_sequence_assets(record, context["asset_base"], annotations_dir)
        _require(assets["source_content_sha256"] == info["assets"]["source_content_sha256"],
                 "asset_changed_during_build", f"{info['sequence_id']}: input changed after preflight")
        data = _load_sequence(record, assets)
        _require(data["object_ids"] == info["object_ids"], "object_track_mismatch",
                 f"{info['sequence_id']}: object slots changed during build")
        feats, valid, _ = object_features(data["poses"], data["bboxes"], data["fps"])
        split_dir = destination / info["split"]
        split_dir.mkdir(exist_ok=True)
        for plan in info["plans"]:
            projection = plan["projection"]
            if not projection["label_valid"] and not include_invalid:
                continue
            sample = _make_output_sample(info, data, feats, valid, plan, effective_n_max, context, include_invalid)
            sample["sample_id"] = plan["sample_id"]
            sample["content_sha256"] = _sample_content_sha256({key: value for key, value in sample.items()
                                                               if key != "content_sha256"})
            path = split_dir / f"{plan['sample_id']}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **sample)
                stream.flush()
                os.fsync(stream.fileno())
            entry = _index_entry(sample, _sha256_file(path))
            entry["path"] = str(path.relative_to(destination))
            index[info["split"]].append(entry)
            written[info["split"]] += 1
    _require(all(written[split] > 0 for split in SPLITS), "valid_split_empty",
             "one or more splits produced no samples")
    index_path = destination / "dataset_index.json"
    index_content = _json_bytes(index)
    _write_new(index_path, index_content)
    sample_integrity = {entry["sample_id"]: {"path": entry["path"],
                                             "sample_sha256": entry["sample_sha256"],
                                             "content_sha256": entry["content_sha256"]}
                        for split in SPLITS for entry in index[split]}
    _require(len(sample_integrity) == sum(written.values()), "duplicate_sample_id",
             "sample IDs must be unique across the dataset")
    meta = {**common, "release_status": "debug_only" if include_invalid else "formal_samples_written",
            "dataset_index_sha256": _sha256_bytes(index_content),
            "sample_counts": {"total": sum(written.values()), **{split: written[split] for split in SPLITS}},
            "sample_integrity": sample_integrity,
            "skipped_invalid": skipped,
            "files": {"dataset_index.json": _sha256_bytes(index_content)}}
    _write_new(destination / "meta.json", _json_bytes(meta))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--horizon-seconds", type=float, default=None)
    parser.add_argument("--n-max", type=int, default=0, help="0 derives capacity from train sequences only")
    parser.add_argument("--include-invalid", action="store_true",
                        help="debug only; output is marked training_allowed=false")
    parser.add_argument("--audit-only", action="store_true", help="preflight without creating out-dir")
    args = parser.parse_args(argv)
    try:
        result = build_formal_samples(
            args.freeze_dir, args.annotations_dir, args.out_dir, window=args.window, stride=args.stride,
            horizon_seconds=args.horizon_seconds, n_max=args.n_max, include_invalid=args.include_invalid,
            audit_only=args.audit_only)
    except FreezeRejected as exc:
        print(json.dumps({"training_allowed": False, "error": str(exc), "freeze_audit": exc.report},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    except (FormalSampleError, OSError, ValueError) as exc:
        print(json.dumps({"training_allowed": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_json_bytes(result))
    return 1 if args.audit_only and not result.get("training_allowed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
