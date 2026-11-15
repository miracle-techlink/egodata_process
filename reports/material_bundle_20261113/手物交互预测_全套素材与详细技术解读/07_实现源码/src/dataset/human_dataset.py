"""Evidence-backed human_pose_dataset_v1 inventory and immutable split freeze.

Run ``python -S src/dataset/human_dataset.py audit LEDGER --output REPORT`` or
``python -S src/dataset/human_dataset.py freeze LEDGER --output NEW_DIRECTORY``.
Audit a frozen directory with the same audit command to recheck its integrity.
Exit codes: 0 = production training gate open, 1 = candidate only, 2 = I/O error.

Paths in assets are relative to the source ledger, never the current directory.
Each asset needs role/path/sha256; optional size_bytes is checked, actual size is
always recorded. Required roles are REQUIRED_ROLES. Shared meshes/calibrations
are not sequence identity; sequence payload hashes and source aliases are.

Formal admission needs additional evidence, not inferred labels. ``evidence``
maps EVIDENCE_KEYS to {asset_role, description}, referring to a verified local
asset (for example a provenance JSON or a signed-off review). timestamp_source
is source_timestamps/video_pts/official_fps. The first two require a timestamps
asset: JSON seconds array (or {timestamps: [...]}) or CSV timestamp_seconds.
Official FPS requires source metadata evidence and a frame_ids asset (JSON
integer array or {frame_ids: [...]} / CSV frame_id), not a default FPS.
duration_definition is frame_span ((N-1)/FPS) or frame_count_over_fps (N/FPS);
both use a one-frame tolerance. asset_verified_long is independent of GT and
observations. Hand-world references can enter verified_auxiliary but cannot
substitute for camera-metric GT; declared_auxiliary records intent separately.
``continuity`` explicitly contains continuous=true and concatenated,
repeated_frames, interpolated, fps_modified=false. ``coordinates`` contains
unit=m, hand_frame=camera, object_frame=camera, handedness=right, aligned=true.
camera_motion is fixed or compensated (the latter needs an actual
camera_motion_compensated asset). gt_independent=true, gt_usage set to
supervision_or_evaluation_only (or the stricter evaluation_only), and independent
reference evidence are required; GT and estimates must be in
separate directories. Quality reviews are external attestations: this module
verifies their files, NOT video decoding, semantic truth or model accuracy.

RGB/depth hashes always link acquisitions. Other sequence payloads only link
when identity_scope=per_sequence; shared label archives do not link. Calibrations
and meshes never link, even when marked per_sequence.
Optional source_aliases/aliases/acquisition_aliases, source_content_id,
source_sequence_id, source_episode_uuid, source_video, parent_sequence_id and
camera_sequence_id join source groups transitively. subject_aliases join people
across datasets. A camera_id alone does not join unrelated acquisitions.
Existing split=test, previous_split=test or previously_evaluated=true taints the
entire connected group against a NEW unseen test. Unknown subjects quarantine
the entire connected group. Split first; no windows or training are produced.

freeze_manifest writes source_ledger.json, config.json, candidate_split.json,
audit_report.json and freeze.json; training_manifest.json exists ONLY on a
production pass. verify_freeze rehashes every file and source asset, recomputes
the split and gate, and optionally checks a trusted freeze.json SHA-256.
Hashes detect changes, not a malicious replacement of the entire bundle unless
the caller retains that trusted digest separately. Fixture mode never opens
the production gate, even when its lowered thresholds pass.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys


SCHEMA_VERSION = "human_pose_dataset_v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/human_dataset_v1.json"
SPLITS = ("train", "val", "test")
MINIMUMS = {"acquisitions": 120, "subjects": 6, "objects": 8, "actions": 4,
            "train": 60, "val": 15, "test": 15}
RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
CAPABILITIES = ("rgb", "depth", "intrinsics", "mesh", "hand_metric_gt",
                "object_pose_gt", "hand_observations", "object_observations")
REQUIRED_ROLES = CAPABILITIES + ("hand_valid", "object_valid")
CONTENT_ROLES = ("rgb", "depth", "hand_metric_gt", "object_pose_gt",
                 "hand_observations", "object_observations", "hand_world_reference", "sequence_data", "raw_sequence")
QUALITY_KEYS = ("decode", "timestamps", "calibration", "coordinates",
                "annotation_alignment", "visual_review")
EVIDENCE_KEYS = ("subject_id", "scene_id", "action_labels", "human_verified",
                 "continuity", "timestamp_source", "coordinates", "gt_independence",
                 "camera", "decode", "calibration", "annotation_alignment", "visual_review")
UNKNOWN = {"", "unknown", "pending", "none", "null", "n/a", "na", "unverified", "?"}


class DatasetError(ValueError):
    """Invalid top-level input or an unverifiable frozen artifact."""


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _read_json(path):
    content = Path(path).read_bytes()
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise DatasetError(f"Invalid JSON in {path}: {exc}") from exc
    return value, content


def _known(value):
    return isinstance(value, str) and value.strip().lower() not in UNKNOWN


def _number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _strings(value):
    return [item for item in value if _known(item)] if isinstance(value, list) else []


def _issue(target, code, message, **details):
    target.append({"code": code, "message": message, **details})


def _load_config(config):
    if config is None:
        config = DEFAULT_CONFIG
    if isinstance(config, dict):
        supplied = copy.deepcopy(config)
        raw = _json_bytes(supplied)
        source = None
    else:
        source = str(Path(config).resolve())
        supplied, raw = _read_json(source)
    if not isinstance(supplied, dict):
        raise DatasetError("Configuration must be an object")
    effective = {"schema_version": SCHEMA_VERSION, "seed": 20261022,
                 "fixture_mode": False, "min_duration_seconds": 10.0,
                 "split_ratios": dict(RATIOS), "minimums": dict(MINIMUMS),
                 "required_roles": list(REQUIRED_ROLES), "required_capabilities": list(CAPABILITIES),
                 "required_quality": list(QUALITY_KEYS), "required_evidence": list(EVIDENCE_KEYS)}
    effective.update(supplied)
    errors = []
    if effective["schema_version"] != SCHEMA_VERSION:
        _issue(errors, "config_schema", "Unsupported configuration schema")
    for key, required in (("required_roles", REQUIRED_ROLES), ("required_capabilities", CAPABILITIES),
                           ("required_quality", QUALITY_KEYS), ("required_evidence", EVIDENCE_KEYS)):
        if effective[key] != list(required):
            _issue(errors, "config_contract_changed", f"{key} cannot weaken or redefine the v1 contract")
            effective[key] = list(required)
    if type(effective["seed"]) is not int:
        _issue(errors, "config_seed", "Seed must be an integer")
        effective["seed"] = 20261022
    if type(effective["fixture_mode"]) is not bool:
        _issue(errors, "config_fixture_mode", "fixture_mode must be a boolean")
        effective["fixture_mode"] = False
    if effective["split_ratios"] != RATIOS:
        _issue(errors, "config_ratios", "V1 requires subject-group targets 70/15/15")
        effective["split_ratios"] = dict(RATIOS)
    duration = effective["min_duration_seconds"]
    if not _number(duration) or duration < 10:
        _issue(errors, "config_duration", "Continuous duration may not be lowered below 10 seconds")
        effective["min_duration_seconds"] = 10.0
    minimums = effective["minimums"]
    if not isinstance(minimums, dict):
        _issue(errors, "config_minimums", "minimums must be an object")
        minimums = {}
    checked = {}
    for key, floor in MINIMUMS.items():
        value = minimums.get(key, floor)
        if type(value) is not int or value < 1:
            _issue(errors, "config_minimum", f"Invalid minimum for {key}")
            value = floor
        if value < floor and not effective["fixture_mode"]:
            _issue(errors, "production_minimum_lowered", f"{key} cannot be below {floor} in production")
        checked[key] = value
    effective["minimums"] = checked
    return effective, errors, {"path": source, "sha256": _digest(raw),
                               "effective_sha256": _digest(_json_bytes(effective))}


def _file_facts(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise DatasetError("Asset must be a regular file")
        head = stream.read(4096)
        digest.update(head)
        size = len(head)
        all_zero = not head.strip(b"\0")
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            all_zero = all_zero and not chunk.strip(b"\0")
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino) or size != after.st_size:
        raise DatasetError("Asset changed while being hashed")
    stripped = head.strip().lower()
    problem = None
    if not size or all_zero:
        problem = "empty_or_zero_payload"
    elif stripped.startswith(b"version https://git-lfs.github.com/spec/v1"):
        problem = "lfs_pointer"
    elif stripped.startswith((b"<!doctype html", b"<html", b"<error", b"<?xml")):
        problem = "download_error_payload"
    elif size <= 4096 and (stripped.startswith((b"http://", b"https://")) or
                          stripped in {b"todo", b"pending", b"placeholder", b"null", b"{}", b"[]"} or
                          stripped.startswith((b"placeholder", b"download pending", b"not downloaded"))):
        problem = "placeholder_payload"
    return {"sha256": digest.hexdigest(), "size_bytes": size, "payload_problem": problem}


def _audit_assets(assets, base):
    results, errors, warnings = [], [], []
    if not isinstance(assets, list):
        return results, [{"code": "assets_invalid", "message": "assets must be a list"}], warnings
    for position, asset in enumerate(assets):
        result = {"asset_index": position, "role": None, "verified": False, "downloaded": False}
        results.append(result)
        failures = []
        if not isinstance(asset, dict):
            _issue(errors, "asset_invalid", "Asset must be an object", asset_index=position)
            continue
        role = asset.get("role")
        result["role"] = role if _known(role) else None
        result["identity_scope"] = asset.get("identity_scope") if isinstance(asset.get("identity_scope"), str) else None
        if not _known(role):
            failures.append("asset_role_missing")
        name = asset.get("path")
        if not _known(name) or "://" in name:
            _issue(errors, "asset_path_invalid", "A local file path is required", asset_index=position)
            continue
        try:
            path = Path(name)
            path = (base / path).resolve() if not path.is_absolute() else path.resolve()
            result["path"] = str(path)
            if not path.is_file():
                raise DatasetError("File missing or not a regular file")
            facts = _file_facts(path)
            result.update(facts)
            result["downloaded"] = facts["payload_problem"] is None
            if facts["payload_problem"]:
                failures.append(facts["payload_problem"])
            if asset.get("placeholder") is True or asset.get("synthetic") is True:
                failures.append("declared_placeholder_or_synthetic")
            if not isinstance(asset.get("sha256"), str) or asset["sha256"].lower() != facts["sha256"]:
                failures.append("asset_sha256_mismatch")
            if "size_bytes" in asset and (type(asset["size_bytes"]) is not int or
                                           asset["size_bytes"] != facts["size_bytes"]):
                failures.append("asset_size_mismatch")
            if "size_bytes" not in asset:
                _issue(warnings, "asset_size_recorded", "Declared size absent; actual size recorded",
                       asset_index=position, size_bytes=facts["size_bytes"])
        except (OSError, ValueError, RuntimeError) as exc:
            _issue(errors, "asset_unreadable", str(exc), asset_index=position)
            continue
        for failure in failures:
            _issue(errors, failure, f"Asset {position} ({role}) failed integrity check", asset_index=position)
        result["verified"] = not failures
    return results, errors, warnings


def _evidence(record, key, verified):
    evidence = record.get("evidence")
    entry = evidence.get(key) if isinstance(evidence, dict) else None
    if key == "timestamp_source" and not entry and isinstance(record.get(key), dict):
        entry = record[key]
    return (isinstance(entry, dict) and _known(entry.get("description")) and
            isinstance(entry.get("asset_role"), str) and entry["asset_role"] in verified)


def _audit_time(record, verified, config, errors):
    frames, fps, duration = (record.get(key) for key in ("frame_count", "fps", "duration_seconds"))
    numeric = (type(frames) is int and 2 <= frames <= sys.maxsize and _number(fps) and fps > 0 and
               _number(duration) and duration > 0)
    if not numeric:
        _issue(errors, "time_values_invalid", "Need frame_count >=2, finite positive fps and duration")
        return False
    initial_errors = len(errors)
    tolerance = 1.0 / fps + 1e-6
    definition = record.get("duration_definition")
    if definition not in ("frame_span", "frame_count_over_fps"):
        _issue(errors, "duration_definition_missing", "Specify frame_span or frame_count_over_fps")
    expected_duration = (frames - 1) / fps if definition == "frame_span" else frames / fps
    if not _number(expected_duration):
        _issue(errors, "time_values_invalid", "Timebase arithmetic must be finite")
        return False
    if abs(expected_duration - duration) > tolerance:
        _issue(errors, "fps_duration_mismatch", "Frame count / FPS disagrees with duration")
    if duration < config["min_duration_seconds"] or expected_duration + 1e-9 < config["min_duration_seconds"]:
        _issue(errors, "sequence_too_short", "Continuous sequence must be at least 10 seconds")
    source = record.get("timestamp_source")
    source = source.get("kind") if isinstance(source, dict) else source
    if source not in ("source_timestamps", "video_pts", "official_fps"):
        _issue(errors, "timestamp_source_invalid", "Source timestamps, video PTS or documented official FPS required")
    if not _evidence(record, "timestamp_source", verified):
        _issue(errors, "timestamp_evidence_missing", "Timebase must reference a verified evidence file")
    if source == "official_fps":
        frame_tables = verified.get("frame_ids", [])
        if len(frame_tables) != 1:
            _issue(errors, "frame_ids_missing", "Official FPS needs a complete, real continuous frame ID table")
        else:
            try:
                path = Path(frame_tables[0]["path"])
                if path.suffix.lower() == ".json":
                    frame_ids, _ = _read_json(path)
                    frame_ids = frame_ids.get("frame_ids") if isinstance(frame_ids, dict) else frame_ids
                elif path.suffix.lower() == ".csv":
                    with path.open(encoding="utf-8", newline="") as stream:
                        frame_ids = [int(row["frame_id"]) for row in csv.DictReader(stream)]
                else:
                    raise DatasetError("Frame IDs need JSON or CSV frame_id")
                if not isinstance(frame_ids, list) or len(frame_ids) != frames or not all(type(value) is int for value in frame_ids):
                    raise DatasetError("Frame ID count/type disagrees with frame_count")
                if frame_ids[0] < 0 or any(right - left != 1 for left, right in zip(frame_ids, frame_ids[1:])):
                    raise DatasetError("Frame IDs must be continuous, nonnegative and unrepeated")
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                _issue(errors, "frame_ids_invalid", str(exc))
    if source in ("source_timestamps", "video_pts"):
        tables = verified.get("timestamps", [])
        if len(tables) != 1:
            _issue(errors, "timestamps_missing", "Exactly one complete timestamps table is required")
        else:
            try:
                path = Path(tables[0]["path"])
                if path.suffix.lower() == ".json":
                    values, _ = _read_json(path)
                    values = values.get("timestamps") if isinstance(values, dict) else values
                elif path.suffix.lower() == ".csv":
                    with path.open(encoding="utf-8", newline="") as stream:
                        values = [float(row["timestamp_seconds"]) for row in csv.DictReader(stream)]
                else:
                    raise DatasetError("Timestamps need JSON seconds or CSV timestamp_seconds")
                if not isinstance(values, list) or len(values) != frames or not all(_number(value) for value in values):
                    raise DatasetError("Timestamp count/type disagrees with frame_count")
                deltas = [right - left for left, right in zip(values, values[1:])]
                if min(deltas) <= 0 or max(deltas) > 2.0 / fps + 1e-6:
                    raise DatasetError("Timestamp repetition, reversal or discontinuity")
                span = values[-1] - values[0]
                if abs(span - duration) > tolerance or abs((frames - 1) / span / fps - 1) > 0.01:
                    raise DatasetError("Timestamp span / FPS disagrees with declared timebase")
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                _issue(errors, "timestamps_invalid", str(exc))
    continuity = record.get("continuity")
    expected = {"continuous": True, "concatenated": False, "repeated_frames": False,
                "interpolated": False, "fps_modified": False}
    if not isinstance(continuity, dict) or any(continuity.get(key) is not value for key, value in expected.items()):
        _issue(errors, "continuity_unverified", "Continuous, unstitched, unrepeated, uninterpolated source required")
    if not _evidence(record, "continuity", verified):
        _issue(errors, "continuity_evidence_missing", "Continuity must reference verified review evidence")
    return len(errors) == initial_errors


def _audit_record(raw, index, base, config):
    errors, warnings = [], []
    record = raw if isinstance(raw, dict) else {}
    if not isinstance(raw, dict):
        _issue(errors, "record_invalid", "Record must be an object")
    assets, asset_errors, asset_warnings = _audit_assets(record.get("assets"), base)
    errors.extend(asset_errors)
    warnings.extend(asset_warnings)
    verified = {}
    for asset in assets:
        if asset["verified"]:
            verified.setdefault(asset["role"], []).append(asset)
    for key in ("sequence_id", "dataset", "acquisition_id", "subject_id", "scene_id", "camera_id", "view"):
        if not _known(record.get(key)):
            _issue(errors, "identity_missing", f"Known {key} required", field=key)
    for key in ("object_ids", "action_labels"):
        values = record.get(key)
        if not isinstance(values, list) or not values or len(_strings(values)) != len(values):
            _issue(errors, "labels_missing", f"Nonempty known {key} required", field=key)
    for key in EVIDENCE_KEYS:
        if not _evidence(record, key, verified):
            _issue(errors, "evidence_missing", f"Verified file evidence required for {key}", field=key)
    human = record.get("human_verified") is True and _evidence(record, "human_verified", verified)
    if not human:
        _issue(errors, "human_unverified", "Real human involvement must be reviewed and evidenced")
    time_ok = _audit_time(record, verified, config, errors)
    capabilities = record.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    missing_roles = [role for role in REQUIRED_ROLES if not verified.get(role)]
    for role in missing_roles:
        _issue(errors, "required_role_missing", f"Verified {role} payload required", role=role)
    for key in CAPABILITIES:
        if capabilities.get(key) is not True:
            _issue(errors, "capability_missing", f"Capability {key} not confirmed", capability=key)
    quality = record.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    for key in QUALITY_KEYS:
        if quality.get(key) != "pass":
            _issue(errors, "quality_not_pass", f"Quality {key} must be pass", field=key)
    motion = record.get("camera_motion")
    if motion not in ("fixed", "compensated"):
        _issue(errors, "camera_motion_unverified", "Fixed camera or verified motion compensation required")
    elif motion == "compensated" and (capabilities.get("camera_motion_compensated") is not True or
                                      not verified.get("camera_motion_compensated")):
        _issue(errors, "camera_compensation_missing", "Compensation needs both capability and actual asset")
    coordinates = record.get("coordinates")
    coordinates = coordinates if isinstance(coordinates, dict) else {}
    expected = {"unit": "m", "hand_frame": "camera", "object_frame": "camera", "handedness": "right"}
    if any(coordinates.get(key) != value for key, value in expected.items()) or coordinates.get("aligned") is not True:
        _issue(errors, "coordinates_unverified", "Aligned right-handed metric camera GT coordinates required")
    if record.get("gt_independent") is not True or record.get("gt_usage") not in (
            "supervision_or_evaluation_only", "evaluation_only"):
        _issue(errors, "gt_independence_missing", "Independent GT is only for train/val supervision or final test evaluation")
    if any(record.get(key) is True for key in ("future_gt_used", "gt_used_for_initialization", "gt_used_for_perception")):
        _issue(errors, "gt_leakage", "Reference GT cannot enter perception or initialization")
    references = [asset for role in ("hand_metric_gt", "object_pose_gt") for asset in verified.get(role, [])]
    inputs = [asset for role in ("rgb", "depth", "intrinsics", "mesh", "hand_observations", "object_observations")
              for asset in verified.get(role, [])]
    for reference in references:
        for observation in inputs:
            if reference["sha256"] == observation["sha256"]:
                _issue(errors, "gt_payload_reused", "GT and perception/estimates cannot reuse the same payload")
            if observation["role"] in ("hand_observations", "object_observations"):
                gt_parent, obs_parent = Path(reference["path"]).parent, Path(observation["path"]).parent
                if gt_parent == obs_parent or gt_parent in obs_parent.parents or obs_parent in gt_parent.parents:
                    _issue(errors, "gt_directory_not_separate", "GT and estimates must occupy separate directory trees")
    prior = record.get("previously_evaluated")
    if prior is not None and type(prior) is not bool:
        _issue(errors, "evaluation_history_invalid", "previously_evaluated must be a boolean")
    old_test = prior is True or any(record.get(key) == "test" for key in
                                    ("split", "previous_split", "original_split", "historical_split"))
    if old_test:
        _issue(warnings, "previously_seen", "Entire linked group is excluded from a new unseen test")
    split = record.get("split")
    if split is not None and split not in (*SPLITS, "candidate", "pending", "unassigned", ""):
        _issue(errors, "split_invalid", "split must be train/val/test/candidate/pending or null")
    long_assets = bool(time_ok and verified.get("rgb") and quality.get("decode") == "pass" and
                       quality.get("timestamps") == "pass")
    auxiliary = bool(long_assets and human and (verified.get("hand_metric_gt") or verified.get("hand_world_reference"))
                     and not verified.get("object_pose_gt"))
    return {"record_index": index, "sequence_id": record.get("sequence_id") if _known(record.get("sequence_id")) else None,
            "assets": assets, "errors": errors, "warnings": warnings, "missing_roles": missing_roles,
            "downloaded": any(asset["downloaded"] and asset["role"] in ("rgb", "depth", "raw_sequence", "sequence_data")
                              for asset in assets), "human_confirmed": human, "asset_verified_long": long_assets,
            "qualified_long_sequence": bool(human and long_assets), "auxiliary": auxiliary,
            "declared_auxiliary": record.get("pool") == "auxiliary", "verified_auxiliary": auxiliary,
            "pool": "joint" if not errors else "auxiliary" if auxiliary else "candidate",
            "joint_pose_ready": not errors, "previously_seen": old_test,
            "subject_known": _known(record.get("subject_id")) and _evidence(record, "subject_id", verified)}


class _Groups:
    def __init__(self, size):
        self.parents = list(range(size))

    def root(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, left, right):
        self.parents[self.root(right)] = self.root(left)


def _tokens(record, result, include_subject):
    tokens = set()
    for key in ("acquisition_id", "sequence_id", "source_content_id", "source_sequence_id", "source_episode_uuid",
                "source_video", "parent_sequence_id", "camera_sequence_id"):
        if _known(record.get(key)):
            tokens.add("source:" + record[key])
    for key in ("aliases", "source_aliases", "acquisition_aliases"):
        tokens.update("source:" + alias for alias in _strings(record.get(key)))
    for asset in result["assets"]:
        if asset["role"] in ("rgb", "depth") or (asset["role"] in CONTENT_ROLES and
                                                    asset.get("identity_scope") == "per_sequence"):
            if asset.get("path"):
                tokens.add("path:" + asset["path"])
            if asset.get("downloaded") and asset.get("sha256"):
                tokens.add("sha256:" + asset["sha256"])
    if include_subject and _known(record.get("subject_id")):
        dataset = record.get("dataset") if _known(record.get("dataset")) else "unknown_dataset"
        tokens.add("subject:" + dataset + ":" + record["subject_id"])
        tokens.update("subject_alias:" + alias for alias in _strings(record.get("subject_aliases")))
    return tokens


def _components(records, results, include_subject):
    groups = _Groups(len(records))
    owners, token_sets = {}, []
    for index, (record, result) in enumerate(zip(records, results)):
        tokens = _tokens(record, result, include_subject)
        token_sets.append(tokens)
        for token in sorted(tokens):
            if token in owners:
                groups.join(owners[token], index)
            else:
                owners[token] = index
    components = {}
    for index in range(len(records)):
        components.setdefault(groups.root(index), []).append(index)
    output = {}
    for members in components.values():
        identities = sorted(token for token in set().union(*(token_sets[index] for index in members))
                            if not token.startswith("path:"))
        if not identities:
            identities = [f"invalid_record:{members[0]}"]
        output[_digest(_json_bytes(identities))] = members
    return output


def _partition(records, results, config):
    source_groups = _components(records, results, False)
    subject_groups = _components(records, results, True)
    acquisition_ids = {index: group for group, members in source_groups.items() for index in members}
    people = _Groups(len(records))
    person_owners = {}
    for index, record in enumerate(records):
        if results[index]["subject_known"]:
            person_tokens = [f"{record.get('dataset')}:{record['subject_id']}"]
            person_tokens.extend("alias:" + alias for alias in _strings(record.get("subject_aliases")))
            for token in person_tokens:
                if token in person_owners:
                    people.join(person_owners[token], index)
                else:
                    person_owners[token] = index
    person_names = {}
    for index, record in enumerate(records):
        if results[index]["subject_known"]:
            person_names.setdefault(people.root(index), []).append(f"{record.get('dataset')}:{record['subject_id']}")
    person_ids = {index: min(person_names[people.root(index)]) if results[index]["subject_known"] else None
                  for index in range(len(records))}
    errors, warnings, groups = [], [], []
    for group_id, members in subject_groups.items():
        declared = {records[index].get("split") for index in members
                    if isinstance(records[index].get("split"), str) and records[index]["split"] in SPLITS}
        if len(declared) > 1:
            _issue(errors, "declared_split_leakage", "A subject/acquisition/content group crosses declared splits",
                   group_id=group_id, splits=sorted(declared))
        unknown = any(not results[index]["subject_known"] for index in members)
        if unknown:
            _issue(warnings, "unknown_subject_quarantined", "Connected group kept candidate-only; no cross-subject claim",
                   group_id=group_id)
        groups.append({"group_id": group_id, "record_indices": members, "candidate_only": unknown,
                       "previously_seen": any(results[index]["previously_seen"] for index in members),
                       "subject_keys": sorted({person_ids[index] for index in members if person_ids[index]}),
                       "weight": len({acquisition_ids[index] for index in members})})
    eligible_weight = sum(group["weight"] for group in groups if not group["candidate_only"])
    counts = dict.fromkeys(SPLITS, 0)
    ordered = sorted(groups, key=lambda group: (not group["previously_seen"], -group["weight"],
                     _digest(_json_bytes([config['seed'], group['subject_keys']]))))
    entries = []
    for group in ordered:
        split = "candidate"
        if not group["candidate_only"]:
            allowed = ("train", "val") if group["previously_seen"] else SPLITS
            split = max(allowed, key=lambda name: (eligible_weight * RATIOS[name] - counts[name], -SPLITS.index(name)))
            counts[split] += group["weight"]
        group["split"] = split
        for index in group["record_indices"]:
            entries.append({"record_index": index, "sequence_id": results[index]["sequence_id"], "split": split,
                            "group_id": group["group_id"], "acquisition_group_id": acquisition_ids[index],
                            "subject_group_id": person_ids[index],
                            "joint_pose_ready": results[index]["joint_pose_ready"] and split != "candidate"})
    return {"schema_version": SCHEMA_VERSION, "status": "candidate_split", "seed": config["seed"],
            "strategy": "subject_components_raw_acquisition_weights_v2",
            "weight_basis": "all_unique_source_acquisitions_before_quality_and_annotation_filtering",
            "target_ratios": dict(RATIOS), "split_before_windows": True,
            "normalization_fit_split": "train", "test_use": "final_only_after_training_protocol_frozen",
            "records": sorted(entries, key=lambda entry: entry["record_index"]),
            "groups": sorted(groups, key=lambda group: group["group_id"])}, errors, warnings


def _summary(indices, records, entries):
    acquisitions, subjects, scenes, objects, actions = set(), set(), set(), set(), set()
    object_groups, action_groups = {}, {}
    for index in indices:
        record, entry = records[index], entries[index]
        group = entry["acquisition_group_id"]
        acquisitions.add(group)
        if entry.get("subject_group_id"):
            subjects.add(entry["subject_group_id"])
        if _known(record.get("scene_id")):
            scenes.add(record["scene_id"])
        for key, values, distribution in (("object_ids", objects, object_groups),
                                          ("action_labels", actions, action_groups)):
            for value in _strings(record.get(key)):
                values.add(value)
                distribution.setdefault(value, set()).add(group)
    return {"records": len(indices), "independent_acquisitions": len(acquisitions), "subjects": len(subjects),
            "scenes": len(scenes), "objects": len(objects), "actions": len(actions),
            "subject_ids": sorted(subjects), "object_ids": sorted(objects), "action_labels": sorted(actions),
            "object_acquisition_distribution": {key: len(value) for key, value in sorted(object_groups.items())},
            "action_acquisition_distribution": {key: len(value) for key, value in sorted(action_groups.items())}}


def _audit(ledger, base, config, config_errors, frozen=False):
    if not isinstance(ledger, dict) or not isinstance(ledger.get("records"), list):
        raise DatasetError("Ledger must be an object with a records list")
    errors, warnings = list(config_errors), []
    if ledger.get("schema_version") != SCHEMA_VERSION:
        _issue(errors, "ledger_schema", "Unsupported ledger schema_version")
    originals = ledger["records"]
    records = [record if isinstance(record, dict) else {} for record in originals]
    results = [_audit_record(record, index, base, config) for index, record in enumerate(originals)]
    sequence_ids = set()
    for result in results:
        sequence_id = result["sequence_id"]
        if sequence_id and sequence_id in sequence_ids:
            _issue(errors, "duplicate_sequence_id", f"Duplicate sequence_id: {sequence_id}")
        sequence_ids.add(sequence_id)
    plan, split_errors, split_warnings = _partition(records, results, config)
    errors.extend(split_errors)
    warnings.extend(split_warnings)
    entries = plan["records"]
    layers = {"candidate": list(range(len(records)))}
    for layer, field in (("downloaded", "downloaded"), ("human_confirmed", "human_confirmed"),
                         ("asset_verified_long", "asset_verified_long"), ("auxiliary", "auxiliary"),
                         ("declared_auxiliary", "declared_auxiliary"), ("verified_auxiliary", "verified_auxiliary"),
                         ("qualified_long_sequences", "qualified_long_sequence"), ("joint_pose_ready", "joint_pose_ready")):
        layers[layer] = [index for index, result in enumerate(results) if result[field]]
    layers["release_eligible"] = [entry["record_index"] for entry in entries if entry["joint_pose_ready"]]
    statistics = {name: _summary(indices, records, entries) for name, indices in layers.items()}
    splits, inventory_splits, declared_splits = {}, {}, {}
    for split in (*SPLITS, "candidate"):
        indices = [entry["record_index"] for entry in entries if entry["split"] == split]
        inventory_splits[split] = _summary(indices, records, entries)
        splits[split] = _summary([index for index in indices if entries[index]["joint_pose_ready"]], records, entries)
        declared_splits[split] = _summary([index for index, record in enumerate(records) if record.get("split") == split],
                                         records, entries)
    eligible = statistics["release_eligible"]
    values = {"acquisitions": eligible["independent_acquisitions"], **{key: eligible[key] for key in
               ("subjects", "objects", "actions")}, **{split: splits[split]["independent_acquisitions"] for split in SPLITS}}
    gates = {key: {"actual": values[key], "minimum": minimum, "pass": values[key] >= minimum}
             for key, minimum in config["minimums"].items()}
    for key, gate in gates.items():
        if not gate["pass"]:
            _issue(errors, "minimum_not_met", f"{key}: {gate['actual']} < {gate['minimum']}", field=key)
    excluded = len(records) - len(layers["release_eligible"])
    if excluded:
        _issue(warnings, "candidate_records_excluded", f"{excluded} records are inventory-only, not training inputs")
    if config["fixture_mode"]:
        _issue(warnings, "fixture_only", "Fixture thresholds never authorize real training or an accepted manifest")
    if not frozen:
        _issue(warnings, "split_not_frozen", "Raw-ledger audit cannot authorize training; freeze and verify first")
    total = eligible["independent_acquisitions"]
    actual_ratios = {split: splits[split]["independent_acquisitions"] / total if total else 0.0 for split in SPLITS}
    unseen = {split: {"objects": sorted(set(splits[split]["object_ids"]) - set(splits["train"]["object_ids"])),
                       "actions": sorted(set(splits[split]["action_labels"]) - set(splits["train"]["action_labels"]))}
              for split in ("val", "test")}
    missing_roles = {role: sum(role in result["missing_roles"] for result in results) for role in REQUIRED_ROLES}
    ready = not errors
    return {"schema_version": SCHEMA_VERSION, "status": "accepted" if ready and frozen and not config["fixture_mode"] else "candidate_only",
            "training_allowed": ready and frozen and not config["fixture_mode"], "readiness_passed": ready,
            "fixture_mode": config["fixture_mode"], "errors": errors, "warnings": warnings,
            "record_error_count": sum(len(result["errors"]) for result in results),
            "record_warning_count": sum(len(result["warnings"]) for result in results),
            "statistics": statistics, "splits": splits, "inventory_splits": inventory_splits,
            "declared_split_distribution": declared_splits, "actual_ratios": actual_ratios,
            "target_ratios": dict(RATIOS), "unseen_categories": unseen, "missing_roles": missing_roles,
            "gates": gates, "records": results, "candidate_split": plan,
            "threshold_source": "project_release_gate_not_official_benchmark",
            "accuracy": {"hand": None, "object_translation": None, "object_rotation": None, "joint": None},
            "validation_scope": "Ledger evidence and file integrity only; external decode/geometry/review attestations required",
            "cross_subject_claim": "Only release_eligible known, evidenced subjects; unknown groups quarantined"}


def audit_manifest(manifest_path, config=None):
    """Inventory all records without admitting a partial/unknown record to training."""
    path = Path(manifest_path).resolve()
    if path.is_dir():
        if config is not None:
            raise DatasetError("A frozen bundle uses its frozen config; config overrides are forbidden")
        return verify_freeze(path)
    ledger, raw = _read_json(path)
    effective, errors, provenance = _load_config(config)
    report = _audit(ledger, path.parent, effective, errors)
    report["source_ledger"] = {"path": str(path), "sha256": _digest(raw)}
    report["config"] = provenance
    return report


def _training_manifest(report):
    return {"schema_version": SCHEMA_VERSION, "status": "accepted", "training_allowed": True,
            "split_before_windows": True, "normalization_fit_split": "train",
            "source_ledger_sha256": report["source_ledger"]["sha256"],
            "config_sha256": report["config"]["effective_sha256"],
            "splits": {split: [entry for entry in report["candidate_split"]["records"]
                                if entry["split"] == split and entry["joint_pose_ready"]] for split in SPLITS}}


def _write_new(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def freeze_manifest(manifest_path, output_dir, config=None):
    """Create a never-overwritten bundle; small/invalid datasets remain candidate-only."""
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Freeze destination already exists: {destination}")
    source = Path(manifest_path).resolve()
    ledger, raw = _read_json(source)
    effective, errors, provenance = _load_config(config)
    report = _audit(ledger, source.parent, effective, errors, frozen=True)
    report["source_ledger"] = {"path": str(source), "sha256": _digest(raw)}
    report["config"] = provenance
    if source.read_bytes() != raw:
        raise DatasetError("Source ledger changed during audit")
    files = {"source_ledger.json": raw, "config.json": _json_bytes(effective),
             "candidate_split.json": _json_bytes(report["candidate_split"]), "audit_report.json": _json_bytes(report)}
    if report["training_allowed"]:
        files["training_manifest.json"] = _json_bytes(_training_manifest(report))
    seal = {"schema_version": SCHEMA_VERSION, "source_ledger": report["source_ledger"], "config": provenance,
            "asset_base": str(source.parent), "files": {name: _digest(content) for name, content in files.items()},
            "training_allowed": report["training_allowed"], "split_before_windows": True,
            "source_config_errors": errors}
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, content in files.items():
            _write_new(destination / name, content)
        _write_new(destination / "freeze.json", _json_bytes(seal))
        verified = verify_freeze(destination, expected_sha256=_digest(_json_bytes(seal)))
        if verified["integrity_errors"]:
            raise DatasetError("Freeze post-write verification failed; bundle is not accepted")
    except (OSError, ValueError):
        training_path = destination / "training_manifest.json"
        if training_path.exists():
            training_path.unlink()
        raise
    return verified


def verify_freeze(output_dir, expected_sha256=None):
    """Rehash bundle, original ledger/config and every asset; recompute the gate."""
    root = Path(output_dir).resolve()
    integrity_errors = []
    try:
        seal, raw_seal = _read_json(root / "freeze.json")
        if not isinstance(seal, dict) or seal.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError("Invalid freeze metadata")
        if expected_sha256 is not None and _digest(raw_seal) != expected_sha256:
            raise DatasetError("Trusted freeze.json SHA-256 mismatch")
        expected_files = {"source_ledger.json", "config.json", "candidate_split.json", "audit_report.json"}
        if seal.get("training_allowed") is True:
            expected_files.add("training_manifest.json")
        if not isinstance(seal.get("files"), dict) or set(seal["files"]) != expected_files:
            raise DatasetError("Frozen file inventory mismatch")
        if {entry.name for entry in root.iterdir()} != expected_files | {"freeze.json"}:
            raise DatasetError("Unexpected or missing files in immutable bundle")
        for name in sorted(expected_files):
            path = root / name
            if path.is_symlink() or _digest(path.read_bytes()) != seal["files"][name]:
                raise DatasetError(f"Frozen file SHA-256 mismatch: {name}")
        ledger, raw_ledger = _read_json(root / "source_ledger.json")
        if _digest(raw_ledger) != seal["source_ledger"]["sha256"]:
            raise DatasetError("Source ledger snapshot hash mismatch")
        original = Path(seal["source_ledger"]["path"])
        if _digest(original.read_bytes()) != seal["source_ledger"]["sha256"]:
            raise DatasetError("Original source ledger changed or unavailable")
        effective, errors, _ = _load_config(root / "config.json")
        if _digest(_json_bytes(effective)) != seal["config"]["effective_sha256"]:
            raise DatasetError("Effective config hash mismatch")
        original_config = seal["config"].get("path")
        if original_config and _digest(Path(original_config).read_bytes()) != seal["config"]["sha256"]:
            raise DatasetError("Original configuration changed or unavailable")
        for error in seal.get("source_config_errors", []):
            if error not in errors:
                errors.append(error)
        report = _audit(ledger, Path(seal["asset_base"]), effective, errors, frozen=True)
        report["source_ledger"] = seal["source_ledger"]
        report["config"] = seal["config"]
        stored_report, _ = _read_json(root / "audit_report.json")
        if report != stored_report:
            _issue(integrity_errors, "audit_changed", "Recomputed file evidence, grouping or gate differs from frozen audit")
        plan, _ = _read_json(root / "candidate_split.json")
        if plan != report["candidate_split"]:
            _issue(integrity_errors, "split_changed", "Frozen assignments differ from deterministic recomputation")
        if seal.get("training_allowed") != report["training_allowed"]:
            _issue(integrity_errors, "gate_changed", "Frozen training gate differs from current evidence")
        if "training_manifest.json" in expected_files:
            training, _ = _read_json(root / "training_manifest.json")
            if training != _training_manifest(report) or not report["training_allowed"]:
                _issue(integrity_errors, "training_manifest_changed", "Accepted training list is not reproducible")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        report = {"schema_version": SCHEMA_VERSION, "training_allowed": False, "status": "candidate_only", "errors": []}
        _issue(integrity_errors, "freeze_integrity", str(exc))
        raw_seal = b""
    report["integrity_errors"] = integrity_errors
    report["freeze_sha256"] = _digest(raw_seal) if raw_seal else None
    report["integrity_verified"] = not integrity_errors
    if integrity_errors:
        report["training_allowed"] = False
        report["status"] = "candidate_only"
        report["errors"].extend(integrity_errors)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "freeze"):
        subparser = commands.add_parser(command)
        subparser.add_argument("manifest", type=Path, help="Source ledger (or frozen directory for audit)")
        subparser.add_argument("--config", type=Path)
        subparser.add_argument("--output", type=Path, required=command == "freeze")
        if command == "audit":
            subparser.add_argument("--expected-freeze-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            report = freeze_manifest(args.manifest, args.output, args.config)
        elif args.expected_freeze_sha256:
            if not args.manifest.is_dir() or args.config is not None:
                raise DatasetError("Trusted freeze hash requires a frozen directory and no config override")
            report = verify_freeze(args.manifest, args.expected_freeze_sha256)
        else:
            report = audit_manifest(args.manifest, args.config)
        if args.command == "audit" and args.output:
            _write_new(args.output, _json_bytes(report))
        else:
            sys.stdout.buffer.write(_json_bytes(report))
        return 0 if report["training_allowed"] else 1
    except (OSError, ValueError) as exc:
        print(json.dumps({"training_allowed": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
