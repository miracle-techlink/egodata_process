"""Evidence-aligned, positive-biased diagnostic windows over frozen v2 arrays.

These indexes select targets using reviewed labels. They are not new acquisitions,
negative examples, or a blind evaluation set. Existing indexes are never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "assistant_annotation_v3_review_windows"
POLICY = "silver_evidence_frame_positive_biased_diagnostic"
CONFIG = {"window_frames": 32, "horizon_seconds": 1.0,
          "max_horizon_overshoot_median_dt": 1.5, "floating_comparison_atol": 1e-9}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_timestamps(values):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)):
        raise ValueError("Native timestamps must be finite and strictly increasing")
    return values


def target_window(timestamps, target, *, window_frames=32, horizon_seconds=1.0,
                  max_overshoot_median_dt=1.5):
    """Find the last real anchor at least the requested time before a target."""
    times = validate_timestamps(timestamps)
    if window_frames < 1 or horizon_seconds <= 0 or max_overshoot_median_dt < 0:
        raise ValueError("Invalid window configuration")
    if type(target) is not int:
        return None, "target_not_integer"
    if not 0 <= target < len(times):
        return None, "target_out_of_range"
    anchor = int(np.searchsorted(times, times[target] - horizon_seconds, side="right")) - 1
    # Compare actual differences: subtracting the desired horizon can round in
    # the opposite direction to computing a timestamp difference.
    while anchor >= 0 and times[target] - times[anchor] < horizon_seconds:
        anchor -= 1
    while anchor + 1 < target and times[target] - times[anchor + 1] >= horizon_seconds:
        anchor += 1
    if anchor < 0:
        return None, "no_anchor_before_horizon"
    if anchor < window_frames - 1:
        return None, "insufficient_observation_history"
    if target <= anchor:
        return None, "target_not_after_anchor"
    actual = float(times[target] - times[anchor])
    tolerance = max_overshoot_median_dt * float(np.median(np.diff(times)))
    if actual - horizon_seconds > tolerance + CONFIG["floating_comparison_atol"]:
        return None, "horizon_overshoot"
    return {"start_frame": anchor - window_frames + 1, "end_frame": anchor + 1,
            "anchor_frame": anchor, "target_frame": target,
            "target_timestamp": float(times[target]), "actual_horizon_s": actual}, None


def evidence_targets(records, frame_count):
    """Collect only explicitly reviewed state/action points, never interval fill."""
    targets = defaultdict(set)
    excluded_records, excluded_frames = Counter(), Counter()
    for index, record in enumerate(records):
        if record.get("layer") not in {"operation", "relation", "cooperation"}:
            excluded_records["not_state_or_operation"] += 1
            continue
        if (record.get("silver_label_valid") is not True or record.get("status") != "supported"
                or record.get("annotation_source") != "assistant_visual_review"
                or record.get("review_status") != "assistant_reviewed"
                or record.get("human_label_valid", False)):
            excluded_records["not_supported_assistant_silver"] += 1
            continue
        if record.get("temporal_coverage") != "sampled_points":
            excluded_records["not_sampled_points"] += 1
            continue
        frames = record.get("evidence_frames")
        if not isinstance(frames, list) or not frames:
            excluded_records["missing_point_evidence"] += 1
            continue
        for target in frames:
            if type(target) is not int:
                excluded_frames["target_not_integer"] += 1
            elif not 0 <= target < frame_count:
                excluded_frames["target_out_of_range"] += 1
            else:
                targets[target].add(index)
    return {target: sorted(indexes) for target, indexes in sorted(targets.items())}, excluded_records, excluded_frames


def export_review_windows(base_root, annotation_root):
    base_root, root = Path(base_root).resolve(), Path(annotation_root).resolve()
    base_path, overlay_path = base_root / "dataset_index.json", root / "mamba_overlay_index.json"
    base_sha, overlay_sha = digest(base_path), digest(overlay_path)
    base, overlay = read(base_path), read(overlay_path)
    if base["schema_version"] != "egodemo_mamba_v2" or overlay["schema_version"] != "assistant_annotation_v3_overlay":
        raise ValueError("Unexpected base/overlay schema")
    if overlay["base_dataset_index_sha256"] != base_sha:
        raise ValueError("Overlay does not bind the frozen base index")
    split_path = base_root / base["split_manifest_path"]
    split_sha = digest(split_path)
    if base["split_manifest_sha256"] != split_sha or overlay["base_split_manifest_sha256"] != split_sha:
        raise ValueError("Frozen split SHA mismatch")
    split = read(split_path)
    if split["source_selection_sha256"] != base["source_selection_sha256"]:
        raise ValueError("Source selection mismatch")
    assignments = split["assignments"]
    entries = {item["sequence_id"]: item for item in base["sequences"]}
    annotations = {item["sequence_id"]: item for item in overlay["sequences"]}
    if len(entries) != len(base["sequences"]) or len(annotations) != len(overlay["sequences"]):
        raise ValueError("Duplicate sequence ID")
    if set(entries) != set(annotations) or set(entries) != set(assignments):
        raise ValueError("Base, overlay, and frozen split must contain the same acquisitions")
    old_windows = {}
    for name, info in base["splits"].items():
        path = base_root / info["window_path"]
        if digest(path) != info["sha256"]:
            raise ValueError("Frozen base windows changed")
        old_windows[name] = {"window_path": info["window_path"], "sha256": info["sha256"]}

    rows = {name: [] for name in ("train", "val", "test")}
    sequence_summaries, totals = [], Counter()
    rejected_targets, rejected_records, rejected_frames = Counter(), Counter(), Counter()
    for uid, entry in sorted(entries.items()):
        annotation_entry = annotations[uid]
        name = assignments[uid]
        if name not in rows or entry["split"] != name or annotation_entry["split"] != name:
            raise ValueError("Sequence split differs from frozen assignment")
        annotation_path = root / annotation_entry["annotation_path"]
        if digest(annotation_path) != annotation_entry["sha256"]:
            raise ValueError("Annotation changed after overlay freeze")
        annotation = read(annotation_path)
        if annotation["sequence_id"] != uid or annotation["split"] != name:
            raise ValueError("Annotation acquisition/split mismatch")
        for kind in ("sequence", "metadata", "events"):
            if digest(base_root / entry[kind + "_path"]) != entry["sha256"][kind]:
                raise ValueError("Frozen base sequence component SHA mismatch")
        bindings = annotation["source_bindings"]
        if (bindings["base_sequence_sha256"] != entry["sha256"]["sequence"]
                or bindings["base_metadata_sha256"] != entry["sha256"]["metadata"]):
            raise ValueError("Annotation numeric source binding mismatch")
        # No geometry, object IDs, target observations, or future slot selection
        # is consulted. The unchanged Dataset selects its objects from history.
        with np.load(base_root / entry["sequence_path"], allow_pickle=False) as arrays:
            times = validate_timestamps(arrays["timestamps"])
            native_frames = arrays["frame_index"]
        if len(times) != entry["frame_count"] or not np.array_equal(native_frames, np.arange(len(times))):
            raise ValueError("Native frame index mismatch")
        targets, record_exclusions, frame_exclusions = evidence_targets(annotation["silver_event_records"], len(times))
        sequence_exclusions = Counter()
        count = 0
        for target, record_indexes in targets.items():
            window, reason = target_window(times, target)
            if window is None:
                sequence_exclusions[reason] += 1
                continue
            rows[name].append({**window,
                **{key: entry[key] for key in ("sequence_id", "sequence_path", "metadata_path", "events_path", "split")},
                "window_id": f"{uid}:{window['start_frame']}:{target}",
                "source_selection_sha256": base["source_selection_sha256"], "split_manifest_sha256": split_sha,
                "sequence_sha256": entry["sha256"]["sequence"],
                "annotation_path": annotation_entry["annotation_path"], "annotation_sha256": annotation_entry["sha256"],
                "evidence_record_indices": record_indexes, "selection_policy": POLICY})
            count += 1
        sequence_summaries.append({"sequence_id": uid, "split": name,
            "annotation_path": annotation_entry["annotation_path"], "annotation_sha256": annotation_entry["sha256"],
            "sequence_sha256": entry["sha256"]["sequence"], "metadata_sha256": entry["sha256"]["metadata"],
            "candidate_target_count": len(targets), "window_count": count,
            "target_exclusion_counts": dict(sequence_exclusions),
            "record_exclusion_counts": dict(record_exclusions), "frame_exclusion_counts": dict(frame_exclusions)})
        totals.update(candidate_target_count=len(targets), window_count=count,
                      sequence_count=1, sequences_with_windows=int(count > 0))
        rejected_targets.update(sequence_exclusions)
        rejected_records.update(record_exclusions)
        rejected_frames.update(frame_exclusions)

    # Inputs must still be frozen when publishing the new, independent index.
    if digest(base_path) != base_sha or digest(overlay_path) != overlay_sha or digest(split_path) != split_sha:
        raise ValueError("Input index changed during export")
    for info in old_windows.values():
        if digest(base_root / info["window_path"]) != info["sha256"]:
            raise ValueError("Frozen base windows changed during export")
    output = root / "review_windows"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "index.json").exists():
        raise FileExistsError("Review windows already published; preserve the frozen version")
    splits = {}
    for name, values in rows.items():
        path = output / f"{name}.jsonl"
        temporary = path.with_suffix(".jsonl.partial")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
        splits[name] = {"window_path": str(path.relative_to(root)), "sha256": digest(path),
                        "window_count": len(values), "sequence_count": len({row["sequence_id"] for row in values})}
    index = {"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "base_dataset_root": str(base_root), "annotation_root": str(root),
        "base_dataset_index_sha256": base_sha, "overlay_index_sha256": overlay_sha,
        "split_manifest_sha256": split_sha, "source_selection_sha256": base["source_selection_sha256"],
        "window_config": CONFIG, "selection_policy": POLICY, "positive_biased_diagnostic": True,
        "blind_evaluation": False, "new_independent_acquisitions": 0, "negative_labels_created": 0,
        "default_loader_windows_replaced": False, "base_windows_modified": False,
        "row_numeric_paths_relative_to": "base_dataset_root", "window_paths_relative_to": "annotation_root",
        "annotation_paths_relative_to": "annotation_root", "splits": splits, "sequences": sequence_summaries,
        "totals": dict(totals), "target_exclusion_counts": dict(rejected_targets),
        "record_exclusion_counts": dict(rejected_records), "frame_exclusion_counts": dict(rejected_frames),
        "base_window_bindings": old_windows, "exporter_sha256": digest(__file__),
        "training_allowed": False, "allowed_as_model_input": False, "diagnostic_loading_allowed": True,
        "human_label_valid": False, "full_annotation_complete": False,
        "scope_zh": "按已复核正例证据帧选择的诊断窗口；非新增采集、负例或盲测。物体槽仍只由观测历史选择。"}
    temporary = output / "index.json.partial"
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "index.json")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    args = parser.parse_args()
    index = export_review_windows(args.base_root, args.annotation_root)
    print(json.dumps({"totals": index["totals"], "target_exclusion_counts": index["target_exclusion_counts"],
                      "index_path": str(args.annotation_root / "review_windows/index.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
