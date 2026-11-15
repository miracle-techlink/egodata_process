"""Freeze evidence-scoped continuous-review windows over unchanged v2 arrays."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.dataset.export_annotation_v3_review_windows import digest, read, target_window, CONFIG


SCHEMA = "assistant_annotation_v4_dense_mamba"


def native_range(record, frame_count):
    start, end = record["start_frame"], record["end_frame"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= frame_count:
        raise ValueError("Invalid native half-open range")
    return set(range(start, end))


def reviewed_record(record):
    return (record.get("annotation_source") == "assistant_visual_review"
            and record.get("review_status") == "assistant_reviewed"
            and record.get("silver_label_valid") is True
            and record.get("human_label_valid") is False
            and record.get("sampling_scope") == "continuous_native_frames_reviewed")


def validate_review(review, frame_count):
    """Fail on fabricated dense coverage or contradictory explicit supervision."""
    if review["frame_count"] != frame_count or review.get("human_label_valid") is not False:
        raise ValueError("Review native frame count or label origin mismatch")
    seen = set().union(*(native_range(r, frame_count) for r in review["reviewed_ranges"]))
    targets, cells = {}, {}
    for index, record in enumerate(review["state_segments"]):
        if not reviewed_record(record):
            raise ValueError("State must be explicit continuous assistant review")
        if type(record.get("value")) is not int or record["value"] not in (0, 1):
            raise ValueError("Unknown state cannot become a negative value")
        frames = native_range(record, frame_count)
        evidence = record.get("evidence_frames", [])
        if any(type(f) is not int for f in evidence) or not frames <= set(evidence) <= seen:
            raise ValueError("State includes a frame without actual reviewed evidence")
        if not frames <= seen:
            raise ValueError("State exceeds actual reviewed range")
        for frame in frames:
            key = (frame, record["layer"], record["type"], record.get("subject"),
                   record.get("object_id"), record.get("secondary_object_id"))
            if key in cells and cells[key] != record["value"]:
                raise ValueError("Contradictory positive and negative states")
            cells[key] = record["value"]
            targets.setdefault(frame, []).append(index)
    for record in review["transition_events"]:
        if not reviewed_record(record):
            raise ValueError("Transition must preserve its reviewed origin")
        boundary = record["boundary_frame_range"]
        if (len(boundary) != 2 or any(type(f) is not int for f in boundary)
                or not 0 <= boundary[0] < boundary[1] <= frame_count):
            raise ValueError("Invalid transition boundary range")
        evidence = record.get("evidence_frames", [])
        if not evidence or any(type(f) is not int for f in evidence) or not set(evidence) <= seen:
            raise ValueError("Transition evidence exceeds actual reviewed frames")
        if record.get("frame_resolved") and (boundary[1] - boundary[0] != 1 or record.get("censored") is not False):
            raise ValueError("A censored boundary range cannot be frame-resolved")
    return seen, {f: sorted(set(rows)) for f, rows in sorted(targets.items())}


def verify_evidence(root, review, reviewed_frames):
    covered, bindings = set(), []
    for ref in review["evidence_manifests"]:
        path = root / ref["path"]
        if digest(path) != ref["sha256"]:
            raise ValueError("Dense evidence manifest changed")
        manifest = read(path)
        if manifest["sequence_id"] != review["sequence_id"]:
            raise ValueError("Evidence belongs to another acquisition")
        pages = {p["path"]: p for p in manifest["pages"]}
        chosen = ref["reviewed_page_paths"]
        hashes = ref["reviewed_page_sha256"]
        if len(chosen) != len(hashes) or len(set(chosen)) != len(chosen):
            raise ValueError("Reviewed page SHA list mismatch")
        for name, expected in zip(chosen, hashes):
            if name not in pages or pages[name]["sha256"] != expected or digest(root / name) != expected:
                raise ValueError("Actually reviewed evidence page changed")
            covered.update(pages[name]["frame_indices"])
        bindings.append({"path": ref["path"], "sha256": ref["sha256"],
                         "reviewed_page_paths": chosen, "reviewed_page_sha256": hashes})
    if not reviewed_frames <= covered:
        raise ValueError("Claimed continuous review contains undisplayed native frames")
    return bindings


def export_windows(base_root, v3_root, dense_root):
    base_root, v3_root, root = (Path(p).resolve() for p in (base_root, v3_root, dense_root))
    index_path = root / "mamba_dense_index.json"
    if index_path.exists():
        raise FileExistsError("Preserve the published dense index; use a new version")
    base_path, overlay_path = base_root / "dataset_index.json", v3_root / "mamba_overlay_index.json"
    base, overlay = read(base_path), read(overlay_path)
    base_sha, overlay_sha = digest(base_path), digest(overlay_path)
    split_path = base_root / base["split_manifest_path"]
    split_sha, split = digest(split_path), read(split_path)
    if (overlay["base_dataset_index_sha256"] != base_sha or base["split_manifest_sha256"] != split_sha
            or overlay["base_split_manifest_sha256"] != split_sha):
        raise ValueError("V2/V3 frozen index or split mismatch")
    entries = {e["sequence_id"]: e for e in base["sequences"]}
    annotations = {e["sequence_id"]: e for e in overlay["sequences"]}
    auxiliary = {}
    for name in ("dense_observables_report.json", "entity_grounding_report.json"):
        auxiliary[name] = {"path": name, "sha256": digest(root / name)}
    audit_path = root / "continuous_review_audit.json"
    audited = None
    if audit_path.exists():
        audit = read(audit_path)
        if audit["failed"] != 0:
            raise ValueError("Continuous review audit has failures")
        auxiliary[audit_path.name] = {"path": audit_path.name, "sha256": digest(audit_path)}
        audited = {r["sequence_id"]: r["sha256"] for r in audit["records"]}
    old_windows = {name: {"path": e["window_path"], "sha256": digest(base_root / e["window_path"])}
                   for name, e in base["splits"].items()}
    if any(old_windows[k]["sha256"] != v["sha256"] for k, v in base["splits"].items()):
        raise ValueError("Base windows differ from their frozen index")
    rows, sequences, totals, excluded = {s: [] for s in ("train", "val", "test")}, [], Counter(), Counter()
    for path in sorted((root / "dense_reviews").glob("*.json")):
        review_sha, review = digest(path), read(path)
        uid = review["sequence_id"]
        if audited is not None and audited.get(uid) != review_sha:
            raise ValueError("Review differs from its independent audit")
        if path.stem != uid or uid not in entries:
            raise ValueError("Unknown or mismatched review acquisition")
        entry, annotation = entries[uid], annotations[uid]
        name = split["assignments"][uid]
        if entry["split"] != name or annotation["split"] != name:
            raise ValueError("Review split differs from frozen acquisition")
        b = review["source_bindings"]
        if (b["v2_dataset_index_sha256"] != base_sha or b["v3_overlay_index_sha256"] != overlay_sha
                or b["v3_annotation_sha256"] != annotation["sha256"]):
            raise ValueError("Review uses stale V2/V3 sources")
        if digest(v3_root / annotation["annotation_path"]) != annotation["sha256"]:
            raise ValueError("Frozen V3 annotation changed")
        if digest(base_root / entry["sequence_path"]) != entry["sha256"]["sequence"]:
            raise ValueError("Frozen native sequence changed")
        with np.load(base_root / entry["sequence_path"], allow_pickle=False) as arrays:
            times = arrays["timestamps"]
        reviewed, targets = validate_review(review, len(times))
        evidence = verify_evidence(root, review, reviewed)
        count = 0
        for target, indices in targets.items():
            window, reason = target_window(times, target)
            if window is None:
                excluded[reason] += 1
                continue
            rows[name].append({**window,
                **{key: entry[key] for key in ("sequence_id", "sequence_path", "metadata_path", "events_path", "split")},
                "window_id": f"{uid}:{window['start_frame']}:{target}",
                "source_selection_sha256": base["source_selection_sha256"], "split_manifest_sha256": split_sha,
                "sequence_sha256": entry["sha256"]["sequence"],
                "annotation_path": annotation["annotation_path"], "annotation_sha256": annotation["sha256"],
                "dense_review_path": str(path.relative_to(root)), "dense_review_sha256": review_sha,
                "state_segment_indices": indices, "selection_policy": "explicit_continuous_review_state_targets"})
            count += 1
        sequences.append({"sequence_id": uid, "split": name, "review_path": str(path.relative_to(root)),
            "review_sha256": review_sha, "v3_annotation_sha256": annotation["sha256"],
            "reviewed_native_frame_count": len(reviewed), "candidate_target_count": len(targets),
            "window_count": count, "evidence_manifests": evidence})
        totals.update(reviewed_sequences=1, reviewed_native_frames=len(reviewed),
                      candidate_target_count=len(targets), window_count=count)
    if not sequences:
        raise ValueError("No actual continuous review exists")
    if audited is not None and set(audited) != {e["sequence_id"] for e in sequences}:
        raise ValueError("Published review set differs from its audit")
    # Recheck all mutable V4 inputs immediately before freezing new windows.
    for entry in sequences:
        if digest(root / entry["review_path"]) != entry["review_sha256"]:
            raise ValueError("Review changed during export")
    for ref in auxiliary.values():
        if digest(root / ref["path"]) != ref["sha256"]:
            raise ValueError("Auxiliary V4 report changed during export")
    if digest(base_path) != base_sha or digest(overlay_path) != overlay_sha or digest(split_path) != split_sha:
        raise ValueError("Frozen source indexes changed during export")
    for ref in old_windows.values():
        if digest(base_root / ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen base windows changed during export")
    dest = root / "dense_review_windows"
    dest.mkdir(exist_ok=True)
    splits = {}
    for name, windows in rows.items():
        path = dest / (name + ".jsonl")
        path.write_text("".join(json.dumps(w, ensure_ascii=False, allow_nan=False) + "\n" for w in windows), encoding="utf-8")
        splits[name] = {"window_path": str(path.relative_to(root)), "sha256": digest(path), "window_count": len(windows)}
    index = {"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "base_dataset_root": str(base_root), "v3_annotation_root": str(v3_root), "dense_root": str(root),
        "base_dataset_index_sha256": base_sha, "v3_overlay_index_sha256": overlay_sha,
        "split_manifest_sha256": split_sha, "source_selection_sha256": base["source_selection_sha256"],
        "base_acquisition_count": len(entries), "sequences": sequences, "splits": splits,
        "source_reports": auxiliary, "base_window_bindings": old_windows,
        "window_config": CONFIG, "totals": dict(totals), "target_exclusion_counts": dict(excluded),
        "review_selected_diagnostic": True, "blind_evaluation": False, "new_independent_acquisitions": 0,
        "numeric_paths_relative_to": "base_dataset_root", "window_paths_relative_to": "dense_root",
        "v3_annotation_paths_relative_to": "v3_annotation_root", "base_windows_modified": False,
        "geometry_binding_policy": "partial_components_never_substitute_whole_visual_entities",
        "training_allowed": False, "allowed_as_model_input": False, "diagnostic_loading_allowed": True,
        "human_label_valid": False, "full_annotation_complete": False, "exporter_sha256": digest(__file__)}
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--v3-root", required=True)
    parser.add_argument("--dense-root", required=True)
    args = parser.parse_args()
    print(json.dumps(export_windows(args.base_root, args.v3_root, args.dense_root)["totals"]))


if __name__ == "__main__":
    main()
