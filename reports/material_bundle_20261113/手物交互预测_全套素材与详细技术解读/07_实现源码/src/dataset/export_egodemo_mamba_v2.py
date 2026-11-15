"""Export private, pending-review EgoDemo geometry for diagnostic Mamba loading.

The output is deliberately separate from the seven-class training dataset. No
interpolation, manufactured object identity, negative labels, or training approval
is introduced. Sparse surface observations retain their tracklet/point keys.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.dataset.complete_egodemo import digest, save


SCHEMA = "egodemo_mamba_v2"
EXPORT_VERSION = "native_sparse_world_v1"
KINDS = {"unknown": 0, "rigid": 1, "deformable": 2, "articulated": 3}
SUBJECTS = {"left_hand": 0, "right_hand": 1, "both_hands": 2, "object": 3}
PHYSICAL_TYPES = {
    "relation": {"contact": "接触", "grasp_control": "抓持控制", "co_motion": "共同运动",
                 "supported_by": "受到支撑", "stabilizes": "稳定工件", "acts_on": "工具作用于工件"},
    "transition": {"contact_onset": "首次接触", "grasp_onset": "建立抓持", "operation_onset": "开始操作",
                   "support_gained": "建立支撑", "support_lost": "失去支撑", "release": "最终脱离"},
    "cooperation": {"single_hand": "单手操作", "bimanual_joint": "双手共同操作",
                    "complementary": "一手操作、一手稳定", "handover": "手间传递", "tool_mediated": "工具介导"},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def native_windows(timestamps, window_frames=32, stride_frames=16,
                   horizon_seconds=1.0, max_overshoot_median_dt=1.5):
    """Use actual timestamps, rejecting targets separated by a missing-frame gap."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if (timestamps.ndim != 1 or len(timestamps) < 2 or
            not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if window_frames < 1 or stride_frames < 1 or horizon_seconds <= 0 or max_overshoot_median_dt < 0:
        raise ValueError("Invalid window configuration")
    tolerance = float(np.median(np.diff(timestamps))) * max_overshoot_median_dt
    for start in range(0, len(timestamps) - window_frames + 1, stride_frames):
        end = start + window_frames
        anchor = end - 1
        desired = float(timestamps[anchor] + horizon_seconds)
        target = int(np.searchsorted(timestamps, desired, side="left"))
        if target >= len(timestamps):
            continue
        if timestamps[target] - desired > tolerance + 1e-9:
            continue
        yield {"start_frame": start, "end_frame": end, "anchor_frame": anchor,
               "target_frame": target, "target_timestamp": float(timestamps[target]),
               "actual_horizon_s": float(timestamps[target] - timestamps[anchor])}


def export_events(proposal, frame_count, object_ids):
    indexes = {value: index for index, value in enumerate(object_ids)}
    records = []
    for interval in proposal.get("source_semantic_intervals", []):
        for operation in interval.get("operations", []):
            records.append({
                **operation, "layer": "operation", "start_frame": interval["start_frame"],
                "end_frame": interval["end_frame"], "subject": -1, "object_index": -1,
                "secondary_object_index": -1, "object_id": None,
                "source_interval_id": interval["source_interval_id"],
                "description_zh": interval.get("description_zh"),
                "description_original": interval.get("description_original"),
                "temporal_precision": "source_interval_only", "review_status": "pending",
                "label_valid": False, "weak_positive": True,
            })
    for row in proposal.get("interactions", []):
        if row["object_id"] not in indexes:
            raise ValueError("Interaction refers to an absent object")
        second = row.get("secondary_object_id")
        if second is not None and second not in indexes:
            raise ValueError("Interaction secondary object is absent")
        records.append({**row, "subject_original": row.get("subject"),
                        "subject": SUBJECTS.get(row.get("subject"), -1),
                        "object_index": indexes[row["object_id"]],
                        "secondary_object_index": indexes.get(second, -1),
                        "review_status": "pending", "label_valid": False, "weak_positive": True})
    for row in records:
        if not 0 <= row["start_frame"] < row["end_frame"] <= frame_count:
            raise ValueError("Candidate interval is outside the native timeline")
        if row["layer"] != "operation" and row["type"] not in PHYSICAL_TYPES[row["layer"]]:
            raise ValueError("Unknown physical candidate type")
    return {"schema_version": SCHEMA, "records": records,
            "source_semantic_intervals": proposal.get("source_semantic_intervals", []),
            "missing_label_means": "unknown", "gold_label_count": 0,
            "review_status": "pending", "training_allowed": False}


def source_paths(source, sequence_id, proposal):
    base = {
        "hands": source / "sequences" / sequence_id / "hands.npz",
        "source_metadata": source / "sequences" / sequence_id / "metadata.json",
        "proposal": source / "joint_annotations" / sequence_id / "proposal.json",
        "objects": source / "joint_annotations" / sequence_id / "objects.npz",
        "joint_receipt": source / "joint_annotations" / sequence_id / "receipt.json",
        "track_manifest": source / "native_tracks" / sequence_id / "tracklets.json",
    }
    for entity in proposal["entities"]:
        for track in entity.get("tracklets", []):
            base[f"track:{track}"] = source / "native_tracks" / sequence_id / (track + ".npz")
    for name in ["visual_quality_overrides.json", "pixel_calibration_audit_summary.json",
                 "calibration_recovery_summary.json", "display_names_zh.json"]:
        if (source / name).exists():
            base[name] = source / name
    return base


def export_sequence(source, output, sequence_id):
    source, output = Path(source), Path(output)
    directory = output / "sequences" / sequence_id
    proposal = read_json(source / "joint_annotations" / sequence_id / "proposal.json")
    paths = source_paths(source, sequence_id, proposal)
    hashes = {role: {"path": str(path), "sha256": digest(path)} for role, path in paths.items()}
    code_sha = digest(__file__)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt.get("source_files") != hashes or receipt.get("exporter_sha256") != code_sha or
                any(digest(directory / artifact["file"]) != artifact["sha256"]
                    for artifact in receipt["artifacts"])):
            raise ValueError(f"Existing shard or source changed: {sequence_id}; use a new destination")
        return receipt
    source_meta = read_json(paths["source_metadata"])
    joint_receipt = read_json(paths["joint_receipt"])
    track_manifest = read_json(paths["track_manifest"])
    track_rows = {row["tracklet_id"]: row for row in track_manifest["records"]}
    selection_sha = proposal["source_selection_sha256"]
    if source_meta["source_selection_sha256"] != selection_sha:
        raise ValueError("Hand and object selection fingerprints differ")
    with np.load(paths["hands"], allow_pickle=False) as z:
        arrays = {"timestamps": z["timestamps"].astype(np.float64),
                  "frame_index": z["frame_index"].astype(np.int32),
                  "hands_world": z["hands_world"].astype(np.float32),
                  "hand_valid": z["hand_valid"].astype(bool),
                  "camera_to_world": z["camera_to_world"].astype(np.float64)}
    times = arrays["timestamps"]
    count = len(times)
    if not np.array_equal(arrays["frame_index"], np.arange(count)):
        raise ValueError("Native frame index is not contiguous")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Native timestamps are invalid")
    if arrays["hands_world"].shape != (count, 2, 21, 3) or arrays["hand_valid"].shape != (count, 2):
        raise ValueError("Unexpected anatomical hand array shape")
    if not np.isfinite(arrays["hands_world"][arrays["hand_valid"]]).all():
        raise ValueError("Valid source hands contain nonfinite joints")
    arrays["hands_world"][~arrays["hand_valid"]] = np.nan
    c2w = arrays["camera_to_world"]
    if c2w.shape != (count, 4, 4) or not np.isfinite(c2w).all():
        raise ValueError("Invalid left camera extrinsics")
    with np.load(paths["objects"], allow_pickle=False) as z:
        object_ids = z["object_ids"].astype(str).tolist()
        camera_pose = z["poses"].transpose(1, 0, 2, 3).copy()
        pose_valid = z["pose_valid"].T.copy().astype(bool)
        if not np.array_equal(z["frame_index"], arrays["frame_index"]):
            raise ValueError("Object and hand native frame indices disagree")
    entities = {row["id"]: row for row in proposal["entities"]}
    if len(entities) != len(object_ids) or set(entities) != set(object_ids):
        raise ValueError("Object pose IDs and proposal entities differ")
    object_count = len(object_ids)
    if camera_pose.shape != (count, object_count, 4, 4) or pose_valid.shape != (count, object_count):
        raise ValueError("Unexpected object pose dimensions")
    metric_allowed = bool(joint_receipt.get("metric_geometry_allowed", False))
    for index, identity in enumerate(object_ids):
        if not metric_allowed or not entities[identity].get("metric_geometry_allowed", False):
            pose_valid[:, index] = False
    if not np.isfinite(camera_pose[pose_valid]).all():
        raise ValueError("Valid filtered object poses contain nonfinite values")
    world_pose = np.full(camera_pose.shape, np.nan, dtype=np.float32)
    frame, obj = np.nonzero(pose_valid)
    world_pose[frame, obj] = (c2w[frame] @ camera_pose[frame, obj]).astype(np.float32)
    arrays.update(object_ids=np.asarray(object_ids, dtype="U64"),
                  object_kind=np.asarray([KINDS[entities[x]["rigidity"]] for x in object_ids], dtype=np.int8),
                  object_pose_world=world_pose, object_pose_valid=pose_valid)

    surface_frames, surface_objects, surface_points, surface_xyz = [], [], [], []
    point_keys, point_objects, assigned_tracks = [], [], set()
    for object_index, identity in enumerate(object_ids):
        entity = entities[identity]
        for track in sorted(entity.get("tracklets", [])):
            if track in assigned_tracks:
                raise ValueError("A tracklet cannot belong to multiple objects")
            assigned_tracks.add(track)
            if hashes[f"track:{track}"]["sha256"] != track_rows[track]["sha256"]:
                raise ValueError("Track file differs from its frozen manifest")
            if not metric_allowed or not entity.get("metric_geometry_allowed", False):
                continue
            with np.load(paths[f"track:{track}"], allow_pickle=False) as z:
                frames = z["frame_index"].astype(np.int32)
                points = z["points_world"]
                valid = z["point_valid"].astype(bool)
                ids = z["point_ids"]
                if (np.any(frames < 0) or np.any(frames >= count) or
                        len(np.unique(frames)) != len(frames) or
                        not np.allclose(z["timestamps"], times[frames], rtol=0, atol=1e-7)):
                    raise ValueError("Surface timestamps do not align with native frames")
                if points.shape != valid.shape + (3,) or valid.shape != (len(frames), len(ids)):
                    raise ValueError("Surface array shapes disagree")
                if len(np.unique(ids)) != len(ids):
                    raise ValueError("Duplicate point IDs within a tracklet")
                valid &= np.isfinite(points).all(axis=-1)
                # Every retained point key has at least one usable observation.
                used = np.flatnonzero(valid.any(axis=0))
                mapping = np.full(len(ids), -1, dtype=np.int32)
                mapping[used] = np.arange(len(point_keys), len(point_keys) + len(used))
                point_keys.extend(f"{track}/{int(ids[p])}" for p in used)
                point_objects.extend([object_index] * len(used))
                local_frame, local_point = np.nonzero(valid)
                surface_frames.append(frames[local_frame])
                surface_objects.append(np.full(len(local_frame), object_index, dtype=np.int32))
                surface_points.append(mapping[local_point])
                surface_xyz.append(points[local_frame, local_point].astype(np.float32))
    sf = np.concatenate(surface_frames) if surface_frames else np.empty(0, dtype=np.int32)
    so = np.concatenate(surface_objects) if surface_objects else np.empty(0, dtype=np.int32)
    sp = np.concatenate(surface_points) if surface_points else np.empty(0, dtype=np.int32)
    xyz = np.concatenate(surface_xyz) if surface_xyz else np.empty((0, 3), dtype=np.float32)
    order = np.lexsort((sp, so, sf))
    observed = pose_valid.copy()
    observed[sf, so] = True
    arrays.update(surface_frame_index=sf[order], surface_object_index=so[order],
                  surface_point_index=sp[order], surface_world=xyz[order],
                  surface_frame_ptr=np.concatenate(([0], np.cumsum(np.bincount(sf, minlength=count)))).astype(np.int64),
                  point_keys=np.asarray(point_keys, dtype="U96"),
                  point_object_index=np.asarray(point_objects, dtype=np.int32), object_observed=observed)
    events = export_events(proposal, count, object_ids)
    metadata = {
        "schema_version": SCHEMA, "sequence_id": sequence_id, "frame_count": count,
        "source_selection_sha256": selection_sha, "task_name": source_meta["task_name"],
        "source_metadata": source_meta, "source_files": hashes,
        "entities": [entities[identity] for identity in object_ids],
        "hand_order": ["left", "right"], "hand_names_zh": ["左手", "右手"],
        "coordinate_frame": "world", "units": "meters",
        "loader_anchor": "left_camera_at_last_observation_frame",
        "object_observed_definition": "at least one quality-retained pose or surface-point candidate; not gold visibility",
        "point_key_definition": "tracklet_id/point_id; no point identity asserted across tracklets",
        "object_kind_codes": KINDS, "metric_geometry_allowed": metric_allowed,
        "pixel_calibration_status": joint_receipt.get("pixel_calibration_status"),
        "annotation_mode": "offline_noncausal", "allowed_as_model_input": False,
        "human_approved": False, "full_annotation_complete": False,
        "review_status": "pending", "training_allowed": False,
        "independent_geometry_accuracy": None, "diagnostic_loading_allowed": True,
    }
    write_npz(directory / "sequence.npz", arrays)
    save(directory / "metadata.json", metadata)
    save(directory / "events.json", events)
    # Recheck the source after reading: a concurrently edited draft is not a frozen export.
    if any(digest(path) != hashes[role]["sha256"] for role, path in paths.items()):
        raise ValueError("Source changed during export")
    receipt = {
        "schema_version": SCHEMA, "export_version": EXPORT_VERSION, "exporter_sha256": code_sha,
        "sequence_id": sequence_id, "source_selection_sha256": selection_sha,
        "source_files": hashes, "frame_count": count, "object_count": object_count,
        "hand_valid_frames": arrays["hand_valid"].sum(axis=0).astype(int).tolist(),
        "object_pose_observations": int(pose_valid.sum()),
        "surface_observations": len(sf), "surface_point_keys": len(point_keys),
        "source_semantic_intervals": len(proposal.get("source_semantic_intervals", [])),
        "candidate_counts": dict(Counter(f'{r["layer"]}:{r["type"]}' for r in events["records"])),
        "gold_label_count": 0, "metric_geometry_allowed": metric_allowed,
        "training_allowed": False, "allowed_as_model_input": False,
        "artifacts": [{"file": name, "sha256": digest(directory / name),
                       "bytes": (directory / name).stat().st_size}
                      for name in ["sequence.npz", "metadata.json", "events.json"]],
    }
    save(receipt_path, receipt)
    return receipt


def build_vocab(source):
    operations = {}
    counts = Counter()
    for path in sorted((Path(source) / "joint_annotations").glob("*/proposal.json")):
        proposal = read_json(path)
        for interval in proposal.get("source_semantic_intervals", []):
            for operation in interval.get("operations", []):
                code, name = operation["type"], operation["name_zh"]
                if code in operations and operations[code] != name:
                    raise ValueError("Inconsistent operation display names")
                operations[code] = name
                counts[f"operation:{code}"] += 1
        counts.update(f'{r["layer"]}:{r["type"]}' for r in proposal.get("interactions", []))
    result = {}
    for layer, mapping in {"operation": operations, **PHYSICAL_TYPES}.items():
        codes = sorted(mapping)
        result[layer] = {"codes": codes, "names_zh": [mapping[code] for code in codes],
                         "source_candidate_counts": [counts[f"{layer}:{code}"] for code in codes]}
    return result


def split_assignments(manifest):
    assignments = manifest.get("assignments", manifest.get("seq_to_split", manifest.get("sequence_to_split")))
    if assignments is None:
        raise ValueError("Frozen split manifest must expose seq_to_split")
    if set(assignments.values()) - {"train", "val", "test"}:
        raise ValueError("Unexpected split names")
    return assignments


def export_dataset(source, output, split_manifest, sequence_ids=None, workers=2,
                   window_frames=32, stride_frames=16, horizon_seconds=1.0):
    source, output, split_manifest = Path(source), Path(output), Path(split_manifest)
    manifest = read_json(split_manifest)
    assignments = split_assignments(manifest)
    uids = sorted(sequence_ids or assignments)
    if not uids or set(uids) - set(assignments):
        raise ValueError("Every sequence must have a frozen split before window generation")
    output.mkdir(parents=True, exist_ok=True)
    # A subset retains the full frozen manifest and is explicitly marked partial.
    frozen_path = output / "splits" / "split_manifest.json"
    if frozen_path.resolve() != split_manifest.resolve():
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = frozen_path.with_suffix(".json.partial")
        temporary.write_bytes(split_manifest.read_bytes())
        temporary.replace(frozen_path)
    frozen_sha = digest(frozen_path)
    vocab = build_vocab(source)
    save(output / "vocab.json", vocab)
    receipts = []
    if workers == 1:
        for uid in uids:
            receipts.append(export_sequence(source, output, uid))
            print(json.dumps({"stage": "sequences", "done": len(receipts), "total": len(uids),
                              "sequence_id": uid}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(export_sequence, source, output, uid): uid for uid in uids}
            for future in as_completed(futures):
                receipts.append(future.result())
                print(json.dumps({"stage": "sequences", "done": len(receipts), "total": len(uids),
                                  "sequence_id": futures[future]}), flush=True)
    receipts.sort(key=lambda row: row["sequence_id"])
    source_shas = {row["source_selection_sha256"] for row in receipts}
    if len(source_shas) != 1:
        raise ValueError("Mixed source selection versions")
    source_sha = next(iter(source_shas))
    if manifest.get("source_selection_sha256", source_sha) != source_sha:
        raise ValueError("Frozen split uses a different source selection")
    window_config = {"window_frames": window_frames, "stride_frames": stride_frames,
                     "horizon_seconds": horizon_seconds, "max_horizon_overshoot_median_dt": 1.5}
    windows_dir = output / "windows"
    windows_dir.mkdir(exist_ok=True)
    counts = Counter()
    coverage = Counter()
    sequence_entries = []
    handles = {split: (windows_dir / f"{split}.jsonl.partial").open("w") for split in ["train", "val", "test"]}
    try:
        for receipt in receipts:
            uid = receipt["sequence_id"]
            split = assignments[uid]
            prefix = f"sequences/{uid}"
            entry = {"sequence_id": uid, "split": split, "frame_count": receipt["frame_count"],
                     "object_count": receipt["object_count"], "sequence_path": prefix + "/sequence.npz",
                     "metadata_path": prefix + "/metadata.json", "events_path": prefix + "/events.json",
                     "sha256": {a["file"].split(".")[0]: a["sha256"] for a in receipt["artifacts"]}}
            count = 0
            events = read_json(output / entry["events_path"])["records"]
            weak_source_at_frame = np.zeros(receipt["frame_count"], dtype=bool)
            for event in events:
                if event["layer"] == "operation" and event.get("weak_positive"):
                    weak_source_at_frame[event["start_frame"]:event["end_frame"]] = True
            with np.load(output / entry["sequence_path"], allow_pickle=False) as z:
                observed = z["object_observed"]
                pose_valid = z["object_pose_valid"]
                surface_ptr = z["surface_frame_ptr"]
                for window in native_windows(z["timestamps"], window_frames, stride_frames, horizon_seconds):
                    a, b, target = window["start_frame"], window["end_frame"], window["target_frame"]
                    coverage["any_object_geometry_in_observation"] += int(observed[a:b].any())
                    coverage["any_object_pose_at_target"] += int(pose_valid[target].any())
                    coverage["any_surface_at_target"] += int(surface_ptr[target + 1] > surface_ptr[target])
                    coverage["weak_source_operation_at_target"] += int(weak_source_at_frame[target])
                    row = {**window, **{key: entry[key] for key in ["sequence_id", "sequence_path", "metadata_path", "events_path", "split"]},
                           "window_id": f'{uid}:{window["start_frame"]}:{window["target_frame"]}',
                           "source_selection_sha256": source_sha, "split_manifest_sha256": frozen_sha,
                           "sequence_sha256": entry["sha256"]["sequence"]}
                    handles[split].write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                    count += 1
            counts[split] += count
            entry["window_count"] = count
            sequence_entries.append(entry)
    finally:
        for handle in handles.values():
            handle.close()
    splits = {}
    for split in handles:
        path = windows_dir / f"{split}.jsonl"
        path.with_suffix(".jsonl.partial").replace(path)
        splits[split] = {"window_path": f"windows/{split}.jsonl", "window_count": counts[split],
                         "sequence_count": sum(row["split"] == split for row in sequence_entries),
                         "sha256": digest(path)}
    index = {"schema_version": SCHEMA, "export_version": EXPORT_VERSION,
             "created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(source),
             "source_selection_sha256": source_sha, "source_sequence_count": len(assignments),
             "exported_sequence_count": len(uids), "is_partial_export": len(uids) != len(assignments),
             "vocab_path": "vocab.json", "vocab_sha256": digest(output / "vocab.json"),
             "split_manifest_path": "splits/split_manifest.json", "split_manifest_sha256": frozen_sha,
             "window_config": window_config, "splits": splits, "sequences": sequence_entries,
             "training_allowed": False, "allowed_as_model_input": False,
             "diagnostic_loading_allowed": True, "human_approved": False,
             "annotation_mode": "offline_noncausal", "full_annotation_complete": False,
             "exporter_sha256": digest(__file__)}
    save(output / "dataset_index.json", index)
    total_candidates = Counter()
    for receipt in receipts:
        total_candidates.update(receipt["candidate_counts"])
    report = {"schema_version": SCHEMA, "sequence_count": len(receipts),
              "native_frame_count": sum(r["frame_count"] for r in receipts),
              "window_count": sum(counts.values()), "windows_by_split": dict(counts),
              "sequences_with_object_pose": sum(r["object_pose_observations"] > 0 for r in receipts),
              "sequences_with_surface": sum(r["surface_observations"] > 0 for r in receipts),
              "object_pose_observations": sum(r["object_pose_observations"] for r in receipts),
              "surface_observations": sum(r["surface_observations"] for r in receipts),
              "surface_point_keys": sum(r["surface_point_keys"] for r in receipts),
              "source_semantic_intervals": sum(r["source_semantic_intervals"] for r in receipts),
              "operation_vocab_count": len(vocab["operation"]["codes"]),
              "window_coverage_before_slot_selection": {**dict(coverage), "gold_event_valid": 0},
              "window_coverage_note": "Candidate availability before N/P selection; not approved joint-training samples",
              "candidate_counts": dict(total_candidates), "gold_label_count": 0,
              "source_selection_sha256": source_sha, "dataset_index_sha256": digest(output / "dataset_index.json"),
              "split_manifest_sha256": frozen_sha, "training_allowed": False,
              "allowed_as_model_input": False, "diagnostic_loading_allowed": True,
              "is_partial_export": index["is_partial_export"],
              "shard_bytes": sum(a["bytes"] for r in receipts for a in r["artifacts"])}
    save(output / "export_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sequence-id", action="append")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    args = parser.parse_args()
    report = export_dataset(args.source, args.output, args.split_manifest, args.sequence_id,
                            args.workers, args.window_frames, args.stride_frames, args.horizon_seconds)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
