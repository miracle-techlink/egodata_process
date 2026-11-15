"""Bind reviewed identity points to uninterrupted native point trajectories.

Only the anchor is a visual judgment. Propagated identity is a separate
geometric candidate, never a dense interaction label or a verified 6D pose.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "assistant_annotation_v4_entity_grounding"
CONFIG = {"minimum_surviving_anchor_points": 4, "maximum_native_time_gap_median_dt": 1.5,
          "maximum_point_step_image_diagonal_fraction": 0.05,
          "point_reappearance_allowed": False, "cross_tracklet_merge_allowed": False}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ranges(frames):
    frames = sorted(set(int(f) for f in frames))
    out = []
    for frame in frames:
        if not out or frame != out[-1]["end_frame"]:
            out.append({"start_frame": frame, "end_frame": frame + 1})
        else:
            out[-1]["end_frame"] += 1
    return out


def inside_polygon(points, polygon):
    """Strict interior membership; polygon boundaries are not assumed visible."""
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("Invalid reviewed pixel polygon")
    x, y = np.asarray(points, dtype=float).T
    result = np.zeros(len(x), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (a[1] > y) != (b[1] > y)
        if b[1] != a[1]:
            result ^= crosses & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
    return result


def valid_pixels(track, image_size):
    width, height = image_size
    uv = track["points_uv"][:, 0]
    valid = track["point_valid"].astype(bool).copy()
    if uv.shape != valid.shape + (2,):
        raise ValueError("Point pixel shape differs from validity")
    valid &= np.isfinite(uv).all(axis=-1)
    valid &= (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    return valid


def propagate_anchor(track, anchor_frame, native_times, image_size, point_ids=None):
    """Keep only points continuously visible all the way from this anchor."""
    frames = np.asarray(track["frame_index"], dtype=int)
    ids = np.asarray(track["point_ids"], dtype=int)
    times = np.asarray(native_times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid native timestamps")
    if (frames.ndim != 1 or np.any(np.diff(frames) <= 0) or np.any(frames < 0)
            or np.any(frames >= len(times)) or len(np.unique(ids)) != len(ids)):
        raise ValueError("Invalid native point identity/frame indices")
    if not np.allclose(track["timestamps"], times[frames], rtol=0, atol=1e-7):
        raise ValueError("Track timestamps differ from native acquisition")
    valid = valid_pixels(track, image_size)
    at = int(np.searchsorted(frames, anchor_frame))
    if at == len(frames) or frames[at] != anchor_frame:
        return [], "anchor_missing_from_track"
    anchored = valid[at].copy()
    if point_ids is not None:
        if not set(point_ids) <= set(ids.tolist()):
            raise ValueError("Anchor references unknown point IDs")
        anchored &= np.isin(ids, point_ids)
    minimum = CONFIG["minimum_surviving_anchor_points"]
    if anchored.sum() < minimum:
        return [], "insufficient_anchor_points"
    gap = CONFIG["maximum_native_time_gap_median_dt"] * float(np.median(np.diff(times)))
    step = CONFIG["maximum_point_step_image_diagonal_fraction"] * float(np.linalg.norm(image_size))
    uv = track["points_uv"][:, 0]
    selected = {at: anchored.copy()}
    stops = {}
    for direction in (-1, 1):
        alive, previous, index = anchored.copy(), at, at + direction
        reason = "track_end"
        while 0 <= index < len(frames):
            if abs(int(frames[index]) - int(frames[previous])) != 1:
                reason = "native_frame_gap"
                break
            if abs(times[frames[index]] - times[frames[previous]]) > gap + 1e-9:
                reason = "native_timestamp_gap"
                break
            alive &= valid[index]
            alive &= np.linalg.norm(uv[index] - uv[previous], axis=-1) <= step
            if alive.sum() < minimum:
                reason = "insufficient_uninterrupted_anchor_points"
                break
            selected[index] = alive.copy()
            previous, index = index, index + direction
        stops["before" if direction < 0 else "after"] = reason
    observations = []
    for index, keep in sorted(selected.items()):
        points = uv[index, keep]
        observations.append({"frame_index": int(frames[index]), "timestamp": float(times[frames[index]]),
            "point_ids": ids[keep].tolist(), "point_count": int(keep.sum()),
            "bbox_xyxy": [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]})
    return observations, stops


def old_anchors(annotation):
    result = []
    for entity in annotation["entities"]:
        if not str(entity["id"]).startswith("obj_"):
            continue
        for assessment_index, assessment in enumerate(entity.get("visual_assessments", [])):
            if (assessment.get("status") != "supported" or assessment.get("silver_label_valid") is not True
                    or assessment.get("annotation_source") != "assistant_visual_review"
                    or assessment.get("review_status") != "assistant_reviewed"
                    or assessment.get("scope") == "role"):
                continue
            for frame in assessment.get("evidence_frames", []):
                if type(frame) is not int:
                    raise ValueError("Visual identity anchor frame is not an integer")
                result.append({"anchor_id": f"v3:{entity['id']}:{assessment_index}:{frame}",
                    "entity_id": entity["id"], "source_object_id": entity["id"], "frame_index": frame,
                    "source_interval_id": assessment.get("source_interval_id"),
                    "mapping_scope": "visible_surface_identity_at_anchor_only",
                    "assessment_scope": assessment.get("scope"), "reason_zh": assessment.get("reason_zh"),
                    "evidence": assessment["evidence"], "reviewer_agent": assessment["reviewer_agent"],
                    "reviewed_at": assessment["reviewed_at"], "role_candidates": assessment.get("role_candidates", []),
                    "annotation_source": "assistant_visual_review", "review_status": "assistant_reviewed",
                    "silver_label_valid": True, "human_label_valid": False, "anchor_source": "frozen_v3_review"})
    return result


def ground_sequence(source, base_root, v3_root, output, entry, overlay_entry, shared_bindings):
    uid = entry["sequence_id"]
    annotation_path = v3_root / overlay_entry["annotation_path"]
    if digest(annotation_path) != overlay_entry["sha256"]:
        raise ValueError("Frozen V3 annotation changed")
    annotation = read(annotation_path)
    proposal_path = source / "joint_annotations" / uid / "proposal.json"
    if digest(proposal_path) != annotation["source_bindings"]["proposal_sha256"]:
        raise ValueError("Frozen proposal changed")
    proposal = read(proposal_path)
    manifest_path = source / "native_tracks" / uid / "tracklets.json"
    manifest = read(manifest_path)
    tracks_by_id = {r["tracklet_id"]: r for r in manifest["records"]}
    native_path = base_root / entry["sequence_path"]
    if digest(native_path) != entry["sha256"]["sequence"]:
        raise ValueError("Frozen native numeric sequence changed")
    with np.load(native_path, allow_pickle=False) as arrays:
        times = arrays["timestamps"]
    source_metadata_path = source / "sequences" / uid / "metadata.json"
    metadata = read(source_metadata_path)
    image_size = metadata["image_size"]
    excluded = {r["object_id"] for r in annotation.get("geometry_exclusions", [])
                if r["scope"] == "entity_until_reidentified"}
    entities = {e["id"]: e for e in proposal["entities"]}
    track_arrays, candidate_tracks = {}, []
    for entity in entities.values():
        for identity in entity["tracklets"]:
            row = tracks_by_id[identity]
            path = manifest_path.parent / (identity + ".npz")
            if digest(path) != row["sha256"]:
                raise ValueError("Frozen native track changed")
            with np.load(path, allow_pickle=False) as arrays:
                track = {k: arrays[k] for k in ("frame_index", "timestamps", "point_ids", "point_valid", "points_uv")}
            track_arrays[identity] = track
            valid = valid_pixels(track, image_size)
            candidate_tracks.append({"tracklet_id": identity, "source_object_id": entity["id"],
                "track_path": str(path), "sha256": row["sha256"], "reference_frame": row["reference_frame"],
                "point_observation_ranges": ranges(track["frame_index"][valid.any(axis=1)]),
                "point_key_count": len(track["point_ids"]), "identity_assistant_verified": False,
                "metric_geometry_allowed": bool(entity.get("metric_geometry_allowed", False)) and entity["id"] not in excluded,
                "pose_scope": entity.get("pose_scope", "unknown"),
                "rejected_by_v3_visual_identity": entity["id"] in excluded,
                "annotation_source": "geometry_rule", "review_status": "estimated", "silver_label_valid": False})
    anchors = old_anchors(annotation)
    manual_path = output / "manual_entity_anchors" / (uid + ".json")
    if manual_path.exists():
        manual = read(manual_path)
        if manual["sequence_id"] != uid or manual["v3_annotation_sha256"] != overlay_entry["sha256"]:
            raise ValueError("Manual anchor source binding mismatch")
        for anchor in manual["anchors"]:
            if (anchor["annotation_source"] != "assistant_visual_review" or anchor["review_status"] != "assistant_reviewed"
                    or anchor["silver_label_valid"] is not True or anchor["human_label_valid"] is not False
                    or not anchor["evidence"] or not anchor.get("pixel_polygon")):
                raise ValueError("Manual mapping requires reviewed pixel evidence")
            if not any(anchor["frame_index"] in evidence.get("frames", []) for evidence in anchor["evidence"]):
                raise ValueError("Manual mapping anchor is absent from viewed evidence")
            anchors.append({**anchor, "anchor_source": "v4_pixel_anchor"})
    bindings, unresolved_anchors = [], []
    for anchor in anchors:
        source_object = anchor["source_object_id"]
        if source_object not in entities:
            raise ValueError("Anchor refers to unknown source object")
        # Each evidence reference binds actual viewed pixels, not a task name.
        for evidence in anchor["evidence"]:
            path = Path(evidence["gallery_path"])
            if not path.is_absolute():
                path = v3_root / path
            if digest(path) != evidence["gallery_sha256"]:
                raise ValueError("Viewed anchor evidence changed")
        if source_object in excluded:
            unresolved_anchors.append({"anchor_id": anchor["anchor_id"], "reason": "v3_identity_excluded"})
            continue
        choices = []
        for track_id in entities[source_object]["tracklets"]:
            if anchor.get("tracklet_id") and anchor["tracklet_id"] != track_id:
                continue
            track = track_arrays[track_id]
            at = int(np.searchsorted(track["frame_index"], anchor["frame_index"]))
            if at == len(track["frame_index"]) or track["frame_index"][at] != anchor["frame_index"]:
                continue
            keep = valid_pixels(track, image_size)[at]
            if anchor.get("pixel_polygon"):
                keep &= inside_polygon(track["points_uv"][at, 0], anchor["pixel_polygon"])
            if keep.sum() >= CONFIG["minimum_surviving_anchor_points"]:
                choices.append((track_id, track["point_ids"][keep].tolist()))
        if len(choices) != 1:
            unresolved_anchors.append({"anchor_id": anchor["anchor_id"],
                "reason": "ambiguous_tracklets_at_anchor" if choices else "insufficient_visible_anchor_points",
                "candidate_tracklets": [c[0] for c in choices]})
            continue
        track_id, point_ids = choices[0]
        observations, stops = propagate_anchor(track_arrays[track_id], anchor["frame_index"], times, image_size, point_ids)
        if not observations:
            unresolved_anchors.append({"anchor_id": anchor["anchor_id"], "reason": stops})
            continue
        anchor["tracklet_id"] = track_id
        anchor["point_keys"] = [f"{track_id}/{p}" for p in point_ids]
        bindings.append({"binding_id": f"binding_{len(bindings):04d}", "anchor_id": anchor["anchor_id"],
            "entity_id": anchor["entity_id"], "source_object_id": source_object, "tracklet_id": track_id,
            "anchor_frame": anchor["frame_index"], "anchor_point_keys": anchor["point_keys"],
            "tracked_candidate_ranges": ranges(o["frame_index"] for o in observations),
            "observations": observations, "stop_reasons": stops,
            "pose_scope": entities[source_object].get("pose_scope", "unknown"),
            "metric_geometry_allowed": bool(entities[source_object].get("metric_geometry_allowed", False)),
            "geometry_representation": "native_point_subset",
            "association_type": anchor.get("association_type", "visible_surface_of"),
            "component_name_zh": anchor.get("component_name_zh"),
            "whole_object_pose_certified": False, "pose_recomputed": False,
            "mapping_scope": anchor["mapping_scope"], "annotation_source": "geometry_rule",
            "review_status": "estimated", "silver_label_valid": False, "human_label_valid": False,
            "role_propagated": False, "interaction_inferred": False,
            "cross_occlusion_identity_verified": False, "allowed_as_model_input": False})
    bound_ids = {a["entity_id"] for a in bindings}
    unresolved = [{"entity_id": e["id"], "reason": "no_unambiguous_pixel_or_existing_id_anchor",
                   "geometry_status": e.get("geometry_status", "unknown")}
                  for e in annotation["entities"] if e.get("kind") == "object" and e["id"] not in bound_ids]
    candidate_frames = {o["frame_index"] for b in bindings for o in b["observations"]}
    identity_frames = {(b["entity_id"], o["frame_index"]) for b in bindings for o in b["observations"]}
    doc = {"schema_version": SCHEMA, "sequence_id": uid, "split": entry["split"], "frame_count": len(times),
        "source_bindings": {**shared_bindings, "v3_annotation_path": str(annotation_path),
            "v3_annotation_sha256": overlay_entry["sha256"], "base_sequence_sha256": entry["sha256"]["sequence"],
            "proposal_path": str(proposal_path), "proposal_sha256": digest(proposal_path),
            "track_manifest_path": str(manifest_path), "track_manifest_sha256": digest(manifest_path),
            "source_metadata_path": str(source_metadata_path), "source_metadata_sha256": digest(source_metadata_path),
            "manual_anchor_sha256": digest(manual_path) if manual_path.exists() else None},
        "anchors": anchors, "bindings": bindings, "candidate_tracks": candidate_tracks,
        "unresolved_anchors": unresolved_anchors, "unresolved_entities": unresolved,
        "exclusions": annotation.get("geometry_exclusions", []), "propagation_config": CONFIG,
        "coverage": {"frame_count": len(times), "source_object_count": len(entities),
            "candidate_track_count": len(candidate_tracks), "anchor_count": len(anchors),
            "v3_anchor_count": sum(a["anchor_source"] == "frozen_v3_review" for a in anchors),
            "v4_pixel_anchor_count": sum(a["anchor_source"] == "v4_pixel_anchor" for a in anchors),
            "binding_count": len(bindings), "unresolved_anchor_count": len(unresolved_anchors),
            "bound_entity_count": len(bound_ids), "unresolved_entity_count": len(unresolved),
            "tracked_candidate_native_frames": len(candidate_frames),
            "tracked_candidate_entity_frames": len(identity_frames), "dense_visual_identity_frames_added": 0},
        "annotator_sha256": digest(__file__), "training_allowed": False, "human_label_valid": False,
        "full_annotation_complete": False, "allowed_as_model_input": False,
        "scope_zh": "身份仅在视觉锚点确认；其余连续point键传播是逐帧候选。缺测即断开，不合并跨轨迹身份，不生成交互或6D真值。"}
    save(output / "entity_grounding" / (uid + ".json"), doc)
    return doc


def export_grounding(source, base_root, v3_root, output, selected=None):
    source, base_root, v3_root, output = map(lambda p: Path(p).resolve(), (source, base_root, v3_root, output))
    if output in (source, base_root, v3_root):
        raise ValueError("V4 output must be a new independent root")
    base_path, overlay_path = base_root / "dataset_index.json", v3_root / "mamba_overlay_index.json"
    base, overlay = read(base_path), read(overlay_path)
    shared = {"base_index_sha256": digest(base_path), "v3_overlay_sha256": digest(overlay_path),
              "split_manifest_sha256": base["split_manifest_sha256"]}
    if overlay["base_dataset_index_sha256"] != shared["base_index_sha256"]:
        raise ValueError("V3 overlay and base index differ")
    overlay_entries = {e["sequence_id"]: e for e in overlay["sequences"]}
    for entry in base["sequences"]:
        uid = entry["sequence_id"]
        if selected and uid not in selected:
            continue
        doc = ground_sequence(source, base_root, v3_root, output, entry, overlay_entries[uid], shared)
        print(json.dumps({"sequence_id": uid, "coverage": doc["coverage"]}), flush=True)
    if digest(base_path) != shared["base_index_sha256"] or digest(overlay_path) != shared["v3_overlay_sha256"]:
        raise ValueError("Source index changed during grounding")
    totals, entries = Counter(), []
    for path in sorted((output / "entity_grounding").glob("*.json")):
        doc = read(path)
        if any(doc["source_bindings"][k] != v for k, v in shared.items()):
            raise ValueError("Existing V4 shard has different source bindings")
        totals.update(doc["coverage"])
        entries.append({"sequence_id": doc["sequence_id"], "split": doc["split"],
                        "path": str(path.relative_to(output)), "sha256": digest(path), **doc["coverage"]})
    report = {"schema_version": SCHEMA, "source_bindings": shared, "sequence_count": len(entries),
        "sequences_with_bindings": sum(e["binding_count"] > 0 for e in entries), "totals": dict(totals),
        "sequences": entries, "created_at": datetime.now(timezone.utc).isoformat(), "propagation_config": CONFIG,
        "training_allowed": False, "human_label_valid": False, "full_annotation_complete": False,
        "allowed_as_model_input": False}
    save(output / "entity_grounding_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--v3-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence-id", action="append")
    args = parser.parse_args()
    report = export_grounding(args.source, args.base_root, args.v3_root, args.output, args.sequence_id)
    print(json.dumps({"sequence_count": report["sequence_count"], "totals": report["totals"]}))


if __name__ == "__main__":
    main()
