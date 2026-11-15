"""Evidence-preserving entity, visibility, role and geometry-gap annotations.

Text rules produce candidates only. Visual confirmation is imported exclusively
from separately authored, image-bound assistant review records. Source identities
and geometry are never merged or modified here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from src.dataset.complete_egodemo import digest, save


SCHEMA = "assistant_annotation_v3_entities"
ROLES = {"hand", "tool", "workpiece", "container", "support", "unknown"}
ROLE_RULES = {
    "tool": r"\b(scissors?|knif(?:e|ves)|cutter|screwdriver|stapler|scanner|marker|pen|shovel|scoop|spoon|tongs|brush|eraser|wrench|tape dispenser)\b|剪刀|刀具|螺丝刀|订书机|扫描器|记号笔|铲子|勺子|夹钳|刷子|橡皮",
    "container": r"\b(box(?:es)?|bags?|baskets?|cups?|mugs?|bowls?|bottles?|jars?|containers?|trays?|envelopes?|drawers?|kettle)\b|盒|箱|袋|篮|杯|碗|瓶|罐|容器|托盘|信封|抽屉|水壶",
    "support": r"\b(table|countertop|shel(?:f|ves)|rack|stand|floor|wall|board)\b|桌|台面|书架|架子|地面|墙|白板",
}
MATERIAL_RULES = {
    "paper": ("纸质材料", r"\b(paper|cardboard|tissue|napkin|carton)\b|纸|纸板"),
    "textile": ("织物", r"\b(cloth|fabric|towel|rag|garment)\b|布|织物|毛巾|抹布|衣物"),
    "flexible_line": ("柔性线状物", r"\b(cable|cord|rope|string|lanyard|wire)\b|电缆|绳|挂绳|电线"),
    "polymer_film": ("薄膜/袋材", r"\b(plastic bag|bubble wrap|film|wrapper)\b|塑料袋|气泡膜|薄膜"),
    "granular": ("粉末/颗粒", r"\b(powder|flour|grain|rice|sand)\b|粉末|面粉|颗粒|米粒"),
    "liquid": ("液体", r"\b(liquid|water|oil)\b|液体|清水|食用油"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    boundary = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [{"start_frame": int(a), "end_frame": int(b)}
            for a, b in zip(np.flatnonzero(boundary == 1), np.flatnonzero(boundary == -1))]


def text_candidates(entity):
    text = " ".join(str(entity.get(key, "")) for key in ["label", "name_zh"])
    roles = []
    for role, pattern in ROLE_RULES.items():
        match = re.search(pattern, text, re.I)
        if match:
            roles.append({"role": role, "annotation_source": "source_text", "review_status": "estimated",
                          "source_quote": match.group(), "silver_label_valid": False,
                          "human_label_valid": False,
                          "reason_zh": "名称只提示可能用途；具体区间角色仍需视觉确认"})
    material = []
    for code, (name, pattern) in MATERIAL_RULES.items():
        match = re.search(pattern, text, re.I)
        if match:
            material.append({"type": code, "name_zh": name, "source_quote": match.group(),
                             "annotation_source": "source_text", "review_status": "estimated",
                             "silver_label_valid": False, "human_label_valid": False})
    return roles, material


def mention_candidates(phrase, entities):
    # Only candidate retrieval: even an exact name does not establish identity.
    ignore = {"the", "a", "an", "new", "old", "another", "empty", "full", "small", "large",
              "black", "white", "blue", "red", "green", "yellow", "orange", "pink", "clear", "plastic"}
    def tokens(value):
        return {word.rstrip("s") for word in re.findall(r"[a-z]+", value.lower())} - ignore
    target = tokens(phrase)
    if not target:
        return []
    matches = []
    for entity in entities:
        words = tokens(entity.get("label", ""))
        if words and (target <= words or words <= target):
            matches.append(entity["id"])
    return matches


def geometry_status(entity, pose_count, surface_count, metric_allowed):
    if not metric_allowed:
        return "blocked_calibration", "像素标定不一致，三维派生禁用"
    if pose_count:
        return "pose_and_surface_candidate" if surface_count else "pose_candidate", "候选几何可用，独立精度及身份尚待确认"
    if surface_count:
        if entity.get("rigidity") == "deformable":
            return "surface_only", "非刚体仅以可见表面表示，整物体6D不适用"
        return "surface_only_pose_missing", "表面候选存在，但缺少通过过滤的刚体/部件6D"
    return "unavailable", "没有通过现有过滤的三维观测，不补造几何"


def verify_review(review, proposal_sha, output):
    if review.get("schema_version") != "assistant_annotation_v3_review":
        raise ValueError("Unknown assistant review schema")
    if review.get("source_bindings", {}).get("proposal_sha256") != proposal_sha:
        raise ValueError("Review proposal SHA mismatch")
    if not review.get("reviewer_agent") or not review.get("reviewed_at"):
        raise ValueError("Review requires agent identity and timestamp")
    for row in review.get("interval_reviews", []):
        if row.get("review_status") != "assistant_reviewed":
            continue
        if row.get("annotation_source") != "assistant_visual_review" or not row.get("evidence"):
            raise ValueError("Visual review requires bound evidence")
        for evidence in row["evidence"]:
            path = Path(evidence["gallery_path"])
            if not path.is_absolute():
                path = output / path
            if digest(path) != evidence["gallery_sha256"] or not evidence.get("frames"):
                raise ValueError("Review image hash or native evidence frames invalid")


def apply_reviews(document, reviews, output):
    by_id = {entity["entity_id"]: entity for entity in document["entities"]}
    intervals = {row["source_interval_id"]: row for row in document["interval_annotations"]}
    review_refs = []
    for review_path in reviews:
        review = read_json(review_path)
        if review.get("sequence_id") != document["sequence_id"]:
            continue
        verify_review(review, document["source_bindings"]["proposal_sha256"], output)
        review_refs.append({"path": str(review_path), "sha256": digest(review_path)})
        for row in review.get("interval_reviews", []):
            identity = row["source_interval_id"]
            if identity not in intervals:
                raise ValueError("Reviewed interval is absent from original source")
            if row.get("review_status") != "assistant_reviewed":
                continue
            interval = intervals[identity]
            interval["review_status"] = "assistant_reviewed"
            interval["evidence_scope"] = row.get("evidence_scope", "sampled_native_frames_only")
            interval["evidence"] = row["evidence"]
            interval["observations_zh"] = row.get("observations_zh", "")
            interval["entity_assessments"] = row.get("entity_assessments", [])
            for entity in row.get("visual_entities", []):
                new_id = entity["entity_id"]
                if not new_id.startswith("visual_"):
                    raise ValueError("New visible entities require visual_ namespace")
                if new_id not in by_id:
                    record = {**entity, "entity_id": new_id, "id": new_id,
                              "kind": "object", "geometry_status": "unavailable", "tracklets": [],
                              "pose_valid_frames": 0, "surface_valid_frames": 0,
                              "annotation_source": "assistant_visual_review", "review_status": "assistant_reviewed",
                              "source_interval_id": identity, "evidence": row["evidence"],
                              "silver_label_valid": True, "human_label_valid": False,
                              "identity_scope": "sampled_visible_instance_only"}
                    by_id[new_id] = record
                    document["entities"].append(record)
            for assessment in row.get("entity_assessments", []):
                object_id = assessment.get("object_id")
                if object_id not in by_id:
                    raise ValueError("Reviewed entity ID absent from frozen entities")
                annotation = {**assessment, "source_interval_id": identity,
                              "evidence": row["evidence"], "reviewer_agent": review["reviewer_agent"],
                              "reviewed_at": review["reviewed_at"], "annotation_source": "assistant_visual_review",
                              "review_status": "assistant_reviewed", "human_label_valid": False}
                # Rejected/unresolved assessments do not create a positive entity label.
                annotation["silver_label_valid"] = assessment.get("status") == "supported"
                by_id[object_id].setdefault("visual_assessments", []).append(annotation)
    document["review_records"] = review_refs
    document["coverage"]["source_intervals_visually_reviewed"] = sum(
        row["review_status"] == "assistant_reviewed" for row in intervals.values())
    document["coverage"]["visually_confirmed_entity_interval_records"] = sum(
        assessment.get("status") == "supported"
        for row in intervals.values() for assessment in row.get("entity_assessments", []))
    return document


def annotate_sequence(source, mamba_root, output, sequence_id, review_paths=()):
    source, mamba_root, output = Path(source), Path(mamba_root), Path(output)
    proposal_path = source / "joint_annotations" / sequence_id / "proposal.json"
    proposal = read_json(proposal_path)
    metadata_path = mamba_root / "sequences" / sequence_id / "metadata.json"
    data_path = mamba_root / "sequences" / sequence_id / "sequence.npz"
    metadata = read_json(metadata_path)
    entities, gaps = [], []
    with np.load(data_path, allow_pickle=False) as z:
        frame_count = len(z["timestamps"])
        hand_valid = z["hand_valid"]
        object_ids = z["object_ids"].tolist()
        observed, poses = z["object_observed"], z["object_pose_valid"]
        sf, so, sp = z["surface_frame_index"], z["surface_object_index"], z["surface_point_index"]
        point_keys = z["point_keys"].tolist()
        point_objects = z["point_object_index"]
        for hand_index, (identity, name) in enumerate([("left_hand", "左手"), ("right_hand", "右手")]):
            entities.append({"entity_id": identity, "id": identity, "kind": "hand", "name_zh": name,
                             "anatomical_side": ["left", "right"][hand_index], "roles": ["hand"],
                             "observability": {"valid_frames": int(hand_valid[:, hand_index].sum()),
                                               "observed_intervals": runs(hand_valid[:, hand_index]),
                                               "unobserved_intervals": runs(~hand_valid[:, hand_index])},
                             "geometry_status": "official_hand_annotation", "annotation_source": "source_annotation",
                             "review_status": "estimated", "silver_label_valid": False, "human_label_valid": False})
        original = {entity["id"]: entity for entity in proposal["entities"]}
        for oi, identity in enumerate(object_ids):
            raw = original[identity]
            rows = so == oi
            surface_valid = np.zeros(frame_count, bool)
            surface_valid[sf[rows]] = True
            pose_valid = poses[:, oi]
            roles, materials = text_candidates(raw)
            status, explanation = geometry_status(raw, int(pose_valid.sum()), int(surface_valid.sum()),
                                                  metadata["metric_geometry_allowed"] and raw.get("metric_geometry_allowed", False))
            point_ids = np.flatnonzero(point_objects == oi)
            tracks = {}
            for pi in point_ids:
                track = point_keys[pi].split("/", 1)[0]
                tracks.setdefault(track, []).append(int(pi))
            fragments = []
            for track, keys in sorted(tracks.items()):
                valid = np.zeros(frame_count, bool)
                selected = rows & np.isin(sp, keys)
                valid[sf[selected]] = True
                fragments.append({"tracklet_id": track, "point_key_count": len(keys),
                                  "visible_intervals": runs(valid), "observation_count": int(selected.sum()),
                                  "identity_status": "continuous_track_candidate", "cross_occlusion_verified": False})
            entity = {**raw, "entity_id": identity, "kind": "object", "roles": [],
                      "role_candidates": roles, "material_candidates": materials,
                      "source_rigidity": raw["rigidity"], "material_review_status": "estimated",
                      "identity_fragments": fragments, "cross_occlusion_identity_verified": False,
                      "identity_scope": "source_tracklet_and_existing_geometric_link_candidates",
                      "name_based_identity_merge_performed": False,
                      "observability": {"definition": "quality-filtered geometry candidates, not visual existence labels",
                          "observed_frames": int(observed[:, oi].sum()), "observed_intervals": runs(observed[:, oi]),
                          "unobserved_intervals": runs(~observed[:, oi]),
                          "pose_valid_intervals": runs(pose_valid), "surface_valid_intervals": runs(surface_valid)},
                      "geometry_status": status, "geometry_explanation_zh": explanation,
                      "pose_valid_frames": int(pose_valid.sum()), "surface_valid_frames": int(surface_valid.sum()),
                      "surface_point_key_count": len(point_ids), "surface_observation_count": int(rows.sum()),
                      "annotation_source": "geometry_rule", "review_status": "estimated",
                      "silver_label_valid": False, "human_label_valid": False,
                      "allowed_as_model_input": False}
            entities.append(entity)
            if not pose_valid.any() or not surface_valid.any():
                gaps.append({"entity_id": identity, "status": status, "reason_zh": explanation,
                             "whole_object_6d_applicable": raw["rigidity"] == "rigid",
                             "pose_scope": raw.get("pose_scope", "unknown"),
                             "missing_pose_is_not_annotation_failure": raw["rigidity"] == "deformable"})
    intervals = []
    for interval in proposal.get("source_semantic_intervals", []):
        bindings = []
        for index, mention in enumerate(interval.get("object_mentions", [])):
            phrase = mention.get("source_phrase", "")
            bindings.append({"mention_index": index, "source_phrase": phrase, "name_zh": mention.get("name_zh"),
                             "role": "unknown", "entity_id": None,
                             "candidate_entity_ids": mention_candidates(phrase, proposal["entities"]),
                             "resolution_status": "unresolved", "annotation_source": "source_text",
                             "silver_label_valid": False, "human_label_valid": False})
        intervals.append({"source_interval_id": interval["source_interval_id"],
                          "start_frame": interval["start_frame"], "end_frame": interval["end_frame"],
                          "description_zh": interval.get("description_zh"),
                          "entity_bindings": bindings, "review_status": "estimated",
                          "annotation_source": "source_text", "human_label_valid": False})
    doc = {"schema_version": SCHEMA, "sequence_id": sequence_id,
           "source_bindings": {"proposal_path": str(proposal_path), "proposal_sha256": digest(proposal_path),
                               "mamba_sequence_path": str(data_path), "mamba_sequence_sha256": digest(data_path),
                               "mamba_metadata_path": str(metadata_path), "mamba_metadata_sha256": digest(metadata_path),
                               "source_selection_sha256": proposal["source_selection_sha256"]},
           "entities": entities, "interval_annotations": intervals, "geometry_gaps": gaps,
           "coverage": {"frame_count": frame_count, "source_object_count": len(object_ids),
                        "source_interval_count": len(intervals), "source_intervals_processed": len(intervals),
                        "source_intervals_visually_reviewed": 0,
                        "visually_confirmed_entity_interval_records": 0,
                        "source_rigidity_unknown_entities": sum(e.get("source_rigidity") == "unknown" for e in entities)},
           "review_records": [], "training_allowed": False, "human_label_valid": False,
           "full_annotation_complete": False, "allowed_as_model_input": False,
           "annotator_version": "entities_native_observability_v1", "annotator_sha256": digest(__file__)}
    apply_reviews(doc, review_paths, output)
    save(output / "entity_annotations" / f"{sequence_id}.json", doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mamba-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-id", action="append")
    parser.add_argument("--apply-reviews", action="store_true")
    args = parser.parse_args()
    uids = sorted(args.sequence_id or [p.name for p in (args.source / "joint_annotations").iterdir() if p.is_dir()])
    coverage = Counter()
    statuses = Counter()
    review_paths = list((args.output / "reviews").glob("*/*.json")) if args.apply_reviews else []
    for index, uid in enumerate(uids):
        selected = [path for path in review_paths if path.stem == uid]
        doc = annotate_sequence(args.source, args.mamba_root, args.output, uid, selected)
        coverage.update(doc["coverage"])
        statuses.update(entity.get("geometry_status") for entity in doc["entities"] if entity["kind"] == "object")
        print(json.dumps({"done": index + 1, "total": len(uids), "sequence_id": uid}), flush=True)
    save(args.output / "entity_annotation_report.json", {"schema_version": SCHEMA, "sequence_count": len(uids),
         "coverage": dict(coverage), "geometry_status_counts": dict(statuses),
         "created_at": datetime.now(timezone.utc).isoformat(), "training_allowed": False,
         "human_label_valid": False, "full_annotation_complete": False})


if __name__ == "__main__":
    main()
