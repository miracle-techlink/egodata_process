"""Join actual entity/interaction reviews into an immutable-v2 annotation layer."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def normalize_edge(record):
    """Retain directed edges while exposing the single-target adapter fields."""
    result = dict(record)
    canonical = {("relation", "grasp"): "grasp_control",
                 ("cooperation", "bimanual"): "bimanual_joint"}.get((result.get("layer"), result.get("type")))
    if canonical:
        result["original_type"] = result["type"]
        result["type"] = canonical
        result["type_normalization"] = "v3_alias_to_frozen_vocab"
    source = result.get("source", result.get("source_entity_id"))
    target = result.get("target", result.get("target_entity_id"))
    original = {key: result.get(key) for key in ("subject", "object_id", "secondary_object_id")}
    if source in ("left_hand", "right_hand", "both_hands") and target is not None:
        result["subject"] = source
        result["object_id"] = target
    elif source is not None and target is not None:
        result["subject"] = "object"
        result["object_id"] = source
        result["secondary_object_id"] = target
    else:
        result.setdefault("object_id", target)
    if result.get("layer") == "cooperation":
        participants = [p.get("entity_id") if isinstance(p, dict) else p
                        for p in result.get("participants", [])]
        hands = {p for p in participants if p in ("left_hand", "right_hand")}
        objects = sorted({p for p in participants if isinstance(p, str)
                          and p not in ("left_hand", "right_hand", "both_hands")})
        if objects:
            result["participant_object_ids"] = objects
            if len(objects) == 1 and result.get("object_id") is None:
                result["object_id"] = objects[0]
        if result.get("subject") in (None, "unknown") and hands:
            result["subject"] = "both_hands" if len(hands) == 2 else next(iter(hands))
    if any(key in record and original[key] != result.get(key) for key in original):
        result["original_object_binding"] = original
        result["binding_normalization"] = ("canonical_directed_source_target" if source is not None and target is not None
                                           else "explicit_cooperation_participants")
    return result


def collect_silver_events(annotation):
    """Do not turn interval-level action presence into per-frame supervision."""
    result = []
    for plural, layer in [("relations", "relation"), ("transitions", "transition"),
                          ("cooperations", "cooperation")]:
        for record in annotation[plural]:
            if not record.get("silver_label_valid", False):
                continue
            if (record.get("annotation_source") != "assistant_visual_review"
                    or record.get("review_status") != "assistant_reviewed"
                    or record.get("status") != "supported"
                    or record.get("human_label_valid", False)):
                raise ValueError("Invalid promotion of unverified physical event")
            if not record.get("evidence_frames") or record.get("temporal_coverage") != "sampled_points":
                raise ValueError("Silver physical event lacks sampled evidence scope")
            result.append(normalize_edge({**record, "layer": layer}))
    review_by_interval = {r["source_interval_id"]: r for r in annotation["review_records"]}
    for interval in annotation["interval_annotations"]:
        for i, record in enumerate(interval.get("visual_operation_assessments", [])):
            if not record.get("silver_label_valid", False):
                continue
            if record.get("status") != "supported":
                raise ValueError("Unsupported operation marked silver")
            # An action can be known to occur somewhere in a source interval
            # without knowing its state at any of the sampled target frames.
            frames = record.get("evidence_frames", [])
            if not frames or record.get("temporal_coverage") != "sampled_points":
                continue
            checked_frames = {f for evidence in interval.get("evidence", []) for f in evidence["frames"]}
            if not set(frames) <= checked_frames:
                raise ValueError("Operation frame was not present in reviewed image")
            review = review_by_interval[interval["source_interval_id"]]
            result.append({**record, "annotation_id": f"visual_{interval['source_interval_id']}_operation_{i}",
                           "layer": "operation", "annotation_source": "assistant_visual_review",
                           "review_status": "assistant_reviewed", "human_label_valid": False,
                           "temporal_coverage": "sampled_points", "evidence_frames": frames,
                           "start_frame": min(frames), "end_frame": max(frames) + 1,
                           "source_interval_ids": [interval["source_interval_id"]],
                           "evidence": interval["evidence"], "reviewer_agent": review["reviewer_agent"],
                           "reviewed_at": review["reviewed_at"],
                           "subject": record.get("subject"), "object_id": record.get("object_id"),
                           "secondary_object_id": record.get("secondary_object_id")})
    return result


def deliver(root, base_root):
    root, base_root = Path(root), Path(base_root)
    base_index = read(base_root / "dataset_index.json")
    entries, totals, classes = [], Counter(), Counter()
    operation_assessments = Counter()
    for source in base_index["sequences"]:
        uid = source["sequence_id"]
        entity_path = root / "entity_annotations" / (uid + ".json")
        event_path = root / "interaction_annotations" / (uid + ".json")
        entity, events = read(entity_path), read(event_path)
        if entity["sequence_id"] != uid or events["sequence_id"] != uid:
            raise ValueError("Mismatched acquisition in component files")
        if entity["source_bindings"]["proposal_sha256"] != events["source_bindings"]["proposal_sha256"]:
            raise ValueError("Entity and interaction components use different proposals")
        if events["source_bindings"]["entity_annotations_sha256"] != digest(entity_path):
            raise ValueError("Interactions must be rebuilt after entity review changes")
        intervals = events["interval_annotations"]
        ids = [r["source_interval_id"] for r in intervals]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicated source interval")
        silver = collect_silver_events(events)
        observed = {r["source_interval_id"] for r in intervals if r.get("visual_evidence_reviewed")}
        assessed = [a for r in intervals for a in r.get("visual_operation_assessments", [])]
        uncertain = sum(r.get("status") == "uncertain" for r in assessed)
        contradicted = sum(r.get("status") == "contradicted" for r in assessed)
        operation_assessments.update(f"{r.get('status', 'unknown')}:{r['type']}" for r in assessed)
        exclusions = [{"object_id": e["entity_id"], "scope": "entity_until_reidentified",
                       "reason_zh": a.get("reason_zh"), "evidence": a.get("evidence", []),
                       "source_interval_id": a.get("source_interval_id")}
                      for e in entity["entities"] for a in e.get("visual_assessments", [])
                      if a.get("status") == "contradicted"
                      and a.get("geometry_exclusion_scope") == "entity_until_reidentified"]
        final = {"schema_version": "assistant_annotation_v3", "sequence_id": uid,
                 "frame_count": source["frame_count"], "split": source["split"],
                 "source_bindings": {"base_sequence_sha256": source["sha256"]["sequence"],
                                     "base_metadata_sha256": source["sha256"]["metadata"],
                                     "entity_component_sha256": digest(entity_path),
                                     "interaction_component_sha256": digest(event_path),
                                     "proposal_sha256": entity["source_bindings"]["proposal_sha256"]},
                 "entities": entity["entities"], "geometry_gaps": entity["geometry_gaps"],
                 "geometry_exclusions": exclusions,
                 "interval_annotations": intervals,
                 **{name: events[name] for name in ["relations", "transitions", "cooperations", "review_records"]},
                 "silver_event_records": silver,
                 "coverage": {"source_interval_count": len(intervals),
                              "source_intervals_visually_reviewed": len(observed),
                              "all_source_intervals_sampled_reviewed": len(observed) == len(intervals),
                              "uncertain_operation_assessments": uncertain,
                              "contradicted_operation_assessments": contradicted,
                              "silver_event_records": len(silver),
                              "unannotated_native_ranges": events["coverage"]["unannotated_native_ranges"],
                              "continuous_frame_annotation_complete": False,
                              "full_annotation_complete": False},
                 "human_label_valid": False, "training_allowed": False,
                 "allowed_as_model_input": False,
                 "scope_zh": "代理抽样视觉标注与原文/几何估计分层保存；不代表连续逐帧确认或人类金标准。"}
        out = root / "annotations" / (uid + ".json")
        save(out, final)
        entries.append({"sequence_id": uid, "split": source["split"],
                        "annotation_path": str(out.relative_to(root)), "sha256": digest(out),
                        "source_interval_count": len(intervals), "visually_reviewed_interval_count": len(observed)})
        totals.update(sequences=1, source_intervals=len(intervals), visually_reviewed_intervals=len(observed),
                      fully_sampled_reviewed_sequences=int(len(observed) == len(intervals)),
                      entities=len(entity["entities"]), visual_only_entities=sum(
                          str(e["entity_id"]).startswith("visual_") for e in entity["entities"]),
                      silver_event_records=len(silver), uncertain_operation_assessments=uncertain,
                      contradicted_operation_assessments=contradicted,
                      geometry_excluded_entities=len({e["object_id"] for e in exclusions}))
        classes.update(f"{r['layer']}:{r['type']}" for r in silver)
    index = {"schema_version": "assistant_annotation_v3_overlay",
             "base_dataset_root": str(base_root),
             "base_dataset_index_sha256": digest(base_root / "dataset_index.json"),
             "base_split_manifest_sha256": base_index["split_manifest_sha256"],
             "sequences": entries, "training_allowed": False, "allowed_as_model_input": False,
             "base_numeric_arrays_modified": False, "base_windows_modified": False}
    save(root / "mamba_overlay_index.json", index)
    report = {"schema_version": "assistant_annotation_v3_delivery", "generated_at": datetime.now(timezone.utc).isoformat(),
              **dict(totals), "silver_event_record_counts_by_type": dict(sorted(classes.items())),
              "visual_operation_assessment_counts_by_status_and_type": dict(sorted(operation_assessments.items())),
              "silver_event_type_count_by_layer": {
                  layer: sum(key.startswith(layer + ":") for key in classes)
                  for layer in ("operation", "relation", "transition", "cooperation")},
              "counting_scope_zh": "银标事件按记录计数，不等于独立视频数或逐帧事件数；区间操作确认另计，只有显式采样点操作进入时间目标。",
              "human_valid_labels": 0, "full_annotation_complete": False, "training_allowed": False,
              "all_source_intervals_sampled_reviewed": totals["source_intervals"] == totals["visually_reviewed_intervals"],
              "base_dataset_index_sha256": index["base_dataset_index_sha256"],
              "overlay_index_sha256": digest(root / "mamba_overlay_index.json"),
              "remaining_zh": ["抽帧之间的连续物理状态和精确边界尚未确认。",
                               "跨遮挡实例身份、visual实体与旧3D轨迹对齐仍有缺口。",
                               "几何为原离线估计，不能自动变成已验证因果预测输入。"]}
    save(root / "annotation_delivery_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.root, args.base_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
