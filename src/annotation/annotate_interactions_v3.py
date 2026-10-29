"""Evidence-tiered interaction annotation for all 300 EgoDemo acquisitions.

Source intent, estimated geometry, and actually reviewed visual evidence remain
separate. This module neither certifies training nor overwrites v2 artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "assistant_annotation_v3_interactions"
LAYERS = ("relations", "transitions", "cooperations")
SOURCE_LAYER = {"relation": "relations", "transition": "transitions", "cooperation": "cooperations"}
RELEASE_RE = re.compile(r"\b(?:release(?:s|d|ing)?|let\s+go\s+of)\b", re.I)
ROLE_WORDS = {
    "tool": re.compile(r"\b(?:screwdriver|scissors|hammer|scanner|knife|drill|scoop|spoon|brush|tape dispenser)\b", re.I),
    "container": re.compile(r"\b(?:box|basket|bag|bowl|cup|container|bin|jar|bottle)\b", re.I),
    "support": re.compile(r"\b(?:table|desk|workbench|shelf|tray|floor|countertop)\b", re.I),
}
EXACT_VERBS = {
    "hold": re.compile(r"(?:holds?|held|holding|stabiliz(?:e|es|ed|ing)|steady|steadies|steadied|steadying)", re.I),
    "wipe": re.compile(r"(?:wipe|wipes|wiped|wiping|erase|erases|erased|erasing|scrub|scrubs|scrubbed|scrubbing)", re.I),
    "arrange": re.compile(r"(?:arrange|arranges|arranged|arranging|organiz(?:e|es|ed|ing)|organis(?:e|es|ed|ing))", re.I),
}


def source_operation_quality(source):
    """Audit literal source evidence; keep rejected originals for traceability."""
    description = source.get("description_original", "")
    output = []
    for operation in source.get("operations", []):
        valid, rejected = [], []
        for evidence in operation.get("text_evidence", []):
            token = evidence["text"]
            a, b = evidence["span"]
            before, after = description[:a], description[b:]
            reason = None
            if description[a:b] != token:
                reason = "原文span不匹配"
            elif operation["type"] in EXACT_VERBS and not EXACT_VERBS[operation["type"]].fullmatch(token):
                reason = "名词或不合法动词词形，不是动作"
            elif token.lower() == "wrapping" and re.match(r"\s+(?:paper|film)\b", after, re.I):
                reason = "wrapping paper/film是物体名词短语"
            elif token.lower() == "stack" and re.match(r"\s+of\b", after, re.I):
                reason = "stack of是物体集合名词，不是堆叠动作"
            elif re.match(r"plugs?\b", token, re.I) and re.search(r"\b(?:the|a|an|these|those|multiple|several|some)\s+(?:\w+\s+){0,2}$", before, re.I):
                reason = "限定词后的plug是插头名词，不是插接动作"
            elif token.lower() in ("installation", "assembly", "inspection", "slot"):
                reason = "名词表示目标/部件，不独立证明该区间动作"
            elif token.lower() == "cutting" and re.match(r"\s+slot\b", after, re.I):
                reason = "cutting slot是切割槽部件名词"
            elif token.lower() in ("wipe", "wipes") and re.search(r"\bwet\s+$", before, re.I):
                reason = "wet wipes是湿巾名词，不是擦拭动作"
            elif token.lower().endswith("ed") and re.search(r"\b(?:the|a|an|these|those)\s+(?:\w+\s+){0,2}$", before, re.I):
                reason = "名词前已完成状态形容词，不是本区间动作"
            elif token.lower() in ("opening", "folding") and re.search(r"\b(?:the|a|an)\s+(?:\w+\s+){0,2}$", before, re.I):
                reason = "限定词后的名词/形容词用法，动作证据不足"
            (rejected if reason else valid).append({**evidence, **({"reason_zh": reason} if reason else {})})
        # Older synthetic fixtures may not carry spans; keep them unresolved as
        # source-only candidates, never infer a visual label from this branch.
        false_match = bool(operation.get("text_evidence")) and not valid
        output.append({**operation, "source_text_quality": "rejected_false_match" if false_match else
                       ("explicit_verb_evidence" if valid else "source_candidate_without_span"),
                       "original_text_evidence": operation.get("text_evidence", []),
                       "validated_text_evidence": valid, "rejected_text_evidence": rejected,
                       "active_weak_candidate": not false_match})
    additions = [("cap", "盖帽", r"\b(?:put|place|fit|replace)\s+(?:the\s+)?caps?\s+(?:back\s+)?on\b"),
                 ("insert", "穿入/插入", r"\bthread\s+[^.;]{0,100}\bthrough\b")]
    for code, name, pattern in additions:
        match = re.search(pattern, description, re.I)
        if match and not any(o["type"] == code and o["active_weak_candidate"] for o in output):
            evidence = {"text": match.group(), "span": list(match.span())}
            output.append({"type": code, "name_zh": name, "source_quote": match.group(),
                           "text_evidence": [evidence], "validated_text_evidence": [evidence],
                           "original_text_evidence": [], "rejected_text_evidence": [],
                           "source_text_quality": "explicit_phrase_added_v3",
                           "active_weak_candidate": True, "label_source": "source_text_rule_v3"})
    return output


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def role_candidates(mention, description):
    """Noun roles are text hypotheses, never physical edge confirmations."""
    phrase = mention.get("source_phrase", "")
    candidates = [role for role, rule in ROLE_WORDS.items() if rule.search(phrase)]
    return candidates or ["workpiece_or_manipulated_object"]


def _unresolved_layers(reason):
    return {layer: {"status": "unresolved", "silver_label_valid": False,
                    "human_label_valid": False, "reason_zh": reason}
            for layer in ("physical_contact", "grasp_control", "support", "transition_boundaries", "cooperation")}


def interval_baseline(source, bindings=None):
    a, b = int(source["start_frame"]), int(source["end_frame"])
    interval_id = source["source_interval_id"]
    mentions = source.get("object_mentions", [])
    bindings = bindings or []
    roles = []
    for i, mention in enumerate(mentions):
        binding = next((x for x in bindings if x.get("mention_index") == i), {})
        roles.append({"mention_index": i, "source_phrase": mention.get("source_phrase"),
                      "name_zh": mention.get("name_zh"),
                      "entity_id": binding.get("entity_id"),
                      "candidate_entity_ids": binding.get("candidate_entity_ids", []),
                      "resolution_status": binding.get("resolution_status", "unresolved"),
                      "role_candidates": role_candidates(mention, source.get("description_original", "")),
                      "annotation_source": "source_text", "review_status": "estimated",
                      "silver_label_valid": False, "human_label_valid": False})
    operations = [{**operation, "annotation_source": "source_text", "review_status": "estimated",
                   "subject": None, "object_id": None, "secondary_object_id": None,
                   "temporal_scope": [a, b], "temporal_precision": "source_interval_only",
                   "silver_label_valid": False, "human_label_valid": False}
                  for operation in source_operation_quality(source)]
    # Preserve source intent as text-role graph candidates, with explicit mention
    # references. A noun pair is not yet a visible physical relation.
    intent_edges = []
    tool_indices = [r["mention_index"] for r in roles if "tool" in r["role_candidates"]]
    for i in tool_indices:
        for j in range(len(roles)):
            if i == j or "support" in roles[j]["role_candidates"]:
                continue
            intent_edges.append({"type": "possible_tool_workpiece_roles", "source": roles[i]["entity_id"],
                                 "target": roles[j]["entity_id"], "secondary_object_id": roles[j]["entity_id"],
                                 "source_mention_refs": [f"{interval_id}/{i}", f"{interval_id}/{j}"],
                                 "annotation_source": "source_text", "review_status": "estimated",
                                 "physical_relation_asserted": False, "silver_label_valid": False,
                                 "human_label_valid": False})
    return {"source_interval_id": interval_id, "source_event_id": source.get("source_event_id"),
            "start_frame": a, "end_frame": b,
            "description_original": source.get("description_original"),
            "description_zh": source.get("description_zh"),
            "processing_status": "processed", "review_status": "estimated",
            "annotation_source": "source_text", "operations": operations,
            "entity_bindings": roles, "intent_graph_candidates": intent_edges,
            "physical_layer_observability": _unresolved_layers("原文描述任务意图；尚无逐帧视觉证据认证实际物理关系。"),
            "visual_evidence_reviewed": False, "evidence": [],
            "silver_label_valid": False, "human_label_valid": False}


def geometry_candidates(proposal, intervals, frame_count):
    result = {layer: [] for layer in LAYERS}
    for i, original in enumerate(proposal.get("interactions", [])):
        a, b = int(original["start_frame"]), int(original["end_frame"])
        if not 0 <= a < b <= frame_count:
            raise ValueError("Geometry candidate outside native timeline")
        overlaps = [x for x in intervals if x["start_frame"] < b and x["end_frame"] > a]
        layer = SOURCE_LAYER[original["layer"]]
        record = {**original, "annotation_id": f"geometry_{i:05d}",
                  "annotation_source": "geometry_rule", "review_status": "estimated",
                  "source": original.get("subject"), "target": original.get("object_id"),
                  "secondary_object_id": original.get("secondary_object_id"),
                  "source_interval_ids": [x["source_interval_id"] for x in overlaps],
                  "silver_label_valid": False, "human_label_valid": False,
                  "physical_relation_verified": False,
                  "reason_zh": "距离或共运动仅为几何候选，不能单独证明接触、抓持或双手协作。"}
        if layer == "transitions":
            record["raw_geometry_candidate_range"] = [a, b]
            # Preserve the estimated trigger, but do not present its one-frame
            # range as a measured transition boundary.
            record["boundary_range"] = ([min(x["start_frame"] for x in overlaps),
                                         max(x["end_frame"] for x in overlaps)]
                                        if overlaps else [0, frame_count])
            record["start_frame"], record["end_frame"] = record["boundary_range"]
            record.update(temporal_precision="unresolved_within_source_interval",
                          censored=True, transition_instant_observed=False)
        result[layer].append(record)
    return result


def source_transition_candidates(intervals):
    output = []
    for interval in intervals:
        a, b = interval["start_frame"], interval["end_frame"]
        if not any(operation.get("active_weak_candidate", True) for operation in interval["operations"]):
            continue
        kinds = ["operation_onset"]
        if RELEASE_RE.search(interval.get("description_original") or ""):
            kinds.append("release")
        for kind in kinds:
            output.append({"annotation_id": f"text_{interval['source_interval_id']}_{kind}",
                           "type": kind, "source": None, "target": None,
                           "secondary_object_id": None, "start_frame": a, "end_frame": b,
                           "boundary_range": [a, b], "source_interval_ids": [interval["source_interval_id"]],
                           "annotation_source": "source_text", "review_status": "estimated",
                           "temporal_precision": "source_interval_only", "censored": True,
                           "transition_instant_observed": False, "physical_relation_verified": False,
                           "silver_label_valid": False, "human_label_valid": False,
                           "reason_zh": "文本只支持操作意图及大致时段，起始/脱离瞬间尚未直接看到。"})
    return output


def uncovered_ranges(intervals, frame_count):
    cursor, missing = 0, []
    for a, b in sorted((x["start_frame"], x["end_frame"]) for x in intervals):
        if a > cursor:
            missing.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < frame_count:
        missing.append([cursor, frame_count])
    return missing


def validate_review(review, proposal_sha, evidence_root):
    if review.get("schema_version") != "assistant_annotation_v3_review":
        raise ValueError("Unexpected visual review schema")
    if review["source_bindings"]["proposal_sha256"] != proposal_sha:
        raise ValueError("Review refers to a different proposal")
    if not review.get("reviewer_agent") or not review.get("reviewed_at"):
        raise ValueError("Visual review lacks agent/time provenance")
    manifest_path = Path(evidence_root) / "evidence" / review["sequence_id"] / "manifest.json"
    if digest(manifest_path) != review["source_bindings"].get("evidence_manifest_sha256"):
        raise ValueError("Evidence manifest SHA mismatch")
    manifest = read_json(manifest_path)
    pages = {str((Path(evidence_root) / p["path"]).resolve()): p for p in manifest["pages"]}
    native_frames = {str((Path(evidence_root) / f["path"]).resolve()): f
                     for f in manifest.get("frames", [])}
    source_intervals = {r["source_interval_id"]: r for r in manifest["intervals"]}
    for interval in review.get("interval_reviews", []):
        if interval.get("review_status") == "assistant_reviewed":
            if interval.get("annotation_source") != "assistant_visual_review" or not interval.get("evidence"):
                raise ValueError("Reviewed interval must reference actual visual evidence")
        visual_assessments = (interval.get("operation_assessments", []) + interval.get("entity_assessments", [])
                              + [r for layer in LAYERS for r in interval.get(layer, [])])
        declared_frames = {frame for e in interval.get("evidence", []) for frame in e.get("frames", [])}
        for assessment in visual_assessments:
            if assessment.get("status") not in ("supported", "contradicted", "uncertain"):
                raise ValueError("Invalid visual assessment status")
            if assessment.get("silver_label_valid", False) and assessment["status"] != "supported":
                raise ValueError("Only supported assessments may become positive silver labels")
            if assessment.get("silver_label_valid", False) and not assessment.get("evidence_frames"):
                raise ValueError("Silver assessment requires explicit native evidence frames")
            if not set(assessment.get("evidence_frames", [])).issubset(declared_frames):
                raise ValueError("Assessment evidence is not a subset of reviewed gallery frames")
        for evidence in interval.get("evidence", []):
            path = Path(evidence["gallery_path"])
            if not path.is_absolute():
                path = Path(evidence_root) / path
            page = pages.get(str(path.resolve()))
            native = native_frames.get(str(path.resolve()))
            if page:
                if interval["source_interval_id"] not in page["interval_ids"]:
                    raise ValueError("Gallery does not contain claimed source interval")
            elif native:
                if evidence.get("frames") != [native["frame_index"]]:
                    raise ValueError("Native image evidence must name exactly its own frame")
            else:
                raise ValueError("Image is not registered in evidence manifest")
            if (page or native)["sha256"] != evidence["gallery_sha256"]:
                raise ValueError("Gallery SHA differs from evidence manifest")
            if digest(path) != evidence["gallery_sha256"]:
                raise ValueError("Evidence image SHA mismatch")
            if not evidence.get("frames"):
                raise ValueError("Evidence needs native frame indices")
            if not set(evidence["frames"]).issubset(source_intervals[interval["source_interval_id"]]["frames"]):
                raise ValueError("Claimed native frame is not on the interval evidence gallery")


def apply_visual_reviews(annotation, reviews, evidence_root):
    intervals = {r["source_interval_id"]: r for r in annotation["interval_annotations"]}
    for review in reviews:
        validate_review(review, annotation["source_bindings"]["proposal_sha256"], evidence_root)
        for checked in review.get("interval_reviews", []):
            if checked["source_interval_id"] not in intervals:
                raise ValueError("Review refers to unknown source interval")
            interval = intervals[checked["source_interval_id"]]
            if checked.get("review_status") != "assistant_reviewed":
                continue
            a, b = interval["start_frame"], interval["end_frame"]
            for evidence in checked["evidence"]:
                if any(not a <= frame < b for frame in evidence["frames"]):
                    raise ValueError("Evidence frame outside source interval")
            interval.update(review_status="assistant_reviewed", annotation_source="assistant_visual_review",
                            visual_evidence_reviewed=True, evidence=checked["evidence"],
                            observations_zh=checked.get("observations_zh", ""),
                            evidence_scope=checked.get("evidence_scope", "sampled_native_frames_only"))
            interval["visual_operation_assessments"] = checked.get("operation_assessments", [])
            interval["visual_entity_assessments"] = checked.get("entity_assessments", [])
            for field in ("handedness_assessment", "boundary_assessment"):
                if field in checked:
                    interval[field] = checked[field]
            if checked.get("corrected_description_zh"):
                interval["description_zh_before_review"] = interval.get("description_zh", "")
                interval["description_zh"] = checked["corrected_description_zh"]
                interval["description_zh_correction_source"] = "assistant_visual_review"
            interval["silver_label_valid"] = any(r.get("silver_label_valid", False)
                                                for r in checked.get("operation_assessments", []))
            for layer in LAYERS:
                for i, item in enumerate(checked.get(layer, [])):
                    evidence_frames = item.get("evidence_frames", [])
                    if any(not a <= frame < b for frame in evidence_frames):
                        raise ValueError("Physical graph evidence outside source interval")
                    entity = {**item, "annotation_id": f"visual_{checked['source_interval_id']}_{layer}_{i}",
                              "annotation_source": "assistant_visual_review", "review_status": "assistant_reviewed",
                              "source_interval_ids": [checked["source_interval_id"]],
                              "evidence": checked["evidence"], "reviewer_agent": review["reviewer_agent"],
                              "reviewed_at": review["reviewed_at"], "human_label_valid": False,
                              "temporal_precision": item.get("temporal_precision", "sampled_frames_only"),
                              "temporal_coverage": item.get("temporal_coverage", "sampled_points"),
                              "evidence_frames": evidence_frames}
                    if layer == "transitions" and not item.get("transition_instant_observed", False):
                        entity["censored"] = True
                        entity["boundary_range"] = item.get("boundary_range", [a, b])
                    annotation[layer].append(entity)
            if "physical_layer_observability" in checked:
                interval["physical_layer_observability"] = checked["physical_layer_observability"]
            annotation["review_records"].append({"source_interval_id": checked["source_interval_id"],
                                                 "reviewer_agent": review["reviewer_agent"],
                                                 "reviewed_at": review["reviewed_at"],
                                                 "evidence": checked["evidence"],
                                                 "human_label_valid": False})
    return annotation


def annotate_sequence(refine_root, v2_root, output_root, uid, review_paths=()):
    refine_root, v2_root, output_root = map(Path, (refine_root, v2_root, output_root))
    proposal_path = refine_root / "joint_annotations" / uid / "proposal.json"
    proposal = read_json(proposal_path)
    metadata = read_json(v2_root / "sequences" / uid / "metadata.json")
    frame_count = metadata["frame_count"]
    entity_path = output_root / "entity_annotations" / (uid + ".json")
    entities = read_json(entity_path) if entity_path.exists() else {}
    binding_by_interval = {r["source_interval_id"]: r.get("entity_bindings", [])
                           for r in entities.get("interval_annotations", [])}
    intervals = [interval_baseline(row, binding_by_interval.get(row["source_interval_id"]))
                 for row in proposal["source_semantic_intervals"]]
    ids = [r["source_interval_id"] for r in intervals]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicated source interval ID")
    for row in intervals:
        if not 0 <= row["start_frame"] < row["end_frame"] <= frame_count:
            raise ValueError("Source interval outside sequence")
    graph = geometry_candidates(proposal, intervals, frame_count)
    graph["transitions"].extend(source_transition_candidates(intervals))
    annotation = {"schema_version": SCHEMA, "sequence_id": uid, "frame_count": frame_count,
                  "source_bindings": {"proposal_path": str(proposal_path), "proposal_sha256": digest(proposal_path),
                                      "v2_metadata_sha256": digest(v2_root / "sequences" / uid / "metadata.json"),
                                      "entity_annotations_sha256": digest(entity_path) if entity_path.exists() else None},
                  "entities": entities.get("entities", proposal["entities"]),
                  "interval_annotations": intervals, **graph, "review_records": [],
                  "missing_label_means": "unknown", "human_label_valid": False,
                  "training_allowed": False, "allowed_as_model_input": False}
    annotation = apply_visual_reviews(annotation, [read_json(p) for p in review_paths], output_root)
    annotation["coverage"] = {
        "source_intervals_total": len(intervals), "source_intervals_processed": len(intervals),
        "source_intervals_visual_reviewed": sum(r["visual_evidence_reviewed"] for r in intervals),
        "source_operations_estimated": sum(len(r["operations"]) for r in intervals),
        "source_operation_false_matches_rejected": sum(not o.get("active_weak_candidate", True)
                                                        for r in intervals for o in r["operations"]),
        "silver_operation_assessments": sum(x.get("silver_label_valid", False) for r in intervals
                                            for x in r.get("visual_operation_assessments", [])),
        "human_valid_labels": 0,
        "unannotated_native_ranges": uncovered_ranges(intervals, frame_count),
        "physical_layers": {layer: dict(Counter("silver" if r.get("silver_label_valid") else "estimated_or_unresolved"
                                                for r in annotation[layer])) for layer in LAYERS},
        "complete_annotation": False,
    }
    write_json(output_root / "interaction_annotations" / (uid + ".json"), annotation)
    return annotation["coverage"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--uid")
    args = parser.parse_args()
    uids = [args.uid] if args.uid else sorted(p.name for p in (args.v2_root / "sequences").iterdir() if p.is_dir())
    coverage = {}
    for uid in uids:
        reviews = sorted((args.output_root / "reviews").glob(f"*/{uid}.json"))
        coverage[uid] = annotate_sequence(args.refine_root, args.v2_root, args.output_root, uid, reviews)
    report = {"schema_version": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
              "sequences_processed": len(coverage),
              "source_intervals_processed": sum(r["source_intervals_processed"] for r in coverage.values()),
              "source_intervals_visual_reviewed": sum(r["source_intervals_visual_reviewed"] for r in coverage.values()),
              "human_valid_labels": 0, "training_allowed": False, "sequences": coverage}
    report_path = (args.output_root / "interaction_annotations" / (args.uid + ".receipt.json")
                   if args.uid else args.output_root / "interaction_annotation_report.json")
    write_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != "sequences"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
