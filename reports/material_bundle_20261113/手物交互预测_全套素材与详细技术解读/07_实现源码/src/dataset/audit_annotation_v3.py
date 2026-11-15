"""Audit evidence-backed assistant silver reviews without turning them into gold."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


STATUSES = {"supported", "contradicted", "uncertain"}
FROZEN_BASE_INDEX_SHA256 = "ae3c08b2590966d949b20ac1224c6b176a63e5b3f91c5355caf8a6477370c981"
FROZEN_SELECTION_SHA256 = "e8d7d57f03cb2c9e251af36f3a86055ca03dd2fe66e65e458b198ad6376b368b"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_gold(value):
    if isinstance(value, dict):
        require(value.get("human_label_valid") is not True, "assistant record promoted to human gold")
        require(value.get("training_allowed") is not True, "annotation review bypasses training gate")
        require(value.get("annotation_source") != "human", "assistant review claims human source")
        if value.get("silver_label_valid") is True:
            require(value.get("status", "supported") == "supported", "unsupported assessment promoted to silver")
        for child in value.values():
            reject_gold(child)
    elif isinstance(value, list):
        for child in value:
            reject_gold(child)


def audit_evidence(root, source, uid, *, verify_video=False):
    root, source = Path(root), Path(source)
    path = root / "evidence" / uid / "manifest.json"
    manifest = json.loads(path.read_text())
    proposal_path = source / "joint_annotations" / uid / "proposal.json"
    proposal = json.loads(proposal_path.read_text())
    metadata = json.loads((source / "sequences" / uid / "metadata.json").read_text())
    bindings = manifest["source_bindings"]
    require(bindings["proposal_sha256"] == digest(proposal_path), "evidence proposal changed")
    require(bindings["hands_sha256"] == digest(source / "sequences" / uid / "hands.npz"), "evidence hands changed")
    require(bindings["track_manifest_sha256"] == digest(source / "native_tracks" / uid / "tracklets.json"), "evidence tracks changed")
    require(bindings["source_selection_sha256"] == proposal["source_selection_sha256"], "evidence selection mismatch")
    require(bindings["video_sha256"] == metadata["video_sha256"], "evidence video binding mismatch")
    if verify_video:
        require(digest(metadata["video_path"]) == bindings["video_sha256"], "actual source video changed")
    source_intervals = {row["source_interval_id"]: row for row in proposal["source_semantic_intervals"]}
    interval_ids = [row["source_interval_id"] for row in manifest["intervals"]]
    require(len(interval_ids) == len(set(interval_ids)) and set(interval_ids) == set(source_intervals), "evidence interval coverage mismatch")
    frames = {row["frame_index"]: row for row in manifest["frames"]}
    for row in manifest["frames"]:
        require(digest(root / row["path"]) == row["sha256"], "full-resolution frame evidence changed")
    for row in manifest["pages"]:
        require(digest(root / row["path"]) == row["sha256"], "page evidence changed")
    for row in manifest["intervals"]:
        original = source_intervals[row["source_interval_id"]]
        start, end = original["start_frame"], original["end_frame"]
        expected = sorted({int(round(start + (end - 1 - start) * f)) for f in [0, .25, .5, .75, 1]})
        require(row["start_frame"] == start and row["end_frame"] == end, "evidence changes source interval")
        require(row["frames"] == expected and set(expected) <= set(frames), "native evidence sample missing")
        require(row["source_interval_id"] in manifest["pages"][row["page_index"]]["interval_ids"], "evidence page mapping mismatch")
    return {"sequence_id": uid, "status": "passed", "intervals": len(source_intervals),
            "frames": len(frames), "pages": len(manifest["pages"]), "manifest_sha256": digest(path),
            "source_video_sha256_verified": verify_video}


def audit_review(root, source, path):
    root, source, path = Path(root), Path(source), Path(path)
    review = json.loads(path.read_text())
    uid = review["sequence_id"]
    evidence_path = root / "evidence" / uid / "manifest.json"
    evidence = json.loads(evidence_path.read_text())
    proposal_path = source / "joint_annotations" / uid / "proposal.json"
    proposal = json.loads(proposal_path.read_text())
    require(review["schema_version"] == "assistant_annotation_v3_review", "wrong review schema")
    require(review["source_bindings"]["evidence_manifest_sha256"] == digest(evidence_path), "evidence manifest SHA mismatch")
    require(review["source_bindings"]["proposal_sha256"] == digest(proposal_path) == evidence["source_bindings"]["proposal_sha256"], "proposal SHA mismatch")
    require(review.get("reviewer_agent") and review.get("reviewed_at"), "missing actual reviewer identity/time")
    reject_gold(review)
    interval_by_id = {row["source_interval_id"]: row for row in proposal["source_semantic_intervals"]}
    evidence_by_id = {row["source_interval_id"]: row for row in evidence["intervals"]}
    pages = {row["path"]: row for row in evidence["pages"]}
    frame_files = {row["path"]: row for row in evidence.get("frames", [])}
    source_entities = {row["id"] for row in proposal["entities"]}
    seen, counts = set(), Counter()
    for row in review["interval_reviews"]:
        iid = row["source_interval_id"]
        require(iid in interval_by_id and iid not in seen, "missing/duplicate/unknown interval ID")
        seen.add(iid)
        require(row["review_status"] == "assistant_reviewed", "not actually visually reviewed")
        require(row["annotation_source"] == "assistant_visual_review", "visual-review source missing")
        require(row["evidence_scope"] == "sampled_native_frames_only", "sampled evidence scope expanded")
        require(row.get("observations_zh"), "missing actual visual finding")
        actual_frames = set()
        for item in row["evidence"]:
            evidence_image = pages.get(item["gallery_path"], frame_files.get(item["gallery_path"]))
            require(evidence_image is not None, "unknown evidence image path")
            if item["gallery_path"] in pages:
                require(iid in evidence_image["interval_ids"], "gallery does not contain reviewed interval")
            else:
                require(item["frames"] == [evidence_image["frame_index"]], "full image evidence claims other frames")
            require(item["gallery_sha256"] == evidence_image["sha256"] == digest(root / evidence_image["path"]), "gallery SHA mismatch")
            frames = item["frames"]
            require(frames and set(frames) <= set(evidence_by_id[iid]["frames"]), "unviewable claimed native evidence frame")
            actual_frames.update(frames)
        require(actual_frames, "no bound visual evidence")
        interval = interval_by_id[iid]
        require(all(interval["start_frame"] <= frame < interval["end_frame"] for frame in actual_frames), "evidence outside source interval")
        for assessment in row.get("operation_assessments", []):
            require(assessment["status"] in STATUSES, "unknown assessment status")
            require(assessment.get("silver_label_valid") is not True or assessment["status"] == "supported", "uncertain/contradicted operation promoted to silver")
            require(assessment.get("reason_zh"), "missing operation evidence reasoning")
            require(set(assessment.get("evidence_frames", [])) <= actual_frames, "operation cites unviewed evidence")
            if assessment.get("silver_label_valid"):
                require(assessment.get("evidence_frames"), "silver operation lacks actual frame evidence")
            counts["operation_" + assessment["status"]] += 1
        visual_ids = set()
        for entity in row.get("visual_entities", []):
            require(entity["entity_id"].startswith("visual_"), "visual-only entity ID collides with geometry")
            require(entity["geometry_status"] == "unavailable", "visual-only entity invents metric geometry")
            require(entity["entity_id"] not in visual_ids, "duplicate visual entity")
            visual_ids.add(entity["entity_id"])
            if entity.get("silver_label_valid"):
                # This is an instance-existence declaration within the reviewed
                # samples, not a per-frame visibility label. A declared frame
                # scope must still be a subset of the actually viewed images.
                if entity.get("evidence_frames"):
                    require(set(entity["evidence_frames"]) <= actual_frames,
                            "visual entity silver cites unviewed frames")
                else:
                    require(entity.get("identity_scope") == "sampled_visible_instance_only",
                            "visual entity without point support expands existence scope")
            for role in entity.get("role_candidates", []):
                if isinstance(role, dict) and role.get("silver_label_valid"):
                    require(role.get("evidence_frames") and set(role["evidence_frames"]) <= actual_frames,
                            "visual entity role cites unviewed frames")
        entities = source_entities | visual_ids | {"left_hand", "right_hand", "both_hands"}
        for assessment in row.get("entity_assessments", []):
            require(assessment["object_id"] in source_entities | visual_ids, "unknown entity assessment reference")
            require(assessment["status"] in STATUSES, "unknown entity assessment status")
            if assessment.get("silver_label_valid"):
                require(assessment.get("evidence_frames") and set(assessment["evidence_frames"]) <= actual_frames,
                        "entity silver lacks actual frame evidence")
            for role in assessment.get("role_candidates", []):
                if isinstance(role, dict) and role.get("silver_label_valid"):
                    require(set(role.get("evidence_frames", [])) <= actual_frames and role.get("evidence_frames"),
                            "role silver lacks actual frame evidence")
            counts["entity_" + assessment["status"]] += 1
        for layer in ["relations", "transitions", "cooperations"]:
            for item in row.get(layer, []):
                require(item.get("annotation_source", "assistant_visual_review") == "assistant_visual_review", "physical silver source missing")
                if item.get("source"):
                    require(item["source"] in entities, "unknown physical source entity")
                if item.get("target"):
                    require(item["target"] in entities, "unknown physical target entity")
                for reference in ["object_id", "secondary_object_id"]:
                    if item.get(reference):
                        require(item[reference] in entities, "unknown physical object reference")
                if item.get("subject"):
                    require(item["subject"] in {"left_hand", "right_hand", "both_hands", "object", "unknown"},
                            "unknown physical subject")
                for participant in item.get("participants", []):
                    identity = participant.get("entity_id") if isinstance(participant, dict) else participant
                    require(identity in entities, "unknown cooperation participant")
                support = set(item.get("evidence_frames", []))
                require(support and support <= actual_frames, "physical label evidence not actually reviewed")
                start = item.get("start_frame", item.get("frame_start"))
                end = item.get("end_frame", item.get("frame_end"))
                if item.get("temporal_coverage") == "sampled_points":
                    if layer == "transitions":
                        require(item.get("silver_label_valid") is not True and item.get("status") == "uncertain"
                                and item.get("censored") is True,
                                "transition cannot masquerade as isolated state snapshots")
                        boundary = item.get("boundary_range", item.get("boundary_frame_range"))
                        require(boundary and len(boundary) == 2 and all(type(f) is int for f in boundary)
                                and interval["start_frame"] <= boundary[0] < boundary[1] <= interval["end_frame"],
                                "unknown transition has invalid uncertainty range")
                    if start is not None or end is not None:
                        require(type(start) is int and type(end) is int and interval["start_frame"] <= start < end <= interval["end_frame"], "invalid sampled-point enclosing range")
                        require(all(start <= frame < end for frame in support), "sample point outside enclosing range")
                    counts[layer + ("_silver" if item.get("silver_label_valid") else "_unknown")] += 1
                    counts[layer + "_sampled_frames"] += len(support)
                    continue
                require(type(start) is int and type(end) is int and interval["start_frame"] <= start < end <= interval["end_frame"], "invalid physical evidence interval")
                require(all(start <= frame < end for frame in support), "physical evidence outside declared support range")
                if item.get("silver_label_valid") is True:
                    require(item.get("status", "supported") == "supported", "uncertain physical label promoted to silver")
                    # Sparse photos support only observed instants, unless the
                    # record explicitly records a censored uncertainty range.
                    if layer != "transitions":
                        require(all(frame in actual_frames for frame in range(start, end)), "silver state extrapolated between sampled frames")
                    else:
                        require(item.get("censored") is True or end - start == 1,
                                "transition precise boundary invented from sparse frames")
                counts[layer + ("_silver" if item.get("silver_label_valid") else "_unknown")] += 1
        counts["reviewed_intervals"] += 1
    require(seen == set(interval_by_id), "sequence has source intervals without a visual review outcome")
    return {"sequence_id": uid, "status": "passed", "review_sha256": digest(path), "counts": dict(counts)}


def check_entity_refs(record, entity_ids):
    """Unknown may be null, but a named object must resolve to this acquisition."""
    for key in ("source", "target", "source_entity_id", "target_entity_id", "object_id", "secondary_object_id"):
        value = record.get(key)
        if value is not None:
            require(value in entity_ids or value in {"left_hand", "right_hand", "both_hands", "object", "unknown"},
                    f"dangling physical entity reference {key}={value}")
    for participant in record.get("participants", []):
        value = participant.get("entity_id") if isinstance(participant, dict) else participant
        require(value in entity_ids or value == "both_hands", "dangling cooperation participant")


def check_silver_record(record, intervals, expected):
    require(record.get("annotation_source") == "assistant_visual_review"
            and record.get("review_status") == "assistant_reviewed"
            and record.get("status") == "supported" and record.get("silver_label_valid") is True,
            "unreviewed record promoted to silver event")
    require(record.get("temporal_coverage") == "sampled_points", "silver event expanded beyond sampled points")
    frames = record.get("evidence_frames", [])
    require(frames and len(frames) == len(set(frames)), "silver event missing/duplicate evidence frames")
    source_ids = record.get("source_interval_ids", [])
    require(source_ids and set(source_ids) <= set(intervals), "silver event has unknown source interval")
    reviewed = {f for iid in source_ids for image in intervals[iid].get("evidence", []) for f in image["frames"]}
    require(set(frames) <= reviewed, "silver event cites unreviewed native frame")
    original = expected.get(record.get("annotation_id"))
    require(original is not None, "silver event invented or interval-only action promoted to frame label")
    aliases = {("relation", "grasp"): "grasp_control", ("cooperation", "bimanual"): "bimanual_joint"}
    expected_type = aliases.get((record["layer"], original["type"]), original["type"])
    require(record["type"] == expected_type and frames == original["evidence_frames"], "silver event changes source evidence support")
    if expected_type != original["type"]:
        require(record.get("original_type") == original["type"] and record.get("type_normalization"),
                "review type alias lacks explicit provenance")
    if record["layer"] == "operation":
        require(original.get("temporal_coverage") == "sampled_points", "interval-only operation promoted to point target")
    else:
        require(record.get("source") == original.get("source") and record.get("target") == original.get("target"),
                "directed physical graph changed during integration")
        source, target = record.get("source"), record.get("target")
        if source in {"left_hand", "right_hand", "both_hands"} and target is not None:
            require(record.get("subject") == source and record.get("object_id") == target, "hand target binding inconsistent")
        elif source is not None and target is not None:
            require(record.get("subject") == "object" and record.get("object_id") == source
                    and record.get("secondary_object_id") == target, "two-object directed binding collapsed")
        if record["layer"] == "cooperation":
            participants = [p.get("entity_id") if isinstance(p, dict) else p
                            for p in original.get("participants", [])]
            hands = {p for p in participants if p in {"left_hand", "right_hand"}}
            objects = {p for p in participants if p not in {"left_hand", "right_hand", "both_hands"}}
            require(record.get("participants", []) == original.get("participants", []),
                    "cooperation loses reviewed participants")
            if objects:
                require(set(record.get("participant_object_ids", [])) == objects,
                        "cooperation loses known object participant bindings")
                if len(objects) == 1:
                    require(record.get("object_id") in objects,
                            "known cooperation object mapped to unknown")
            if original.get("subject") in {None, "unknown"} and source is None and hands:
                subject = "both_hands" if len(hands) == 2 else next(iter(hands))
                require(record.get("subject") == subject, "cooperation loses hand participant bindings")


def filtered_geometry_counts(entities, exclusions):
    excluded = {row["object_id"] for row in exclusions if row["scope"] == "entity_until_reidentified"}
    counts = Counter(geometry_excluded_entities=len(excluded))
    for entity in entities:
        if entity.get("kind") != "object" or str(entity["entity_id"]).startswith("visual_"):
            continue
        for key in ("pose_valid_frames", "surface_valid_frames", "surface_observation_count"):
            count = int(entity.get(key, 0))
            require(count >= 0, "negative geometry coverage")
            counts["before_filter_" + key] += count
            if entity["entity_id"] not in excluded:
                counts["retained_candidate_" + key] += count
    counts["retained_candidate_pose_sequences"] = int(counts["retained_candidate_pose_valid_frames"] > 0)
    counts["retained_candidate_surface_sequences"] = int(counts["retained_candidate_surface_valid_frames"] > 0)
    return counts


def audit_video_ledger(root, source, expected_ids):
    """Reuse a retained full-byte video hash pass without silently rehashing it."""
    root, source = Path(root), Path(source)
    path = root / "source_video_sha_ledger.json"
    ledger = json.loads(path.read_text())
    rows = ledger["records"]
    require(ledger["schema_version"] == "assistant_annotation_v3_source_video_sha_ledger"
            and ledger["hash_algorithm"] == "sha256", "invalid video SHA ledger")
    require(ledger["failed"] == 0 and ledger["checked_sequences"] == len(rows)
            == len(expected_ids), "source video SHA pass incomplete")
    require({r["sequence_id"] for r in rows} == set(expected_ids), "video SHA ledger sequence set mismatch")
    for record in rows:
        uid = record["sequence_id"]
        metadata_path = source / "sequences" / uid / "metadata.json"
        manifest_path = root / "evidence" / uid / "manifest.json"
        metadata = json.loads(metadata_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        require(digest(metadata_path) == record["metadata_sha256"]
                and digest(manifest_path) == record["evidence_manifest_sha256"], "video ledger source binding changed")
        require(record["status"] == "passed" and record["actual_sha256"] == record["expected_sha256"]
                == record["evidence_expected_sha256"] == metadata["video_sha256"]
                == manifest["source_bindings"]["video_sha256"], "video ledger SHA mismatch")
        require(record["video_path"] == metadata["video_path"], "video ledger path changed")
        stat = Path(record["video_path"]).stat()
        require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]),
                "video changed since retained full-byte SHA verification")
    return {"path": str(path), "sha256": digest(path), "checked_sequences": len(rows),
            "verified_at": ledger["completed_at"], "full_video_bytes_rehashed_in_current_audit": False,
            "current_size_mtime_and_bindings_match": True}


def audit_delivery(root, source, base_root):
    """Independently bind the final overlay to components and the audited v2 index."""
    root, source, base_root = map(Path, (root, source, base_root))
    read = lambda path: json.loads(Path(path).read_text())
    base_path, overlay_path = base_root / "dataset_index.json", root / "mamba_overlay_index.json"
    base, overlay = read(base_path), read(overlay_path)
    require(digest(base_path) == FROZEN_BASE_INDEX_SHA256, "frozen v2 index changed")
    require(base["source_selection_sha256"] == FROZEN_SELECTION_SHA256, "wrong source selection")
    require(overlay["base_dataset_index_sha256"] == digest(base_path), "overlay base index SHA mismatch")
    require(overlay["base_split_manifest_sha256"] == base["split_manifest_sha256"]
            == digest(base_root / base["split_manifest_path"]), "overlay split SHA mismatch")
    require(overlay.get("base_numeric_arrays_modified") is False and overlay.get("base_windows_modified") is False,
            "overlay claims to modify immutable v2 inputs")
    reject_gold(overlay)
    for split in base["splits"].values():
        require(digest(base_root / split["window_path"]) == split["sha256"], "frozen v2 windows changed")
    by_id = {row["sequence_id"]: row for row in base["sequences"]}
    video_ledger = audit_video_ledger(root, source, by_id)
    review_paths = {}
    for review_path in (root / "reviews").glob("*/*.json"):
        review_uid = read(review_path)["sequence_id"]
        require(review_uid not in review_paths, "multiple review files for final acquisition")
        review_paths[review_uid] = review_path
    overlay_ids = [row["sequence_id"] for row in overlay["sequences"]]
    require(len(overlay_ids) == len(set(overlay_ids)) == 300 and set(overlay_ids) == set(by_id), "overlay omits or duplicates frozen sequences")
    require({p.stem for p in (root / "annotations").glob("*.json")} == set(by_id), "final annotation file set mismatch")
    counts, classes, results = Counter(), Counter(), []
    for entry in overlay["sequences"]:
        uid, frozen = entry["sequence_id"], by_id[entry["sequence_id"]]
        try:
            path = root / entry["annotation_path"]
            final = read(path)
            entity_path, event_path = root / "entity_annotations" / (uid + ".json"), root / "interaction_annotations" / (uid + ".json")
            entities, events = read(entity_path), read(event_path)
            reviewed_document = read(review_paths[uid])
            raw_reviews = {r["source_interval_id"]: r for r in reviewed_document["interval_reviews"]}
            proposal_path = source / "joint_annotations" / uid / "proposal.json"
            proposal = read(proposal_path)
            require(digest(path) == entry["sha256"], "final annotation SHA differs from overlay")
            require(final["schema_version"] == "assistant_annotation_v3" and final["sequence_id"] == uid, "final annotation identity mismatch")
            require(final["split"] == entry["split"] == frozen["split"] and final["frame_count"] == frozen["frame_count"], "final annotation changes split or native timeline")
            bindings = final["source_bindings"]
            require(bindings["base_sequence_sha256"] == frozen["sha256"]["sequence"], "final base sequence binding mismatch")
            require(bindings["base_metadata_sha256"] == frozen["sha256"]["metadata"]
                    == digest(base_root / frozen["metadata_path"]), "base metadata changed")
            require(bindings["entity_component_sha256"] == digest(entity_path)
                    == events["source_bindings"]["entity_annotations_sha256"], "stale entity component binding")
            require(bindings["interaction_component_sha256"] == digest(event_path), "stale event component binding")
            require(bindings["proposal_sha256"] == digest(proposal_path)
                    == entities["source_bindings"]["proposal_sha256"] == events["source_bindings"]["proposal_sha256"], "final proposal SHA mismatch")
            require(entities["source_bindings"]["mamba_sequence_sha256"] == frozen["sha256"]["sequence"], "entity component binds different numeric source")
            require(entities["source_bindings"]["source_selection_sha256"] == FROZEN_SELECTION_SHA256,
                    "entity component uses another source selection")
            require(events["source_bindings"]["v2_metadata_sha256"] == frozen["sha256"]["metadata"],
                    "event component binds different metadata")
            for reference in entities.get("review_records", []):
                require(digest(reference["path"]) == reference["sha256"], "entity component uses stale review")
            reject_gold(final)
            require(final.get("allowed_as_model_input") is False, "offline annotations enabled as model input")
            require(final["entities"] == entities["entities"], "final entities diverge from reviewed component")
            for key in ("interval_annotations", "relations", "transitions", "cooperations", "review_records"):
                require(final[key] == events[key], f"final {key} differs from reviewed component")
            ids = [e["entity_id"] for e in final["entities"]]
            require(len(ids) == len(set(ids)), "duplicate final entity ID")
            entity_ids = set(ids)
            reviewed_visual_ids = {e["entity_id"] for r in raw_reviews.values() for e in r.get("visual_entities", [])}
            require(entity_ids == {e["id"] for e in proposal["entities"]} | reviewed_visual_ids
                    | {"left_hand", "right_hand"}, "final entity set omits reviewed identity or invents new identity")
            for entity in final["entities"]:
                if str(entity["entity_id"]).startswith("visual_"):
                    require(entity["geometry_status"] == "unavailable" and entity.get("pose_valid_frames", 0) == 0
                            and entity.get("surface_valid_frames", 0) == 0 and not entity.get("tracklets"), "visual-only entity invents 3D geometry")
            original_intervals = {r["source_interval_id"]: r for r in proposal["source_semantic_intervals"]}
            intervals = {r["source_interval_id"]: r for r in final["interval_annotations"]}
            require(len(intervals) == len(final["interval_annotations"]) and set(intervals) == set(original_intervals), "final source interval coverage mismatch")
            expected_silver = {}
            for iid, interval in intervals.items():
                original = original_intervals[iid]
                require((interval["start_frame"], interval["end_frame"]) == (original["start_frame"], original["end_frame"]), "final source interval changed")
                require(interval.get("visual_evidence_reviewed") is True and interval.get("evidence_scope") == "sampled_native_frames_only", "source interval not actually visually reviewed")
                require(interval.get("visual_operation_assessments", []) == raw_reviews[iid].get("operation_assessments", [])
                        and interval.get("evidence") == raw_reviews[iid]["evidence"], "integration changes actual operation review or evidence")
                require(interval.get("visual_entity_assessments", []) == raw_reviews[iid].get("entity_assessments", []),
                        "integration changes actual entity assessment")
                for i, assessment in enumerate(interval.get("visual_operation_assessments", [])):
                    if assessment.get("silver_label_valid") and assessment.get("temporal_coverage") == "sampled_points" and assessment.get("evidence_frames"):
                        expected_silver[f"visual_{iid}_operation_{i}"] = assessment
            for layer in ("relations", "transitions", "cooperations"):
                raw_physical = {f"visual_{iid}_{layer}_{i}": r
                                for iid, checked in raw_reviews.items() for i, r in enumerate(checked.get(layer, []))}
                for record in final[layer]:
                    check_entity_refs(record, entity_ids)
                    if record.get("annotation_source") == "assistant_visual_review":
                        raw = raw_physical.pop(record["annotation_id"], None)
                        require(raw is not None, "physical event invented outside actual review")
                        for key in ("type", "source", "target", "evidence_frames", "status", "silver_label_valid", "participants"):
                            require(record.get(key) == raw.get(key), "physical event changes actual review finding")
                    if record.get("silver_label_valid"):
                        expected_silver[record["annotation_id"]] = record
                require(not raw_physical, "reviewed physical events omitted during integration")
            actual_silver = final["silver_event_records"]
            require(len(actual_silver) == len(expected_silver), "final frame-silver records omitted or overpromoted")
            seen_silver = set()
            for record in actual_silver:
                require(record["annotation_id"] not in seen_silver, "duplicate final silver record")
                seen_silver.add(record["annotation_id"])
                check_entity_refs(record, entity_ids)
                check_silver_record(record, intervals, expected_silver)
                classes[record["layer"] + ":" + record["type"]] += 1
            exclusions = final.get("geometry_exclusions", [])
            expected_exclusions = {a["object_id"] for r in raw_reviews.values() for a in r.get("entity_assessments", [])
                                   if a.get("status") == "contradicted" and a.get("geometry_exclusion_scope") == "entity_until_reidentified"}
            component_exclusions = {e["entity_id"] for e in entities["entities"] for a in e.get("visual_assessments", [])
                                    if a.get("status") == "contradicted" and a.get("geometry_exclusion_scope") == "entity_until_reidentified"}
            require(component_exclusions == expected_exclusions, "entity component drops explicit reviewed geometry exclusion")
            require({e["object_id"] for e in exclusions} == expected_exclusions <= entity_ids, "wrong geometry exclusion identities")
            counts.update(filtered_geometry_counts(final["entities"], exclusions))
            coverage = final["coverage"]
            require(coverage["source_interval_count"] == coverage["source_intervals_visually_reviewed"]
                    == entry["source_interval_count"] == entry["visually_reviewed_interval_count"] == len(intervals), "final interval coverage overclaim")
            require(coverage["continuous_frame_annotation_complete"] is False and coverage["full_annotation_complete"] is False, "sparse review misrepresented as dense completion")
            counts.update(sequences=1, native_frames=final["frame_count"], source_intervals=len(intervals), silver_records=len(actual_silver))
            results.append({"sequence_id": uid, "status": "passed", "annotation_sha256": digest(path)})
        except Exception as error:
            results.append({"sequence_id": uid, "status": "failed", "error": str(error)})
    failures = [r for r in results if r["status"] != "passed"]
    return {"schema_version": "assistant_annotation_v3_delivery_audit", "overlay_sha256": digest(overlay_path),
            "base_index_sha256": digest(base_path), "checked_sequences": len(results), "failed": len(failures),
            "source_video_verification": video_ledger,
            "counts": dict(counts), "silver_counts_by_type": dict(sorted(classes.items())), "records": results,
            "format_ready": not failures, "training_ready": False, "full_annotation_complete": False,
            "geometry_counting_scope_zh": "逐实体有效候选帧数求和，排除明确误框实体；不是独立帧数、经验证准确姿态或因果观测。",
            "base_numeric_verification": "Bound to previously independently audited frozen index; unchanged split/windows/metadata rehashed; large NPZ arrays not re-read."}


def audit(root, source, *, verify_video=False):
    root, source = Path(root), Path(source)
    expected = {path.parent.name for path in (source / "joint_annotations").glob("*/proposal.json")}
    expected_intervals = sum(len(json.loads(path.read_text())["source_semantic_intervals"])
                             for path in (source / "joint_annotations").glob("*/proposal.json"))
    evidence_rows = []
    for uid in sorted(expected):
        if not (root / "evidence" / uid / "manifest.json").exists():
            continue
        try:
            evidence_rows.append(audit_evidence(root, source, uid, verify_video=verify_video))
        except Exception as error:
            evidence_rows.append({"sequence_id": uid, "status": "failed", "error": str(error)})
    evidence_failures = [row for row in evidence_rows if row["status"] != "passed"]
    paths = sorted((root / "reviews").glob("*/*.json"))
    rows, seen, counts = [], set(), Counter()
    for path in paths:
        try:
            row = audit_review(root, source, path)
            require(row["sequence_id"] not in seen, "duplicate reviewer ownership of sequence")
            seen.add(row["sequence_id"])
            counts.update(row["counts"])
        except Exception as error:
            row = {"path": str(path.relative_to(root)), "status": "failed", "error": str(error)}
        rows.append(row)
    failures = [row for row in rows if row["status"] != "passed"]
    return {"schema_version": "assistant_annotation_v3_audit", "generated_at": datetime.now(timezone.utc).isoformat(),
            "auditor_sha256": digest(__file__), "expected_sequences": len(expected),
            "expected_source_intervals": expected_intervals, "checked_reviews": len(rows), "passed": len(rows) - len(failures),
            "failed": len(failures), "reviewed_sequences": len(seen), "remaining_sequences": sorted(expected - seen),
            "counts": dict(counts), "records": rows,
            "evidence_checked_sequences": len(evidence_rows), "evidence_failed": len(evidence_failures),
            "evidence_source_intervals": sum(row.get("intervals", 0) for row in evidence_rows),
            "evidence_native_frames": sum(row.get("frames", 0) for row in evidence_rows),
            "evidence_pages": sum(row.get("pages", 0) for row in evidence_rows),
            "evidence_records": evidence_rows,
            "all_source_intervals_have_visual_review_outcome": seen == expected and not failures and not evidence_failures,
            "full_annotation_complete": False, "human_label_valid": False, "training_allowed": False,
            "scope_zh": "审核记录及实际图像SHA、原生证据帧和未知/银标门禁一致性；采样帧覆盖不是逐帧精确标注。"}


def review_window_expectations(timestamps, annotation):
    """Derive eligible evidence targets from actual timestamp differences."""
    import numpy as np

    timestamps = np.asarray(timestamps, dtype=np.float64)
    require(np.isfinite(timestamps).all() and np.all(np.diff(timestamps) > 0), "invalid native timestamps")
    dt = float(np.median(np.diff(timestamps)))
    candidates, skipped = {}, Counter()
    for index, record in enumerate(annotation["silver_event_records"]):
        if (record.get("layer") not in {"operation", "relation", "cooperation"}
                or record.get("silver_label_valid") is not True or record.get("status") != "supported"
                or record.get("annotation_source") != "assistant_visual_review"
                or record.get("review_status") != "assistant_reviewed"
                or record.get("temporal_coverage") != "sampled_points"):
            continue
        for frame in record.get("evidence_frames", []):
            require(type(frame) is int, "noninteger evidence target")
            candidates.setdefault(frame, []).append(index)
    expected = {}
    for target, indices in candidates.items():
        if not 0 <= target < len(timestamps):
            skipped["out_of_range"] += 1
            continue
        # Deliberately do not share the exporter's inverse-search helper. The
        # actual differences independently define the latest eligible anchor.
        eligible = np.flatnonzero(timestamps[target] - timestamps[:target] >= 1.0)
        if not len(eligible) or int(eligible[-1]) < 31:
            skipped["early_history"] += 1
            continue
        anchor = int(eligible[-1])
        horizon = float(timestamps[target] - timestamps[anchor])
        if horizon - 1.0 > 1.5 * dt + 1e-9:
            skipped["horizon_tolerance"] += 1
            continue
        expected[target] = {"start_frame": anchor - 31, "end_frame": anchor + 1, "anchor_frame": anchor,
                            "target_timestamp": float(timestamps[target]), "actual_horizon_s": horizon,
                            "evidence_record_indices": sorted(set(indices))}
    return expected, skipped


def audit_review_windows(root, base_root):
    """Audit the additional positive-biased windows, independently of export."""
    import numpy as np

    root, base_root = Path(root), Path(base_root)
    read = lambda path: json.loads(Path(path).read_text())
    path = root / "review_windows" / "index.json"
    index, base, overlay = read(path), read(base_root / "dataset_index.json"), read(root / "mamba_overlay_index.json")
    require(index["schema_version"] == "assistant_annotation_v3_review_windows", "wrong review-window schema")
    require(index["base_dataset_index_sha256"] == digest(base_root / "dataset_index.json")
            == FROZEN_BASE_INDEX_SHA256, "review windows bind different v2 source")
    require(index["overlay_index_sha256"] == digest(root / "mamba_overlay_index.json"), "review windows bind stale overlay")
    require(index["source_selection_sha256"] == FROZEN_SELECTION_SHA256, "review windows change source selection")
    require(index["split_manifest_sha256"] == base["split_manifest_sha256"]
            == digest(base_root / base["split_manifest_path"]), "review windows change frozen split")
    require(index.get("training_allowed") is False, "diagnostic windows bypass training gate")
    reject_gold(index)
    by_id = {r["sequence_id"]: r for r in base["sequences"]}
    annotations = {r["sequence_id"]: r for r in overlay["sequences"]}
    grouped, ids, counts = {}, set(), Counter()
    for split in ("train", "val", "test"):
        frozen = base["splits"][split]
        require(digest(base_root / frozen["window_path"]) == frozen["sha256"], "original v2 window file changed")
        entry = index["splits"][split]
        window_path = root / entry["window_path"]
        require(digest(window_path) == entry["sha256"], "review window JSONL SHA mismatch")
        rows = [json.loads(line) for line in window_path.read_text().splitlines() if line.strip()]
        require(len(rows) == entry["window_count"] and len({r["sequence_id"] for r in rows}) == entry["sequence_count"],
                "review window split counts mismatch")
        for row in rows:
            uid = row["sequence_id"]
            require(uid in by_id and row["split"] == by_id[uid]["split"] == split, "review window split leakage")
            key = (uid, row["target_frame"])
            require(key not in ids, "duplicate acquisition/evidence target window")
            ids.add(key)
            grouped.setdefault(uid, []).append(row)
            require(row["selection_policy"] == "silver_evidence_frame_positive_biased_diagnostic", "review window selection bias hidden")
            for field in ("sequence_path", "metadata_path", "events_path"):
                require(row[field] == by_id[uid][field], "review window changes frozen source path")
            require(row["source_selection_sha256"] == FROZEN_SELECTION_SHA256
                    and row["split_manifest_sha256"] == base["split_manifest_sha256"]
                    and row["sequence_sha256"] == by_id[uid]["sha256"]["sequence"], "review window source SHA mismatch")
            require(row["annotation_path"] == annotations[uid]["annotation_path"]
                    and row["annotation_sha256"] == annotations[uid]["sha256"], "review window annotation binding mismatch")
        counts["windows_" + split] = len(rows)
    require(len({r["window_id"] for rows in grouped.values() for r in rows}) == len(ids), "duplicate review window ID")
    sequence_entries = index["sequences"]
    require(len({r["sequence_id"] for r in sequence_entries}) == len(sequence_entries), "duplicate review manifest sequence")
    for entry in sequence_entries:
        uid = entry["sequence_id"]
        require(uid in by_id and entry["annotation_sha256"] == annotations[uid]["sha256"]
                and entry["sequence_sha256"] == by_id[uid]["sha256"]["sequence"]
                and entry["metadata_sha256"] == by_id[uid]["sha256"]["metadata"], "review window sequence binding mismatch")
    skipped = Counter()
    for uid, frozen in by_id.items():
        entry = annotations[uid]
        annotation_path = root / entry["annotation_path"]
        require(digest(annotation_path) == entry["sha256"], "annotation changed after review window selection")
        with np.load(base_root / frozen["sequence_path"], allow_pickle=False) as numeric:
            timestamps = numeric["timestamps"]
        expected, omissions = review_window_expectations(timestamps, read(annotation_path))
        skipped.update(omissions)
        rows = grouped.get(uid, [])
        require({r["target_frame"] for r in rows} == set(expected), "eligible evidence targets omitted or non-evidence target included")
        for row in rows:
            target = expected[row["target_frame"]]
            for key in ("start_frame", "end_frame", "anchor_frame", "evidence_record_indices"):
                require(row[key] == target[key], "review window history or evidence mapping mismatch")
            for key in ("target_timestamp", "actual_horizon_s"):
                require(abs(row[key] - target[key]) <= 1e-12, "review window time/horizon mismatch")
        counts["sequences_checked"] += 1
        counts["sequences_with_review_windows"] += int(bool(rows))
    return {"schema_version": "assistant_annotation_v3_review_window_audit", "generated_at": datetime.now(timezone.utc).isoformat(),
            "auditor_sha256": digest(__file__), "index_sha256": digest(path), "overlay_sha256": digest(root / "mamba_overlay_index.json"),
            "failed": 0, "counts": dict(counts), "total_windows": len(ids), "excluded_evidence_targets": dict(skipped),
            "all_eligible_evidence_targets_covered": True, "format_ready": True, "training_allowed": False,
            "scope_zh": "逐窗口核验实际原生时基、最大满足一秒间隔的历史锚点、32帧历史、明确银标证据点、固定划分和SHA；新增窗口按正例选择，有选择偏置，不替代原评测窗口。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--verify-source-videos", action="store_true")
    parser.add_argument("--base-root", type=Path, help="Also audit the fully assembled v3 overlay against frozen v2")
    parser.add_argument("--review-windows-only", action="store_true", help="Audit only the additional evidence-target window index")
    args = parser.parse_args()
    if args.review_windows_only:
        require(args.base_root is not None, "review windows require --base-root")
        result = audit_review_windows(args.root, args.base_root)
        (args.root / "review_window_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))
        return
    result = audit(args.root, args.source, verify_video=args.verify_source_videos)
    if args.base_root:
        try:
            result["delivery"] = audit_delivery(args.root, args.source, args.base_root)
        except Exception as error:
            result["delivery"] = {"failed": 1, "error": str(error), "format_ready": False, "training_ready": False}
    (args.root / "annotation_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: (dict(v, records="omitted") if k == "delivery" and "records" in v else v)
                      for k, v in result.items() if k not in {"records", "remaining_sequences", "evidence_records"}}, ensure_ascii=False))
    if result["failed"] or result["evidence_failed"] or result.get("delivery", {}).get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
