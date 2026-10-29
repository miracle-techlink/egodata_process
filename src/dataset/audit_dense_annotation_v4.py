"""Read-only native-frame gap accounting for the independent dense V4 effort.

Scene geometry, sampled positives, source interval semantics, and continuous
review are deliberately different denominators. No missing label becomes zero.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


LAYERS = ("operation", "relation", "transition", "cooperation")
HANDS = {"left_hand", "right_hand", "both_hands", "object", "unknown", None}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def mask_ranges(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [[int(a), int(b)] for a, b in edges.reshape(-1, 2)]


def range_mask(ranges, length):
    mask = np.zeros(length, bool)
    for item in ranges:
        a, b = (item["start_frame"], item["end_frame"]) if isinstance(item, dict) else item
        require(type(a) is int and type(b) is int and 0 <= a < b <= length, "range outside native timeline")
        mask[a:b] = True
    return mask


def point_mask(frames, length):
    require(all(type(f) is int and 0 <= f < length for f in frames), "evidence point outside native timeline")
    mask = np.zeros(length, bool)
    mask[list(frames)] = True
    return mask


def entity_references(record):
    refs = {record.get(k) for k in ("object_id", "secondary_object_id", "source", "target")}
    refs.update(record.get("participant_object_ids", []))
    refs.update(p.get("entity_id") if isinstance(p, dict) else p for p in record.get("participants", []))
    return refs - HANDS


def geometry_representation(entity, excluded):
    identity = entity["entity_id"]
    if identity in excluded:
        return "unsupported_rejected_identity", "既有框/身份已被明确否定，需重新辨认；不是不存在负例。"
    if identity.startswith("visual_"):
        return "unsupported_unmapped_visual_identity", "仅有视觉实例，尚无明确旧三维ID映射。"
    material = entity.get("rigidity", entity.get("material_kind", "unknown"))
    if material == "deformable":
        return "not_applicable_whole_object_6d", "非刚体整物体6D不适用；应审查表面或部件表示。"
    return "candidate_geometry_unverified", "有效数组只表示候选几何；无观测帧仍未知，不能标不存在。"


def summarize_sequence(uid, base, overlay_entry, base_root, v3_root, vocab):
    annotation_path = v3_root / overlay_entry["annotation_path"]
    require(digest(annotation_path) == overlay_entry["sha256"], "V3 annotation SHA changed")
    annotation = read(annotation_path)
    length = int(base["frame_count"])
    require(annotation["frame_count"] == length, "native timeline mismatch")
    with np.load(base_root / base["sequence_path"], allow_pickle=False) as data:
        timestamps = data["timestamps"]
        hand = data["hand_valid"].astype(bool)
        ids = [str(i) for i in data["object_ids"]]
        pose = data["object_pose_valid"].astype(bool)
        surface = np.zeros_like(pose)
        frame_idx, object_idx = data["surface_frame_index"], data["surface_object_index"]
        surface[frame_idx, object_idx] = True
        point_count = len(frame_idx)
    require(hand.shape == (length, 2) and pose.shape == (length, len(ids)), "invalid native mask shapes")
    require(len(timestamps) == length and np.all(np.diff(timestamps) > 0), "invalid native timestamps")
    object_columns = {identity: i for i, identity in enumerate(ids)}
    excluded = {r["object_id"] for r in annotation.get("geometry_exclusions", [])
                if r["scope"] == "entity_until_reidentified"}
    rejected_point_count = int(np.isin(object_idx, [object_columns[i] for i in excluded if i in object_columns]).sum())
    for identity in excluded:
        if identity in object_columns:
            pose[:, object_columns[identity]] = False
            surface[:, object_columns[identity]] = False
    geometry = pose | surface
    entities = {r["entity_id"]: r for r in annotation["entities"] if r.get("kind") == "object"}
    intervals = {r["source_interval_id"]: r for r in annotation["interval_annotations"]}
    semantic = range_mask(list(intervals.values()), length)
    sampled = np.zeros(length, bool)
    for interval in intervals.values():
        for evidence in interval.get("evidence", []):
            sampled |= point_mask(evidence["frames"], length)
    positive = {layer: np.zeros(length, bool) for layer in LAYERS}
    by_type = {layer: defaultdict(set) for layer in LAYERS}
    by_entity = defaultdict(lambda: {layer: set() for layer in LAYERS})
    requests = set()
    physical_record_frames_without_identity = set()
    for record in annotation["silver_event_records"]:
        require(record.get("silver_label_valid") is True and record.get("status") == "supported"
                and record.get("temporal_coverage") == "sampled_points", "V3 silver scope is not sampled points")
        layer, frames = record["layer"], record["evidence_frames"]
        positive[layer] |= point_mask(frames, length)
        by_type[layer][record["type"]].update(frames)
        refs = entity_references(record)
        require(refs <= set(entities), "physical label has unresolved entity reference")
        for identity in refs:
            by_entity[identity][layer].update(frames)
        if layer in {"relation", "cooperation"}:
            if not refs:
                physical_record_frames_without_identity.update(frames)
            requests.update((f, identity) for f in frames for identity in refs)
    requested_by_frame = defaultdict(set)
    request_counts = Counter()
    for frame, identity in requests:
        requested_by_frame[frame].add(identity)
        request_counts["physical_entity_frame_requests"] += 1
        column = object_columns.get(identity)
        if column is None:
            request_counts["physical_entity_frame_requests_unmapped_visual"] += 1
        elif identity in excluded:
            request_counts["physical_entity_frame_requests_rejected_identity"] += 1
        else:
            request_counts["physical_entity_frame_requests_with_v2_id"] += 1
            request_counts["physical_entity_frame_requests_with_pose_candidate"] += int(pose[frame, column])
            request_counts["physical_entity_frame_requests_with_surface_candidate"] += int(surface[frame, column])
            request_counts["physical_entity_frame_requests_with_geometry_candidate"] += int(geometry[frame, column])
    all_requested_geometry = np.zeros(length, bool)
    for frame, refs in requested_by_frame.items():
        all_requested_geometry[frame] = frame not in physical_record_frames_without_identity and all(
            identity in object_columns and geometry[frame, object_columns[identity]] for identity in refs)
    supported_span = np.zeros(length, bool)
    operation_assessments = Counter()
    for interval in intervals.values():
        outcomes = interval.get("visual_operation_assessments", [])
        operation_assessments.update(r["status"] for r in outcomes)
        if any(r["status"] == "supported" for r in outcomes):
            supported_span[interval["start_frame"]:interval["end_frame"]] = True
    object_rows = []
    for identity, entity in entities.items():
        column = object_columns.get(identity)
        source_interval = intervals.get(entity.get("source_interval_id"))
        domain = range_mask([source_interval] if source_interval else [[0, length]], length)
        # Physical evidence can legitimately extend an explicitly tracked
        # instance; preserve those points without claiming the intervening gap.
        for frames in by_entity[identity].values():
            domain |= point_mask(sorted(frames), length)
        identity_supported, identity_rejected = set(), set()
        for assessment in entity.get("visual_assessments", []):
            if assessment.get("status") == "supported":
                identity_supported.update(assessment.get("evidence_frames", []))
            if assessment.get("status") == "contradicted":
                identity_rejected.update(assessment.get("evidence_frames", []))
        visibility_points = set(entity.get("evidence_frames", [])) | identity_supported
        visible = point_mask(sorted(visibility_points), length) & domain
        pm = np.zeros(length, bool) if column is None else pose[:, column]
        sm = np.zeros(length, bool) if column is None else surface[:, column]
        state_layers = {}
        for layer in ("relation", "transition", "cooperation"):
            positive_points = point_mask(sorted(by_entity[identity][layer]), length)
            state_layers[layer] = {
                "frames_with_any_positive": int(positive_points.sum()), "explicit_negative_frames": 0,
                "fully_classified_frames": 0,
                "frames_without_any_known_state": int((domain & ~positive_points).sum()),
                "unknown_ranges": mask_ranges(domain & ~positive_points),
                "positive_frames": sorted(by_entity[identity][layer]),
                "scope_zh": "仅表示至少一条涉及该实体的正例；同帧其他关系/主体组合仍未知。"}
        status, reason = geometry_representation(entity, excluded)
        object_rows.append({"entity_id": identity, "name_zh": entity.get("name_zh"),
            "material_kind": entity.get("rigidity", entity.get("material_kind", "unknown")),
            "pose_scope": entity.get("pose_scope", "unavailable"),
            "analysis_domain_ranges": mask_ranges(domain), "domain_does_not_assert_presence": True,
            "geometry_status": status, "geometry_reason_zh": reason,
            "v2_id_mapping": identity if column is not None else None,
            "visual_identity_mapping_status": "existing_v2_candidate_id" if column is not None else "unknown_no_explicit_mapping",
            "pose_candidate_frames": int(pm.sum()), "surface_candidate_frames": int(sm.sum()),
            "pose_candidate_ranges": mask_ranges(pm), "surface_candidate_ranges": mask_ranges(sm),
            "geometry_unobserved_unknown_ranges": [] if status.startswith("unsupported_") else mask_ranges(domain & ~(pm | sm)),
            "geometry_unsupported_ranges": mask_ranges(domain) if status.startswith("unsupported_") else [],
            "visibility_supported_points": sorted(visibility_points), "visibility_explicit_absent_frames": 0,
            "visibility_unknown_frames": int((domain & ~visible).sum()),
            "visibility_unknown_ranges": mask_ranges(domain & ~visible),
            "identity_supported_points": sorted(identity_supported),
            "identity_rejected_claim_points": sorted(identity_rejected),
            "rejected_claim_is_not_absence_negative": True, "layers": state_layers})
    any_positive = np.logical_or.reduce(list(positive.values()))
    counts = Counter(native_frames=length, hand_left_valid_frames=int(hand[:, 0].sum()),
        hand_right_valid_frames=int(hand[:, 1].sum()), hand_both_valid_frames=int(hand.all(1).sum()),
        hand_any_valid_frames=int(hand.any(1).sum()), scene_any_pose_candidate_frames=int(pose.any(1).sum()),
        scene_any_surface_candidate_frames=int(surface.any(1).sum()),
        scene_any_geometry_candidate_frames=int(geometry.any(1).sum()),
        pose_candidate_object_frames=int(pose.sum()), surface_candidate_object_frames=int(surface.sum()),
        surface_candidate_point_observations=point_count - rejected_point_count,
        v3_reviewed_sample_frames=int(sampled.sum()), v3_any_silver_positive_frames=int(any_positive.sum()),
        semantic_source_interval_frames=int(semantic.sum()), semantic_supported_action_interval_span_frames=int(supported_span.sum()),
        semantic_source_intervals=len(intervals), visual_entities=sum(i.startswith("visual_") for i in entities),
        visual_entities_with_explicit_v2_mapping=0, geometry_excluded_entities=len(excluded),
        physical_frames_all_explicit_entities_have_candidate_geometry=int(all_requested_geometry.sum()),
        physical_frames_with_explicit_entity_references=len(requested_by_frame),
        relation_explicit_negative_frames=0, transition_explicit_negative_frames=0,
        v3_confirmed_transition_boundaries=0, v3_continuously_reviewed_frames=0,
        v3_all_layers_fully_classified_frames=0)
    counts.update(request_counts)
    for entity in object_rows:
        counts["objects_material_" + entity["material_kind"]] += 1
        counts["objects_representation_" + entity["geometry_status"]] += 1
    for layer in LAYERS:
        counts["v3_" + layer + "_positive_frames"] = int(positive[layer].sum())
    unknown = ~any_positive
    return {"schema_version": "assistant_annotation_v4_dense_gap_sequence", "sequence_id": uid,
        "split": base["split"], "source_bindings": {"v2_sequence_sha256": base["sha256"]["sequence"],
            "v2_metadata_sha256": base["sha256"]["metadata"], "v3_annotation_sha256": overlay_entry["sha256"]},
        "counts": dict(counts), "source_operation_assessments": dict(operation_assessments),
        "positive_frame_counts_by_type": {layer: {kind: len(frames) for kind, frames in values.items()}
                                           for layer, values in by_type.items()},
        "frame_gaps": {"hand_left_invalid": mask_ranges(~hand[:, 0]), "hand_right_invalid": mask_ranges(~hand[:, 1]),
            "no_scene_geometry_candidate": mask_ranges(~geometry.any(1)),
            "no_v3_point_silver_of_any_layer": mask_ranges(unknown),
            "no_source_interval_semantics": mask_ranges(~semantic)},
        "entities": object_rows, "intervals": [{"source_interval_id": i, "start_frame": r["start_frame"],
            "end_frame": r["end_frame"], "description_zh": r.get("description_zh"),
            "v3_evidence_points": sorted({f for e in r.get("evidence", []) for f in e["frames"]}),
            "v3_point_positive_frames": int(any_positive[r["start_frame"]:r["end_frame"]].sum()),
            "continuous_frames_remaining": r["end_frame"] - r["start_frame"],
            "boundary_review_status": r.get("boundary_assessment"),
            "scope_zh": "源区间/区间内动作存在不等于逐帧状态。"} for i, r in intervals.items()],
        "dense_index_complete": False, "continuous_visual_review_complete": False,
        "full_annotation_complete": False, "training_allowed": False,
        "scope_zh": "这是V4开始时的V2/V3逐帧缺口基线；场景候选几何、物理边参与实体几何和当前工件完成不能互换。"}


def audit_dense_baseline(base_root, v3_root, output, workers=4):
    base_root, v3_root, output = map(Path, (base_root, v3_root, output))
    base, overlay = read(base_root / "dataset_index.json"), read(v3_root / "mamba_overlay_index.json")
    require(overlay["base_dataset_index_sha256"] == digest(base_root / "dataset_index.json"), "V3 belongs to different V2")
    require(overlay["base_split_manifest_sha256"] == base["split_manifest_sha256"], "split binding changed")
    annotations = {r["sequence_id"]: r for r in overlay["sequences"]}
    require(set(annotations) == {r["sequence_id"] for r in base["sequences"]}, "sequence coverage mismatch")
    vocab = read(base_root / "vocab.json")
    def work(entry):
        return summarize_sequence(entry["sequence_id"], entry, annotations[entry["sequence_id"]], base_root, v3_root, vocab)
    totals, classes, assessments, records, priorities = Counter(), Counter(), Counter(), [], []
    with ThreadPoolExecutor(max_workers=min(4, workers)) as executor:
        for done, report in enumerate(executor.map(work, base["sequences"]), 1):
            path = output / "dense_gap_sequences" / (report["sequence_id"] + ".json")
            write(path, report)
            totals.update(report["counts"])
            assessments.update(report["source_operation_assessments"])
            for layer, values in report["positive_frame_counts_by_type"].items():
                classes.update({layer + ":" + k: n for k, n in values.items()})
            records.append({"sequence_id": report["sequence_id"], "path": str(path.relative_to(output)), "sha256": digest(path)})
            for interval in report["intervals"]:
                frames = interval["end_frame"] - interval["start_frame"]
                priorities.append({"sequence_id": report["sequence_id"], "split": report["split"], **interval,
                    "priority_tier": 0 if interval["v3_point_positive_frames"] and frames <= 600 else (1 if frames <= 600 else 2),
                    "next_action_zh": "逐帧确认主体/客体、关系正负和转换；先补连续证据，再核当前工件几何，未见状态保留未知。"})
            if done % 25 == 0:
                print(json.dumps({"sequences_checked": done, "native_frames": totals["native_frames"]}), flush=True)
    priorities.sort(key=lambda r: (r["priority_tier"], r["continuous_frames_remaining"], r["sequence_id"], r["start_frame"]))
    priority_path = output / "dense_annotation_priority.jsonl"
    priority_path.write_text("".join(json.dumps(dict(row, priority_rank=i + 1), ensure_ascii=False) + "\n"
                                     for i, row in enumerate(priorities)))
    report = {"schema_version": "assistant_annotation_v4_dense_gap_audit", "created_at": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": digest(__file__), "source_bindings": {"v2_index_sha256": digest(base_root / "dataset_index.json"),
            "v3_overlay_sha256": digest(v3_root / "mamba_overlay_index.json"), "split_manifest_sha256": base["split_manifest_sha256"]},
        "sequence_count": len(records), "counts": dict(totals), "point_positive_frame_counts_by_type": dict(classes),
        "source_operation_assessments": dict(assessments), "sequences": records,
        "priority_path": str(priority_path.relative_to(output)), "priority_sha256": digest(priority_path),
        "priority_policy_zh": "优先有点级正例且≤600帧的区间，再短区间，再长区间；这是补标效率顺序，不修改数据划分或评测采样。",
        "unknown_semantics_zh": "unknown=未观测/未判定；rejected_claim=已有提案被否定，不是不存在负例；unsupported=缺少身份映射或错误实体停用；非刚体整物体6D另记不适用。",
        "current_workpiece_complete_frames": None,
        "current_workpiece_limitation_zh": "尚无覆盖全部帧的当前工件身份与可见性清单，因此不能从任意场景pose或少量物理边推导完成帧数。",
        "v3_interval_semantics_are_dense_frame_labels": False,
        "gap_summary_zh": [
            {"层": "左右手", "总帧": totals["native_frames"], "双手有效帧": totals["hand_both_valid_frames"],
             "至少一手缺失帧": totals["native_frames"] - totals["hand_both_valid_frames"],
             "说明": "官方有效掩码覆盖，不等于已独立逐帧验证精度。"},
            {"层": "物体几何", "任意场景位姿候选帧": totals["scene_any_pose_candidate_frames"],
             "任意场景表面候选帧": totals["scene_any_surface_candidate_frames"],
             "任意场景几何也缺失帧": totals["native_frames"] - totals["scene_any_geometry_candidate_frames"],
             "说明": "这些可能包含背景实体，不能替代当前工件完成率。"},
            {"层": "交互时序", "抽样已看帧": totals["v3_reviewed_sample_frames"],
             "至少一条点级银标帧": totals["v3_any_silver_positive_frames"],
             "完全没有点级银标帧": totals["native_frames"] - totals["v3_any_silver_positive_frames"],
             "明确关系负例帧": totals["relation_explicit_negative_frames"],
             "精确转换边界": totals["v3_confirmed_transition_boundaries"],
             "说明": "区间动作存在与抽样点不能扩展成连续状态，负例仍需明确对象与关系。"},
            {"层": "视觉到三维身份", "视觉实体": totals["visual_entities"],
             "明确旧ID映射": totals["visual_entities_with_explicit_v2_mapping"],
             "物理边实体帧请求": totals["physical_entity_frame_requests"],
             "有当帧候选几何的请求": totals["physical_entity_frame_requests_with_geometry_candidate"],
             "说明": "同名或同角色不能替代实例映射，未映射视觉实例不造三维姿态。"}],
        "dense_index_complete": False, "continuous_visual_review_complete": False,
        "full_annotation_complete": False, "training_allowed": False,
        "numeric_scope_zh": "读取全部原生hand_valid/object_pose_valid以及COO surface_frame/object_index，应用V3明确排除；未重复读取全部浮点三维坐标或更改旧源。",
        "baseline_only": True}
    write(output / "dense_gap_audit.json", report)
    return report


def validate_observable_masks(arrays, hand_valid, pose_valid, excluded_columns):
    counts = arrays["surface_point_count"]
    distance, observed = arrays["nearest_hand_surface_m"], arrays["distance_observed"]
    require(counts.shape == pose_valid.shape and np.issubdtype(counts.dtype, np.integer)
            and np.all(counts >= 0), "invalid dense surface count shape/type")
    require(np.array_equal(arrays["hand_valid"], hand_valid), "dense observables change official hand mask")
    require(np.array_equal(arrays["pose_valid"], pose_valid), "dense observables change filtered pose mask")
    require(distance.shape == observed.shape == counts.shape + (2,), "invalid hand/object distance shape")
    require(np.array_equal(np.isfinite(distance), observed) and np.isnan(distance[~observed]).all(),
            "dense distance NaN/observed mismatch")
    require(np.all(distance[observed] >= 0), "negative metric distance")
    expected_observed = (counts[:, :, None] >= 8) & hand_valid[:, None, :]
    require(np.array_equal(observed, expected_observed), "distance eligibility does not match valid hand/surface")
    require(np.array_equal(arrays["near_surface_candidate"], observed & (distance < .03)),
            "proximity candidate threshold/mask mismatch")
    for column in excluded_columns:
        require(not counts[:, column].any() and not observed[:, column].any() and not pose_valid[:, column].any(),
                "rejected entity reappears in dense observables")


def audit_dense_observables(base_root, v3_root, output):
    base_root, v3_root, output = map(Path, (base_root, v3_root, output))
    base, overlay = read(base_root / "dataset_index.json"), read(v3_root / "mamba_overlay_index.json")
    report_path = output / "dense_observables_report.json"
    delivery = read(report_path)
    require(delivery["base_index_sha256"] == digest(base_root / "dataset_index.json")
            and delivery["v3_overlay_sha256"] == digest(v3_root / "mamba_overlay_index.json"), "dense index source mismatch")
    base_entries = {r["sequence_id"]: r for r in base["sequences"]}
    annotation_entries = {r["sequence_id"]: r for r in overlay["sequences"]}
    rows = delivery["records"]
    require(len(rows) == len(base_entries) == len({r["sequence_id"] for r in rows})
            and {r["sequence_id"] for r in rows} == set(base_entries), "dense native acquisition coverage mismatch")
    counts, results = Counter(), []
    for row in rows:
        uid = row["sequence_id"]
        directory = output / "dense_sequences" / uid
        metadata_path = directory / "metadata.json"
        metadata = read(metadata_path)
        require(digest(metadata_path) == row["metadata_sha256"], "dense metadata SHA mismatch")
        entry = base_entries[uid]
        annotation_path = v3_root / annotation_entries[uid]["annotation_path"]
        require(digest(annotation_path) == annotation_entries[uid]["sha256"], "dense source annotation changed")
        annotation = read(annotation_path)
        require(metadata["source_bindings"]["base_sequence_sha256"] == entry["sha256"]["sequence"]
                and metadata["source_bindings"]["v3_annotation_sha256"] == annotation_entries[uid]["sha256"], "dense sequence source mismatch")
        require(metadata["split"] == entry["split"] and metadata["training_allowed"] is False
                and metadata["full_annotation_complete"] is False, "dense numeric processing overclaims completion")
        require(metadata["thresholds"] == {"minimum_surface_points": 8, "proximity_m": .03}, "dense thresholds changed")
        require(digest(directory / "observables.npz") == metadata["sha256"]["observables"]
                and digest(directory / "boundary_candidates.json") == metadata["sha256"]["boundary_candidates"], "dense artifact SHA mismatch")
        with np.load(directory / "observables.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(base_root / entry["sequence_path"], allow_pickle=False) as z:
            ids, times, hand, pose = z["object_ids"].tolist(), z["timestamps"], z["hand_valid"], z["object_pose_valid"].copy()
        excluded = {r["object_id"] for r in annotation.get("geometry_exclusions", []) if r["scope"] == "entity_until_reidentified"}
        require(set(metadata["geometry_excluded_object_ids"]) == excluded, "dense exclusions differ from frozen review")
        columns = [i for i, identity in enumerate(ids) if identity in excluded]
        pose[:, columns] = False
        require(np.array_equal(arrays["frame_index"], np.arange(len(times)))
                and np.array_equal(arrays["timestamps"], times) and arrays["object_ids"].tolist() == ids,
                "dense native timeline/entity order changed")
        validate_observable_masks(arrays, hand, pose, columns)
        pairs = sorted({(frame, i) for i, event in enumerate(annotation["silver_event_records"]) for frame in event["evidence_frames"]})
        frame_counts = np.bincount([f for f, _ in pairs], minlength=len(times))
        require(np.array_equal(arrays["v3_silver_frame_ptr"], np.r_[0, np.cumsum(frame_counts)])
                and np.array_equal(arrays["v3_silver_record_index"], [i for _, i in pairs]), "sampled silver expanded or dropped in dense index")
        boundaries = read(directory / "boundary_candidates.json")["records"]
        expected_boundary = set()
        observed, near = arrays["distance_observed"], arrays["near_surface_candidate"]
        contiguous = np.diff(times) <= 1.5 * np.median(np.diff(times))
        for column, identity in enumerate(ids):
            for h, subject in enumerate(("left_hand", "right_hand")):
                transition = observed[1:, column, h] & observed[:-1, column, h] & contiguous & (near[1:, column, h] != near[:-1, column, h])
                expected_boundary.update((identity, subject, int(frame), "proximity_enter" if near[frame, column, h] else "proximity_exit")
                                         for frame in np.flatnonzero(transition) + 1)
        actual_boundary = set()
        for event in boundaries:
            require(event["annotation_source"] == "geometry_rule" and event["silver_label_valid"] is False
                    and event["human_label_valid"] is False, "proximity automatically promoted to physical label")
            actual_boundary.add((event["object_id"], event["subject"], event["candidate_boundary_frame"], event["candidate_type"]))
        require(len(actual_boundary) == len(boundaries) and actual_boundary == expected_boundary, "proximity candidate boundary mismatch")
        baseline = read(output / "dense_gap_sequences" / (uid + ".json"))["counts"]
        require(int(arrays["surface_point_count"].sum()) == baseline["surface_candidate_point_observations"]
                and int((arrays["surface_point_count"] > 0).sum()) == baseline["surface_candidate_object_frames"],
                "dense surface counts differ from independent COO audit")
        actual = dict(frames_indexed=len(times), frames_with_any_pose=int(pose.any(1).sum()),
            frames_with_any_surface=int((arrays["surface_point_count"] > 0).any(1).sum()),
            frame_object_hand_distances=int(observed.sum()), proximity_candidate_values=int(near.sum()),
            candidate_boundary_count=len(boundaries), v3_labeled_native_frames=int((frame_counts > 0).sum()))
        require(actual == metadata["counts"] and all(row[k] == v for k, v in actual.items()), "dense counts overclaim")
        counts.update(actual)
        results.append({"sequence_id": uid, "metadata_sha256": digest(metadata_path), "status": "passed"})
    for key, value in counts.items():
        require(delivery["totals"][key] == value, "aggregate dense count mismatch")
    result = {"schema_version": "assistant_annotation_v4_dense_observables_audit", "created_at": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": digest(__file__), "delivery_sha256": digest(report_path), "checked_sequences": len(results),
        "failed": 0, "counts": dict(counts), "records": results, "dense_index_complete": True,
        "continuous_visual_review_complete": False, "full_annotation_complete": False, "training_allowed": False,
        "scope_zh": "独立核全部原生索引、源/产物SHA、实际手/位姿mask、表面COO统计、接近候选mask/边界及排除；数值处理齐全不是逐帧语义标注完成。"}
    write(output / "dense_observables_audit.json", result)
    return result


def resolved(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def audit_continuous_evidence(output, reference, timestamps):
    """Re-decode the few actually reviewed frames and bind them to pages."""
    import cv2
    from PIL import Image

    output = Path(output)
    path = resolved(output, reference["path"])
    require(digest(path) == reference["sha256"], "dense evidence manifest SHA changed")
    manifest = read(path)
    require(manifest["schema_version"] == "assistant_annotation_v4_dense_evidence", "invalid dense evidence schema")
    binding = manifest["source_bindings"]
    require(digest(binding["metadata_path"]) == binding["metadata_sha256"]
            and digest(binding["hands_path"]) == binding["hands_sha256"], "dense evidence source metadata/hands changed")
    metadata = read(binding["metadata_path"])
    require(metadata["video_path"] == binding["video_path"] and metadata["video_sha256"] == binding["video_sha256"], "dense evidence video source mismatch")
    frames = manifest["frames"]
    start, end = manifest["native_frame_range"]
    require([r["frame_index"] for r in frames] == list(range(start, end)), "dense evidence skipped native frames")
    pages = {p["path"]: p for p in manifest["pages"]}
    viewed_paths = reference["reviewed_page_paths"]
    require(len(viewed_paths) == len(set(viewed_paths)) and set(viewed_paths) <= set(pages), "unknown/duplicate viewed page")
    if reference.get("reviewed_page_sha256") is not None:
        require(reference["reviewed_page_sha256"] == [pages[p]["sha256"] for p in viewed_paths], "viewed page SHA list differs")
    seen = set()
    for page_path in viewed_paths:
        page = pages[page_path]
        require(digest(resolved(output, page_path)) == page["sha256"], "actual reviewed page changed")
        require(page["frame_indices"] == [c["frame_index"] for c in page["cells"]], "page cells lose frame identity")
        seen.update(page["frame_indices"])
    require(seen <= {r["frame_index"] for r in frames}, "page claims absent native evidence")
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(binding["video_path"])
    require(cap.isOpened(), "cannot decode original evidence video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for record in frames:
            frame = record["frame_index"]
            before = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, bgr = cap.read()
            require(ok and abs(before - frame) < .1, "native source decode position differs")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            require(hashlib.sha256(rgb.tobytes()).hexdigest() == record["decoded_native_rgb_sha256"], "native decoded image differs from evidence")
            require(abs(record["timestamp_s"] - timestamps[frame]) <= 1e-9, "dense evidence timestamp differs")
            crop_path = resolved(output, record["crop_path"])
            require(digest(crop_path) == record["crop_sha256"], "native crop SHA changed")
            x0, y0, x1, y1 = record["crop_xyxy_native"]
            require(0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0], "crop outside original image")
            with Image.open(crop_path) as image:
                require(np.array_equal(np.asarray(image.convert("RGB")), rgb[y0:y1, x0:x1]), "crop pixels do not match original video")
            page = manifest["pages"][record["page_index"]]
            require(page["cells"][record["page_cell_index"]]["frame_index"] == frame, "native frame/page cell association differs")
    finally:
        cap.release()
    return seen, {"path": str(path), "sha256": digest(path), "decoded_native_frames_verified": len(frames),
                  "reviewed_pages_verified": len(viewed_paths), "reviewed_native_frames": len(seen)}


def audit_dense_review_labels(review, viewed_frames, frame_count):
    require(review["annotation_source"] == "assistant_visual_review" and review.get("reviewer_agent")
            and review.get("reviewed_at") and review["human_label_valid"] is False
            and review["training_allowed"] is False, "dense review lacks assistant provenance/gates")
    reviewed = range_mask(review["reviewed_ranges"], frame_count)
    require(set(np.flatnonzero(reviewed)) <= set(viewed_frames), "declared continuous review includes unviewed native frame")
    require(review.get("reviewed_native_frame_count", int(reviewed.sum())) == int(reviewed.sum()), "dense reviewed frame count overclaim")
    require(not review.get("full_sequence_continuous_review_complete") or reviewed.all(), "partial review claims complete sequence")
    if "unreviewed_ranges" in review:
        require(np.array_equal(range_mask(review["unreviewed_ranges"], frame_count), ~reviewed), "unreviewed gaps differ from actual review coverage")
    ids = {r["entity_id"] for r in review["entities"]} | {"left_hand", "right_hand"}
    known, counts, by_kind = {}, Counter(reviewed_frames=int(reviewed.sum())), Counter()
    for alias in review.get("visual_identity_aliases", []):
        source, canonical = alias["alias_id"], alias["canonical_entity_id"]
        require(source.startswith("visual_") and canonical.startswith("visual_") and source != canonical
                and canonical in ids, "visual alias merges geometry identity or has unknown target")
        a, b = alias["start_frame"], alias["end_frame"]
        mask = range_mask([[a, b]], frame_count)
        require(alias["evidence_frames"] == list(range(a, b)) and reviewed[mask].all()
                and alias["sampling_scope"] == "continuous_native_frames_reviewed"
                and alias["annotation_source"] == "assistant_visual_review"
                and alias["silver_label_valid"] is True and alias["human_label_valid"] is False,
                "visual alias exceeds actual continuous identity evidence")
        counts["visual_identity_alias_links"] += 1
    positive_frames, negative_frames = set(), set()
    for record in review["state_segments"]:
        require(record["value"] in (0, 1) and type(record["value"]) is int, "unknown state encoded as negative")
        require(record["sampling_scope"] == "continuous_native_frames_reviewed"
                and record["annotation_source"] == "assistant_visual_review"
                and record["silver_label_valid"] is True and record["human_label_valid"] is False,
                "state segment lacks continuous assistant evidence")
        a, b = record["start_frame"], record["end_frame"]
        mask = range_mask([[a, b]], frame_count)
        require(np.all(reviewed[mask]) and record["evidence_frames"] == list(range(a, b)), "state segment interpolates across unviewed frames")
        require(entity_references(record) <= ids, "dense state refers to unknown entity")
        if record["value"] == 0:
            require(record.get("object_id") is not None and record.get("subject") is not None,
                    "negative relation lacks explicit subject/object scope")
        key = tuple(record.get(k) for k in ("layer", "type", "subject", "object_id", "secondary_object_id"))
        for frame in range(a, b):
            address = key + (frame,)
            require(address not in known or known[address] == record["value"], "conflicting positive and negative state labels")
            known[address] = record["value"]
        (positive_frames if record["value"] else negative_frames).update(range(a, b))
        counts["state_segments"] += 1
    for address, value in known.items():
        counts["positive_state_values" if value else "negative_state_values"] += 1
        by_kind[f"{address[0]}:{address[1]}:value={value}"] += 1
    for unknown in review.get("unresolved_ranges", []):
        require(unknown.get("value") is None and unknown.get("silver_label_valid") is not True, "unknown converted to supervised state")
        if "start_frame" in unknown and "end_frame" in unknown:
            mask = range_mask([[unknown["start_frame"], unknown["end_frame"]]], frame_count)
            key = tuple(unknown.get(k) for k in ("layer", "type", "subject", "object_id", "secondary_object_id"))
            require(not any(key + (int(f),) in known for f in np.flatnonzero(mask)), "unknown overlaps a known state for the same entity/relation")
    for event in review["transition_events"]:
        require(entity_references(event) <= ids, "dense transition refers to unknown entity")
        require(event["sampling_scope"] == "continuous_native_frames_reviewed"
                and event["annotation_source"] == "assistant_visual_review" and event["human_label_valid"] is False,
                "transition lacks continuous assistant evidence")
        support = set(event["evidence_frames"])
        require(support and support <= set(np.flatnonzero(reviewed)), "transition cites unviewed native evidence")
        lo, hi = event["boundary_frame_range"]
        range_mask([[lo, hi]], frame_count)
        require(set(range(lo, hi)) <= set(np.flatnonzero(reviewed)), "transition uncertainty range extends beyond reviewed frames")
        family = event.get("event_family", "physical_interaction")
        if event["frame_resolved"]:
            require(event.get("silver_label_valid") is True and event.get("censored") is False and hi - lo == 1 and lo > 0
                    and {lo - 1, lo} <= support, "precise transition lacks consecutive before/after evidence")
            counts["confirmed_" + family + "_transition_bounds"] += 1
        else:
            require(event.get("censored") is True, "unresolved transition not marked censored")
            counts[family + "_transition_uncertainty_ranges"] += 1
        if family == "visual_signal":
            require(event["type"] not in {"contact_onset", "grasp_onset", "operation_onset", "release", "support_gained", "support_lost"},
                    "visual signal masquerades as physical interaction boundary")
        counts["transition_records"] += 1
    counts.update(frames_with_positive_state=len(positive_frames), frames_with_negative_state=len(negative_frames),
                  positive_negative_labeled_frames=len(positive_frames | negative_frames), all_layers_fully_classified_frames=0)
    return {"counts": dict(counts), "state_values_by_type": dict(by_kind),
            "reviewed_ranges": mask_ranges(reviewed), "remaining_unreviewed_ranges": mask_ranges(~reviewed)}


def audit_continuous_reviews(base_root, v3_root, output):
    base_root, v3_root, output = map(Path, (base_root, v3_root, output))
    base, overlay = read(base_root / "dataset_index.json"), read(v3_root / "mamba_overlay_index.json")
    entries = {r["sequence_id"]: r for r in base["sequences"]}
    annotations = {r["sequence_id"]: r for r in overlay["sequences"]}
    counts, records = Counter(), []
    for path in sorted((output / "dense_reviews").glob("*.json")):
        review_sha = digest(path)
        review = read(path)
        uid, bindings = review["sequence_id"], review["source_bindings"]
        require(uid in entries and review["frame_count"] == entries[uid]["frame_count"], "dense review timeline differs")
        require(bindings["v3_annotation_sha256"] == annotations[uid]["sha256"]
                and bindings["v3_overlay_index_sha256"] == digest(v3_root / "mamba_overlay_index.json")
                and bindings["v2_dataset_index_sha256"] == digest(base_root / "dataset_index.json"), "dense review source binding changed")
        v3_ids = {e["entity_id"] for e in read(v3_root / annotations[uid]["annotation_path"])["entities"]}
        require(all(alias["alias_id"] in v3_ids for alias in review.get("visual_identity_aliases", [])),
                "visual alias source absent from frozen V3 identities")
        with np.load(base_root / entries[uid]["sequence_path"], allow_pickle=False) as data:
            timestamps = data["timestamps"]
        viewed, evidence_rows, by_manifest = set(), [], {}
        for reference in review["evidence_manifests"]:
            frames, checked = audit_continuous_evidence(output, reference, timestamps)
            require(read(resolved(output, reference["path"]))["sequence_id"] == uid, "review image belongs to another acquisition")
            viewed.update(frames)
            by_manifest[reference["path"]] = frames
            evidence_rows.append(checked)
        for row in review["reviewed_ranges"]:
            require(row["evidence_scope"] == "continuous_native_frames_reviewed", "review range is not continuous")
            support = set().union(*(by_manifest[p] for p in row["evidence_manifest_paths"]))
            require(set(range(row["start_frame"], row["end_frame"])) <= support, "range has missing referenced evidence page")
        checked = audit_dense_review_labels(review, viewed, len(timestamps))
        require(digest(path) == review_sha, "review changed during independent audit; retry frozen revision")
        counts.update(checked["counts"])
        records.append({"sequence_id": uid, "path": str(path.relative_to(output)), "sha256": review_sha,
                        "status": "passed", **checked, "evidence": evidence_rows})
    result = {"schema_version": "assistant_annotation_v4_continuous_review_audit", "created_at": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": digest(__file__), "reviewed_sequences": len(records), "failed": 0, "counts": dict(counts),
        "total_native_frames": sum(r["frame_count"] for r in base["sequences"]), "records": records,
        "continuous_visual_review_complete": counts["reviewed_frames"] == sum(r["frame_count"] for r in base["sequences"]),
        "full_annotation_complete": False, "training_allowed": False,
        "scope_zh": "重新解码全部已声明连续证据、核每帧RGB/裁剪与联系表SHA；明确正负与unknown分开，视觉信号与物理交互边界分开，不把任一已知关系计作整帧全层完成。"}
    write(output / "continuous_review_audit.json", result)
    return result


def audit_binding_points(binding, track, timestamps, image_size):
    """Independent check that surviving anchor points never resurrect."""
    frame_ids = track["frame_index"].astype(int)
    positions = {int(frame): i for i, frame in enumerate(frame_ids)}
    point_columns = {int(identity): i for i, identity in enumerate(track["point_ids"])}
    uv = track["points_uv"][:, 0]
    valid = track["point_valid"].astype(bool) & np.isfinite(uv).all(-1)
    valid &= (uv[..., 0] >= 0) & (uv[..., 0] < image_size[0]) & (uv[..., 1] >= 0) & (uv[..., 1] < image_size[1])
    require(np.allclose(track["timestamps"], timestamps[frame_ids], atol=1e-7, rtol=0), "binding track/native timestamps differ")
    observations = binding["observations"]
    frames = [o["frame_index"] for o in observations]
    require(frames == sorted(set(frames)) and binding["anchor_frame"] in frames, "binding duplicates frames or omits anchor")
    require(np.array_equal(range_mask(binding["tracked_candidate_ranges"], len(timestamps)), point_mask(frames, len(timestamps))),
            "binding ranges interpolate missing observations")
    anchor_ids = {int(key.split("/")[-1]) for key in binding["anchor_point_keys"]}
    require(all(key.startswith(binding["tracklet_id"] + "/") for key in binding["anchor_point_keys"]), "binding mixes point tracklets")
    by_frame = {o["frame_index"]: o for o in observations}
    require(set(by_frame[binding["anchor_frame"]]["point_ids"]) == anchor_ids, "binding anchor point identity changed")
    for observation in observations:
        frame, ids = observation["frame_index"], observation["point_ids"]
        require(frame in positions and len(ids) == len(set(ids)) >= 4
                and set(ids) <= anchor_ids <= set(point_columns), "binding contains invented point identity")
        at, columns = positions[frame], [point_columns[i] for i in ids]
        require(valid[at, columns].all() and observation["point_count"] == len(ids), "binding includes invalid/hidden source points")
        points = uv[at, columns]
        require(np.allclose(observation["bbox_xyxy"], np.r_[points.min(0), points.max(0)], atol=1e-5, rtol=0)
                and abs(observation["timestamp"] - timestamps[frame]) <= 1e-9, "binding bbox/time does not match actual source points")
    max_step = .05 * np.linalg.norm(image_size)
    max_gap = 1.5 * np.median(np.diff(timestamps))
    anchor = binding["anchor_frame"]
    for direction in (-1, 1):
        previous = anchor
        previous_ids = anchor_ids
        sequence = sorted((f for f in frames if (f - anchor) * direction > 0), reverse=direction < 0)
        for frame in sequence:
            require(frame == previous + direction and abs(timestamps[frame] - timestamps[previous]) <= max_gap + 1e-9,
                    "binding crosses native missing frame/time gap")
            at, before = positions[frame], positions[previous]
            surviving = {i for i in previous_ids if valid[at, point_columns[i]]
                         and np.linalg.norm(uv[at, point_columns[i]] - uv[before, point_columns[i]]) <= max_step}
            require(set(by_frame[frame]["point_ids"]) == surviving, "binding point resurrected, jumped, or silently dropped")
            previous, previous_ids = frame, surviving
    return frames


def audit_entity_grounding(base_root, v3_root, output):
    base_root, v3_root, output = map(Path, (base_root, v3_root, output))
    base, overlay = read(base_root / "dataset_index.json"), read(v3_root / "mamba_overlay_index.json")
    report_path = output / "entity_grounding_report.json"
    report_sha, report = digest(report_path), read(report_path)
    require(report["source_bindings"]["base_index_sha256"] == digest(base_root / "dataset_index.json")
            and report["source_bindings"]["v3_overlay_sha256"] == digest(v3_root / "mamba_overlay_index.json")
            and report["source_bindings"]["split_manifest_sha256"] == base["split_manifest_sha256"], "grounding source mismatch")
    entries = {r["sequence_id"]: r for r in base["sequences"]}
    annotations = {r["sequence_id"]: r for r in overlay["sequences"]}
    rows = report["sequences"]
    require(len(rows) == len(entries) == len({r["sequence_id"] for r in rows})
            and {r["sequence_id"] for r in rows} == set(entries), "grounding sequence coverage mismatch")
    counts, records = Counter(), []
    for row in rows:
        path = output / row["path"]
        require(digest(path) == row["sha256"], "grounding component SHA changed")
        document = read(path)
        uid, binding_source = document["sequence_id"], document["source_bindings"]
        require(document["split"] == entries[uid]["split"] and document["frame_count"] == entries[uid]["frame_count"], "grounding timeline/split changed")
        require(binding_source["v3_annotation_sha256"] == annotations[uid]["sha256"]
                and binding_source["base_sequence_sha256"] == entries[uid]["sha256"]["sequence"], "grounding frozen source differs")
        for prefix in ("proposal", "track_manifest", "source_metadata", "v3_annotation"):
            require(digest(binding_source[prefix + "_path"]) == binding_source[prefix + "_sha256"], "grounding source file changed")
        if binding_source["manual_anchor_sha256"] is not None:
            require(digest(output / "manual_entity_anchors" / (uid + ".json")) == binding_source["manual_anchor_sha256"], "manual identity anchor changed")
        require(document["human_label_valid"] is False and document["training_allowed"] is False
                and document["full_annotation_complete"] is False and document["allowed_as_model_input"] is False,
                "grounding candidate promoted to human/dense truth")
        excluded = {r["object_id"] for r in document["exclusions"] if r["scope"] == "entity_until_reidentified"}
        anchors = {a["anchor_id"]: a for a in document["anchors"]}
        tracks = {t["tracklet_id"]: t for t in document["candidate_tracks"]}
        with np.load(base_root / entries[uid]["sequence_path"], allow_pickle=False) as data:
            timestamps = data["timestamps"]
        source_meta = read(binding_source["source_metadata_path"])
        cached = {}
        identity_frames, native_frames = set(), set()
        for binding in document["bindings"]:
            require(binding["annotation_source"] == "geometry_rule" and binding["review_status"] == "estimated"
                    and binding["silver_label_valid"] is False and binding["human_label_valid"] is False
                    and binding["interaction_inferred"] is False and binding["role_propagated"] is False
                    and binding["whole_object_pose_certified"] is False and binding["pose_recomputed"] is False,
                    "tracked identity candidate overclaims reviewed interaction or new geometry")
            require(binding["source_object_id"] not in excluded and binding["anchor_id"] in anchors, "binding revives rejected identity")
            anchor = anchors[binding["anchor_id"]]
            require(anchor["entity_id"] == binding["entity_id"] and anchor["source_object_id"] == binding["source_object_id"]
                    and anchor["frame_index"] == binding["anchor_frame"], "binding changes anchor identity")
            require(anchor["annotation_source"] == "assistant_visual_review" and anchor["silver_label_valid"] is True
                    and anchor["human_label_valid"] is False, "binding seeded by unreviewed name/geometry")
            require(any(binding["anchor_frame"] in e["frames"] for e in anchor["evidence"]), "anchor lacks viewed native evidence")
            for evidence in anchor["evidence"]:
                require(digest(resolved(v3_root, evidence["gallery_path"])) == evidence["gallery_sha256"], "anchor evidence image changed")
            track_id = binding["tracklet_id"]
            require(track_id in tracks and tracks[track_id]["source_object_id"] == binding["source_object_id"], "binding mixes source object tracklets")
            if track_id not in cached:
                track = tracks[track_id]
                require(digest(track["track_path"]) == track["sha256"], "actual binding track changed")
                with np.load(track["track_path"], allow_pickle=False) as data:
                    cached[track_id] = {key: data[key] for key in ("frame_index", "timestamps", "point_ids", "point_valid", "points_uv")}
            frames = audit_binding_points(binding, cached[track_id], timestamps, source_meta["image_size"])
            identity_frames.update((binding["entity_id"], f) for f in frames)
            native_frames.update(frames)
            counts["bindings"] += 1
            counts["component_only_bindings"] += int(binding["association_type"] == "component_of")
            counts["visual_entity_candidate_bindings"] += int(binding["entity_id"].startswith("visual_"))
        coverage = document["coverage"]
        require(coverage["tracked_candidate_native_frames"] == len(native_frames)
                and coverage["tracked_candidate_entity_frames"] == len(identity_frames), "grounding candidate coverage overclaim")
        counts.update(sequences=1, sequences_with_bindings=int(bool(document["bindings"])),
                      tracked_candidate_native_frames=len(native_frames), tracked_candidate_entity_frames=len(identity_frames),
                      anchors=len(anchors), v3_anchors=coverage["v3_anchor_count"], pixel_anchors=coverage["v4_pixel_anchor_count"])
        records.append({"sequence_id": uid, "sha256": row["sha256"], "status": "passed"})
    require(digest(report_path) == report_sha, "grounding report changed during audit")
    result = {"schema_version": "assistant_annotation_v4_entity_grounding_audit", "created_at": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": digest(__file__), "grounding_report_sha256": report_sha, "failed": 0,
        "counts": dict(counts), "records": records, "full_annotation_complete": False, "training_allowed": False,
        "scope_zh": "独立核实际锚点与图像/源轨迹SHA、逐帧point键/有效性/边界框/时间、缺测不复活和来源身份；传播仍是未复核候选，部件不冒充整物体6D。"}
    write(output / "entity_grounding_audit.json", result)
    return result


def dense_window_expectations(review, timestamps):
    targets = defaultdict(list)
    reviewed = range_mask(review["reviewed_ranges"], len(timestamps))
    for index, state in enumerate(review["state_segments"]):
        if state.get("silver_label_valid") is not True or state.get("value") not in (0, 1):
            continue
        a, b = state["start_frame"], state["end_frame"]
        require(state["evidence_frames"] == list(range(a, b)) and reviewed[a:b].all(), "window source state has unreviewed internal frame")
        for frame in range(a, b):
            targets[frame].append(index)
    expectations, excluded = {}, Counter()
    for target, segments in targets.items():
        anchors = np.flatnonzero(timestamps[target] - timestamps[:target] >= 1.0)
        if not len(anchors):
            excluded["no_anchor_before_horizon"] += 1
            continue
        anchor = int(anchors[-1])
        if anchor < 31:
            excluded["insufficient_observation_history"] += 1
            continue
        horizon = float(timestamps[target] - timestamps[anchor])
        if horizon - 1 > 1.5 * np.median(np.diff(timestamps)) + 1e-9:
            excluded["horizon_overshoot"] += 1
            continue
        expectations[target] = {"start_frame": anchor - 31, "end_frame": anchor + 1, "anchor_frame": anchor,
            "target_timestamp": float(timestamps[target]), "actual_horizon_s": horizon, "state_segment_indices": segments}
    return expectations, excluded, len(targets)


def audit_dense_windows(base_root, v3_root, output):
    base_root, v3_root, output = map(Path, (base_root, v3_root, output))
    path = output / "mamba_dense_index.json"
    index = read(path)
    base, overlay = read(base_root / "dataset_index.json"), read(v3_root / "mamba_overlay_index.json")
    require(index["schema_version"] == "assistant_annotation_v4_dense_mamba", "invalid dense Mamba schema")
    require(index["base_dataset_index_sha256"] == digest(base_root / "dataset_index.json")
            and index["v3_overlay_index_sha256"] == digest(v3_root / "mamba_overlay_index.json")
            and index["split_manifest_sha256"] == base["split_manifest_sha256"]
            and index["source_selection_sha256"] == base["source_selection_sha256"], "dense Mamba frozen source changed")
    require(index["training_allowed"] is False and index["full_annotation_complete"] is False
            and index["base_windows_modified"] is False and index["review_selected_diagnostic"] is True
            and index["blind_evaluation"] is False, "review-selected data hides selection or training limitations")
    for entry in index["source_reports"].values():
        require(digest(output / entry["path"]) == entry["sha256"], "dense Mamba report binding changed")
    review_audit_path = output / "continuous_review_audit.json"
    reviewed = {r["sequence_id"]: r["sha256"] for r in read(review_audit_path)["records"]}
    sequences = {r["sequence_id"]: r for r in index["sequences"]}
    require(len(sequences) == len(index["sequences"]) and set(sequences) == set(reviewed), "dense index omits/duplicates audited reviews")
    source_entries = {r["sequence_id"]: r for r in base["sequences"]}
    annotations = {r["sequence_id"]: r for r in overlay["sequences"]}
    by_uid, seen, window_ids, counts = defaultdict(list), set(), set(), Counter()
    for split in ("train", "val", "test"):
        old, exported = base["splits"][split], index["base_window_bindings"][split]
        require(exported["path"] == old["window_path"] and exported["sha256"] == old["sha256"]
                == digest(base_root / old["window_path"]), "old frozen windows changed")
        entry = index["splits"][split]
        window_path = output / entry["window_path"]
        require(digest(window_path) == entry["sha256"], "dense window JSONL SHA changed")
        rows = [json.loads(line) for line in window_path.read_text().splitlines() if line.strip()]
        require(len(rows) == entry["window_count"], "dense window count differs")
        for row in rows:
            uid = row["sequence_id"]
            require(uid in sequences and row["split"] == source_entries[uid]["split"] == sequences[uid]["split"] == split,
                    "dense window split leakage")
            address = (uid, row["target_frame"])
            require(address not in seen and row["window_id"] not in window_ids, "duplicate dense target/window")
            seen.add(address); window_ids.add(row["window_id"])
            by_uid[uid].append(row)
            require(row["selection_policy"] == "explicit_continuous_review_state_targets", "dense target selection scope differs")
            for key in ("sequence_path", "metadata_path", "events_path"):
                require(row[key] == source_entries[uid][key], "dense window numeric source path changed")
            require(row["sequence_sha256"] == source_entries[uid]["sha256"]["sequence"]
                    and row["source_selection_sha256"] == base["source_selection_sha256"]
                    and row["split_manifest_sha256"] == base["split_manifest_sha256"], "dense window numeric SHA changed")
            require(row["annotation_path"] == annotations[uid]["annotation_path"]
                    and row["annotation_sha256"] == annotations[uid]["sha256"], "dense window V3 source changed")
            require(row["dense_review_path"] == sequences[uid]["review_path"]
                    and row["dense_review_sha256"] == sequences[uid]["review_sha256"] == reviewed[uid], "dense window uses unaudited review revision")
        counts["windows_" + split] = len(rows)
    exclusions = Counter()
    for uid, entry in sequences.items():
        review_path = output / entry["review_path"]
        require(digest(review_path) == entry["review_sha256"] == reviewed[uid], "continuous review changed after audit")
        review = read(review_path)
        require(entry["v3_annotation_sha256"] == annotations[uid]["sha256"], "dense review V3 source differs")
        for reference in entry["evidence_manifests"]:
            require(digest(resolved(output, reference["path"])) == reference["sha256"], "dense index evidence manifest changed")
        with np.load(base_root / source_entries[uid]["sequence_path"], allow_pickle=False) as numeric:
            timestamps = numeric["timestamps"]
        expected, skipped, candidates = dense_window_expectations(review, timestamps)
        require({r["target_frame"] for r in by_uid[uid]} == set(expected), "dense eligible target omitted or unknown frame selected")
        for row in by_uid[uid]:
            target = expected[row["target_frame"]]
            for key in ("start_frame", "end_frame", "anchor_frame", "state_segment_indices"):
                require(row[key] == target[key], "dense history/known state target mismatch")
            for key in ("target_timestamp", "actual_horizon_s"):
                require(abs(row[key] - target[key]) <= 1e-12, "dense timestamp/horizon differs")
        reviewed_count = int(range_mask(review["reviewed_ranges"], len(timestamps)).sum())
        require(entry["candidate_target_count"] == candidates and entry["window_count"] == len(expected)
                and entry["reviewed_native_frame_count"] == reviewed_count, "dense sequence coverage overclaim")
        counts.update(reviewed_sequences=1, reviewed_native_frames=reviewed_count, candidate_target_count=candidates, window_count=len(expected))
        exclusions.update(skipped)
    require(all(index["totals"][k] == counts[k] for k in index["totals"]), "dense aggregate count mismatch")
    require(sum(exclusions.values()) == sum(index["target_exclusion_counts"].values()), "dense skipped target count mismatch")
    result = {"schema_version": "assistant_annotation_v4_dense_window_audit", "created_at": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": digest(__file__), "dense_index_sha256": digest(path), "continuous_review_audit_sha256": digest(review_audit_path),
        "failed": 0, "counts": dict(counts), "independently_derived_exclusions": dict(exclusions),
        "all_eligible_known_state_targets_covered": True, "training_allowed": False, "full_annotation_complete": False,
        "scope_zh": "从已核验的连续正/负片段独立构造target集合及真实一秒历史锚点；核原划分、32帧历史、去重、全部来源SHA，unknown不作为负例。"}
    write(output / "dense_window_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--observables-only", action="store_true")
    parser.add_argument("--continuous-reviews-only", action="store_true")
    parser.add_argument("--grounding-only", action="store_true")
    parser.add_argument("--dense-windows-only", action="store_true")
    args = parser.parse_args()
    if args.dense_windows_only:
        result = audit_dense_windows(args.base_root, args.v3_root, args.output)
    elif args.grounding_only:
        result = audit_entity_grounding(args.base_root, args.v3_root, args.output)
    elif args.continuous_reviews_only:
        result = audit_continuous_reviews(args.base_root, args.v3_root, args.output)
    elif args.observables_only:
        result = audit_dense_observables(args.base_root, args.v3_root, args.output)
    else:
        result = audit_dense_baseline(args.base_root, args.v3_root, args.output, args.workers)
    print(json.dumps({k: v for k, v in result.items() if k not in {"sequences", "records", "point_positive_frame_counts_by_type"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
