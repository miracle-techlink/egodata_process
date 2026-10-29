import copy
import json

import pytest

from src.annotation.render_annotation_evidence_v3 import review_shard, sample_frames
from src.dataset.audit_annotation_v3 import (audit_review, audit_video_ledger, check_entity_refs, check_silver_record,
                                             digest, filtered_geometry_counts, review_window_expectations)


def test_native_sampling_and_reviewer_partition():
    assert sample_frames(10, 20) == [10, 12, 14, 17, 19]
    assert sample_frames(7, 8) == [7]
    with pytest.raises(ValueError):
        sample_frames(7, 7)
    assert [review_shard(i) for i in [0, 99, 100, 199, 200, 279, 280, 299]] == [
        "entity_shard_a", "entity_shard_a", "event_shard_b", "event_shard_b",
        "audit_shard_c", "audit_shard_c", "root_shard_d", "root_shard_d"]


@pytest.fixture
def evidence(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "out"
    folder = source / "joint_annotations" / "test"
    folder.mkdir(parents=True)
    proposal_path = folder / "proposal.json"
    proposal_path.write_text(json.dumps(dict(entities=[dict(id="obj_001")], source_semantic_intervals=[
        dict(source_interval_id="source_0000", start_frame=0, end_frame=10)])))
    directory = root / "evidence" / "test"
    directory.mkdir(parents=True)
    page = directory / "page_000.jpg"
    page.write_bytes(b"synthetic-hash-fixture-not-a-real-review")
    relative = "evidence/test/page_000.jpg"
    manifest = dict(source_bindings=dict(proposal_sha256=digest(proposal_path)),
                    intervals=[dict(source_interval_id="source_0000", frames=[0, 4, 9])],
                    pages=[dict(path=relative, sha256=digest(page), interval_ids=["source_0000"])])
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    review = dict(schema_version="assistant_annotation_v3_review", sequence_id="test",
                  source_bindings=dict(proposal_sha256=digest(proposal_path), evidence_manifest_sha256=digest(manifest_path)),
                  reviewer_agent="test_only", reviewed_at="2026-01-01T00:00:00Z", human_label_valid=False,
                  interval_reviews=[dict(source_interval_id="source_0000", review_status="assistant_reviewed",
                       annotation_source="assistant_visual_review", evidence_scope="sampled_native_frames_only",
                       observations_zh="合成测试，不是真实审核", evidence=[dict(gallery_path=relative,
                       gallery_sha256=digest(page), frames=[0, 4, 9])], operation_assessments=[dict(type="place",
                       status="uncertain", silver_label_valid=False, reason_zh="未看到转换")])])
    path = root / "review.json"
    return root, source, path, review


def run(evidence, mutate=None):
    root, source, path, review = evidence
    review = copy.deepcopy(review)
    if mutate:
        mutate(review)
    path.write_text(json.dumps(review))
    return audit_review(root, source, path)


def test_bound_review_and_uncertainty_pass(evidence):
    assert run(evidence)["counts"]["reviewed_intervals"] == 1


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(human_label_valid=True),
    lambda r: r["source_bindings"].update(proposal_sha256="wrong"),
    lambda r: r["interval_reviews"][0]["evidence"][0].update(frames=[3]),
    lambda r: r["interval_reviews"][0]["operation_assessments"][0].update(silver_label_valid=True),
    lambda r: r.update(interval_reviews=[]),
])
def test_invalid_promotion_and_unviewed_claims_rejected(evidence, mutate):
    with pytest.raises(ValueError):
        run(evidence, mutate)


def test_sparse_frames_do_not_justify_dense_contact(evidence):
    def mutate(review):
        review["interval_reviews"][0]["relations"] = [dict(type="contact", source="left_hand", target="obj_001",
            evidence_frames=[4], start_frame=0, end_frame=10, silver_label_valid=True, status="supported")]
    with pytest.raises(ValueError, match="extrapolated"):
        run(evidence, mutate)


def test_discrete_multi_frame_contact_and_visual_id_assessment(evidence):
    def mutate(review):
        row = review["interval_reviews"][0]
        row["visual_entities"] = [dict(entity_id="visual_a", geometry_status="unavailable")]
        row["entity_assessments"] = [dict(object_id="visual_a", status="supported")]
        row["relations"] = [dict(type="contact", source="both_hands", target="visual_a",
            evidence_frames=[0, 9], start_frame=0, end_frame=10, temporal_coverage="sampled_points",
            silver_label_valid=True, status="supported")]
    assert run(evidence, mutate)["counts"]["relations_sampled_frames"] == 2


def test_single_frame_label_range_cannot_refer_to_another_viewed_frame(evidence):
    def mutate(review):
        review["interval_reviews"][0]["relations"] = [dict(type="contact", source="left_hand", target="obj_001",
            evidence_frames=[4], start_frame=0, end_frame=1, silver_label_valid=True, status="supported")]
    with pytest.raises(ValueError, match="outside declared"):
        run(evidence, mutate)


def test_supported_entity_requires_its_own_point_evidence(evidence):
    def mutate(review):
        review["interval_reviews"][0]["entity_assessments"] = [dict(object_id="obj_001", status="supported",
                                                                  silver_label_valid=True)]
    with pytest.raises(ValueError, match="entity silver lacks"):
        run(evidence, mutate)


def test_sampled_instance_existence_does_not_claim_per_frame_visibility(evidence):
    def mutate(review):
        review["interval_reviews"][0]["visual_entities"] = [dict(entity_id="visual_paper", geometry_status="unavailable",
            identity_scope="sampled_visible_instance_only", silver_label_valid=True)]
    assert run(evidence, mutate)["status"] == "passed"
    def bad(review):
        mutate(review)
        review["interval_reviews"][0]["visual_entities"][0]["evidence_frames"] = [3]
    with pytest.raises(ValueError, match="unviewed"):
        run(evidence, bad)


def test_censored_unknown_transition_can_cite_sparse_context_images(evidence):
    def mutate(review):
        review["interval_reviews"][0]["transitions"] = [dict(type="release", source="left_hand", target="obj_001",
            status="uncertain", silver_label_valid=False, evidence_frames=[0, 9], boundary_range=[0, 9],
            censored=True, temporal_coverage="sampled_points")]
    assert run(evidence, mutate)["counts"]["transitions_unknown"] == 1
    def bad(review):
        mutate(review)
        review["interval_reviews"][0]["transitions"][0].update(status="supported", silver_label_valid=True)
    with pytest.raises(ValueError, match="masquerade"):
        run(evidence, bad)


def test_original_full_frame_image_evidence_must_match_single_frame(evidence):
    root, source, path, review = evidence
    manifest_path = root / "evidence/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frame = root / "evidence/test/frame_4.jpg"
    frame.write_bytes(b"synthetic-full-frame-fixture")
    manifest["frames"] = [dict(path="evidence/test/frame_4.jpg", frame_index=4, sha256=digest(frame))]
    manifest_path.write_text(json.dumps(manifest))
    review["source_bindings"]["evidence_manifest_sha256"] = digest(manifest_path)
    review["interval_reviews"][0]["evidence"] = [dict(gallery_path="evidence/test/frame_4.jpg",
                                                      gallery_sha256=digest(frame), frames=[4])]
    assert run(evidence)["counts"]["reviewed_intervals"] == 1
    review["interval_reviews"][0]["evidence"][0]["frames"] = [0, 4]
    with pytest.raises(ValueError, match="claims other frames"):
        run(evidence)


def test_final_audit_rejects_interval_action_promoted_to_point_label():
    record = dict(annotation_id="op", layer="operation", type="place", status="supported",
                  annotation_source="assistant_visual_review", review_status="assistant_reviewed",
                  silver_label_valid=True, temporal_coverage="sampled_points", evidence_frames=[0, 9],
                  source_interval_ids=["s0"])
    intervals = {"s0": {"evidence": [{"frames": [0, 9]}]}}
    with pytest.raises(ValueError, match="interval-only"):
        check_silver_record(record, intervals, {"op": {**record, "temporal_coverage": "interval_presence"}})
    check_silver_record(record, intervals, {"op": record})
    record["evidence_frames"] = [4]
    with pytest.raises(ValueError, match="unreviewed native"):
        check_silver_record(record, intervals, {"op": record})


def test_final_graph_preserves_two_object_direction_and_geometry_filter_counts():
    r = dict(annotation_id="r", layer="relation", type="acts_on", status="supported",
             annotation_source="assistant_visual_review", review_status="assistant_reviewed",
             silver_label_valid=True, temporal_coverage="sampled_points", evidence_frames=[4],
             source_interval_ids=["s0"], source="visual_tool", target="obj_a",
             subject="object", object_id="obj_a", secondary_object_id="obj_a")
    with pytest.raises(ValueError, match="collapsed"):
        check_silver_record(r, {"s0": {"evidence": [{"frames": [4]}]}}, {"r": r})
    with pytest.raises(ValueError, match="dangling"):
        check_entity_refs(r, {"obj_a"})
    counts = filtered_geometry_counts([
        dict(entity_id="obj_a", kind="object", pose_valid_frames=10, surface_valid_frames=20),
        dict(entity_id="obj_b", kind="object", pose_valid_frames=0, surface_valid_frames=8),
        dict(entity_id="visual_tool", kind="object", pose_valid_frames=0, surface_valid_frames=0)],
        [dict(object_id="obj_a", scope="entity_until_reidentified")])
    assert counts["before_filter_pose_valid_frames"] == 10
    assert counts["retained_candidate_pose_sequences"] == 0
    assert counts["retained_candidate_surface_sequences"] == 1
    assert counts["retained_candidate_surface_valid_frames"] == 8


def test_cooperation_keeps_known_participants_out_of_unknown_object_slot():
    original = dict(annotation_id="c", layer="cooperation", type="dual_grasp", status="supported",
                    annotation_source="assistant_visual_review", review_status="assistant_reviewed",
                    silver_label_valid=True, temporal_coverage="sampled_points", evidence_frames=[4],
                    source_interval_ids=["s0"], participants=["left_hand", {"entity_id": "right_hand"},
                                                           {"entity_id": "visual_cable"}])
    intervals = {"s0": {"evidence": [{"frames": [4]}]}}
    with pytest.raises(ValueError, match="participant bindings"):
        check_silver_record(original, intervals, {"c": original})
    fixed = {**original, "subject": "both_hands", "object_id": "visual_cable",
             "participant_object_ids": ["visual_cable"]}
    check_silver_record(fixed, intervals, {"c": original})
    with pytest.raises(ValueError, match="mapped to unknown"):
        check_silver_record({**fixed, "object_id": None}, intervals, {"c": original})
    original["participants"].append("visual_tie")
    with pytest.raises(ValueError, match="participant bindings"):
        check_silver_record(fixed, intervals, {"c": original})


def test_retained_video_hash_receipt_rejects_changed_source_file(evidence):
    root, source, _, _ = evidence
    video = source / "synthetic_video_bytes"
    video.write_bytes(b"hash-test-fixture")
    metadata_path = source / "sequences/test/metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(dict(video_path=str(video), video_sha256=digest(video))))
    manifest_path = root / "evidence/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_bindings"]["video_sha256"] = digest(video)
    manifest_path.write_text(json.dumps(manifest))
    stat = video.stat()
    ledger = dict(schema_version="assistant_annotation_v3_source_video_sha_ledger", hash_algorithm="sha256",
                  failed=0, checked_sequences=1, completed_at="test-fixture", records=[dict(sequence_id="test",
                  video_path=str(video), metadata_sha256=digest(metadata_path), evidence_manifest_sha256=digest(manifest_path),
                  status="passed", actual_sha256=digest(video), expected_sha256=digest(video),
                  evidence_expected_sha256=digest(video), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)])
    (root / "source_video_sha_ledger.json").write_text(json.dumps(ledger))
    report = audit_video_ledger(root, source, ["test"])
    assert report["current_size_mtime_and_bindings_match"]
    assert not report["full_video_bytes_rehashed_in_current_audit"]
    video.write_bytes(b"changed-video-test-fixture")
    with pytest.raises(ValueError, match="changed since"):
        audit_video_ledger(root, source, ["test"])


def test_grasp_alias_is_limited_to_relation_not_operation():
    original = dict(annotation_id="r", layer="relation", type="grasp", status="supported",
                    annotation_source="assistant_visual_review", review_status="assistant_reviewed",
                    silver_label_valid=True, temporal_coverage="sampled_points", evidence_frames=[4],
                    source_interval_ids=["s0"], source="left_hand", target="obj_a")
    intervals = {"s0": {"evidence": [{"frames": [4]}]}}
    final = {**original, "type": "grasp_control", "subject": "left_hand", "object_id": "obj_a",
             "original_type": "grasp", "type_normalization": "frozen_vocab_alias"}
    check_silver_record(final, intervals, {"r": original})
    with pytest.raises(ValueError, match="provenance"):
        check_silver_record({**final, "original_type": None}, intervals, {"r": original})
    with pytest.raises(ValueError, match="changes source"):
        check_silver_record({**final, "layer": "operation"}, intervals, {"r": {**original, "layer": "operation"}})


def test_evidence_windows_use_actual_time_maximal_historical_anchor():
    import numpy as np

    timestamps = np.arange(100, dtype=np.float64) / 30
    record = dict(layer="relation", silver_label_valid=True, status="supported",
                  annotation_source="assistant_visual_review", review_status="assistant_reviewed",
                  temporal_coverage="sampled_points", evidence_frames=[4, 90, 150])
    expected, skipped = review_window_expectations(timestamps, {"silver_event_records": [record, record,
        {**record, "status": "uncertain", "silver_label_valid": False, "evidence_frames": [80]}]})
    assert set(expected) == {90}
    row = expected[90]
    assert row["end_frame"] - row["start_frame"] == 32
    assert timestamps[90] - timestamps[row["anchor_frame"]] >= 1
    assert timestamps[90] - timestamps[row["anchor_frame"] + 1] < 1
    assert row["evidence_record_indices"] == [0, 1]
    assert skipped == {"early_history": 1, "out_of_range": 1}
