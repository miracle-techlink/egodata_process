import pytest

from src.dataset.deliver_annotation_v3 import collect_silver_events, normalize_edge


def annotation():
    return {"relations": [], "transitions": [], "cooperations": [],
            "review_records": [{"source_interval_id": "s0", "reviewer_agent": "/test",
                                "reviewed_at": "2026-10-23T00:00:00Z"}],
            "interval_annotations": [{"source_interval_id": "s0",
                                      "evidence": [{"frames": [3, 10]}],
                                      "visual_operation_assessments": []}]}


def test_interval_action_presence_without_frame_never_becomes_dense_target():
    a = annotation()
    a["interval_annotations"][0]["visual_operation_assessments"] = [
        {"type": "place", "status": "supported", "silver_label_valid": True}]
    assert collect_silver_events(a) == []


def test_operation_supervision_rejects_unseen_frame_and_keeps_sparse_scope():
    a = annotation()
    item = {"type": "place", "status": "supported", "silver_label_valid": True,
            "temporal_coverage": "sampled_points", "evidence_frames": [5]}
    a["interval_annotations"][0]["visual_operation_assessments"] = [item]
    with pytest.raises(ValueError, match="not present"):
        collect_silver_events(a)
    item["evidence_frames"] = [3, 10]
    result = collect_silver_events(a)
    assert result[0]["temporal_coverage"] == "sampled_points"
    assert result[0]["evidence_frames"] == [3, 10]
    assert result[0]["human_label_valid"] is False


def test_rule_generated_or_uncertain_relation_cannot_become_silver():
    a = annotation()
    a["relations"] = [{"type": "contact", "silver_label_valid": True,
                       "status": "supported", "annotation_source": "geometry_rule"}]
    with pytest.raises(ValueError, match="promotion"):
        collect_silver_events(a)


def test_directed_two_object_edge_retains_both_identities():
    r = normalize_edge({"source": "visual_tool", "target": "visual_workpiece", "type": "acts_on"})
    assert r["object_id"] == "visual_tool" and r["secondary_object_id"] == "visual_workpiece"
    assert r["source"] == "visual_tool" and r["target"] == "visual_workpiece"
    r = normalize_edge({"source": "left_hand", "target": "obj_1"})
    assert r["subject"] == "left_hand" and r["object_id"] == "obj_1"


def test_interval_presence_frames_do_not_imply_point_labels():
    a = annotation()
    a["interval_annotations"][0]["visual_operation_assessments"] = [
        {"type": "place", "status": "supported", "silver_label_valid": True,
         "evidence_frames": [3, 10], "temporal_coverage": "sampled_evidence_interval_presence"}]
    assert collect_silver_events(a) == []


def test_directed_edge_repairs_legacy_primary_object_without_losing_provenance():
    r = normalize_edge({"source": "tool", "target": "workpiece", "object_id": "workpiece",
                        "secondary_object_id": "workpiece"})
    assert r["object_id"] == "tool" and r["secondary_object_id"] == "workpiece"
    assert r["original_object_binding"]["object_id"] == "workpiece"


@pytest.mark.parametrize("participants", [
    ["left_hand", "right_hand", "visual_workpiece"],
    [{"entity_id": "left_hand"}, {"entity_id": "right_hand"}, {"entity_id": "visual_workpiece"}]])
def test_cooperation_preserves_known_participant_instead_of_unknown_object(participants):
    r = normalize_edge({"layer": "cooperation", "participants": participants})
    assert r["object_id"] == "visual_workpiece" and r["subject"] == "both_hands"
    assert r["participant_object_ids"] == ["visual_workpiece"]


def test_cooperation_keeps_all_participating_objects_without_inventing_direction():
    r = normalize_edge({"layer": "cooperation", "participants": ["left_hand", "right_hand", "tool", "workpiece"]})
    assert r["participant_object_ids"] == ["tool", "workpiece"]
    assert r["object_id"] is None and "secondary_object_id" not in r


def test_review_aliases_do_not_inflate_classes_or_rename_operation_grasp():
    for layer, old, new in [("relation", "grasp", "grasp_control"), ("cooperation", "bimanual", "bimanual_joint")]:
        r = normalize_edge({"layer": layer, "type": old})
        assert r["type"] == new and r["original_type"] == old
    assert normalize_edge({"layer": "operation", "type": "grasp"})["type"] == "grasp"
