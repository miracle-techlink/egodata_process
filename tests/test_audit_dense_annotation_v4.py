import json

import numpy as np
import pytest

from src.dataset.audit_dense_annotation_v4 import (
    audit_binding_points, audit_dense_review_labels, dense_window_expectations, digest, entity_references, geometry_representation, mask_ranges, range_mask, summarize_sequence, validate_observable_masks,
)


def test_native_masks_preserve_gaps_and_half_open_edges():
    mask = range_mask([[0, 2], [4, 7]], 8)
    assert mask_ranges(mask) == [[0, 2], [4, 7]]
    with pytest.raises(ValueError):
        range_mask([[4, 9]], 8)


def test_visual_identity_and_nonrigid_6d_are_not_generic_missing_data():
    assert geometry_representation(dict(entity_id="visual_paper"), set())[0] == "unsupported_unmapped_visual_identity"
    assert geometry_representation(dict(entity_id="obj1", rigidity="deformable"), set())[0] == "not_applicable_whole_object_6d"
    assert geometry_representation(dict(entity_id="obj1"), {"obj1"})[0] == "unsupported_rejected_identity"
    assert entity_references(dict(source="both_hands", target="visual_paper", participants=["left_hand", {"entity_id": "obj1"}])) == {"visual_paper", "obj1"}


def test_background_pose_and_interval_action_do_not_complete_workpiece(tmp_path):
    base_root, v3_root = tmp_path / "base", tmp_path / "v3"
    base_root.mkdir(); v3_root.mkdir()
    np.savez(base_root / "sequence.npz", timestamps=np.arange(8) / 30,
             hand_valid=np.ones((8, 2), bool), object_ids=np.array(["obj_background", "obj_rejected"]),
             object_pose_valid=np.ones((8, 2), bool), surface_frame_index=np.array([1, 2, 4]),
             surface_object_index=np.array([0, 0, 1]))
    interval = dict(source_interval_id="s0", start_frame=0, end_frame=8, description_zh="移动纸张",
                    evidence=[dict(frames=[0, 4, 7])], visual_operation_assessments=[dict(status="supported")])
    annotation = dict(frame_count=8, entities=[
        dict(entity_id="obj_background", kind="object", rigidity="rigid"),
        dict(entity_id="obj_rejected", kind="object", rigidity="rigid"),
        dict(entity_id="visual_paper", kind="object", source_interval_id="s0", material_kind="deformable")],
        interval_annotations=[interval], geometry_exclusions=[dict(object_id="obj_rejected", scope="entity_until_reidentified")],
        silver_event_records=[dict(layer="relation", type="grasp_control", source="right_hand", target="visual_paper",
            evidence_frames=[4], silver_label_valid=True, status="supported", temporal_coverage="sampled_points")])
    path = v3_root / "annotation.json"
    path.write_text(json.dumps(annotation))
    base = dict(frame_count=8, split="train", sequence_path="sequence.npz", sha256=dict(sequence="bound-prior-sha", metadata="metadata-sha"))
    entry = dict(annotation_path="annotation.json", sha256=digest(path))
    report = summarize_sequence("test", base, entry, base_root, v3_root, {})
    c = report["counts"]
    assert c["scene_any_pose_candidate_frames"] == 8
    assert c["physical_frames_all_explicit_entities_have_candidate_geometry"] == 0
    assert c["semantic_supported_action_interval_span_frames"] == 8
    assert c["v3_any_silver_positive_frames"] == 1
    assert c["pose_candidate_object_frames"] == 8
    assert c["surface_candidate_point_observations"] == 2
    assert c["relation_explicit_negative_frames"] == 0
    entity = next(r for r in report["entities"] if r["entity_id"] == "visual_paper")
    assert entity["geometry_unsupported_ranges"] == [[0, 8]]
    assert not entity["geometry_unobserved_unknown_ranges"]
    assert entity["visibility_unknown_frames"] == 8  # relation evidence is not a visibility label
    assert entity["layers"]["relation"]["frames_without_any_known_state"] == 7
    assert entity["layers"]["relation"]["fully_classified_frames"] == 0
    assert not report["full_annotation_complete"]


def test_proximity_mask_requires_observed_hand_and_enough_surface_points():
    hand = np.array([[True, False], [True, True]])
    pose = np.zeros((2, 1), bool)
    arrays = dict(hand_valid=hand.copy(), pose_valid=pose.copy(), surface_point_count=np.array([[8], [3]]),
                  nearest_hand_surface_m=np.array([[[.02, np.nan]], [[np.nan, np.nan]]]),
                  distance_observed=np.array([[[True, False]], [[False, False]]]),
                  near_surface_candidate=np.array([[[True, False]], [[False, False]]]))
    validate_observable_masks(arrays, hand, pose, [])
    arrays["near_surface_candidate"][1, 0, 0] = True
    with pytest.raises(ValueError, match="threshold"):
        validate_observable_masks(arrays, hand, pose, [])
    arrays["near_surface_candidate"][1, 0, 0] = False
    arrays["nearest_hand_surface_m"][0, 0, 1] = .01
    with pytest.raises(ValueError, match="NaN"):
        validate_observable_masks(arrays, hand, pose, [])


def dense_review_fixture():
    state = dict(layer="relation", type="contact", subject="right_hand", object_id="visual_tool", secondary_object_id=None,
                 start_frame=0, end_frame=2, value=0, evidence_frames=[0, 1], sampling_scope="continuous_native_frames_reviewed",
                 annotation_source="assistant_visual_review", silver_label_valid=True, human_label_valid=False)
    return dict(annotation_source="assistant_visual_review", reviewer_agent="test", reviewed_at="test",
                human_label_valid=False, training_allowed=False, entities=[dict(entity_id="visual_tool")],
                reviewed_ranges=[[0, 4]], unreviewed_ranges=[], state_segments=[state], transition_events=[], unresolved_ranges=[])


def test_dense_negative_has_explicit_scope_and_does_not_make_all_layers_complete():
    review = dense_review_fixture()
    checked = audit_dense_review_labels(review, {0, 1, 2, 3}, 4)
    assert checked["counts"]["negative_state_values"] == 2
    assert checked["counts"]["all_layers_fully_classified_frames"] == 0
    with pytest.raises(ValueError, match="unviewed"):
        audit_dense_review_labels(review, {0, 1, 3}, 4)
    review["state_segments"][0]["object_id"] = None
    with pytest.raises(ValueError, match="explicit subject/object"):
        audit_dense_review_labels(review, {0, 1, 2, 3}, 4)


def test_continuous_state_cannot_skip_internal_evidence_or_conflict_with_unknown():
    review = dense_review_fixture()
    review["state_segments"][0]["evidence_frames"] = [0]
    with pytest.raises(ValueError, match="interpolates"):
        audit_dense_review_labels(review, set(range(4)), 4)
    review["state_segments"][0]["evidence_frames"] = [0, 1]
    review["unresolved_ranges"] = [{**review["state_segments"][0], "value": None, "silver_label_valid": False}]
    with pytest.raises(ValueError, match="unknown overlaps"):
        audit_dense_review_labels(review, set(range(4)), 4)


def test_exact_visual_signal_does_not_count_as_physical_interaction_boundary():
    review = dense_review_fixture()
    event = dict(layer="transition", type="illumination_onset", event_family="visual_signal", subject="right_hand",
                 object_id="visual_tool", sampling_scope="continuous_native_frames_reviewed",
                 annotation_source="assistant_visual_review", silver_label_valid=True, human_label_valid=False,
                 evidence_frames=[1, 2], boundary_frame_range=[2, 3], frame_resolved=True, censored=False)
    review["transition_events"] = [event]
    checked = audit_dense_review_labels(review, set(range(4)), 4)
    assert checked["counts"]["confirmed_visual_signal_transition_bounds"] == 1
    assert checked["counts"].get("confirmed_physical_interaction_transition_bounds", 0) == 0
    event["evidence_frames"] = [2]
    with pytest.raises(ValueError, match="before/after"):
        audit_dense_review_labels(review, set(range(4)), 4)


def test_grounding_identity_point_cannot_resurrect_after_occlusion():
    uv = np.zeros((3, 1, 5, 2), np.float32)
    uv[:, 0, :, 0] = np.arange(5) * 10 + 10
    uv[:, 0, :, 1] = 10
    valid = np.ones((3, 5), bool)
    valid[1, 4] = False
    timestamps = np.arange(3) * .03
    track = dict(frame_index=np.arange(3), timestamps=timestamps, point_ids=np.arange(5), point_valid=valid, points_uv=uv)
    observations = [dict(frame_index=f, timestamp=float(timestamps[f]), point_ids=list(range(5 if f == 0 else 4)),
                         point_count=5 if f == 0 else 4, bbox_xyxy=[10, 10, 50 if f == 0 else 40, 10]) for f in range(3)]
    binding = dict(tracklet_id="t", anchor_frame=0, anchor_point_keys=[f"t/{i}" for i in range(5)],
                   observations=observations, tracked_candidate_ranges=[[0, 3]])
    assert audit_binding_points(binding, track, timestamps, [100, 100]) == [0, 1, 2]
    observations[2].update(point_ids=list(range(5)), point_count=5, bbox_xyxy=[10, 10, 50, 10])
    with pytest.raises(ValueError, match="resurrected"):
        audit_binding_points(binding, track, timestamps, [100, 100])


def test_dense_window_targets_include_explicit_negatives_but_not_unknown():
    review = dict(reviewed_ranges=[[0, 100]], state_segments=[
        dict(start_frame=70, end_frame=74, evidence_frames=list(range(70, 74)), value=0, silver_label_valid=True),
        dict(start_frame=72, end_frame=76, evidence_frames=list(range(72, 76)), value=1, silver_label_valid=True),
        dict(start_frame=80, end_frame=90, evidence_frames=list(range(80, 90)), value=None, silver_label_valid=False)])
    timestamps = np.arange(100) / 10
    expected, skipped, candidates = dense_window_expectations(review, timestamps)
    assert set(expected) == set(range(70, 76))
    assert candidates == 6 and not skipped
    assert expected[72]["state_segment_indices"] == [0, 1]
    assert expected[75]["state_segment_indices"] == [1]
    for target, row in expected.items():
        assert row["end_frame"] - row["start_frame"] == 32
        assert timestamps[target] - timestamps[row["anchor_frame"]] >= 1
        assert timestamps[target] - timestamps[row["anchor_frame"] + 1] < 1
    review["state_segments"][0]["evidence_frames"] = [70, 73]
    with pytest.raises(ValueError, match="unreviewed internal"):
        dense_window_expectations(review, timestamps)


def test_visual_alias_requires_continuous_evidence_and_cannot_merge_geometry_id():
    review = dense_review_fixture()
    alias = dict(alias_id="visual_old", canonical_entity_id="visual_tool", start_frame=0, end_frame=4,
                 evidence_frames=list(range(4)), sampling_scope="continuous_native_frames_reviewed",
                 annotation_source="assistant_visual_review", silver_label_valid=True, human_label_valid=False)
    review["visual_identity_aliases"] = [alias]
    assert audit_dense_review_labels(review, set(range(4)), 4)["counts"]["visual_identity_alias_links"] == 1
    alias["alias_id"] = "obj_001"
    with pytest.raises(ValueError, match="merges geometry"):
        audit_dense_review_labels(review, set(range(4)), 4)
