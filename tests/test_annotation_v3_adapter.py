import torch
import pytest

from src.mamba.annotation_v3 import silver_targets, temporal_support, collate_annotation_v3, exclude_rejected_geometry
from src.mamba.annotation_v3 import validate_window_bindings


def record(**kwargs):
    return dict(annotation_source="assistant_visual_review", review_status="assistant_reviewed",
                silver_label_valid=True, layer="relation", type="grasp_control",
                object_id="obj_1", subject="left_hand", temporal_coverage="sampled_points",
                evidence_frames=[20, 40], **kwargs)


def test_sampled_positives_never_fill_unseen_frames():
    r = record()
    assert temporal_support(r, 10, 20)
    assert not temporal_support(r, 10, 30)
    r["annotation_source"] = "geometry_rule"
    assert not temporal_support(r, 10, 20)


def test_partial_uncertain_boundary_does_not_become_known_transition():
    r = record(censored=True, boundary_frame_range=[16, 26])
    r["layer"] = "transition"
    assert not temporal_support(r, 10, 20)
    assert not temporal_support(r, 20, 30)
    assert temporal_support(r, 10, 30)


def test_unobserved_known_entity_never_becomes_unknown_object():
    r = record()
    candidate, valid, graph, missing = silver_targets([r], {"object_ids": ["", ""]},
        [{"layer": "relation", "code": "grasp_control"}], 10, 20)
    assert not valid.any() and not candidate.any()
    assert len(graph) == 1 and graph[0]["object_slot"] is None
    assert missing[0]["reason"] == "known_entity_not_observed_in_window"


def test_second_object_edge_is_not_collapsed_to_single_target():
    r = record(secondary_object_id="obj_2")
    candidate, valid, graph, missing = silver_targets([r], {"object_ids": ["obj_1", "obj_2"]},
        [{"layer": "relation", "code": "grasp_control"}], 10, 20)
    assert not valid.any() and not candidate.any()
    assert graph[0]["secondary_object_slot"] == 1
    assert missing[0]["reason"] == "two_object_edge_preserved_in_graph"


def test_silver_mask_only_marks_observed_positives():
    candidate, valid, _, _ = silver_targets([record()], {"object_ids": ["obj_1", ""]},
        [{"layer": "relation", "code": "grasp_control"}], 10, 20)
    assert valid.sum() == 1 and candidate[0, 0, 0] == 1
    assert not valid[2].any()
    sample = {"silver_event_candidates": candidate, "metadata": {}, "graph_targets": []}
    batch = collate_annotation_v3([sample, sample])
    assert torch.equal(batch["silver_event_candidates"][0], candidate)
    assert batch["graph_targets"] == [[], []]


def test_multi_object_participants_stay_in_graph_when_secondary_field_is_absent():
    r = record(participant_object_ids=["obj_1", "obj_2"])
    r.update(layer="cooperation", type="complementary", subject="both_hands")
    _, valid, graph, missing = silver_targets([r], {"object_ids": ["obj_1", "obj_2"]},
        [{"layer": "cooperation", "code": "complementary"}], 10, 20)
    assert not valid.any() and graph[0]["participant_object_slots"] == [0, 1]
    assert missing[0]["reason"] == "two_object_edge_preserved_in_graph"


def test_rejected_identity_masks_input_future_and_weak_labels():
    sample = {"obj_pose_seq": torch.ones(3, 1, 4, 4), "surface_seq": torch.ones(3, 1, 2, 3),
              "target_obj_pose": torch.ones(1, 4, 4), "target_surface": torch.ones(1, 2, 3),
              "obj_valid": torch.ones(3, 1, dtype=torch.bool), "surface_valid": torch.ones(3, 1, 2, dtype=torch.bool),
              "object_mask": torch.ones(1, dtype=torch.bool), "point_mask": torch.ones(1, 2, dtype=torch.bool),
              "target_obj_valid": torch.ones(1, dtype=torch.bool), "target_surface_valid": torch.ones(1, 2, dtype=torch.bool),
              "object_kind": torch.ones(1, dtype=torch.int64),
              "event_candidates": torch.ones(2, 5, 1), "event_weak_mask": torch.ones(2, 5, 1, dtype=torch.bool),
              "event_label_valid": torch.zeros(2, 5, 1, dtype=torch.bool),
              "metadata": {"object_ids": ["wrong"], "object_names_zh": ["错框"], "object_indices": [0],
                           "point_keys": [["a", "b"]], "point_indices": [[0, 1]]}}
    exclude_rejected_geometry(sample, [{"object_id": "wrong", "scope": "entity_until_reidentified"}])
    assert torch.isnan(sample["obj_pose_seq"]).all() and torch.isnan(sample["target_surface"]).all()
    assert not sample["object_mask"].any() and not sample["event_weak_mask"][0].any()
    assert sample["event_weak_mask"][1].all()  # unrelated unknown-object candidates survive
    assert sample["metadata"]["geometry_exclusions_applied"] == ["wrong"]


@pytest.mark.parametrize("change", [{"split": "test"}, {"sequence_path": "other.npz"}, {"annotation_sha256": "stale"}])
def test_custom_window_rejects_split_mix_source_swap_and_stale_annotation(change):
    source = dict(sequence_id="s", split="train", sequence_path="s.npz", metadata_path="s.json",
                  events_path="s.events", sha256={"sequence": "data"})
    index = dict(sequences=[source], source_selection_sha256="selection", split_manifest_sha256="split")
    annotations = {"s": {"split": "train", "sha256": "annotation"}}
    row = {k: v for k, v in source.items() if k != "sha256"}
    row.update(sequence_sha256="data", source_selection_sha256="selection", split_manifest_sha256="split",
               annotation_sha256="annotation")
    validate_window_bindings([row], index, annotations, "train")
    row.update(change)
    with pytest.raises(ValueError):
        validate_window_bindings([row], index, annotations, "train")
