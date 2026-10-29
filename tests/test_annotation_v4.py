import json

import pytest

from src.dataset.export_annotation_v4_windows import digest, validate_review, verify_evidence
from src.mamba.annotation_v4 import dense_targets


VOCAB = [{"layer": "relation", "code": "contact"}, {"layer": "transition", "code": "release"}]


def state(value=1, identity="obj_001", start=4, end=8, **kwargs):
    return {"id": "state", "layer": "relation", "type": "contact", "subject": "right_hand",
        "object_id": identity, "secondary_object_id": None, "start_frame": start, "end_frame": end,
        "value": value, "evidence_frames": list(range(start, end)),
        "sampling_scope": "continuous_native_frames_reviewed", "annotation_source": "assistant_visual_review",
        "review_status": "assistant_reviewed", "silver_label_valid": True, "human_label_valid": False, **kwargs}


def review(states=None, transitions=None):
    return {"sequence_id": "uid", "frame_count": 20, "human_label_valid": False,
            "reviewed_ranges": [{"start_frame": 0, "end_frame": 12}],
            "state_segments": states or [], "transition_events": transitions or []}


def transition(boundary=(6, 7), **kwargs):
    record = state()
    for key in ("start_frame", "end_frame", "value"):
        record.pop(key)
    record.update(id="transition", layer="transition", type="release", boundary_frame_range=list(boundary),
                  evidence_frames=list(range(4, 10)), frame_resolved=True, censored=False)
    record.update(kwargs)
    return record


def test_explicit_negative_is_valid_but_unmentioned_cells_remain_unknown():
    values, mask, graph, _ = dense_targets(review([state(0)]), {"object_ids": ["obj_001", "obj_002"]}, VOCAB, 2, 5)
    assert mask.sum().item() == 1
    assert mask[0, 1, 0] and values[0, 1, 0] == 0
    assert not mask[1:].any()
    assert graph[0]["value"] == 0 and graph[0]["temporal_supervision_valid"]


def test_positive_and_negative_are_scoped_to_each_actual_target_frame():
    doc = review([state(0, start=1, end=4), state(1, start=6, end=9)])
    for target, expected in [(2, 0), (7, 1)]:
        values, mask, _, _ = dense_targets(doc, {"object_ids": ["obj_001"]}, VOCAB, 0, target)
        assert mask[0, 1, 0] and values[0, 1, 0] == expected
    _, mask, graph, _ = dense_targets(doc, {"object_ids": ["obj_001"]}, VOCAB, 0, 5)
    assert not mask.any() and graph == []


def test_partial_geometry_and_unknown_ids_never_fill_whole_object_slot():
    doc = review([state(1, identity="visual_whole_scanner")])
    doc["entities"] = [{"id": "visual_whole_scanner", "source_geometry_parts": [{"source_object_id": "obj_001", "relation": "component_of"}]}]
    _, mask, graph, unmapped = dense_targets(doc, {"object_ids": ["obj_001"]}, VOCAB, 2, 5)
    assert not mask.any() and graph[0]["object_id"] == "visual_whole_scanner"
    assert graph[0]["object_slot"] is None and "component_substitution" in unmapped[0]["reason"]


def test_two_object_relation_is_preserved_without_single_object_compression():
    doc = review([state(1, secondary_object_id="obj_002")])
    _, mask, graph, _ = dense_targets(doc, {"object_ids": ["obj_001", "obj_002"]}, VOCAB, 2, 5)
    assert not mask.any()
    assert graph[0]["object_slot"] == 0 and graph[0]["secondary_object_slot"] == 1


def test_censored_transition_keeps_range_and_cannot_be_a_precise_positive():
    event = transition((5, 9), frame_resolved=False, censored=True)
    _, mask, graph, _ = dense_targets(review(transitions=[event]), {"object_ids": ["obj_001"]}, VOCAB, 3, 7)
    assert not mask.any() and graph[0]["boundary_frame_range"] == [5, 9]
    assert graph[0]["supervision_value"] is None and not graph[0]["temporal_supervision_valid"]


def test_only_true_single_frame_transition_in_future_horizon_is_valid():
    doc = review(transitions=[transition()])
    values, mask, _, _ = dense_targets(doc, {"object_ids": ["obj_001"]}, VOCAB, 3, 7)
    assert mask[0, 1, 1] and values[0, 1, 1] == 1
    _, mask, graph, _ = dense_targets(doc, {"object_ids": ["obj_001"]}, VOCAB, 6, 7)
    assert not mask.any() and not graph


def test_new_visual_signal_is_not_mapped_to_physical_transition_vocab():
    doc = review(transitions=[transition(type="illumination_onset", event_family="visual_signal")])
    _, mask, graph, unmapped = dense_targets(doc, {"object_ids": ["obj_001"]}, VOCAB, 3, 7)
    assert not mask.any() and graph[0]["type"] == "illumination_onset"
    assert unmapped[0]["reason"] == "event_outside_frozen_v2_vocab"


def test_review_validation_rejects_unseen_frame_and_contradictory_states():
    with pytest.raises(ValueError, match="reviewed evidence"):
        validate_review(review([state(start=10, end=14)]), 20)
    with pytest.raises(ValueError, match="Contradictory"):
        validate_review(review([state(0), state(1)]), 20)
    invalid = state()
    invalid["evidence_frames"].remove(5)
    with pytest.raises(ValueError, match="reviewed evidence"):
        validate_review(review([invalid]), 20)
    with pytest.raises(ValueError, match="negative value"):
        validate_review(review([state(value=None)]), 20)


def test_review_validation_keeps_unknown_gap_out_of_target_set():
    seen, targets = validate_review(review([state(start=1, end=3), state(0, start=7, end=9)]), 20)
    assert seen == set(range(12)) and set(targets) == {1, 2, 7, 8}
    with pytest.raises(ValueError, match="frame-resolved"):
        validate_review(review(transitions=[transition((5, 8))]), 20)


def test_actual_page_coverage_and_content_sha_are_required(tmp_path):
    page = tmp_path / "page.jpg"
    page.write_bytes(b"viewed image bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sequence_id": "uid", "pages": [{"path": "page.jpg", "sha256": digest(page), "frame_indices": [1, 2, 3]}]}))
    doc = review()
    doc["evidence_manifests"] = [{"path": "manifest.json", "sha256": digest(manifest),
        "reviewed_page_paths": ["page.jpg"], "reviewed_page_sha256": [digest(page)]}]
    assert len(verify_evidence(tmp_path, doc, {1, 2, 3})) == 1
    with pytest.raises(ValueError, match="undisplayed"):
        verify_evidence(tmp_path, doc, {1, 2, 3, 4})
    page.write_bytes(b"different pixels")
    with pytest.raises(ValueError, match="page changed"):
        verify_evidence(tmp_path, doc, {1, 2})
