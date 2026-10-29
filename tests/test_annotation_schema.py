import json

import pytest

from src.annotation.schema import (
    project_frame_label, read_jsonl, validate_event, validate_none_interval, write_jsonl,
)


def event(**updates):
    result = {
        "event_id": "e1", "sequence_id": "s1", "subject": "left",
        "object_id": "track_1", "object_hint": "blue adapter", "interaction_type": "grasp",
        "start_frame": 2, "end_frame": 5, "evidence_frames": [2, 4],
        "confidence": 0.7, "label_source": "vlm", "review_status": "pending",
        "geometry_status": "not_checked", "source_event_ids": [], "notes": "",
    }
    result.update(updates)
    return result


def none_interval(**updates):
    result = {
        "interval_id": "n1", "sequence_id": "s1", "start_frame": 0, "end_frame": 2,
        "evidence_frames": [0], "confidence": None, "label_source": "human",
        "review_status": "accepted", "reviewer": "reviewer1", "reviewed_at": "2026-10-22T00:00:00Z",
        "notes": "",
    }
    result.update(updates)
    return result


def accepted(**updates):
    return event(review_status="accepted", reviewer="reviewer1",
                 reviewed_at="2026-10-22T00:00:00Z", geometry_status="passed", **updates)


def test_pending_and_unlabelled_are_not_none():
    pending = project_frame_label([event()], [], 3, ["track_1"])
    missing = project_frame_label([], [], 3, ["track_1"])
    rejected = project_frame_label([event(review_status="rejected")], [], 3, ["track_1"])
    assert not pending["valid"] and not missing["valid"] and not rejected["valid"]
    assert pending["interaction_name"] == "none"


def test_reviewed_event_and_explicit_none():
    actual = project_frame_label([accepted()], [], 3, ["track_0", "track_1"])
    negative = project_frame_label([], [none_interval()], 1, ["track_1"])
    assert actual["valid"] and actual["object_slot"] == 1 and actual["interaction"] == 2
    assert negative["valid"] and negative["subject"] == 2 and negative["object_slot"] == -1


def test_overlapping_accepted_events_do_not_get_arbitrarily_selected():
    second = accepted(event_id="e2", subject="right", interaction_type="hold")
    projected = project_frame_label([accepted(), second], [], 3, ["track_1"])
    assert not projected["valid"] and projected["reason"] == "overlapping_accepted_labels"


def test_pending_overlap_invalidates_accepted_event():
    projected = project_frame_label([accepted(), event(event_id="e2")], [], 3, ["track_1"])
    assert not projected["valid"] and projected["reason"] == "pending_or_needs_revision"


def test_accepted_requires_identity_and_track():
    with pytest.raises(ValueError, match="reviewer"):
        validate_event(event(review_status="accepted", reviewed_at="2026-10-22T00:00:00Z"))
    projected = project_frame_label([accepted(object_id="unknown")], [], 3, ["track_1"])
    assert not projected["valid"] and projected["reason"] == "accepted_event_object_not_in_tracks"


@pytest.mark.parametrize("boundary", [(-1, 2), (2, 2), (3, 2), (0, 6)])
def test_bad_boundaries_fail(boundary):
    candidate = event(start_frame=boundary[0], end_frame=boundary[1], evidence_frames=[2])
    with pytest.raises(ValueError):
        validate_event(candidate, frame_count=5)


def test_jsonl_validation_and_no_overwrite(tmp_path):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, [event()], frame_count=5)
    assert read_jsonl(path, frame_count=5)[0]["event_id"] == "e1"
    with pytest.raises(FileExistsError):
        write_jsonl(path, [event()], frame_count=5)
    assert len(path.read_text().splitlines()) == 1
    path.write_text(json.dumps(event(end_frame=8, evidence_frames=[2])) + "\n")
    with pytest.raises(ValueError, match="frame_count"):
        read_jsonl(path, frame_count=5)


@pytest.mark.parametrize("field", ["subject", "object_id", "object_hint"])
def test_missing_interaction_type_cannot_turn_event_into_none(tmp_path, field):
    candidate = accepted()
    candidate.pop("interaction_type")
    for other in ("subject", "object_id", "object_hint"):
        if other != field:
            candidate.pop(other)
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(candidate) + "\n")
    with pytest.raises(ValueError):
        read_jsonl(path, frame_count=5)


@pytest.mark.parametrize("updates", [
    {"subject": "left"},
    {"object_id": "track_1"},
    {"object_hint": "blue adapter"},
    {"interaction_type": "grasp"},
])
def test_none_interval_rejects_actual_interaction_fields(updates):
    with pytest.raises(ValueError, match="must not contain interaction fields"):
        validate_none_interval(none_interval(**updates))


def test_valid_projection_carries_offline_provenance():
    provenance = {"annotation_mode": "offline_noncausal", "allowed_as_model_input": False}
    actual = project_frame_label([accepted(**provenance)], [], 3, ["track_1"])
    negative = project_frame_label([], [none_interval(**provenance)], 1, ["track_1"])
    for projected in (actual, negative):
        assert projected["valid"]
        assert projected["annotation_mode"] == "offline_noncausal"
        assert projected["allowed_as_model_input"] is False


@pytest.mark.parametrize("updates", [
    {"occluded": True},
    {"occlusion_status": "occluded"},
    {"occlusion_status": "unknown"},
    {"occlusion_status": "pending"},
])
def test_occluded_or_unknown_occlusion_cannot_be_valid(updates):
    actual = project_frame_label([accepted(**updates)], [], 3, ["track_1"])
    negative = project_frame_label([], [none_interval(**updates)], 1, ["track_1"])
    for projected in (actual, negative):
        assert not projected["valid"]
        assert projected["reason"] == "occluded"


@pytest.mark.parametrize("updates", [
    {"annotation_mode": "online_causal"},
    {"allowed_as_model_input": True},
])
def test_invalid_causal_provenance_is_rejected_and_not_projected(updates):
    with pytest.raises(ValueError, match="annotation_mode"):
        validate_event(accepted(**updates))
    with pytest.raises(ValueError, match="annotation_mode"):
        validate_none_interval(none_interval(**updates))
    actual = project_frame_label([accepted(**updates)], [], 3, ["track_1"])
    negative = project_frame_label([], [none_interval(**updates)], 1, ["track_1"])
    for projected in (actual, negative):
        assert not projected["valid"]
        assert projected["reason"] == "invalid_annotation_provenance"
        assert projected["annotation_mode"] == "offline_noncausal"
        assert projected["allowed_as_model_input"] is False


@pytest.mark.parametrize("updates", [{"occluded": True}, {"allowed_as_model_input": True}])
def test_invalid_accepted_label_does_not_mask_overlap_defenses(updates):
    pending = project_frame_label(
        [accepted(**updates), event(event_id="e2")], [], 3, ["track_1"]
    )
    assert pending["reason"] == "pending_or_needs_revision"
    overlapping = project_frame_label(
        [accepted(**updates), accepted(event_id="e2")], [], 3, ["track_1"]
    )
    assert overlapping["reason"] == "overlapping_accepted_labels"
