import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.annotation.review_checks import review_sequence, sequence_dependencies


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def fixture(tmp_path, *, objects=1, frames=8):
    root = tmp_path / "base"
    directory = root / "sequences/uid"
    directory.mkdir(parents=True)
    pose = np.broadcast_to(np.eye(4), (frames, objects, 4, 4)).astype(np.float32).copy()
    hands = np.zeros((frames, 2, 21, 3), np.float32)
    for finger in range(5):
        for joint in range(4):
            hands[:, :, 1 + 4 * finger + joint] = [(finger + 1) * .02, (joint + 1) * .02, .5]
    hands[:, :, 0, 2] = .5
    ids = np.asarray([f"obj_{i:03d}" for i in range(objects)], dtype="U32")
    # A real stable point key repeats over native frames; no hidden dense tensor.
    sf = np.arange(frames, dtype=np.int32) if objects else np.empty(0, np.int32)
    arrays = dict(timestamps=np.arange(frames, dtype=float) / 30, frame_index=np.arange(frames, dtype=np.int32),
        hands_world=hands, hand_valid=np.ones((frames, 2), bool),
        camera_to_world=np.broadcast_to(np.eye(4), (frames, 4, 4)).copy(),
        object_ids=ids, object_kind=np.ones(objects, np.int8), object_pose_world=pose,
        object_pose_valid=np.ones((frames, objects), bool), object_observed=np.ones((frames, objects), bool),
        surface_frame_index=sf, surface_object_index=np.zeros(len(sf), np.int32),
        surface_point_index=np.zeros(len(sf), np.int32), surface_world=np.ones((len(sf), 3), np.float32),
        surface_frame_ptr=np.arange(frames + 1, dtype=np.int64) if objects else np.zeros(frames + 1, np.int64),
        point_keys=np.asarray(["tracklet_0/7"] if objects else [], dtype="U32"),
        point_object_index=np.zeros(int(bool(objects)), np.int32))
    meta = {"sequence_id": "uid", "frame_count": frames,
        "entities": [{"id": identity, "pose_scope": "whole_object_candidate"} for identity in ids.tolist()]}
    np.savez_compressed(directory / "sequence.npz", **arrays)
    write(directory / "metadata.json", meta)
    row = {"sequence_id": "uid", "split": "train", "frame_count": frames,
        "sequence_path": "sequences/uid/sequence.npz", "metadata_path": "sequences/uid/metadata.json"}
    return root, row, arrays, meta


def save(root, row, arrays, meta=None):
    np.savez_compressed(root / row["sequence_path"], **arrays)
    if meta is not None:
        write(root / row["metadata_path"], meta)


def state(value=1, **kwargs):
    return {"id": "state", "layer": "relation", "type": "contact", "subject": "left_hand",
        "object_id": "obj_000", "secondary_object_id": None, "start_frame": 1, "end_frame": 4,
        "value": value, "evidence_frames": [1, 2, 3], "silver_label_valid": True,
        "annotation_source": "assistant_visual_review", "human_label_valid": False, **kwargs}


def dense_review(tmp_path, states=(), transitions=(), *, frame_count=8, covered=None):
    root = tmp_path / "v5"
    page = root / "evidence/page.jpg"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_bytes(b"frozen actual evidence bytes")
    covered = list(range(frame_count)) if covered is None else covered
    manifest = write(root / "evidence/manifest.json", {"sequence_id": "uid", "pages": [
        {"path": "evidence/page.jpg", "sha256": digest(page), "frame_indices": covered}]})
    doc = {"sequence_id": "uid", "frame_count": frame_count, "annotation_source": "assistant_visual_review",
        "reviewer_agent": "test_reviewer", "reviewed_at": "2026-10-24T00:00:00Z",
        "human_label_valid": False, "reviewed_ranges": [{"start_frame": 0, "end_frame": frame_count}],
        "entities": [{"entity_id": "obj_000"}, {"entity_id": "obj_001"}],
        "state_segments": list(states), "transition_events": list(transitions),
        "evidence_manifests": [{"path": "evidence/manifest.json", "sha256": digest(manifest),
            "reviewed_page_paths": ["evidence/page.jpg"], "reviewed_page_sha256": [digest(page)]}]}
    path = write(root / "dense_reviews/uid.json", doc)
    return path, doc


def codes(report, status="fail"):
    return {x["code"] for x in report["checks"] if x["status"] == status}


def test_no_geometry_or_review_is_unknown_not_structural_failure(tmp_path):
    root, row, arrays, _ = fixture(tmp_path, objects=0)
    report = review_sequence(root, row)
    assert not codes(report)
    assert report["metrics"]["semantic_pending_frames"] == 8
    assert report["metrics"]["continuous_reviewed_frames"] == 0
    assert report["metrics"]["known_state_negative_frame_cells"] == 0
    assert report["metrics"]["frames_without_effective_geometry"] == 8
    assert report["metrics"]["full_annotation_complete"] is False
    assert report["metrics"]["whole_object_6d_certified"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("corruption,expected", [
    ("timestamp", "native_timeline"), ("frame", "native_timeline"),
    ("hand_nan", "hand_mask_finite"), ("hand_hidden_finite", "hand_mask_finite"),
    ("duplicate_id", "object_identity"), ("reflection", "object_se3"),
    ("camera_last_row", "camera_se3"), ("csr", "surface_csr"),
    ("point_bounds", "surface_csr"), ("point_owner", "surface_csr"),
    ("false_observed", "object_observed_mask"), ("integer_mask", "array_schema"),
])
def test_real_structural_corruptions_fail(tmp_path, corruption, expected):
    root, row, a, _ = fixture(tmp_path, objects=2)
    if corruption == "timestamp": a["timestamps"][3] = a["timestamps"][2]
    if corruption == "frame": a["frame_index"][3] = 2
    if corruption == "hand_nan": a["hands_world"][2, 0, 1, 0] = np.nan
    if corruption == "hand_hidden_finite": a["hand_valid"][2, 0] = False
    if corruption == "duplicate_id": a["object_ids"][1] = a["object_ids"][0]
    if corruption == "reflection": a["object_pose_world"][3, 0, 0, 0] = -1
    if corruption == "camera_last_row": a["camera_to_world"][0, 3, 0] = 1
    if corruption == "csr": a["surface_frame_ptr"][2] = 1
    if corruption == "point_bounds": a["surface_point_index"][2] = 50
    if corruption == "point_owner": a["point_object_index"][0] = 1
    if corruption == "false_observed": a["object_observed"][0, 0] = False
    if corruption == "integer_mask": a["hand_valid"] = a["hand_valid"].astype(np.int8)
    save(root, row, a)
    report = review_sequence(root, row)
    assert expected in codes(report)
    assert report["metrics"]["automated_checks_passed"] is False
    json.dumps(report, allow_nan=False)


def test_geometry_exclusion_changes_only_effective_counts_and_respects_range(tmp_path):
    root, row, a, _ = fixture(tmp_path)
    before = digest(root / row["sequence_path"])
    annotation = write(tmp_path / "v3/annotations/uid.json", {"sequence_id": "uid", "frame_count": 8,
        "geometry_exclusions": [{"object_id": "obj_000", "scope": "frame_range", "start_frame": 2, "end_frame": 5}]})
    result = review_sequence(root, row, annotation)
    assert not codes(result)
    m = result["metrics"]
    assert (m["raw_pose_observations"], m["effective_pose_observations"], m["geometry_excluded_pose_observations"]) == (8, 5, 3)
    assert (m["raw_surface_observations"], m["effective_surface_observations"], m["geometry_excluded_surface_observations"]) == (8, 5, 3)
    assert digest(root / row["sequence_path"]) == before
    doc = json.loads(annotation.read_text());doc["geometry_exclusions"][0] = {"object_id": "obj_000", "scope": "entity_until_reidentified"}
    write(annotation, doc)
    result = review_sequence(root, row, annotation)
    assert result["metrics"]["frames_with_effective_geometry"] == 0


def test_nonrigid_surface_is_appropriate_and_whole_pose_is_rejected(tmp_path):
    root, row, a, meta = fixture(tmp_path)
    a["object_kind"][0] = 2
    save(root, row, a)
    assert "nonrigid_pose_scope" in codes(review_sequence(root, row))
    a["object_pose_valid"][:] = False
    a["object_pose_world"][:] = np.nan
    save(root, row, a)
    report = review_sequence(root, row)
    assert not codes(report)
    assert report["metrics"]["nonrigid_pose_not_applicable_objects"] == 1
    assert report["metrics"]["effective_surface_observations"] == 8
    assert not any(i["code"] == "rigid_pose_missing" for i in report["issues"])


def test_motion_thresholds_only_warn_and_do_not_bridge_missing_frames(tmp_path):
    root, row, a, _ = fixture(tmp_path)
    a["hands_world"][3:, 0] += 2
    a["object_pose_world"][3:, 0, 0, 3] = 2
    a["object_pose_world"][4:, 0, :2, :2] = [[-1, 0], [0, -1]]
    save(root, row, a)
    report = review_sequence(root, row)
    assert not codes(report)
    assert "motion_outliers" in codes(report, "warn")
    assert report["metrics"]["object_rotation_warning_observations"] == 1
    a["object_pose_valid"][2:5] = False
    a["object_pose_world"][2:5] = np.nan
    save(root, row, a)
    report = review_sequence(root, row)
    assert report["metrics"]["object_speed_warning_observations"] == 0
    assert report["metrics"]["object_rotation_warning_observations"] == 0


def test_known_negative_and_unmentioned_relation_are_separate(tmp_path):
    root, row, _, _ = fixture(tmp_path, objects=2)
    path, doc = dense_review(tmp_path, [state(0), state(1, id="second", object_id="obj_001")])
    report = review_sequence(root, row, review_path=path)
    assert not codes(report)
    m = report["metrics"]
    assert m["continuous_reviewed_frames"] == 8
    assert m["known_state_positive_frame_cells"] == 3
    assert m["known_state_negative_frame_cells"] == 3
    assert not m["full_annotation_complete"]
    json.dumps(report, allow_nan=False)


def test_conflicting_labels_fail_even_when_evidence_files_missing(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    path, doc = dense_review(tmp_path, [state(0), state(1, id="conflict")])
    (path.parent.parent / "evidence/page.jpg").unlink()
    report = review_sequence(root, row, review_path=path)
    assert "state_conflict" in codes(report)
    assert report["metrics"]["continuous_reviewed_frames"] == 0
    assert report["metrics"]["known_state_positive_frame_cells"] == 0


@pytest.mark.parametrize("change,expected", [
    ("identity", "review_entity_reference"), ("range", "review_label_range"),
    ("evidence_gap", "state_evidence_range"), ("unknown_value", "state_value"),
    ("human", "review_provenance"), ("page_tampered", "review_evidence"),
])
def test_review_identity_ranges_and_actual_evidence_are_checked(tmp_path, change, expected):
    root, row, _, _ = fixture(tmp_path)
    path, doc = dense_review(tmp_path, [state()])
    record = doc["state_segments"][0]
    if change == "identity": record["object_id"] = "not_declared"
    if change == "range": record["end_frame"] = 30
    if change == "evidence_gap": record["evidence_frames"].remove(2)
    if change == "unknown_value": record["value"] = None
    if change == "human": record["human_label_valid"] = True
    if change == "page_tampered": (path.parent.parent / "evidence/page.jpg").write_bytes(b"different")
    write(path, doc)
    assert expected in codes(review_sequence(root, row, review_path=path))


def test_partial_evidence_cannot_complete_claimed_review(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    path, _ = dense_review(tmp_path, covered=[0, 1, 4, 5])
    report = review_sequence(root, row, review_path=path)
    assert report["metrics"]["continuous_reviewed_frames"] == 4
    assert report["metrics"]["semantic_pending_frames"] == 4
    assert "review_evidence" in codes(report, "pending")


def test_censored_transition_is_pending_and_cannot_be_frame_resolved(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    event = {"id": "t", "layer": "transition", "type": "release", "subject": "left_hand",
        "object_id": "obj_000", "boundary_frame_range": [2, 5], "evidence_frames": [1, 2, 3, 4, 5],
        "censored": True, "frame_resolved": False, "silver_label_valid": True,
        "annotation_source": "assistant_visual_review", "human_label_valid": False}
    path, doc = dense_review(tmp_path, transitions=[event])
    report = review_sequence(root, row, review_path=path)
    assert not codes(report)
    assert report["metrics"]["censored_transition_events"] == 1
    doc["transition_events"][0]["frame_resolved"] = True
    write(path, doc)
    assert "transition_precision" in codes(review_sequence(root, row, review_path=path))


def test_confirmed_range_event_requires_observed_before_after_not_exact_frame(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    event = {"id": "t", "layer": "transition", "type": "release", "subject": "left_hand",
        "object_id": "obj_000", "boundary_frame_range": [2, 5], "evidence_frames": [1, 2, 3, 4],
        "censored": False, "frame_resolved": False, "silver_label_valid": True,
        "annotation_source": "assistant_visual_review", "occurrence_confirmed": True,
        "before_after_observed": True, "before_evidence_frame": 1, "after_evidence_frame": 4}
    path, doc = dense_review(tmp_path, transitions=[event])
    assert not codes(review_sequence(root, row, review_path=path))
    doc["transition_events"][0]["after_evidence_frame"] = 6
    write(path, doc)
    assert "transition_before_after" in codes(review_sequence(root, row, review_path=path))


def test_overlapping_censored_events_keep_distinct_source_context(tmp_path):
    root, row, _, _ = fixture(tmp_path, objects=2)
    events = [
        {"id": "first_contact", "layer": "transition", "type": "contact_onset", "subject": "left_hand",
         "object_id": "obj_000", "secondary_object_id": None, "boundary_frame_range": [2, 4],
         "evidence_frames": [1, 2, 3, 4], "censored": True, "frame_resolved": False,
         "silver_label_valid": True, "annotation_source": "assistant_visual_review",
         "reason_zh": "手指首次接触被遮挡。"},
        {"id": "support_disappears", "layer": "transition", "type": "support_lost", "subject": "object",
         "object_id": "obj_000", "secondary_object_id": "obj_001", "boundary_frame_range": [3, 6],
         "evidence_frames": [2, 3, 4, 5, 6], "censored": True, "frame_resolved": False,
         "silver_label_valid": True, "annotation_source": "assistant_visual_review",
         "reason_zh": "底部离开支撑面的瞬间不可见。"},
    ]
    path, _ = dense_review(tmp_path, transitions=events)
    before = digest(path)
    report = review_sequence(root, row, review_path=path)
    assert not codes(report)
    issues = [i for i in report["issues"] if i["code"] == "transition_censored"]
    assert len(issues) == 2
    for issue, event in zip(issues, events):
        assert issue["source_record_id"] == event["id"]
        assert issue["event_type"] == event["type"]
        assert issue["subject"] == event["subject"]
        assert issue["secondary_object_id"] == event["secondary_object_id"]
        assert issue["source_reason_zh"] == event["reason_zh"]
        assert [issue["start_frame"], issue["end_frame"]] == event["boundary_frame_range"]
    assert digest(path) == before


def test_source_sha_failure_and_missing_sequence_keep_original_frame_count(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    row["sha256"] = {"sequence": "wrong"}
    assert "source_sha256" in codes(review_sequence(root, row))
    (root / row["sequence_path"]).unlink()
    report = review_sequence(root, row)
    assert "sequence_files" in codes(report)
    assert report["frame_count"] == report["metrics"]["semantic_pending_frames"] == 8


def test_dependencies_include_missing_nested_pages_and_survive_bad_json(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    review_path, doc = dense_review(tmp_path)
    annotation_path = write(tmp_path / "v3/annotations/uid.json", {"sequence_id": "uid", "frame_count": 8,
        "entities": [{"evidence": [{"gallery_path": "evidence/missing.jpg", "gallery_sha256": "x"}]}]})
    dependency = set(sequence_dependencies(root, row, annotation_path, review_path))
    assert annotation_path.parent.parent / "evidence/missing.jpg" in dependency
    assert review_path.parent.parent / "evidence/page.jpg" in dependency
    assert review_path.parent.parent / "evidence/manifest.json" in dependency
    (review_path.parent.parent / "evidence/manifest.json").write_text("invalid json")
    assert dependency == set(sequence_dependencies(root, row, annotation_path, review_path))
    review_path.write_text("invalid json")
    assert review_path in sequence_dependencies(root, row, annotation_path, review_path)


def test_invalid_thresholds_fail_without_computing_or_editing_data(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    for config in ({"hand_speed_mps": float("nan")}, {"hand_speed_mps": -1}, {"auto_approve": 1}):
        assert "threshold_config" in codes(review_sequence(root, row, thresholds=config))


def test_weak_events_are_bound_but_never_count_as_reviewed_supervision(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    path = write(root / "sequences/uid/events.json", {"records": [
        {"start_frame": 1, "end_frame": 3, "object_index": -1, "secondary_object_index": -1,
         "layer": "operation", "type": "hold", "label_valid": False, "weak_positive": True}]})
    row["events_path"] = "sequences/uid/events.json"
    row["sha256"] = {"events": digest(path)}
    result = review_sequence(root, row)
    assert not codes(result)
    assert result["metrics"]["known_state_positive_frame_cells"] == 0
    assert path in sequence_dependencies(root, row)
    path.write_text('{"records":[]}')
    assert "source_sha256" in codes(review_sequence(root, row))


def test_generated_contact_sheet_without_saved_reviewer_is_not_review(tmp_path):
    root, row, _, _ = fixture(tmp_path)
    path, doc = dense_review(tmp_path, [state()])
    del doc["reviewer_agent"]
    write(path, doc)
    result = review_sequence(root, row, review_path=path)
    assert "review_provenance" in codes(result)
    assert result["metrics"]["continuous_reviewed_frames"] == 0


def test_forbidden_metric_geometry_and_reversed_hand_order_are_not_accepted(tmp_path):
    root, row, arrays, meta = fixture(tmp_path)
    meta.update(metric_geometry_allowed=False, hand_order=["right", "left"])
    save(root, row, arrays, meta)
    report = review_sequence(root, row)
    assert {"calibration_geometry_gate", "hand_order"} <= codes(report)
