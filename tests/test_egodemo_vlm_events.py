import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.annotation.egodemo_events import (
    AnnotationError,
    convert_vlm_events,
    convert_vlm_none_intervals,
    load_native_timeline,
    prepare_annotations,
)
from src.annotation.review import REVIEW_COLUMNS, apply_review
from src.annotation.schema import project_frame_label
from src.annotation.vlm_annotate import VideoTimeline, parse_model_response


REVIEWER = "reviewer-a"
REVIEWED_AT = "2026-10-22T12:00:00+00:00"


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _vlm_events():
    return [
        {
            "event_id": "phase-grasp",
            "subject": "left",
            "object_hint": "semantic cup, not a track ID",
            "object_id": "obj_000",
            "interaction_type": "grasp",
            "start_time": 0.0,
            "end_time": 1.0,
            "evidence_frames": [0, 1],
            "source_event_ids": ["1"],
            "confidence": 0.9,
        },
        {
            "event_id": "phase-hold",
            "subject": "right",
            "object_hint": "semantic cup, not a track ID",
            "object_id": "obj_000",
            "interaction_type": "hold",
            "start_time": 0.5,
            "end_time": 1.5,
            "evidence_frames": [1, 2],
            "confidence": None,
        },
    ]


@pytest.fixture
def case(tmp_path):
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    timestamps = np.arange(6, dtype=np.float64) * 0.5
    np.savez(
        sequence / "sequence.npz",
        timestamps=timestamps,
        frame_index=np.arange(6, dtype=np.int64),
        hand_valid=np.ones((6, 2), dtype=bool),
    )
    source = {
        "episode_description": "original text is preserved, never mapped to HOI labels",
        "subtask_annotations": [
            {"subtask_id": 1, "start_time": 0.0, "end_time": 1.5,
             "description": "原文：完整的拿取过程，不拆成左右手或七类标签"},
            {"subtask_id": "2", "start_time": 1.0, "end_time": 2.5,
             "description": "original overlapping semantic phase"},
        ],
    }
    _write_json(sequence / "events.json", source)
    _write_json(sequence / "meta.json", {"source_episode_uuid": "seq", "fps": 2.0})
    vlm = tmp_path / "vlm.json"
    _write_json(vlm, {
        "run_id": "run-a",
        "model": "test-vlm",
        "sequence_id": "seq",
        "timeline": "native_annotated",
        "frame_count": 6,
        "events": _vlm_events(),
    })
    pose = tmp_path / "pose6d.npz"
    bboxes = np.tile(np.asarray([10, 10, 50, 50], dtype=np.int32), (2, 6, 1))
    np.savez(pose, object_ids=np.asarray(["obj_000", "obj_001"]), bboxes=bboxes)
    hand = tmp_path / "hand.npz"
    left = np.full((6, 21, 3), np.nan, dtype=np.float32)
    right = np.full((6, 21, 3), np.nan, dtype=np.float32)
    left[:, 0, :] = [0.2, 0.2, 0.0]
    right[:, 0, :] = [0.8, 0.8, 0.0]
    np.savez(
        hand, left_hand=left, right_hand=right,
        hand_valid=np.ones((6, 2), dtype=bool),
        image_width=np.asarray(100), image_height=np.asarray(100),
    )
    return {
        "root": tmp_path, "sequence": sequence, "vlm": vlm, "pose": pose, "hand": hand,
        "source": source, "output": tmp_path / "prepared",
    }


def _prepare(case, **kwargs):
    options = {"hand_pose": case["hand"], "pose6d": case["pose"]}
    options.update(kwargs)
    return prepare_annotations(case["sequence"], case["vlm"], case["output"], **options)


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _accept(row, timestamp=REVIEWED_AT):
    row.update(review_status="accepted", reviewer=REVIEWER, reviewed_at=timestamp)


def test_half_open_native_mapping_source_preservation_and_overlap(case):
    native = load_native_timeline(case["sequence"])
    events = convert_vlm_events(native, [
        (event, {"sequence_id": "seq", "timeline": "native_annotated"}, "test")
        for event in _vlm_events()
    ])
    assert (events[0]["start_frame"], events[0]["end_frame"]) == (0, 2)
    assert (events[1]["start_frame"], events[1]["end_frame"]) == (1, 3)
    assert events[0]["source_event_ids"] == ["1"]
    assert events[1]["source_event_ids"] == ["1", "2"]
    assert events[0]["source_events"][0]["description"] == case["source"]["subtask_annotations"][0]["description"]
    assert events[0]["label_source"] == "source_text+vlm"
    assert all(event["review_status"] == "pending" for event in events)
    assert native["source_maps"][0]["mapped_frame_count"] == 3
    assert native["source_maps"][1]["start_frame"] == 2


@pytest.mark.parametrize("change,match", [
    ({"timeline": "raw"}, "timeline"),
    ({"frame_count": 18}, "frame_count"),
    ({"start_time": 8.0, "end_time": 9.0, "evidence_frames": [0]}, "native domain"),
    ({"source_event_ids": ["2"], "end_time": 0.5, "evidence_frames": [0]},
     "temporally overlap|timeline conflict"),
    ({"evidence_frames": [2]}, "evidence frame"),
    ({"evidence_frames": []}, "evidence_frames"),
])
def test_raw_annotated_or_invalid_time_alignment_fails(case, change, match):
    native = load_native_timeline(case["sequence"])
    event = {**_vlm_events()[0], **change}
    with pytest.raises(AnnotationError, match=match):
        convert_vlm_events(native, [(event, {}, "test")])


def test_sparse_or_raw_frame_indices_are_not_guessed(case):
    path = case["sequence"] / "sequence.npz"
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files}
    values["frame_index"] = np.asarray([10, 11, 12, 13, 14, 15])
    np.savez(path, **values)
    with pytest.raises(AnnotationError, match="explicit mapping"):
        load_native_timeline(case["sequence"])


def test_description_is_never_used_to_fill_vlm_hoi_fields(case):
    native = load_native_timeline(case["sequence"])
    event = _vlm_events()[0]
    del event["subject"]
    event["notes"] = "left hand grasp cup"
    with pytest.raises(AnnotationError, match="subject"):
        convert_vlm_events(native, [(event, {}, "test")])


def test_prepare_tracks_geometry_and_pending_are_not_labels(case):
    result = _prepare(case)
    events = _events(case["output"] / "events.jsonl")
    assert [event["geometry_status"] for event in events] == ["passed", "failed"]
    assert all(event["object_id"] == "obj_000" for event in events)
    assert all(event["track_link_status"] == "linked" for event in events)
    assert all(event["geometry_evidence"]["3d_verified"] is False for event in events)
    assert result["structurally_valid_candidates"] == 2
    assert result["track_linked_candidates"] == 2
    assert result["source_events"]["unlabelled_frames"] == 1
    assert result["accepted_none_coverage_frames"] == 0
    assert result["semantic_projection"]["valid_frames"] == 0
    pending = project_frame_label(events, [], 1, ["obj_000"])
    assert pending["valid"] is False
    assert pending["reason"] == "pending_or_needs_revision"
    gap = project_frame_label(events, [], 5, ["obj_000"])
    assert gap["valid"] is False
    assert gap["reason"] == "unlabelled_or_rejected"
    assert _events(case["output"] / "none_intervals.jsonl") == []
    provenance = json.loads((case["output"] / "provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["inputs"]) == 6
    assert all(len(item["sha256"]) == 64 for item in provenance["inputs"])


def test_object_hint_cannot_become_a_stable_track_id(case):
    payload = json.loads(case["vlm"].read_text(encoding="utf-8"))
    payload["events"] = [payload["events"][0]]
    payload["events"][0].pop("object_id")
    payload["events"][0]["object_hint"] = "obj_000"
    _write_json(case["vlm"], payload)
    _prepare(case)
    event = _events(case["output"] / "events.jsonl")[0]
    assert event["object_id"] is None
    assert event["track_link_status"] == "not_provided"
    assert event["object_hint"] == "obj_000"


def test_unknown_proposed_track_is_not_linked(case):
    payload = json.loads(case["vlm"].read_text(encoding="utf-8"))
    payload["events"] = [payload["events"][0]]
    payload["events"][0]["object_id"] = "cup"
    _write_json(case["vlm"], payload)
    _prepare(case)
    event = _events(case["output"] / "events.jsonl")[0]
    assert event["object_id"] is None
    assert event["proposed_object_id"] == "cup"
    assert event["track_link_status"] == "not_found"


def test_track_frame_length_mismatch_fails(case):
    with np.load(case["pose"]) as archive:
        pose = {key: archive[key] for key in archive.files}
    pose["bboxes"] = pose["bboxes"][:, :-1]
    np.savez(case["pose"], **pose)
    with pytest.raises(AnnotationError, match="same native timeline"):
        _prepare(case)


def test_track_timestamps_conflict_with_native_fails(case):
    with np.load(case["pose"]) as archive:
        pose = {key: archive[key] for key in archive.files}
    pose["timestamps"] = np.arange(6, dtype=np.float64) * 0.5 + 10.0
    np.savez(case["pose"], **pose)
    with pytest.raises(AnnotationError, match="timestamps conflict"):
        _prepare(case)


def test_geometry_not_checked_and_inconclusive_are_distinct(case):
    pose_only = case["root"] / "pose-only"
    prepare_annotations(case["sequence"], case["vlm"], pose_only, pose6d=case["pose"])
    events = _events(pose_only / "events.jsonl")
    assert all(event["track_link_status"] == "linked" for event in events)
    assert all(event["geometry_status"] == "not_checked" for event in events)

    with np.load(case["hand"]) as archive:
        hand = {key: archive[key] for key in archive.files if not key.startswith("image_")}
    without_dimensions = case["root"] / "hand-no-dimensions.npz"
    np.savez(without_dimensions, **hand)
    inconclusive = case["root"] / "inconclusive"
    prepare_annotations(
        case["sequence"], case["vlm"], inconclusive,
        hand_pose=without_dimensions, pose6d=case["pose"],
    )
    events = _events(inconclusive / "events.jsonl")
    assert all(event["geometry_status"] == "inconclusive" for event in events)
    assert all(event["review_status"] == "pending" for event in events)


def test_accepted_requires_reviewer_and_timezone_timestamp(case):
    _prepare(case)
    rows = _rows(case["output"] / "review.csv")
    _accept(rows[0])
    rows[0]["reviewer"] = ""
    _save_rows(case["output"] / "review.csv", rows)
    with pytest.raises(AnnotationError, match="requires reviewer"):
        apply_review(case["output"], case["root"] / "missing-reviewer")
    _accept(rows[0], timestamp="")
    _save_rows(case["output"] / "review.csv", rows)
    with pytest.raises(AnnotationError, match="requires reviewed_at"):
        apply_review(case["output"], case["root"] / "missing-time")
    _accept(rows[0], timestamp="2026-10-22 12:00:00")
    _save_rows(case["output"] / "review.csv", rows)
    with pytest.raises(AnnotationError, match="timezone"):
        apply_review(case["output"], case["root"] / "naive-time")


def test_apply_preserves_original_candidate_and_revision_history(case):
    _prepare(case)
    rows = _rows(case["output"] / "review.csv")
    _accept(rows[0])
    rows[0].update(end_frame="1", evidence_frames="[0]")
    _save_rows(case["output"] / "review.csv", rows)
    reviewed = case["root"] / "reviewed"
    apply_review(case["output"], reviewed)
    original = _events(case["output"] / "events.jsonl")[0]
    event = _events(reviewed / "events.jsonl")[0]
    assert original["review_status"] == "pending"
    assert event["original_candidate"]["end_frame"] == 2
    assert event["original_candidate"]["label_source"] == "source_text+vlm"
    assert event["original_candidate"]["review_status"] == "pending"
    assert event["end_frame"] == 1
    assert event["label_source"] == "human"
    assert len(event["revision_history"]) == 1
    assert event["revision_history"][0]["before"]["review_status"] == "pending"
    assert project_frame_label([event], [], 0, ["obj_000"])["valid"] is True

    rows = _rows(reviewed / "review.csv")
    rows[0].update(review_status="needs_revision", reviewed_at="2026-10-22T13:00:00+00:00",
                   end_frame="2", evidence_frames="[0,1]")
    _save_rows(reviewed / "review.csv", rows)
    second = case["root"] / "second-review"
    apply_review(reviewed, second)
    revised = _events(second / "events.jsonl")[0]
    assert len(revised["revision_history"]) == 2
    assert revised["original_candidate"]["end_frame"] == 2
    assert revised["review_status"] == "needs_revision"
    assert project_frame_label([revised], [], 0, ["obj_000"])["valid"] is False


def test_overlapping_accepted_events_remain_overlapping_and_invalid(case):
    _prepare(case)
    rows = _rows(case["output"] / "review.csv")
    for row in rows:
        _accept(row)
    _save_rows(case["output"] / "review.csv", rows)
    reviewed = case["root"] / "accepted-overlap"
    apply_review(case["output"], reviewed)
    events = _events(reviewed / "events.jsonl")
    assert len(events) == 2
    assert project_frame_label(events, [], 0, ["obj_000"])["interaction_name"] == "grasp"
    overlap = project_frame_label(events, [], 1, ["obj_000"])
    assert overlap["valid"] is False
    assert overlap["reason"] == "overlapping_accepted_labels"
    assert set(overlap["event_ids"]) == {"phase-grasp", "phase-hold"}
    summary = json.loads((reviewed / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_event_overlap_frames"] == 1


def test_changing_track_requires_pose_and_can_use_real_alternate_id(case):
    _prepare(case)
    rows = _rows(case["output"] / "review.csv")
    _accept(rows[0])
    rows[0]["object_id"] = "obj_001"
    _save_rows(case["output"] / "review.csv", rows)
    with pytest.raises(AnnotationError, match="requires matching --pose6d"):
        apply_review(case["output"], case["root"] / "unverified-track")
    reviewed = case["root"] / "verified-track"
    apply_review(
        case["output"], reviewed, hand_pose=case["hand"], pose6d=case["pose"],
    )
    event = _events(reviewed / "events.jsonl")[0]
    assert event["object_id"] == "obj_001"
    assert event["track_link_status"] == "linked"


def test_only_explicit_human_accepted_none_covers_unlabelled_gap(case):
    _prepare(case)
    rows = _rows(case["output"] / "review.csv")
    none_row = {column: "" for column in REVIEW_COLUMNS}
    none_row.update(
        record_type="none", record_id="human-none-5", sequence_id="seq",
        review_status="accepted", reviewer=REVIEWER, reviewed_at=REVIEWED_AT,
        start_frame="5", end_frame="6", evidence_frames="[5]", source_event_ids="[]",
        original_start_frame="5", original_end_frame="6", label_source="human",
        geometry_status="not_checked", notes="visually verified no interaction",
    )
    rows.append(none_row)
    _save_rows(case["output"] / "review.csv", rows)
    reviewed = case["root"] / "with-none"
    result = apply_review(case["output"], reviewed)
    events = _events(reviewed / "events.jsonl")
    none = _events(reviewed / "none_intervals.jsonl")
    assert len(none) == 1
    assert none[0]["label_source"] == "human"
    assert none[0]["original_candidate"]["review_status"] == "pending"
    label = project_frame_label(events, none, 5, ["obj_000"])
    assert label["valid"] is True
    assert label["reason"] == "reviewed_none"
    assert label["interaction_name"] == "none"
    assert result["accepted_none_coverage_frames"] == 1
    assert result["source_events"]["unlabelled_frames"] == 1


def test_pending_rows_cannot_silently_modify_boundaries(case):
    _prepare(case)
    rows = _rows(case["output"] / "review.csv")
    rows[0]["start_frame"] = "1"
    rows[0]["evidence_frames"] = "[1]"
    _save_rows(case["output"] / "review.csv", rows)
    with pytest.raises(AnnotationError, match="pending rows cannot"):
        apply_review(case["output"], case["root"] / "pending-edit")


def test_source_review_is_provenance_and_candidates_remain_pending(case):
    source_review = case["root"] / "annotation_review.csv"
    with source_review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_event_id", "review_status", "subject"])
        writer.writeheader()
        writer.writerows([
            {"source_event_id": "1", "review_status": "pending", "subject": ""},
            {"source_event_id": "2", "review_status": "pending", "subject": ""},
        ])
    _prepare(case, source_review=source_review)
    assert all(event["review_status"] == "pending" for event in _events(case["output"] / "events.jsonl"))
    provenance = json.loads((case["output"] / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_review"]["row_count"] == 2


def test_prepare_apply_and_render_refuse_overwrite(case):
    _prepare(case)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _prepare(case)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        apply_review(case["output"], case["output"])


def test_contact_sheet_and_video_frame_count_validation(case):
    cv2 = pytest.importorskip("cv2")
    from src.annotation.render_review import RenderError, render_review

    _prepare(case)
    video = case["root"] / "annotated.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (100, 80))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4 writer is unavailable")
    for frame in range(6):
        writer.write(np.full((80, 100, 3), frame * 30, dtype=np.uint8))
    writer.release()
    rendered_dir = case["root"] / "prepared-with-video"
    prepare_annotations(
        case["sequence"], case["vlm"], rendered_dir,
        hand_pose=case["hand"], pose6d=case["pose"], video=video,
        render_format="contact-sheet",
    )
    rendered_summary = json.loads((rendered_dir / "summary.json").read_text(encoding="utf-8"))
    assert rendered_summary["render"]["path"] == str(rendered_dir / "contact_sheet.jpg")
    assert (rendered_dir / "contact_sheet.jpg").stat().st_size > 0
    output = case["root"] / "sheet.jpg"
    result = render_review(case["output"], video, output, output_format="contact-sheet")
    assert output.stat().st_size > 0
    assert result["output_verified"] is True
    assert result["decoded_annotated_frames"] == 6
    assert result["panel_count"] == 4
    with pytest.raises(FileExistsError):
        render_review(case["output"], video, output, output_format="contact-sheet")

    short_video = case["root"] / "raw-wrong-length.mp4"
    writer = cv2.VideoWriter(str(short_video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (100, 80))
    for frame in range(5):
        writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
    writer.release()
    with pytest.raises(RenderError, match="refusing raw/positional"):
        render_review(case["output"], short_video, case["root"] / "bad.jpg", output_format="contact-sheet")


def _native_event():
    event = _vlm_events()[0]
    event.update(
        start_frame=1, end_frame=3, start_time=0.5, end_time=1.5,
        evidence_frames=[1, 2], source_event_ids=["1"],
    )
    return event


@pytest.mark.parametrize("change,match", [
    ({"start_frame": True}, "native integers"),
    ({"start_frame": 1.0}, "native integers"),
    ({"start_frame": 3}, "start_frame < end_frame"),
    ({"end_frame": 7}, "end_frame <= 6"),
    ({"evidence_frames": [3]}, "evidence frame"),
    ({"evidence_frames": [9]}, "evidence frame"),
    ({"end_time": 1.0}, "boundaries disagree"),
])
def test_invalid_native_boundaries_evidence_and_mixed_seconds_fail(case, change, match):
    native = load_native_timeline(case["sequence"])
    event = {**_native_event(), **change}
    with pytest.raises(AnnotationError, match=match):
        convert_vlm_events(native, [(event, {}, "native-fixture")])


def test_partial_native_or_seconds_boundaries_fail(case):
    native = load_native_timeline(case["sequence"])
    event = _native_event()
    del event["end_frame"]
    with pytest.raises(AnnotationError, match="both start_frame and end_frame"):
        convert_vlm_events(native, [(event, {}, "partial-native")])
    event = _native_event()
    del event["end_time"]
    with pytest.raises(AnnotationError, match="both start and end"):
        convert_vlm_events(native, [(event, {}, "partial-seconds")])


def test_native_source_overlap_uses_derived_native_timestamps(case):
    native = load_native_timeline(case["sequence"])
    event = _native_event()
    event.update(start_frame=2, end_frame=4, start_time=0.75, end_time=2.0,
                 evidence_frames=[2, 3])
    converted = convert_vlm_events(native, [(event, {}, "native-source")])
    assert (converted[0]["start_time"], converted[0]["end_time"]) == (1.0, 2.0)
    assert converted[0]["source_event_ids"] == ["1"]

    native["source_maps"].append({
        "source_event_id": "boundary", "start_time": 0.0, "end_time": 1.0,
        "description": "ends exactly at the native start", "start_frame": 0,
        "end_frame": 2, "mapped_frame_count": 2,
        "original_event": {"subtask_id": "boundary", "start_time": 0.0, "end_time": 1.0},
    })
    event["source_event_ids"] = ["boundary"]
    with pytest.raises(AnnotationError, match="temporally overlap"):
        convert_vlm_events(native, [(event, {}, "native-source-boundary")])


@pytest.mark.parametrize("field,value", [
    ("review_status", "accepted"),
    ("geometry_status", "passed"),
    ("annotation_mode", "causal"),
    ("allowed_as_model_input", True),
])
def test_prepare_does_not_launder_reviewed_or_causal_inputs(case, field, value):
    native = load_native_timeline(case["sequence"])
    event = {**_native_event(), field: value}
    with pytest.raises(AnnotationError, match="must"):
        convert_vlm_events(native, [(event, {}, "event-provenance")])
    interval = {
        "interval_id": "none-native", "start_frame": 5, "end_frame": 6,
        "evidence_frames": [5], "confidence": None, "source_event_ids": [],
        "notes": "explicit fixture", field: value,
    }
    with pytest.raises(AnnotationError, match="must"):
        convert_vlm_none_intervals(native, [(interval, {}, "none-provenance")])


def test_prepare_rejects_event_none_identifier_collision(case):
    none_path = case["root"] / "none.json"
    _write_json(none_path, [{
        "interval_id": "phase-grasp", "start_frame": 5, "end_frame": 6,
        "evidence_frames": [5], "confidence": None, "source_event_ids": [],
        "review_status": "pending", "geometry_status": "not_checked",
        "annotation_mode": "offline_noncausal", "allowed_as_model_input": False,
    }])
    with pytest.raises(AnnotationError, match="collide across inputs"):
        _prepare(case, vlm_none_intervals=none_path)


def test_generic_qwen_native_events_none_prepare_review_apply(tmp_path):
    pts = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
    timeline = VideoTimeline(
        pts_seconds=pts, duration_seconds=2.0, stream={},
        format_duration_seconds=2.0, duration_source="native_fixture",
    )
    evidence = [
        {"frame_index": frame, "pts_seconds": pts[frame], "image_sha256": f"fixture-{frame}"}
        for frame in (0, 2, 4, 7)
    ]
    source = {
        "episode_description": "source text remains provenance",
        "subtask_annotations": [{
            "subtask_id": "source-1", "start_time": 0.0, "end_time": 2.0,
            "description": "original source interval",
        }],
    }
    model_payload = {
        "events": [{
            "subject": "left", "object_hint": "semantic cup", "interaction_type": "grasp",
            "start_sec": 0.25, "end_sec": 1.25, "evidence_times_sec": [0.5, 1.0],
            "confidence": 0.8, "notes": "native candidate", "source_event_ids": [],
        }],
        "none_intervals": [{
            "start_sec": 1.5, "end_sec": 2.0, "evidence_times_sec": [1.75],
            "confidence": 0.7, "notes": "explicit none candidate",
            "source_event_ids": ["source-1"],
        }],
    }
    response, raw_events, raw_none = parse_model_response(
        json.dumps(model_payload), "generic-seq", "fixture-run", timeline, evidence,
        source_events=source,
    )
    assert response["status"] == "valid"
    assert raw_events[0]["start_frame"] == 1
    assert raw_events[0]["end_frame"] == 5
    assert raw_events[0]["evidence_frames"] == [2, 4]
    assert raw_events[0]["object_id"] is None

    sequence = tmp_path / "sequence"
    sequence.mkdir()
    np.savez(
        sequence / "sequence.npz", timestamps=np.asarray(pts),
        frame_index=np.arange(8, dtype=np.int64),
    )
    _write_json(sequence / "events.json", source)
    _write_json(sequence / "meta.json", {"source_episode_uuid": "generic-seq"})
    events_path = tmp_path / "events.jsonl"
    none_path = tmp_path / "none_intervals.jsonl"
    events_path.write_text("".join(json.dumps(item) + "\n" for item in raw_events), encoding="utf-8")
    none_path.write_text("".join(json.dumps(item) + "\n" for item in raw_none), encoding="utf-8")

    prepared = tmp_path / "prepared"
    result = prepare_annotations(
        sequence, events_path, prepared, vlm_none_intervals=none_path,
    )
    prepared_event = _events(prepared / "events.jsonl")[0]
    prepared_none = _events(prepared / "none_intervals.jsonl")[0]
    assert (prepared_event["start_frame"], prepared_event["end_frame"]) == (1, 5)
    assert (prepared_event["start_time"], prepared_event["end_time"]) == (0.25, 1.25)
    assert prepared_event["evidence_frames"] == [2, 4]
    assert prepared_event["object_id"] is None
    assert prepared_event["source_event_ids"] == []
    assert prepared_event["label_source"] == "vlm"
    assert prepared_none["source_event_ids"] == ["source-1"]
    for record in (prepared_event, prepared_none):
        assert record["review_status"] == "pending"
        assert record["annotation_mode"] == "offline_noncausal"
        assert record["allowed_as_model_input"] is False
    assert not {"subject", "object_id", "object_hint", "interaction_type"} & set(prepared_none)
    assert result["raw_vlm_none_candidates"] == 1
    assert result["structurally_valid_none_candidates"] == 1
    assert result["accepted_none_coverage_frames"] == 0
    provenance = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
    assert {"vlm_events", "vlm_none_intervals"} <= {item["role"] for item in provenance["inputs"]}
    assert (prepared / "vlm_input.jsonl").is_file()
    assert (prepared / "vlm_none_input.jsonl").is_file()

    pending_applied = tmp_path / "pending-applied"
    apply_review(prepared, pending_applied)
    assert _events(pending_applied / "events.jsonl") == [prepared_event]
    assert _events(pending_applied / "none_intervals.jsonl") == [prepared_none]

    rows = _rows(prepared / "review.csv")
    assert [row["record_type"] for row in rows] == ["event", "none"]
    rows[0]["object_id"] = "obj_000"
    for row in rows:
        _accept(row)
    _save_rows(prepared / "review.csv", rows)
    pose = tmp_path / "pose.npz"
    np.savez(
        pose, object_ids=np.asarray(["obj_000"]),
        bboxes=np.tile(np.asarray([0, 0, 20, 20]), (1, 8, 1)),
    )
    reviewed = tmp_path / "reviewed"
    reviewed_result = apply_review(prepared, reviewed, pose6d=pose)
    reviewed_event = _events(reviewed / "events.jsonl")[0]
    reviewed_none = _events(reviewed / "none_intervals.jsonl")[0]
    assert reviewed_event["review_status"] == "accepted"
    assert reviewed_event["object_id"] == "obj_000"
    assert reviewed_event["track_link_status"] == "linked"
    assert reviewed_event["original_candidate"]["object_id"] is None
    assert reviewed_none["review_status"] == "accepted"
    assert reviewed_none["label_source"] == "human"
    assert reviewed_none["source_event_ids"] == ["source-1"]
    assert reviewed_none["original_candidate"]["label_source"] == "vlm"
    assert reviewed_result["raw_vlm_none_candidates"] == 1
    assert reviewed_result["structurally_valid_none_candidates"] == 1
    assert reviewed_result["semantic_projection"]["valid_frames"] == 6
    assert (reviewed / "vlm_none_input.jsonl").is_file()

    interaction = project_frame_label([reviewed_event], [reviewed_none], 2, ["obj_000"])
    negative = project_frame_label([reviewed_event], [reviewed_none], 7, ["obj_000"])
    assert interaction["valid"] is True
    assert interaction["reason"] == "reviewed_event"
    assert negative["valid"] is True
    assert negative["reason"] == "reviewed_none"
    for frame in (0, 5):
        unknown = project_frame_label([reviewed_event], [reviewed_none], frame, ["obj_000"])
        assert unknown["valid"] is False
        assert unknown["reason"] == "unlabelled_or_rejected"


def _responses_payload(**overrides):
    payload = {
        "response_id": "run-a:response:0000",
        "sequence_id": "seq",
        "candidate_counts": {"events": 4, "none_intervals": 0},
        "parsed_json": {"events": [{"candidate": index} for index in range(4)],
                        "none_intervals": []},
        "event_ids": ["phase-grasp", "phase-hold"],
        "none_interval_ids": [],
        "rejected_candidate_count": 2,
    }
    payload.update(overrides)
    return payload


def _write_responses(case, **overrides):
    path = case["root"] / "responses.jsonl"
    path.write_text(json.dumps(_responses_payload(**overrides)) + "\n", encoding="utf-8")
    return path


def test_exact_raw_counts_from_matching_responses_survive_pending_apply(case):
    _write_responses(case)
    result = _prepare(case)
    assert result["raw_vlm_candidates"] == 4
    assert result["structurally_valid_candidates"] == 2
    assert result["structurally_rejected_candidates"] == 2
    assert result["raw_vlm_candidates_exact"] is True
    assert result["raw_vlm_count_source"] == "same_vlm_run_responses_jsonl_candidate_counts"
    assert result["raw_vlm_count_provenance"]["events"]["run_id"] == "run-a"
    provenance = json.loads((case["output"] / "provenance.json").read_text(encoding="utf-8"))
    responses_inputs = [item for item in provenance["inputs"] if item["role"] == "vlm_run_responses"]
    assert len(responses_inputs) == 1
    assert responses_inputs[0]["path"] == str((case["root"] / "responses.jsonl").resolve())

    applied = case["root"] / "applied"
    applied_result = apply_review(case["output"], applied)
    assert applied_result["raw_vlm_candidates"] == 4
    assert applied_result["structurally_valid_candidates"] == 2
    assert applied_result["structurally_rejected_candidates"] == 2
    assert applied_result["raw_vlm_candidates_exact"] is True
    assert applied_result["raw_vlm_none_candidates_exact"] is True
    applied_provenance = json.loads((applied / "provenance.json").read_text(encoding="utf-8"))
    assert "vlm_run_responses" in {item["role"] for item in applied_provenance["inputs"]}
    assert applied_provenance["raw_vlm_count_provenance"]["events"]["count"] == 4


def test_cli_main_builds_prepare_and_apply_parsers():
    from src.annotation import egodemo_events

    with pytest.raises(SystemExit) as excinfo:
        egodemo_events.main(["--help"])
    assert excinfo.value.code == 0
    parser = egodemo_events.argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    egodemo_events._add_prepare_parser(subparsers)
    egodemo_events._add_apply_parser(subparsers)
    prepare = parser.parse_args([
        "prepare", "--sequence-dir", "s", "--vlm-events", "v", "--output-dir", "o",
        "--render-format", "continuous", "--vlm-responses", "r.jsonl",
    ])
    assert prepare.render_format == "continuous"
    assert str(prepare.vlm_responses) == "r.jsonl"
    applied = parser.parse_args([
        "apply", "--annotation-dir", "a", "--output-dir", "o",
        "--render-format", "continuous",
    ])
    assert applied.render_format == "continuous"


def test_missing_responses_fall_back_to_non_exact_counts(case):
    result = _prepare(case)
    assert result["raw_vlm_candidates"] == 2
    assert result["raw_vlm_candidates_exact"] is False
    assert result["structurally_rejected_candidates"] is None
    assert result["structurally_rejected_none_candidates"] is None
    assert result["raw_vlm_count_source"] == "responses_unavailable_structurally_valid_input_fallback"
    provenance = json.loads((case["output"] / "provenance.json").read_text(encoding="utf-8"))
    assert "vlm_run_responses" not in {item["role"] for item in provenance["inputs"]}

    applied = case["root"] / "applied-fallback"
    applied_result = apply_review(case["output"], applied)
    assert applied_result["raw_vlm_candidates"] == 2
    assert applied_result["raw_vlm_candidates_exact"] is False
    assert applied_result["structurally_rejected_candidates"] is None


def test_explicit_missing_responses_path_fails(case):
    with pytest.raises(AnnotationError, match="Missing VLM responses input"):
        _prepare(case, vlm_responses=case["root"] / "missing-responses.jsonl")


@pytest.mark.parametrize("overrides,match", [
    ({"sequence_id": "other-sequence"}, "sequence_id"),
    ({"response_id": "run-b:response:0000"}, "do not match"),
    ({"event_ids": ["phase-grasp"]}, "dangling, incomplete"),
    ({"event_ids": ["phase-grasp", "phase-hold", "ghost-event"]}, "dangling, incomplete"),
    ({"event_ids": ["phase-grasp", "phase-grasp"]}, "unique nonempty"),
    ({"parsed_json": {"events": [{}], "none_intervals": []}}, "parsed_json.events length"),
    ({"rejected_candidate_count": 5}, "rejected_candidate_count disagrees"),
    ({"candidate_counts": {"events": 1, "none_intervals": 0},
      "parsed_json": {"events": [{}], "none_intervals": []}}, "exceeds the raw candidate count"),
])
def test_mismatched_responses_are_rejected(case, overrides, match):
    _write_responses(case, **overrides)
    with pytest.raises(AnnotationError, match=match):
        _prepare(case)


def _write_fixture_video(case, cv2, name="annotated.mp4", frames=6, fps=2.0):
    video = case["root"] / name
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (100, 80))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4 writer is unavailable")
    for frame in range(frames):
        writer.write(np.full((80, 100, 3), frame * 30, dtype=np.uint8))
    writer.release()
    return video


def test_continuous_render_verifies_input_and_output_pts(case):
    cv2 = pytest.importorskip("cv2")
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for continuous review rendering")
    from src.annotation.render_review import render_continuous_review

    _prepare(case)
    video = _write_fixture_video(case, cv2)
    output = case["root"] / "continuous.mp4"
    result = render_continuous_review(
        case["output"], video, output, fps=2.0,
        pts_tolerance_seconds=0.01, output_pts_tolerance_seconds=0.01,
    )
    assert result["decoded_annotated_frames"] == 6
    assert result["displayed_frame_count"] == 6
    assert result["output_decoded_frames"] == 6
    assert result["panel_count"] == 0
    assert result["human_reviewed_frames"] == 0
    assert result["human_reviewed_candidates"] == 0
    assert result["human_review_complete"] is False
    assert result["input_pts"]["status"] == "verified"
    assert result["input_pts"]["frame_count"] == 6
    assert result["output_pts"]["status"] == "verified"
    assert result["output_pts"]["frame_count"] == 6
    assert result["output_timebase"]["mode"] == "constant_frame_rate_playback"
    with pytest.raises(FileExistsError):
        render_continuous_review(case["output"], video, output, fps=2.0)


def test_prepare_continuous_format_writes_static_and_continuous(case):
    cv2 = pytest.importorskip("cv2")
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for continuous review rendering")

    video = _write_fixture_video(case, cv2)
    rendered_dir = case["root"] / "prepared-continuous"
    prepare_annotations(
        case["sequence"], case["vlm"], rendered_dir,
        hand_pose=case["hand"], pose6d=case["pose"], video=video,
        render_format="continuous",
    )
    assert (rendered_dir / "review.mp4").is_file()
    assert (rendered_dir / "review_continuous.mp4").is_file()
    summary = json.loads((rendered_dir / "summary.json").read_text(encoding="utf-8"))
    render = summary["render"]
    assert render["format"] == "continuous"
    assert render["path"] == str(rendered_dir / "review_continuous.mp4")
    assert render["decoded_annotated_frames"] == 6
    assert render["output_decoded_frames"] == 6
    assert render["input_pts"]["status"] == "verified"
    assert render["output_pts"]["status"] == "verified"
    assert render["static_review"]["format"] == "mp4"
    assert render["static_review"]["panel_count"] == 4
    assert render["static_review"]["output_verified"] is True


def test_continuous_render_rejects_shifted_native_timeline(case):
    cv2 = pytest.importorskip("cv2")
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for continuous review rendering")
    from src.annotation.render_review import RenderError, render_continuous_review

    _prepare(case)
    video = _write_fixture_video(case, cv2)
    timeline_path = case["output"] / "timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline["timestamps"] = [value + 0.1 for value in timeline["timestamps"]]
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
    with pytest.raises(RenderError, match="does not match the native timeline"):
        render_continuous_review(case["output"], video, case["root"] / "shifted.mp4", fps=2.0)


def test_continuous_render_requires_ffprobe(case, monkeypatch):
    pytest.importorskip("cv2")
    import src.annotation.render_review as render_review_module

    _prepare(case)
    monkeypatch.setattr(
        render_review_module.shutil, "which", lambda *args, **kwargs: None,
    )
    with pytest.raises(render_review_module.RenderError, match="ffprobe is required"):
        render_review_module.render_continuous_review(
            case["output"], case["root"] / "annotated.mp4", case["root"] / "continuous.mp4",
        )
