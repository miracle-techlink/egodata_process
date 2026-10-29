"""CPU-only unit tests for src/annotation/apply_review.py (fast-review-kit decisions)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.annotation import apply_review
from src.annotation.egodemo_events import AnnotationError
from src.annotation.review import REVIEW_COLUMNS, write_review_csv
from src.annotation.schema import write_jsonl


SEQ_ID = "abcd1234-0000-4000-8000-000000000001"
SEQ8 = SEQ_ID[:8]
REVIEWER = "daniel"


def _event(index, start, end, *, subject="left", hint="test object"):
    return {
        "event_id": f"{SEQ8}_qwen25vl7b_20261023:event:{index:04d}",
        "sequence_id": SEQ_ID,
        "subject": subject,
        "object_id": None,
        "object_hint": hint,
        "interaction_type": "grasp",
        "start_frame": start,
        "end_frame": end,
        "evidence_frames": [start],
        "confidence": 0.8,
        "label_source": "vlm",
        "review_status": "pending",
        "geometry_status": "not_checked",
        "track_link_status": "not_provided",
        "source_event_ids": [],
        "notes": "fixture",
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
    }


def _none_interval(start, end):
    return {
        "interval_id": f"{SEQ8}_qwen25vl7b_20261023:none:{start:04d}",
        "sequence_id": SEQ_ID,
        "start_frame": start,
        "end_frame": end,
        "evidence_frames": [start],
        "confidence": None,
        "label_source": "vlm",
        "review_status": "pending",
        "geometry_status": "not_checked",
        "source_event_ids": [],
        "notes": "fixture none",
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
    }


def _timeline(frame_count):
    return {
        "schema_version": "hoi_annotation_v1",
        "sequence_id": SEQ_ID,
        "timeline": "native_annotated",
        "raw_frame_alignment": False,
        "frame_count": frame_count,
        "frame_index": list(range(frame_count)),
        "timestamps": [frame / 30.0 for frame in range(frame_count)],
        "source_events": [],
    }


def _summary(events, none_intervals, frame_count):
    return {
        "schema_version": "hoi_annotation_v1",
        "sequence_id": SEQ_ID,
        "frame_count": frame_count,
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
        "training_ready": False,
        "semantic_projection_only": True,
        "raw_vlm_candidates": len(events),
        "raw_vlm_none_candidates": len(none_intervals),
        "review_status_counts": {
            "pending": len(events) + len(none_intervals), "accepted": 0,
            "rejected": 0, "needs_revision": 0,
        },
        "render": {"status": "rendered", "format": "mp4", "panel_count": 1},
    }


@pytest.fixture
def collection(tmp_path):
    """One review dir: 3 events + 1 none interval over 40 native frames."""
    frame_count = 40
    events = [_event(0, 0, 10), _event(1, 5, 15, subject="right"), _event(2, 20, 30)]
    none_intervals = [_none_interval(30, 40)]
    root = tmp_path / "vlm_annotation_20261023"
    review_dir = root / f"{SEQ8}_qwen25vl7b_20261023_review"
    review_dir.mkdir(parents=True)
    write_jsonl(review_dir / "events.jsonl", events, frame_count)
    write_jsonl(review_dir / "none_intervals.jsonl", none_intervals, frame_count)
    (review_dir / "timeline.json").write_text(json.dumps(_timeline(frame_count)), encoding="utf-8")
    (review_dir / "summary.json").write_text(
        json.dumps(_summary(events, none_intervals, frame_count)), encoding="utf-8"
    )
    write_review_csv(review_dir / "review.csv", events, none_intervals)
    return root, review_dir, events, none_intervals


def _write_decisions(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(apply_review.DECISION_COLUMNS)
        for row in rows:
            writer.writerow(row)
    return Path(path)


def _row(record_id, status, *, object_id="obj_test_object", notes="", reviewer=REVIEWER):
    return [SEQ_ID, record_id, status, reviewer, object_id, notes]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_review_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_events_jsonl(review_dir):
    """Load events.jsonl as it stands after an apply; empty if the file was never written."""
    path = review_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestLoadDecisions:
    def test_accepted_requires_object_id(self, tmp_path):
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(f"{SEQ8}_qwen25vl7b_20261023:event:0000", "accepted", object_id=""),
        ])
        with pytest.raises(AnnotationError, match="accepted.*object_id"):
            apply_review.load_decisions(csv_path)

    def test_bad_status_rejected(self, tmp_path):
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(f"{SEQ8}_qwen25vl7b_20261023:event:0000", "maybe"),
        ])
        with pytest.raises(AnnotationError, match="review_status"):
            apply_review.load_decisions(csv_path)

    def test_empty_reviewer_rejected(self, tmp_path):
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(f"{SEQ8}_qwen25vl7b_20261023:event:0000", "rejected", reviewer=""),
        ])
        with pytest.raises(AnnotationError, match="reviewer"):
            apply_review.load_decisions(csv_path)

    def test_duplicate_record_id_rejected(self, tmp_path):
        record_id = f"{SEQ8}_qwen25vl7b_20261023:event:0000"
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(record_id, "rejected"), _row(record_id, "accepted"),
        ])
        with pytest.raises(AnnotationError, match="duplicate"):
            apply_review.load_decisions(csv_path)


class TestApply:
    def test_all_accept(self, collection, tmp_path):
        root, review_dir, events, _ = collection
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[0]["event_id"], "accepted", object_id="paper_roll"),
            _row(events[1]["event_id"], "accepted", object_id="paper_roll"),
            _row(events[2]["event_id"], "accepted", object_id="printer_cover"),
        ])
        report = apply_review.run(csv_path, root)
        assert len(report) == 1 and report[0]["updated"] == 3
        summary = json.loads((review_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["review_status_counts"] == {
            "pending": 1, "accepted": 3, "rejected": 0, "needs_revision": 0,
        }
        assert summary["human_review_completed"] is False  # none interval still pending
        assert summary["accepted_object_ids"] == ["paper_roll", "printer_cover"]
        labels = _read_jsonl(review_dir / "labels_projected.jsonl")
        assert len(labels) == 40
        # frames 0-4: single accepted left grasp on paper_roll
        for frame in range(0, 5):
            label = labels[frame]
            assert label["valid"] and label["reason"] == "reviewed_event"
            assert label["object_id"] == "paper_roll"
            assert label["subject_name"] == "left"
        # frames 5-9: two overlapping accepted events -> label_valid=0
        for frame in range(5, 10):
            assert labels[frame]["valid"] is False
            assert labels[frame]["reason"] == "overlapping_accepted_labels"
        # frames 30-39: none interval still pending
        for frame in range(30, 40):
            assert labels[frame]["reason"] == "pending_or_needs_revision"
        csv_rows = _read_review_csv(review_dir / "review.csv")
        assert len(csv_rows) == 4  # unchanged row count, columns intact
        assert (review_dir / "review.csv.bak_apply").is_file()

    def test_partial_reject_keeps_undecided_pending(self, collection, tmp_path):
        root, review_dir, events, none_intervals = collection
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[0]["event_id"], "accepted", object_id="paper_roll"),
            _row(events[1]["event_id"], "rejected"),
        ])
        apply_review.run(csv_path, root)
        updated = {row["event_id"]: row for row in _read_events_jsonl(review_dir)}
        assert updated[events[0]["event_id"]]["review_status"] == "accepted"
        assert updated[events[0]["event_id"]]["object_id"] == "paper_roll"
        assert updated[events[1]["event_id"]]["review_status"] == "rejected"
        assert updated[events[2]["event_id"]]["review_status"] == "pending"
        # events.jsonl is rewritten in place so review.csv stays consistent
        rows = {row["record_id"]: row for row in _read_review_csv(review_dir / "review.csv")}
        assert rows[events[0]["event_id"]]["review_status"] == "accepted"
        assert rows[events[0]["event_id"]]["object_id"] == "paper_roll"
        assert rows[events[0]["event_id"]]["reviewer"] == REVIEWER
        assert rows[events[0]["event_id"]]["reviewed_at"].endswith("Z")
        assert rows[events[1]["event_id"]]["review_status"] == "rejected"
        assert rows[events[2]["event_id"]]["review_status"] == "pending"
        assert rows[events[2]["event_id"]]["reviewer"] == ""
        assert rows[none_intervals[0]["interval_id"]]["review_status"] == "pending"
        summary = json.loads((review_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["review_status_counts"] == {
            "pending": 2, "accepted": 1, "rejected": 1, "needs_revision": 0,
        }
        labels = _read_jsonl(review_dir / "labels_projected.jsonl")
        # frames 0-4 accepted; 5-9 rejected+accepted mix? events[1] rejected so only events[0]
        # frames 5-9: only event0 accepted (event1 rejected) -> valid again
        for frame in range(0, 10):
            assert labels[frame]["valid"] is True, labels[frame]
        # frames 10-14: only rejected event1 -> unlabelled
        for frame in range(10, 15):
            assert labels[frame]["reason"] == "unlabelled_or_rejected"

    def test_unknown_record_id_lists_all(self, collection, tmp_path):
        root, _, events, _ = collection
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[0]["event_id"], "accepted"),
            _row("deadbeef_qwen25vl7b_20261023:event:0099", "rejected", reviewer=REVIEWER),
        ])
        with pytest.raises(AnnotationError, match="deadbeef"):
            apply_review.run(csv_path, root, dry_run=True)

    def test_dry_run_writes_nothing(self, collection, tmp_path):
        root, review_dir, events, _ = collection
        before = (review_dir / "review.csv").read_text(encoding="utf-8")
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[0]["event_id"], "accepted", object_id="paper_roll"),
        ])
        report = apply_review.run(csv_path, root, dry_run=True)
        assert report[0]["updated"] == 1
        assert (review_dir / "review.csv").read_text(encoding="utf-8") == before
        assert not (review_dir / "labels_projected.jsonl").exists()
        assert not (review_dir / "review.csv.bak_apply").exists()

    def test_accepted_needs_revision_keeps_proposed_only(self, collection, tmp_path):
        root, review_dir, events, _ = collection
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[2]["event_id"], "needs_revision", object_id="holder_thing", notes="check boundary"),
        ])
        apply_review.run(csv_path, root)
        rows = {row["record_id"]: row for row in _read_review_csv(review_dir / "review.csv")}
        row = rows[events[2]["event_id"]]
        assert row["review_status"] == "needs_revision"
        assert row["object_id"] == ""
        assert row["proposed_object_id"] == "holder_thing"
        assert row["notes"] == "check boundary"
        labels = _read_jsonl(review_dir / "labels_projected.jsonl")
        for frame in range(20, 30):
            assert labels[frame]["reason"] == "pending_or_needs_revision"

    def test_overlap_of_two_accepted_is_invalid(self, collection, tmp_path):
        root, review_dir, events, _ = collection
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[0]["event_id"], "accepted", object_id="paper_roll"),
            _row(events[1]["event_id"], "accepted", object_id="holder"),
        ])
        apply_review.run(csv_path, root)
        labels = _read_jsonl(review_dir / "labels_projected.jsonl")
        assert labels[2]["valid"] is True
        overlap = [label["reason"] for label in labels[5:10]]
        assert set(overlap) == {"overlapping_accepted_labels"}
        summary = json.loads((review_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["accepted_event_overlap_frames"] == 5
        assert summary["semantic_projection"]["label_valid_zero_frames"] > 0

    def test_refuses_to_overwrite_existing_projection(self, collection, tmp_path):
        root, review_dir, events, _ = collection
        (review_dir / "labels_projected.jsonl").write_text("{}\n", encoding="utf-8")
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[0]["event_id"], "accepted", object_id="paper_roll"),
        ])
        with pytest.raises(AnnotationError, match="refusing to overwrite"):
            apply_review.run(csv_path, root)

    def test_review_csv_columns_preserved(self, collection, tmp_path):
        root, review_dir, events, _ = collection
        csv_path = _write_decisions(tmp_path / "d.csv", [
            _row(events[0]["event_id"], "accepted", object_id="paper_roll"),
        ])
        apply_review.run(csv_path, root)
        with (review_dir / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
            fieldnames = csv.reader(stream).__next__()
        assert fieldnames == REVIEW_COLUMNS
