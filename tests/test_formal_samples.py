import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_formal_samples as formal
from src.dataset import human_dataset


FRAME_IDS = list(range(100, 113))
TIMESTAMPS = [0.0, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 12.0]
REVIEWED_AT = "2026-10-22T12:00:00+00:00"


def _asset(root, path, role, content=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if content is not None:
        path.write_bytes(content)
    payload = path.read_bytes()
    return {"role": role, "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _json_asset(root, path, role, value):
    return _asset(root, path, role, json.dumps(value, sort_keys=True).encode("utf-8"))


def _jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records) or "\n"
    path.write_text(content, encoding="utf-8")


def _event(sequence_id, event_id, object_id="obj_000", start=102, end=106, status="accepted",
           interaction="grasp", subject="left", geometry="passed", **updates):
    record = {
        "schema_version": "hoi_annotation_v1",
        "event_id": event_id,
        "sequence_id": sequence_id,
        "subject": subject,
        "object_id": object_id,
        "object_hint": "unknown object" if object_id is None else "tracked object",
        "interaction_type": interaction,
        "start_frame": start,
        "end_frame": end,
        "evidence_frames": [start],
        "confidence": 0.5,
        "label_source": "source_text+vlm",
        "review_status": status,
        "geometry_status": geometry,
        "source_event_ids": [f"raw-{event_id}"],
        "notes": "",
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
    }
    if status == "accepted":
        record.update(reviewer="human-reviewer", reviewed_at=REVIEWED_AT)
    record.update(updates)
    return record


def _none(sequence_id, event_id, start=102, end=106, status="accepted", geometry="not_checked", **updates):
    record = _event(sequence_id, event_id, object_id=None, start=start, end=end, status=status,
                    interaction="grasp", subject="left", geometry=geometry, **updates)
    record["interval_id"] = record.pop("event_id")
    for key in ("subject", "object_id", "object_hint", "interaction_type"):
        record.pop(key)
    record["source_event_ids"] = []
    record["label_source"] = "human"
    return record


def _annotations_for(index, sequence_id, object_ids):
    events = [_event(sequence_id, "base", object_id=object_ids[-1])]
    none_intervals = []
    if index == 1:
        events = [_event(sequence_id, "pending", status="pending"),
                  _event(sequence_id, "accepted-elsewhere", start=105, end=106)]
    elif index == 2:
        events = [_event(sequence_id, "overlap-a"),
                  _event(sequence_id, "overlap-b", subject="right", interaction="move"),
                  _event(sequence_id, "accepted-elsewhere", start=105, end=106)]
    elif index == 3:
        events = []
        none_intervals = [_none(sequence_id, "accepted-none")]
    elif index == 4:
        events = [_event(sequence_id, "accepted-elsewhere", start=105, end=106)]
        none_intervals = [_none(sequence_id, "pending-none", status="pending")]
    elif index == 5:
        events = [_event(sequence_id, "accepted-elsewhere", start=105, end=106)]
        none_intervals = [_none(sequence_id, "rejected-none", status="rejected")]
    elif index == 6:
        events = [_event(sequence_id, "unknown-object", object_id=None),
                  _event(sequence_id, "accepted-elsewhere", start=105, end=106)]
    elif index == 7:
        events = [_event(sequence_id, "bad-geometry", geometry="inconclusive"),
                  _event(sequence_id, "accepted-elsewhere", start=105, end=106)]
    elif index == 8:
        events = [_event(sequence_id, "occluded", occluded=True),
                  _event(sequence_id, "accepted-elsewhere", start=105, end=106)]
    elif index == 9:
        none_intervals = [_none(sequence_id, "conflicting-none")]
    elif index == 10:
        events.append(_event(sequence_id, "pending-overlap", status="pending", subject="right"))
    elif index == 12:
        events = [_event(sequence_id, "needs-revision", status="needs_revision"),
                  _event(sequence_id, "accepted-elsewhere", start=105, end=106)]
    for event in events:
        if event["object_id"] == "obj_000" and "obj_000" not in object_ids:
            event["object_id"] = object_ids[-1]
    return events, none_intervals


def _make_release(root):
    shared = root / "shared"
    timestamps = _json_asset(root, shared / "source/timestamps.json", "timestamps", TIMESTAMPS)
    frame_ids = _json_asset(root, shared / "source/frame_ids.json", "frame_ids", FRAME_IDS)
    common = {
        "intrinsics": _asset(root, shared / "source/intrinsics.bin", "intrinsics", b"calibrated intrinsics"),
        "mesh": _asset(root, shared / "source/mesh.bin", "mesh", b"metric object mesh"),
        "hand_metric_gt": _asset(root, shared / "gt/hand.bin", "hand_metric_gt", b"hand reference"),
        "object_pose_gt": _asset(root, shared / "gt/object.bin", "object_pose_gt", b"object reference"),
        "hand_valid": _asset(root, shared / "observations/hand_valid.bin", "hand_valid", b"hand validity"),
        "object_valid": _asset(root, shared / "observations/object_valid.bin", "object_valid", b"object validity"),
    }
    provenance = _json_asset(root, shared / "source/provenance.json", "provenance",
                             {"review": "production-scale synthetic fixture with real review fields"})
    records = []
    for index in range(120):
        sequence_id = f"seq_{index:03d}"
        object_ids = ["obj_000", "obj_001"] if index == 0 else [f"obj_{index % 8:03d}"]
        sequence_root = root / "assets" / sequence_id
        rgb = _asset(root, sequence_root / "source/rgb.bin", "rgb", f"unique RGB {index}".encode())
        depth = _asset(root, sequence_root / "source/depth.bin", "depth", f"unique depth {index}".encode())
        left = np.zeros((len(FRAME_IDS), 21, 3), np.float32)
        right = np.zeros_like(left)
        left[:, 0, 0] = np.arange(len(FRAME_IDS))
        right[:, 0, 0] = -np.arange(len(FRAME_IDS))
        hand_path = sequence_root / "observations/hand.npz"
        hand_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hand_path, left_hand=left, right_hand=right,
                            conf=np.ones((len(FRAME_IDS), 2), np.float32))
        hand = _asset(root, hand_path, "hand_observations")
        count = len(object_ids)
        poses = np.broadcast_to(np.eye(4, dtype=np.float32), (count, len(FRAME_IDS), 4, 4)).copy()
        poses[..., 0, 3] = np.arange(count)[:, None]
        poses[..., 2, 3] = 1.0
        bboxes = np.zeros((count, len(FRAME_IDS), 4), np.int32)
        bboxes[..., 2:] = 10
        if index == 11:
            poses[0, 2] = np.nan
            bboxes[0, 2] = -1
        pose_path = sequence_root / "observations/pose.npz"
        np.savez_compressed(pose_path, object_ids=np.asarray(object_ids), poses=poses, bboxes=bboxes)
        pose = _asset(root, pose_path, "object_observations")
        meta = {"sequence_id": sequence_id, "frame_count": len(FRAME_IDS), "fps": 1.0,
                "width": 640, "height": 480, "timestamp_source": "source_timestamps",
                "timestamps_sha256": timestamps["sha256"], "frame_ids_sha256": frame_ids["sha256"]}
        hand_meta = _json_asset(root, hand_path.with_suffix(".meta.json"), "hand_pose_meta", meta)
        pose_meta = _json_asset(root, pose_path.with_suffix(".meta.json"), "pose6d_meta", meta)
        events, none_intervals = _annotations_for(index, sequence_id, object_ids)
        events_path = root / "annotations" / sequence_id / "events.jsonl"
        none_path = root / "annotations" / sequence_id / "none_intervals.jsonl"
        _jsonl(events_path, events)
        _jsonl(none_path, none_intervals)
        event_asset = _asset(root, events_path, "hoi_labels")
        none_asset = _asset(root, none_path, "hoi_none_intervals")
        assets = [rgb, depth, common["intrinsics"], common["mesh"], common["hand_metric_gt"],
                  common["object_pose_gt"], hand, pose, common["hand_valid"], common["object_valid"],
                  hand_meta, pose_meta, timestamps, frame_ids, provenance, event_asset, none_asset]
        records.append({
            "sequence_id": sequence_id,
            "dataset": "FormalFixture",
            "acquisition_id": f"acq_{index:03d}",
            "subject_id": f"person_{index // 20}",
            "scene_id": f"scene_{index % 3}",
            "object_ids": object_ids,
            "action_labels": [f"action_{index % 4}"],
            "camera_id": "camera_0",
            "view": "third_person",
            "source_video": f"source_{index:03d}",
            "frame_count": len(FRAME_IDS),
            "duration_seconds": TIMESTAMPS[-1] - TIMESTAMPS[0],
            "fps": 1.0,
            "duration_definition": "frame_span",
            "timestamp_source": "source_timestamps",
            "human_verified": True,
            "assets": assets,
            "capabilities": {key: True for key in human_dataset.CAPABILITIES},
            "quality": {key: "pass" for key in human_dataset.QUALITY_KEYS},
            "split": None,
            "evidence": {key: {"asset_role": "provenance", "description": f"reviewed {key}"}
                         for key in human_dataset.EVIDENCE_KEYS},
            "continuity": {"continuous": True, "concatenated": False, "repeated_frames": False,
                           "interpolated": False, "fps_modified": False},
            "coordinates": {"unit": "m", "hand_frame": "camera", "object_frame": "camera",
                            "handedness": "right", "aligned": True},
            "camera_motion": "fixed",
            "gt_independent": True,
            "gt_usage": "supervision_or_evaluation_only",
        })
        records[-1]["capabilities"]["camera_motion_compensated"] = False
    ledger = root / "candidate.json"
    ledger.write_text(json.dumps({"schema_version": "human_pose_dataset_v1", "records": records}), encoding="utf-8")
    freeze = root / "freeze"
    report = human_dataset.freeze_manifest(ledger, freeze, {})
    assert report["training_allowed"], report["errors"]
    assert (freeze / "training_manifest.json").is_file()
    return records


def _build(root, name, **kwargs):
    options = {"window": 1, "stride": 12, "horizon_seconds": 1.0, "n_max": 2}
    options.update(kwargs)
    return formal.build_formal_samples(root / "freeze", root / "annotations", root / name, **options)


def _entries(output, sequence_id):
    index = json.loads((output / "dataset_index.json").read_text())
    return [entry for entries in index.values() for entry in entries if entry["sequence_id"] == sequence_id]


def _entry(output, sequence_id):
    entries = _entries(output, sequence_id)
    assert len(entries) == 1
    return entries[0]


def _load_entry(output, entry):
    return np.load(output / entry["path"], allow_pickle=False)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("formal_release")
    records = _make_release(root)
    output = root / "samples"
    meta = _build(root, "samples")
    debug_output = root / "debug_samples"
    debug_meta = _build(root, "debug_samples", include_invalid=True)
    return {"root": root, "records": records, "output": output, "meta": meta,
            "debug_output": debug_output, "debug_meta": debug_meta}


def test_production_freeze_and_valid_roundtrip(release):
    output, meta = release["output"], release["meta"]
    assert meta["training_allowed"] is True
    assert meta["debug_only"] is False
    assert meta["stats"]["valid_counts"]["total"] == 109
    assert meta["stats"]["invalid_counts"]["total"] == 11
    assert meta["stats"]["dropped_objects"] == 0
    index = json.loads((output / "dataset_index.json").read_text())
    entries = [entry for split in formal.SPLITS for entry in index[split]]
    assert len(entries) == 109
    assert meta["sample_counts"]["total"] == len(entries)
    for entry in entries:
        path = output / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sample_sha256"]
        integrity = meta["sample_integrity"][entry["sample_id"]]
        assert integrity["sample_sha256"] == entry["sample_sha256"]
        assert integrity["content_sha256"] == entry["content_sha256"]
        with _load_entry(output, entry) as sample:
            assert sample["label_valid"] == 1
            assert sample["training_allowed"]
            assert sample["content_sha256"].item() == entry["content_sha256"]
            arrays = {key: sample[key] for key in sample.files}
            assert formal._sample_content_sha256(arrays) == entry["content_sha256"]
            assert sample["annotation_sha256"].item() == entry["annotation_sha256"]
            assert len(sample["freeze_sha256"].item()) == 64
            assert len(sample["config_sha256"].item()) == 64
            assert len(sample["source_ledger_sha256"].item()) == 64
            assert len(sample["code_sha256"].item()) == 64
            assert sample["split"].item() == entry["split"]
    assert set(meta["sample_counts"]) >= set(formal.SPLITS)
    assert all(meta["sample_counts"][split] > 0 for split in formal.SPLITS)


def test_native_timestamp_target_not_nominal_next_frame(release):
    entry = _entry(release["output"], "seq_000")
    assert entry["start_frame"] == 100
    assert entry["observation_end_frame"] == 100
    assert entry["target_frame"] == 102
    assert entry["target_index"] == 2
    assert entry["requested_target_timestamp"] == pytest.approx(1.0)
    assert entry["target_timestamp"] == pytest.approx(1.9)
    assert entry["horizon_seconds"] == pytest.approx(1.9)
    with _load_entry(release["output"], entry) as sample:
        assert sample["hand_seq"].shape == (1, 2, 21, 3)
        assert sample["hand_seq"][0, 0, 0, 0] == 0
        assert sample["target_frame"] > sample["observation_end_frame"]


def test_object_slot_is_stable_and_never_semantic_hint(release):
    entry = _entry(release["output"], "seq_000")
    assert entry["label_object_id"] == "obj_001"
    assert entry["label_object_slot"] == 1
    with _load_entry(release["output"], entry) as sample:
        assert sample["object_ids"].tolist() == ["obj_000", "obj_001"]
        assert sample["label_object_id"].item() == sample["object_ids"][1].item()
        assert sample["obj_mask"].tolist() == [1.0, 1.0]
        assert sample["label_source"].item() == "source_text+vlm"
        assert sample["annotation_mode"].item() == "offline_noncausal"
        assert sample["label_event_ids"].tolist() == ["base"]
        assert sample["reviewer"].item() == "human-reviewer"


def test_pending_is_invalid_and_skipped_not_none(release):
    assert _entries(release["output"], "seq_001") == []
    skipped = [item for item in release["meta"]["skipped_invalid"]
               if item["sequence_id"] == "seq_001" and item["reason"] == "pending_annotation"]
    assert len(skipped) == 1
    entry = _entry(release["debug_output"], "seq_001")
    assert entry["training_allowed"] is False
    with _load_entry(release["debug_output"], entry) as sample:
        assert sample["label_valid"] == 0
        assert sample["label_subject"] == -1
        assert sample["label_interaction"] == -1
        assert sample["label_interaction_name"].item() != "none"
        assert sample["invalid_reason"].item() == "pending_annotation"
        assert not sample["training_allowed"]
    assert release["debug_meta"]["training_allowed"] is False
    assert release["debug_meta"]["debug_only"] is True


def test_overlapping_accepted_events_conflict(release):
    assert _entries(release["output"], "seq_002") == []
    entry = _entry(release["debug_output"], "seq_002")
    assert entry["label_valid"] == 0
    assert entry["invalid_reason"] == "multiple_accepted_events"
    assert release["meta"]["stats"]["invalid_counts"]["by_reason"]["multiple_accepted_events"] == 1


def test_only_accepted_none_projects_to_none(release):
    accepted = _entry(release["output"], "seq_003")
    assert accepted["label_subject"] == 2
    assert accepted["label_object_slot"] == -1
    assert accepted["label_interaction_name"] == "none"
    assert accepted["label_event_ids"] == ["accepted-none"]
    assert _entries(release["output"], "seq_004") == []
    assert _entries(release["output"], "seq_005") == []
    pending = _entry(release["debug_output"], "seq_004")
    rejected = _entry(release["debug_output"], "seq_005")
    assert pending["invalid_reason"] == "pending_annotation"
    assert rejected["invalid_reason"] == "unannotated_or_rejected"
    assert pending["label_interaction_name"] == ""
    assert rejected["label_interaction_name"] == ""


@pytest.mark.parametrize("sequence_id,reason", [
    ("seq_006", "unknown_object"),
    ("seq_007", "geometry_not_passed"),
    ("seq_008", "occluded"),
    ("seq_009", "accepted_event_none_conflict"),
    ("seq_010", "pending_annotation"),
    ("seq_011", "object_pose_missing_at_target"),
    ("seq_012", "annotation_needs_revision"),
])
def test_all_other_invalid_causes_are_explicit(release, sequence_id, reason):
    assert _entries(release["output"], sequence_id) == []
    entry = _entry(release["debug_output"], sequence_id)
    assert entry["label_valid"] == 0
    assert entry["invalid_reason"] == reason


def test_manifest_groups_and_content_aliases_cannot_cross_splits():
    first = {"split": "train", "entry": {"group_id": "group-a"},
             "record": {"sequence_id": "a", "dataset": "d", "subject_id": "p",
                        "acquisition_id": "acq", "assets": []}}
    same = {"split": "train", "entry": {"group_id": "group-a"},
            "record": {"sequence_id": "a-view", "dataset": "d", "subject_id": "p",
                       "acquisition_id": "acq", "assets": []}}
    formal._validate_split_isolation([first, same])
    leaked = {**same, "split": "val"}
    with pytest.raises(formal.FormalSampleError, match="split_leakage"):
        formal._validate_split_isolation([first, leaked])


def test_actual_output_respects_frozen_split_assignment(release):
    manifest = json.loads((release["root"] / "freeze/training_manifest.json").read_text())
    expected = {entry["sequence_id"]: (split, entry["group_id"])
                for split in formal.SPLITS for entry in manifest["splits"][split]}
    index = json.loads((release["output"] / "dataset_index.json").read_text())
    for split in formal.SPLITS:
        for entry in index[split]:
            assert (entry["split"], entry["split_group"]) == expected[entry["sequence_id"]]


def test_candidate_freeze_is_rejected_and_audit_explains(release, tmp_path):
    root = release["root"]
    ledger = root / "candidate.json"
    one = json.loads(ledger.read_text())
    one["records"] = one["records"][:1]
    one_path = tmp_path / "one.json"
    one_path.write_text(json.dumps(one))
    original_base = root
    for record in one["records"]:
        for asset in record["assets"]:
            asset["path"] = str(original_base / asset["path"])
    one_path.write_text(json.dumps(one))
    candidate = tmp_path / "candidate_freeze"
    report = human_dataset.freeze_manifest(one_path, candidate, {})
    assert not report["training_allowed"]
    assert not (candidate / "training_manifest.json").exists()
    with pytest.raises(formal.FreezeRejected, match="training_manifest.json is missing"):
        formal.build_formal_samples(candidate, root / "annotations", tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    audit = formal.build_formal_samples(candidate, root / "annotations", tmp_path / "audit", audit_only=True)
    assert audit["training_allowed"] is False
    assert audit["release_status"] == "blocked"
    assert not (tmp_path / "audit").exists()


def test_frozen_file_hash_tampering_is_rejected(release, tmp_path):
    tampered = tmp_path / "tampered_freeze"
    shutil.copytree(release["root"] / "freeze", tampered)
    with (tampered / "config.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(formal.FreezeRejected, match="integrity"):
        formal.build_formal_samples(tampered, release["root"] / "annotations", tmp_path / "samples")
    assert not (tmp_path / "samples").exists()


def test_existing_output_is_never_overwritten(release, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        formal.build_formal_samples(
            release["root"] / "freeze", release["root"] / "annotations", destination)
    assert sentinel.read_text() == "keep"
    assert list(destination.iterdir()) == [sentinel]


def test_native_frame_ids_do_not_accept_float_truncation(tmp_path):
    path = tmp_path / "frames.json"
    path.write_text(json.dumps([0, 1.5, 2]))
    with pytest.raises(formal.FormalSampleError, match="literal integers"):
        formal._read_table(path, "frame_ids", "frame_id", integer=True)


def test_source_aliases_join_other_source_identity_fields():
    first = {"split": "train", "entry": {"group_id": "group-a"},
             "record": {"sequence_id": "a", "dataset": "d", "subject_id": "p1",
                        "acquisition_id": "acq-a", "source_aliases": ["shared-original"], "assets": []}}
    second = {"split": "val", "entry": {"group_id": "group-b"},
              "record": {"sequence_id": "b", "dataset": "d", "subject_id": "p2",
                         "acquisition_id": "acq-b", "source_content_id": "shared-original", "assets": []}}
    with pytest.raises(formal.FormalSampleError, match="source:shared-original"):
        formal._validate_split_isolation([first, second])


def test_auto_n_max_uses_only_train_and_overflow_is_fatal(release):
    train_counts = release["meta"]["stats"]["train_object_track_counts"]
    train_max = max(train_counts.values())
    destination = release["root"] / "auto_capacity_audit"
    if train_max == 2:
        audit = _build(release["root"], destination.name, n_max=0, audit_only=True)
        assert audit["stats"]["n_max"] == 2
        assert audit["stats"]["n_max_source"] == "train_max_object_tracks"
    else:
        assert train_max == 1
        with pytest.raises(formal.FormalSampleError, match="n_max_capacity_exceeded"):
            _build(release["root"], destination.name, n_max=0, audit_only=True)
    assert not destination.exists()
    with pytest.raises(formal.FormalSampleError, match="n_max_capacity_exceeded"):
        _build(release["root"], "too_small", n_max=1, audit_only=True)
    assert not (release["root"] / "too_small").exists()


def test_pending_only_sequence_cannot_supply_reviewed_release_labels(tmp_path):
    events = tmp_path / "events.jsonl"
    none = tmp_path / "none_intervals.jsonl"
    _jsonl(events, [_event("seq_pending", "pending", status="pending")])
    _jsonl(none, [])
    assets = {"events": {"path": events}, "none": {"path": none}}
    with pytest.raises(formal.FormalSampleError, match="accepted_annotations_missing"):
        formal._load_annotation_set(assets, "seq_pending", FRAME_IDS, ["obj_000"])


def test_cli_audit_only_success_writes_no_samples(release, capsys):
    destination = release["root"] / "cli_audit"
    result = formal.main([
        "--freeze-dir", str(release["root"] / "freeze"),
        "--annotations-dir", str(release["root"] / "annotations"),
        "--out-dir", str(destination),
        "--window", "1", "--stride", "12", "--horizon-seconds", "1", "--n-max", "2",
        "--audit-only",
    ])
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training_allowed"] is True
    assert report["release_status"] == "audited_not_written"
    assert not destination.exists()
