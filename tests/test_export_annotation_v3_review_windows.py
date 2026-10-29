import json

import numpy as np
import pytest

from src.dataset.export_annotation_v3_review_windows import (
    digest, evidence_targets, export_review_windows, target_window,
)


def point_record(frames, **changes):
    return {"layer": "relation", "status": "supported", "silver_label_valid": True,
            "annotation_source": "assistant_visual_review", "review_status": "assistant_reviewed",
            "temporal_coverage": "sampled_points", "evidence_frames": frames,
            "start_frame": 0, "end_frame": 100, **changes}


def test_last_anchor_obeys_actual_seconds_and_32_native_frames():
    times = np.arange(100, dtype=float) / 30
    for target in range(62, 100):
        row, reason = target_window(times, target)
        assert reason is None
        choices = [i for i in range(target) if times[target] - times[i] >= 1.0]
        assert row["anchor_frame"] == max(choices)
        assert row["end_frame"] - row["start_frame"] == 32
        assert row["actual_horizon_s"] >= 1.0
    assert target_window(times, 20)[1] == "no_anchor_before_horizon"
    assert target_window(times, 40)[1] == "insufficient_observation_history"


def test_irregular_timestamp_gap_is_rejected_without_interpolation():
    times = np.arange(100, dtype=float) * 0.02
    times[50:] += 0.5
    row, reason = target_window(times, 80)
    assert row is None and reason == "horizon_overshoot"
    row, reason = target_window(times, 60)
    assert reason is None and row["target_frame"] == 60
    assert row["target_timestamp"] == times[60]


@pytest.mark.parametrize("times", [[0, 0, 1], [0, 2, 1], [0, float("nan")], [[0, 1]], [0]])
def test_invalid_native_timestamps_are_fatal(times):
    with pytest.raises(ValueError, match="timestamps"):
        target_window(times, 1)


def test_points_are_deduplicated_without_interval_or_transition_expansion():
    records = [point_record([70, 70, 80]), point_record([70]),
               point_record([75], layer="transition"),
               point_record([78], temporal_coverage="sampled_evidence_interval_presence"),
               point_record([77], silver_label_valid=False), point_record([]),
               point_record([True, 1.2, -1, 101])]
    targets, excluded_records, excluded_frames = evidence_targets(records, 100)
    assert targets == {70: [0, 1], 80: [0]}
    assert excluded_records == {"not_state_or_operation": 1, "not_sampled_points": 1,
                                "not_supported_assistant_silver": 1, "missing_point_evidence": 1}
    assert excluded_frames == {"target_not_integer": 2, "target_out_of_range": 2}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_dataset(tmp_path):
    base, root = tmp_path / "v2", tmp_path / "v3"
    prefix = base / "sequences/uid"
    prefix.mkdir(parents=True)
    # Absence of all object arrays verifies target creation does not inspect them.
    np.savez(prefix / "sequence.npz", timestamps=np.arange(100) / 30, frame_index=np.arange(100))
    save(prefix / "metadata.json", {"source": "frozen"})
    save(prefix / "events.json", {"records": []})
    entry = {"sequence_id": "uid", "split": "val", "frame_count": 100,
             **{kind + "_path": f"sequences/uid/{kind}.{ext}" for kind, ext in
                [("sequence", "npz"), ("metadata", "json"), ("events", "json")]},
             "sha256": {kind: digest(prefix / f"{kind}.{ext}") for kind, ext in
                        [("sequence", "npz"), ("metadata", "json"), ("events", "json")]}}
    split_path = base / "splits/split_manifest.json"
    save(split_path, {"assignments": {"uid": "val"}, "source_selection_sha256": "source-sha"})
    splits = {}
    for name in ["train", "val", "test"]:
        path = base / "windows" / (name + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        path.write_text('{"original_window":true}\n')
        splits[name] = {"window_path": f"windows/{name}.jsonl", "sha256": digest(path)}
    index = {"schema_version": "egodemo_mamba_v2", "sequences": [entry], "splits": splits,
             "source_selection_sha256": "source-sha", "split_manifest_path": "splits/split_manifest.json",
             "split_manifest_sha256": digest(split_path)}
    save(base / "dataset_index.json", index)
    annotation_path = root / "annotations/uid.json"
    save(annotation_path, {"sequence_id": "uid", "split": "val",
         "source_bindings": {"base_sequence_sha256": entry["sha256"]["sequence"],
                             "base_metadata_sha256": entry["sha256"]["metadata"]},
         "silver_event_records": [point_record([5, 70, 80]), point_record([70])]})
    save(root / "mamba_overlay_index.json", {"schema_version": "assistant_annotation_v3_overlay",
         "base_dataset_index_sha256": digest(base / "dataset_index.json"),
         "base_split_manifest_sha256": digest(split_path),
         "sequences": [{"sequence_id": "uid", "split": "val", "annotation_path": "annotations/uid.json",
                        "sha256": digest(annotation_path)}]})
    return base, root


def test_export_binds_frozen_sources_keeps_split_and_leaves_old_windows(tmp_path):
    base, root = fixture_dataset(tmp_path)
    before = {str(p): digest(p) for p in base.rglob("*") if p.is_file()}
    index = export_review_windows(base, root)
    rows = [json.loads(line) for line in (root / "review_windows/val.jsonl").read_text().splitlines()]
    assert [r["target_frame"] for r in rows] == [70, 80]
    assert rows[0]["evidence_record_indices"] == [0, 1]
    assert all(r["split"] == "val" for r in rows)
    assert index["totals"]["candidate_target_count"] == 3
    assert index["target_exclusion_counts"] == {"no_anchor_before_horizon": 1}
    assert index["positive_biased_diagnostic"] and not index["training_allowed"]
    assert not index["blind_evaluation"] and not index["default_loader_windows_replaced"]
    assert digest(root / index["splits"]["val"]["window_path"]) == index["splits"]["val"]["sha256"]
    assert before == {str(p): digest(p) for p in base.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        export_review_windows(base, root)


def test_changed_annotation_cannot_be_published(tmp_path):
    base, root = fixture_dataset(tmp_path)
    with (root / "annotations/uid.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="Annotation changed"):
        export_review_windows(base, root)
    assert not (root / "review_windows/index.json").exists()
