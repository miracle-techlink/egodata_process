"""Independent orchestration checks using private, disposable source fixtures.

The fixture has no claimed semantic supervision.  No frozen data directory is
written by these tests; the sequence checker is replaced only where controlling
failure/concurrency is the behavior under test.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from src.dataset.egodemo_v2_split import freeze_split


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_pool(tmp_path, count=3):
    """Create a self-consistent tiny inventory, retaining native unknown masks."""
    source, base, v3 = (tmp_path / name for name in ("source", "base", "v3"))
    frames = 64
    ids = [f"00000000-0000-4000-8000-{i:012d}" for i in range(count)]
    records = [dict(source_episode_uuid=uid, scene_id=f"scene_{i}",
                    environment_id=f"environment_{i}", subject_id=None,
                    task_name="测试动作", manifest_frame_count=frames,
                    planned_assets=[], previously_downloaded=False)
               for i, uid in enumerate(ids)]
    selection = write_json(source / "selection_frozen.json", {
        "schema_version": "egodemo_expansion_selection", "records": records,
        "subject_identity_verified": False,
    })
    for uid in ids:
        staging = source / "staging" / uid
        staging.mkdir(parents=True)
        (staging / "sequence.npz").write_bytes(("staging-" + uid).encode())
        write_json(staging / "meta.json", {"sequence_id": uid})
    split = freeze_split(selection, base)
    split_path = base / "splits/split_manifest.json"
    vocabulary = {layer: {"codes": [code], "names_zh": [name]}
                  for layer, code, name in [("operation", "hold", "持有"),
                      ("relation", "contact", "接触"),
                      ("transition", "contact_onset", "开始接触"),
                      ("cooperation", "bimanual_joint", "双手共同操作")]}
    vocab_path = write_json(base / "vocab.json", vocabulary)
    sequence_rows, overlay_rows = [], []
    for uid in ids:
        directory = base / "sequences" / uid
        directory.mkdir(parents=True)
        sequence_path = directory / "sequence.npz"
        np.savez(sequence_path, timestamps=np.arange(frames, dtype=np.float64) / 30,
                 frame_index=np.arange(frames, dtype=np.int32),
                 hands_world=np.full((frames, 2, 21, 3), np.nan, np.float32),
                 hand_valid=np.zeros((frames, 2), bool),
                 camera_to_world=np.broadcast_to(np.eye(4), (frames, 4, 4)).copy(),
                 object_ids=np.array([], dtype="U16"), object_kind=np.array([], np.int8),
                 object_pose_world=np.empty((frames, 0, 4, 4), np.float32),
                 object_pose_valid=np.empty((frames, 0), bool),
                 object_observed=np.empty((frames, 0), bool),
                 surface_frame_index=np.array([], np.int32),
                 surface_object_index=np.array([], np.int32),
                 surface_point_index=np.array([], np.int32),
                 surface_world=np.empty((0, 3), np.float32),
                 surface_frame_ptr=np.zeros(frames + 1, np.int64),
                 point_keys=np.array([], dtype="U16"),
                 point_object_index=np.array([], np.int32))
        metadata_path = write_json(directory / "metadata.json", {
            "sequence_id": uid, "frame_count": frames,
            "source_selection_sha256": sha(selection), "entities": [],
            "training_allowed": False,
        })
        events_path = write_json(directory / "events.json", {"records": []})
        row = dict(sequence_id=uid, split=split["assignments"][uid],
                   frame_count=frames, object_count=0, window_count=0,
                   sequence_path=str(sequence_path.relative_to(base)),
                   metadata_path=str(metadata_path.relative_to(base)),
                   events_path=str(events_path.relative_to(base)),
                   sha256={"sequence": sha(sequence_path), "metadata": sha(metadata_path),
                           "events": sha(events_path)})
        sequence_rows.append(row)
        annotation_path = write_json(v3 / "annotations" / f"{uid}.json", {
            "schema_version": "assistant_annotation_v3", "sequence_id": uid,
            "frame_count": frames, "split": row["split"],
            "source_bindings": {"base_sequence_sha256": row["sha256"]["sequence"],
                                "base_metadata_sha256": row["sha256"]["metadata"]},
            "entities": [], "geometry_exclusions": [], "geometry_gaps": [],
            "interval_annotations": [], "silver_event_records": [],
            "relations": [], "transitions": [], "cooperations": [],
            "review_records": [], "human_label_valid": False,
            "training_allowed": False, "allowed_as_model_input": False,
            "coverage": {"continuous_frame_annotation_complete": False,
                         "full_annotation_complete": False},
        })
        overlay_rows.append(dict(sequence_id=uid, split=row["split"],
            annotation_path=str(annotation_path.relative_to(v3)), sha256=sha(annotation_path),
            source_interval_count=0, visually_reviewed_interval_count=0))
    windows = {}
    for name in ("train", "val", "test"):
        path = base / "windows" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        windows[name] = dict(window_path=str(path.relative_to(base)), sha256=sha(path),
                             window_count=0, sequence_count=0)
    base_index = write_json(base / "dataset_index.json", {
        "schema_version": "egodemo_mamba_v2", "source_selection_sha256": sha(selection),
        "source_sequence_count": count, "exported_sequence_count": count,
        "is_partial_export": False, "sequences": sequence_rows,
        "split_manifest_path": "splits/split_manifest.json",
        "split_manifest_sha256": sha(split_path), "vocab_path": "vocab.json",
        "vocab_sha256": sha(vocab_path), "splits": windows,
        "window_config": {"window_frames": 32, "stride_frames": 16,
                          "horizon_seconds": 1., "max_horizon_overshoot_median_dt": 1.5},
        "training_allowed": False, "allowed_as_model_input": False,
        "human_approved": False, "full_annotation_complete": False,
    })
    write_json(v3 / "mamba_overlay_index.json", {
        "schema_version": "assistant_annotation_v3_mamba_overlay",
        "base_dataset_index_sha256": sha(base_index),
        "base_split_manifest_sha256": sha(split_path), "sequences": overlay_rows,
        "training_allowed": False, "allowed_as_model_input": False,
    })
    return dict(base_root=str(base), v3_root=str(v3), source_pool=str(source),
                expected_sequences=count, workers=3, max_attempts=1,
                queue_chunk_frames=32, thresholds={})


def module():
    from src.annotation import review_pipeline
    return review_pipeline


def fake_report(base_root, base_row, annotation_path=None, review_path=None, thresholds=None):
    return {"sequence_id": base_row["sequence_id"], "split": base_row["split"],
            "frame_count": base_row["frame_count"],
            "checks": [{"code": "fixture_structure", "status": "pass", "message_zh": "结构有效"}],
            "metrics": {"native_frames": base_row["frame_count"],
                        "continuous_reviewed_frames": 0, "full_annotation_complete": False},
            "issues": [{"code": "fixture_unknown", "severity": "pending", "layer": "all",
                        "start_frame": 0, "end_frame": base_row["frame_count"],
                        "entity_id": None, "message_zh": "此序列尚无连续语义标注",
                        "next_action_zh": "逐帧审核"}]}


@pytest.fixture
def pipeline(monkeypatch):
    implementation = module()
    # Other agents can be editing unrelated checker modules during this suite.
    # JobRunner's code-change behavior is tested separately below.
    monkeypatch.setattr(implementation, "code_bindings", lambda: {"fixture_checker": {"sha256": "fixed"}})
    return implementation


def _job(config, output, pipeline, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    return pipeline.JobRunner(output, pipeline.normalized_config(config), resume)


def test_job_resume_requires_matching_nested_input_and_output_hash(tmp_path, pipeline):
    config = fixture_pool(tmp_path / "data", 1)
    out = tmp_path / "run"
    document = write_json(tmp_path / "root.json", {"evidence_path": "nested/page.json"})
    nested = write_json(tmp_path / "nested/page.json", {"pixels": 1})
    calls = []

    def dependencies():
        return [document, document.parent / read_json(document)["evidence_path"]]

    def work():
        calls.append(1)
        return {"checks": [{"status": "pass"}], "metrics": {}, "issues": []}

    _job(config, out, pipeline).run("job", dependencies, work, "result.json")
    _job(config, out, pipeline, True).run("job", dependencies, work, "result.json")
    assert len(calls) == 1
    assert read_json(out / "jobs.json")["jobs"]["job"]["cache_hit"] is True
    write_json(nested, {"pixels": 2})
    _job(config, out, pipeline, True).run("job", dependencies, work, "result.json")
    assert len(calls) == 2
    (out / "result.json").write_text('{"tampered": true}\n')
    _job(config, out, pipeline, True).run("job", dependencies, work, "result.json")
    assert len(calls) == 3


@pytest.mark.parametrize("damage", ["change", "delete"])
def test_job_resume_rechecks_produced_auxiliary_reports(tmp_path, pipeline, damage):
    config = fixture_pool(tmp_path / "data", 1)
    out = tmp_path / "run"
    source = write_json(tmp_path / "input.json", {"v": 0})
    calls = []

    def work():
        calls.append(1)
        auxiliary = write_json(out / "training_check" / "loaded_windows.json", {"checked": 7})
        return {"checks": [{"status": "pass"}], "issues": [], "artifact_paths": [str(auxiliary)]}

    _job(config, out, pipeline).run("job", lambda: [source], work, "result.json")
    _job(config, out, pipeline, True).run("job", lambda: [source], work, "result.json")
    assert len(calls) == 1
    if damage == "change":
        write_json(out / "training_check" / "loaded_windows.json", {"checked": 999})
    else:
        (out / "training_check" / "loaded_windows.json").unlink()
    _job(config, out, pipeline, True).run("job", lambda: [source], work, "result.json")
    assert len(calls) == 2
    assert read_json(out / "training_check" / "loaded_windows.json")["checked"] == 7


def test_job_rejects_source_changes_during_execution_and_retries(tmp_path, pipeline):
    config = fixture_pool(tmp_path / "data", 1)
    config["max_attempts"] = 2
    path = write_json(tmp_path / "input.json", {"v": 0})
    calls = []

    def unstable():
        calls.append(1)
        if len(calls) == 1:
            write_json(path, {"v": 1})
        return {"checks": [{"status": "pass"}], "metrics": {}, "issues": []}

    out = tmp_path / "run"
    report = _job(config, out, pipeline).run("job", lambda: [path], unstable, "result.json")
    assert len(calls) == 2 and report["status"] == "pass"
    assert read_json(out / "jobs.json")["jobs"]["job"]["attempt"] == 2
    # With no remaining retry the mutable input must be a failure, not cached.
    config["max_attempts"] = 1
    calls.clear()
    write_json(path, {"v": 0})
    report = _job(config, tmp_path / "one_attempt", pipeline).run(
        "job", lambda: [path], unstable, "result.json")
    assert report["status"] == "fail"
    assert "变化" in report["issues"][0]["message_zh"]


@pytest.mark.parametrize("old_status", ["running", "fail"])
def test_interrupted_or_failed_job_is_not_reused(tmp_path, pipeline, old_status):
    config = fixture_pool(tmp_path / "data", 1)
    out = tmp_path / "run"
    source = write_json(tmp_path / "input.json", {"v": 0})
    calls = []

    def work():
        calls.append(1)
        return {"checks": [{"status": "pass"}], "issues": []}

    _job(config, out, pipeline).run("job", lambda: [source], work, "result.json")
    state = read_json(out / "jobs.json")
    state["jobs"]["job"]["status"] = old_status
    write_json(out / "jobs.json", state)
    _job(config, out, pipeline, True).run("job", lambda: [source], work, "result.json")
    assert len(calls) == 2


def test_checker_code_change_invalidates_resume(tmp_path, pipeline, monkeypatch):
    config = fixture_pool(tmp_path / "data", 1)
    out = tmp_path / "run"
    source = write_json(tmp_path / "input.json", {})
    calls = []

    def work():
        calls.append(1)
        return {"checks": [{"status": "pass"}], "issues": []}

    _job(config, out, pipeline).run("job", lambda: [source], work, "result.json")
    monkeypatch.setattr(pipeline, "code_bindings", lambda: {"fixture_checker": {"sha256": "new"}})
    _job(config, out, pipeline, True).run("job", lambda: [source], work, "result.json")
    assert len(calls) == 2


@pytest.mark.parametrize("location", ["same", "inside", "ancestor", "symlink"])
def test_output_cannot_overlap_source(tmp_path, pipeline, location):
    config = fixture_pool(tmp_path / "data", 1)
    source = Path(config["base_root"])
    if location == "same":
        output = source
    elif location == "inside":
        output = source / "run"
    elif location == "ancestor":
        output = source.parent
    else:
        output = tmp_path / "source_alias"
        output.symlink_to(source, target_is_directory=True)
    before = {p: sha(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="重叠|包含"):
        pipeline.run_pipeline(config, output)
    assert before == {p: sha(p) for p in source.rglob("*") if p.is_file()}


def test_same_output_lock_rejects_concurrent_writer_and_releases(tmp_path, pipeline):
    config = fixture_pool(tmp_path / "data", 1)
    output = tmp_path / "run"
    with pipeline.output_lock(output):
        with pytest.raises(RuntimeError, match="正在运行"):
            pipeline.run_pipeline(config, output)
    with pipeline.output_lock(output):
        pass


def test_output_child_symlink_cannot_redirect_reports_into_sources(tmp_path, pipeline):
    config = fixture_pool(tmp_path / "data", 1)
    output = tmp_path / "run"
    output.mkdir()
    source = Path(config["base_root"])
    (output / "sequence_reports").symlink_to(source / "sequences", target_is_directory=True)
    before = {p: sha(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="软链接"):
        pipeline.run_pipeline(config, output)
    assert before == {p: sha(p) for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage, expected_code", [
    ("duplicate", "unique_ids"), ("missing", "acquisition_count"),
    ("illegal_split", "split_assignments"), ("expected_300", "acquisition_count"),
])
def test_inventory_rejects_bad_ids_counts_and_split(tmp_path, pipeline, damage, expected_code):
    config = fixture_pool(tmp_path / "data", 3)
    path = Path(config["base_root"]) / "dataset_index.json"
    index = read_json(path)
    if damage == "duplicate":
        index["sequences"][1]["sequence_id"] = index["sequences"][0]["sequence_id"]
    elif damage == "missing":
        index["sequences"].pop()
    elif damage == "illegal_split":
        index["sequences"][1]["split"] = "validation"
    else:
        config["expected_sequences"] = 300
    write_json(path, index)
    report = pipeline.inspect_inventory(pipeline.normalized_config(config))
    assert any(c["code"] == expected_code and c["status"] == "fail" for c in report["checks"])


def test_inventory_checks_each_native_frame_count_against_source_selection(tmp_path, pipeline):
    config = fixture_pool(tmp_path / "data", 3)
    path = Path(config["base_root"]) / "dataset_index.json"
    index = read_json(path)
    # Keep the total unchanged: a sum-only check cannot find this mismatch.
    index["sequences"][0]["frame_count"] -= 1
    index["sequences"][1]["frame_count"] += 1
    write_json(path, index)
    config.pop("v3_root")  # Isolate native-source verification from the index SHA binding.
    report = pipeline.inspect_inventory(pipeline.normalized_config(config))
    assert any(c["status"] == "fail" and "frame" in c["code"] for c in report["checks"])


def test_parallel_sequence_failure_does_not_drop_other_frames(tmp_path, pipeline, monkeypatch):
    from src.annotation import review_checks
    config = fixture_pool(tmp_path / "data", 3)
    rows = read_json(Path(config["base_root"]) / "dataset_index.json")["sequences"]
    rendezvous = threading.Barrier(3, timeout=10)
    calls = Counter()
    guard = threading.Lock()

    def controlled(*args, **kwargs):
        row = args[1]
        with guard:
            calls[row["sequence_id"]] += 1
        rendezvous.wait()
        if row["sequence_id"] == rows[1]["sequence_id"]:
            raise RuntimeError("isolated fixture failure")
        return fake_report(*args, **kwargs)

    monkeypatch.setattr(review_checks, "review_sequence", controlled)
    out = tmp_path / "run"
    summary = pipeline.run_pipeline(config, out)
    reports = [read_json(out / "sequence_reports" / (row["sequence_id"] + ".json")) for row in rows]
    assert calls == Counter({row["sequence_id"]: 1 for row in rows})
    assert summary["processed_sequences"] == 3 and summary["execution_complete"] is True
    assert summary["exit_code"] == 2 and summary["automated_checks_passed"] is False
    assert summary["metrics"]["native_frames"] == 192
    assert Counter(r["status"] for r in reports)["fail"] == 1
    assert all(r["frame_count"] == 64 for r in reports)
    for filename in ["summary.json", "jobs.json", "issues.jsonl", "review_queue.jsonl",
                     "复核汇总.csv", "report.html"]:
        assert (out / filename).is_file()


def test_actual_gallery_dependency_invalidates_only_affected_sequence(tmp_path, pipeline, monkeypatch):
    from src.annotation import review_checks
    config = fixture_pool(tmp_path / "data", 3)
    base, root = Path(config["base_root"]), Path(config["v3_root"])
    rows = read_json(base / "dataset_index.json")["sequences"]
    uid = rows[0]["sequence_id"]
    page = root / "evidence" / uid / "page.jpg"
    page.parent.mkdir(parents=True)
    page.write_bytes(b"fixture-image-version-1")
    path = root / "annotations" / (uid + ".json")
    annotation = read_json(path)
    annotation["review_records"] = [{"evidence": [{
        "gallery_path": str(page.relative_to(root)), "gallery_sha256": sha(page), "frames": [0]}]}]
    write_json(path, annotation)
    overlay_path = root / "mamba_overlay_index.json"
    overlay = read_json(overlay_path)
    next(r for r in overlay["sequences"] if r["sequence_id"] == uid)["sha256"] = sha(path)
    write_json(overlay_path, overlay)
    calls = Counter()

    def controlled(*args, **kwargs):
        calls[args[1]["sequence_id"]] += 1
        return fake_report(*args, **kwargs)

    monkeypatch.setattr(review_checks, "review_sequence", controlled)
    output = tmp_path / "run"
    initial = pipeline.run_pipeline(config, output)
    resumed = pipeline.run_pipeline(config, output, resume=True)
    assert initial["exit_code"] == resumed["exit_code"] == 0
    assert calls == Counter({row["sequence_id"]: 1 for row in rows})
    assert resumed["cache_hits"] == 4  # inventory plus all three sequences
    page.write_bytes(b"fixture-image-version-2")
    pipeline.run_pipeline(config, output, resume=True)
    assert calls[uid] == 2
    assert all(calls[r["sequence_id"]] == 1 for r in rows[1:])
    job = read_json(output / "jobs.json")["jobs"]["sequence:" + uid]
    assert str(page.absolute()) in job["input_bindings"] and job["cache_hit"] is False


def test_review_dependencies_include_manifest_and_nested_missing_pages(tmp_path):
    from src.annotation.review_checks import sequence_dependencies
    config = fixture_pool(tmp_path / "data", 1)
    base = Path(config["base_root"])
    row = read_json(base / "dataset_index.json")["sequences"][0]
    v5 = tmp_path / "v5"
    manifest = write_json(v5 / "dense_evidence" / "manifest.json", {
        "pages": [{"path": "dense_evidence/page_exists.jpg"},
                  {"path": "dense_evidence/page_missing.jpg"}]})
    page = v5 / "dense_evidence/page_exists.jpg"
    page.write_bytes(b"image")
    review = write_json(v5 / "dense_reviews" / (row["sequence_id"] + ".json"), {
        "evidence_manifests": [{"path": str(manifest.relative_to(v5)), "sha256": sha(manifest),
            "reviewed_page_paths": ["dense_evidence/extra_missing.jpg"]}]})
    dependencies = set(sequence_dependencies(base, row, review_path=review))
    assert {review, manifest, page, v5 / "dense_evidence/page_missing.jpg",
            v5 / "dense_evidence/extra_missing.jpg"} <= dependencies


def test_invalid_source_identity_does_not_abort_other_sequence_jobs(tmp_path, pipeline, monkeypatch):
    from src.annotation import review_checks
    config = fixture_pool(tmp_path / "data", 3)
    path = Path(config["base_root"]) / "dataset_index.json"
    index = read_json(path)
    del index["sequences"][1]["frame_count"]
    write_json(path, index)
    monkeypatch.setattr(review_checks, "review_sequence", fake_report)
    output = tmp_path / "run"
    summary = pipeline.run_pipeline(config, output)
    assert summary["processed_sequences"] == 3 and summary["exit_code"] == 2
    reports = [read_json(output / "sequence_reports" / (r["sequence_id"] + ".json"))
               for r in index["sequences"]]
    assert Counter(r["status"] for r in reports)["fail"] == 1


def test_cli_require_complete_exits_three_for_valid_unknown_data(tmp_path):
    config = fixture_pool(tmp_path / "data", 1)
    config_path = write_json(tmp_path / "config.json", config)
    output = tmp_path / "run"
    completed = subprocess.run([sys.executable, "-m", "src.annotation.review_pipeline",
        "--config", str(config_path), "--output", str(output), "--require-complete"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 3, completed.stdout + completed.stderr
    report = read_json(output / "summary.json")
    assert report["automated_checks_passed"] is True
    assert report["full_annotation_complete"] is False and report["exit_code"] == 3


def queue_issue(lo, hi, message, **changes):
    return {"sequence_id": "seq", "split": "train", "code": "transition_censored",
            "entity_id": "scanner", "layer": "transition", "severity": "pending",
            "start_frame": lo, "end_frame": hi, "message_zh": message,
            "next_action_zh": "核对" + message, **changes}


def test_review_queue_merges_overlapping_and_adjacent_ranges_with_context(pipeline):
    issues = [queue_issue(0, 5, "接触边界"), queue_issue(3, 8, "抓持边界"),
              queue_issue(8, 12, "失去支撑边界")]
    before = json.loads(json.dumps(issues))
    queue = pipeline.build_review_queue(issues, 5)
    assert [(r["start_frame"], r["end_frame"]) for r in queue] == [(0, 5), (5, 10), (10, 12)]
    expected = [{1, 2}, {2, 3}, {3}]
    for row, lines in zip(queue, expected):
        assert set(row["source_issue_lines"]) == lines
        assert set(row["messages_zh"]) == {issues[i - 1]["message_zh"] for i in lines}
        assert set(row["next_actions_zh"]) == {issues[i - 1]["next_action_zh"] for i in lines}
    assert issues == before  # Source events remain separate records.


@pytest.mark.parametrize("field,value", [
    ("sequence_id", "other_sequence"), ("split", "test"), ("code", "other_code"),
    ("entity_id", "carton"), ("layer", "relation"), ("severity", "warning"),
])
def test_review_queue_does_not_merge_distinct_context_groups(pipeline, field, value):
    issues = [queue_issue(0, 8, "扫描枪"), queue_issue(3, 5, "另一问题", **{field: value})]
    queue = pipeline.build_review_queue(issues, 100)
    assert len(queue) == 2
    assert {tuple(r["source_issue_lines"]) for r in queue} == {(1,), (2,)}
    zero = pipeline.build_review_queue([queue_issue(0, 0, "全局问题", sequence_id=None)], 100)
    assert len(zero) == 1 and zero[0]["start_frame"] == zero[0]["end_frame"] == 0


def test_summary_writer_keeps_original_issues_when_queue_is_merged(tmp_path, pipeline):
    config = pipeline.normalized_config({"base_root": str(tmp_path / "source"),
        "expected_sequences": 1, "queue_chunk_frames": 100})
    report = fake_report(None, {"sequence_id": "seq", "split": "train", "frame_count": 64})
    report["issues"] = [queue_issue(1, 5, "接触边界"), queue_issue(3, 8, "抓持边界")]
    original = json.loads(json.dumps(report["issues"]))
    output = tmp_path / "run"
    output.mkdir()
    pipeline.write_reports(output, config, [report], {"checks": [], "issues": []},
        {"status": "pending", "checks": [], "issues": [], "metrics": {}},
        {"jobs": {"sequence:seq": {"status": "pending", "cache_hit": False}}})
    source = [json.loads(line) for line in (output / "issues.jsonl").read_text().splitlines()]
    queued = [json.loads(line) for line in (output / "review_queue.jsonl").read_text().splitlines()]
    assert source == original and report["issues"] == original
    assert len(queued) == 1
    assert set(queued[0]["source_issue_lines"]) == {1, 2}
    assert [queued[0]["start_frame"], queued[0]["end_frame"]] == [1, 8]
