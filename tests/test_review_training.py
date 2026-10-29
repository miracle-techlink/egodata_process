"""Regression tests for isolated audit writes and complete cache dependencies.

The reused numeric fixture is synthetic and isolates index auditing; its small
placeholder provenance reports are not claims about real model/visual quality.
"""
import hashlib
import json
from pathlib import Path

import pytest

from test_audit_training_v5 import graph_fixture  # Existing actual numeric audit fixture.
from src.annotation.review_training import check_training_subset, training_dependencies, REQUIRED_EVIDENCE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def released_fixture(graph_fixture):
    from src.dataset.audit_training_v5 import audit_training_index
    root = graph_fixture
    audit_training_index(root)
    for name in REQUIRED_EVIDENCE:
        path = root / name
        if not path.exists():
            write(path, {"synthetic_test_evidence": True})
    index = read(root / "training_index.json")
    write(root / "training_release.json", {
        "training_index_sha256": sha(root / "training_index.json"),
        "training_scope": index["training_scope"], "interaction_training_allowed": True,
        "full_joint_training_allowed": False, "full_annotation_complete": False,
        "whole_object_6d_certified": False, "positive_types_absent_from_train": [],
        "evidence_sha256": {name: sha(root / name) for name in REQUIRED_EVIDENCE},
    })
    return root


def tree_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}


def test_actual_full_audit_and_three_default_datasets_do_not_write_source(released_fixture, monkeypatch):
    import src.mamba.dataset_v5 as dataset_module
    source = released_fixture
    before = tree_hashes(source)
    original = dataset_module.EgoDemoGraphV5Dataset
    calls = []

    def record_calls(root, split="train", **kwargs):
        calls.append((split, kwargs.get("diagnostic", False), Path(root)))
        return original(root, split, **kwargs)

    monkeypatch.setattr(dataset_module, "EgoDemoGraphV5Dataset", record_calls)
    output = source.parent / "stage_output"
    result = check_training_subset(source, output)
    assert result["status"] == "pass", result
    assert result["metrics"]["actual_audited_windows"] > 0
    assert result["metrics"]["actual_audited_windows"] == result["metrics"]["default_dataset_windows_loaded"]
    assert result["metrics"]["optimizer_steps"] == 0
    assert [(s, d) for s, d, _ in calls] == [(s, False) for s in ("train", "val", "test")] + [
        (s, True) for s in ("train", "val", "test")]
    assert all(path != source and output in path.parents for _, _, path in calls)
    assert tree_hashes(source) == before
    assert sha(output / "training_index_audit_recomputed.json") != before["training_index_audit.json"]
    assert not list(output.glob("training_audit_overlay_*"))
    assert all(Path(p).is_absolute() and output in Path(p).parents and Path(p).is_file()
               for p in result["artifact_paths"])


def test_source_report_symlink_is_copied_before_auditor_writes(released_fixture):
    source = released_fixture
    audit = source / "training_index_audit.json"
    target = source.parent / "separately_frozen_audit.json"
    target.write_bytes(audit.read_bytes())
    audit.unlink(); audit.symlink_to(target)
    before = sha(target)
    result = check_training_subset(source, source.parent / "audit_output")
    assert result["status"] == "pass", result
    assert audit.is_symlink() and sha(target) == before


def test_missing_external_original_array_is_pending_not_pass(released_fixture):
    source = released_fixture
    original = source.parent / "base" / "sequence.npz"
    original.unlink()
    result = check_training_subset(source, source.parent / "missing_output")
    assert result["status"] == "pending"
    assert str(original) in result["metrics"]["missing_dependencies"]
    assert "actual_audited_windows" not in result["metrics"]
    assert result["issues"][0]["severity"] == "pending"


def test_nested_evidence_is_fingerprinted_and_tamper_fails_actual_audit(released_fixture):
    source = released_fixture
    dependencies = training_dependencies(source)
    expected = {source / "tracking_manifest.json", source / "gallery.jpg", source / "native_metadata.json",
                source.parent / "base" / "sequence.npz", source / "visual_observations/uid/seeds.json",
                source / "visual_observations/uid/tracking_review.json", source / "sequences/uid/hands.npz"}
    assert expected <= set(dependencies)
    assert {source / name for name in REQUIRED_EVIDENCE} <= set(dependencies)
    assert all(p.is_absolute() and p.is_file() for p in dependencies)
    before = {str(p): sha(p) for p in dependencies}
    (source / "gallery.jpg").write_bytes(b"changed real evidence bytes")
    after = {str(p): sha(p) for p in training_dependencies(source)}
    assert before != after
    result = check_training_subset(source, source.parent / "tampered_output")
    assert result["status"] == "fail"
    assert "tracking page changed" in result["issues"][0]["message_zh"]


def test_dependency_discovery_follows_seed_manifest_crop_and_external_intrinsics(released_fixture):
    source = released_fixture
    outside = source.parent / "source_camera"
    outside.mkdir()
    intrinsics = outside / "intrinsics.json"
    write(intrinsics, {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    meta = outside / "metadata.json"
    write(meta, {"intrinsics_path": str(intrinsics)})
    crop = source / "evidence" / "frame.png"
    crop.parent.mkdir(); crop.write_bytes(b"synthetic pixels")
    manifest = source / "evidence" / "manifest.json"
    write(manifest, {"source_bindings": {"metadata_path": str(meta)},
                     "frames": [{"crop_path": "evidence/frame.png"}]})
    seeds = source / "visual_observations/uid/seeds.json"
    doc = read(seeds)
    doc["seeds"][0]["evidence_manifest_path"] = "evidence/manifest.json"
    write(seeds, doc)
    assert {manifest, crop, meta, intrinsics} <= set(training_dependencies(source))
    manifest.unlink()
    assert manifest in training_dependencies(source)  # Missing must also affect resume fingerprints.


def test_dependency_discovery_ignores_debug_and_unread_historical_paths(released_fixture):
    source = released_fixture
    metadata = source / "native_metadata.json"
    doc = read(metadata)
    doc.update(debug_path="missing/debug.png", notes={"path": "missing/note.json"})
    write(metadata, doc)
    review = source / "dense_reviews/uid.json"
    doc = read(review)
    doc.update(parent_review_path="missing/legacy_review.json",
               source_bindings={"v3_annotation_path": "missing/legacy_annotation.json"})
    write(review, doc)
    dependencies = set(training_dependencies(source))
    assert metadata in dependencies and review in dependencies
    assert source / "tracking_manifest.json" in dependencies
    assert source / "gallery.jpg" in dependencies
    assert not any(source / "missing" in p.parents for p in dependencies)


@pytest.mark.parametrize("change", ["missing_key", "changed_evidence", "scope", "overclaim", "unseen_type"])
def test_release_integrity_and_cross_split_type_claims_are_not_silently_accepted(released_fixture, change):
    source = released_fixture
    path = source / "training_release.json"
    release = read(path)
    if change == "missing_key":
        del release["evidence_sha256"]["audit_sources/model_report_crosscheck_v2.json"]
    elif change == "changed_evidence":
        write(source / "model_checks/real_graph_mamba_smoke.json", {"tampered": True})
    elif change == "scope":
        release["training_scope"] = "all_300_full_joint_training"
    elif change == "overclaim":
        release["whole_object_6d_certified"] = True
    else:
        release["positive_types_absent_from_train"] = ["operation:invented"]
    write(path, release)
    before = tree_hashes(source)
    result = check_training_subset(source, source.parent / "invalid_output")
    assert result["status"] == "fail", result
    assert tree_hashes(source) == before


@pytest.mark.parametrize("relative", [".", "inside_output", ".."])
def test_output_cannot_overlap_frozen_source(released_fixture, relative):
    source = released_fixture
    before = tree_hashes(source)
    with pytest.raises(ValueError, match="完全分离"):
        check_training_subset(source, source / relative)
    assert tree_hashes(source) == before


def test_malformed_index_is_failure_not_missing_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "training_index.json").write_text("{broken json")
    result = check_training_subset(source, tmp_path / "output")
    assert result["status"] == "fail"
