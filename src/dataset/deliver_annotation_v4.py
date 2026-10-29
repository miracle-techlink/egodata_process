"""Package audited native-frame candidates and actual continuous annotations."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def checked(root, name):
    data = read(root / name)
    if data.get("failed") != 0:
        raise ValueError(f"Audit did not pass: {name}")
    return data


AUDITED_REPORTS = {
    "dense_gap_audit.json", "dense_observables_audit.json", "continuous_review_audit.json",
    "entity_grounding_audit.json", "dense_window_audit.json",
}


def checked_audit_lock(root):
    """The independent auditor freezes this lock; delivery never creates it."""
    lock = read(root / "audit_lock.json")
    if (lock.get("schema_version") != "assistant_annotation_v4_audit_lock"
            or set(lock.get("reports", {})) != AUDITED_REPORTS):
        raise ValueError("Missing or incomplete independently frozen audit lock")
    for name, expected in lock["reports"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Audited report changed after independent freeze: {name}")
    return lock


def unique_rows(rows, description):
    result = {row["sequence_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate acquisition in {description}")
    return result


def sum_counts(rows):
    total = Counter()
    for row in rows:
        total.update({key: value for key, value in row.items() if type(value) is int})
    return total


def deliver(root, repo, *, check_only=False):
    root, repo = Path(root), Path(repo)
    audit_lock = checked_audit_lock(root)
    baseline = read(root / "dense_gap_audit.json")
    numeric = read(root / "dense_observables_report.json")
    numerical_audit = checked(root, "dense_observables_audit.json")
    continuous = checked(root, "continuous_review_audit.json")
    grounding = read(root / "entity_grounding_report.json")
    grounding_audit = checked(root, "entity_grounding_audit.json")
    window_audit = checked(root, "dense_window_audit.json")
    index = read(root / "mamba_dense_index.json")
    coverage = read(root / "dense_target_coverage.json")
    smoke = read(root / "dense_mamba_smoke.json")
    index_sha = sha(root / "mamba_dense_index.json")
    if audit_lock["mamba_dense_index_sha256"] != index_sha:
        raise ValueError("Mamba index differs from independent audit lock")
    if grounding_audit["grounding_report_sha256"] != sha(root / "entity_grounding_report.json"):
        raise ValueError("Grounding report changed after audit")
    if window_audit["dense_index_sha256"] != index_sha or any(
        r["mamba_dense_index_sha256"] != index_sha for r in (coverage, smoke)
    ):
        raise ValueError("Window audit, coverage or smoke uses a different index")
    if window_audit["continuous_review_audit_sha256"] != sha(root / "continuous_review_audit.json"):
        raise ValueError("Window audit uses an obsolete continuous review")
    for row in index["source_reports"].values():
        if sha(root / row["path"]) != row["sha256"]:
            raise ValueError("Mamba index source report changed")
    for split, row in index["splits"].items():
        if sha(root / row["window_path"]) != row["sha256"]:
            raise ValueError("Window content changed after audit")
        if coverage["splits"][split]["windows_sha256"] != row["sha256"]:
            raise ValueError("Coverage describes different window content")
        if coverage["splits"][split]["windows"] != row["window_count"]:
            raise ValueError("Coverage window count differs from index")
    if not smoke["passed"] or smoke["optimizer_steps"] != 0 or smoke["training_executed"]:
        raise ValueError("Expected successful diagnostic-only forward")
    if numerical_audit["delivery_sha256"] != sha(root / "dense_observables_report.json"):
        raise ValueError("Numeric report changed after audit")
    if numeric["sequences"] != baseline["sequence_count"] or numerical_audit["checked_sequences"] != numeric["sequences"]:
        raise ValueError("Dataset count differs across reports")
    baseline_rows = unique_rows(baseline["sequences"], "baseline")
    numeric_rows = unique_rows(numeric["records"], "numeric report")
    numeric_audit_rows = unique_rows(numerical_audit["records"], "numeric audit")
    grounding_rows = unique_rows(grounding["sequences"], "grounding report")
    grounding_audit_rows = unique_rows(grounding_audit["records"], "grounding audit")
    if (len(baseline_rows) != baseline["sequence_count"]
            or not all(set(rows) == set(baseline_rows) for rows in
                       (numeric_rows, numeric_audit_rows, grounding_rows, grounding_audit_rows))):
        raise ValueError("Acquisition identities differ across audited components")
    if Counter(numeric["totals"]) != sum_counts(numeric_rows.values()):
        raise ValueError("Numeric aggregate differs from per-sequence counts")
    if Counter(numerical_audit["counts"]) != Counter(numeric["totals"]):
        raise ValueError("Numeric audit counts differ from delivery counts")
    if Counter(grounding["totals"]) != sum_counts(grounding_rows.values()):
        raise ValueError("Grounding aggregate differs from per-sequence counts")
    for uid, row in grounding_rows.items():
        if sha(root / row["path"]) != row["sha256"] or grounding_audit_rows[uid]["sha256"] != row["sha256"]:
            raise ValueError("Grounding component changed after audit")
    review_rows = unique_rows(continuous["records"], "continuous review audit")
    if not set(review_rows) <= set(baseline_rows):
        raise ValueError("Continuous review refers to another source acquisition")
    if Counter(continuous["counts"]) != sum_counts(row["counts"] for row in review_rows.values()):
        raise ValueError("Continuous aggregate differs from audited sequence counts")
    index_rows = unique_rows(index["sequences"], "Mamba index")
    if set(index_rows) != set(review_rows):
        raise ValueError("Mamba index and audited continuous reviews differ")
    for uid, entry in index_rows.items():
        if (entry["review_sha256"] != review_rows[uid]["sha256"]
                or entry["review_path"] != review_rows[uid]["path"]
                or entry["reviewed_native_frame_count"] != review_rows[uid]["counts"]["reviewed_frames"]):
            raise ValueError("Mamba index uses a different audited review")
    if {p.stem for p in (root / "dense_reviews").glob("*.json")} != set(review_rows):
        raise ValueError("An unaudited continuous review is present")
    for row in review_rows.values():
        if sha(root / row["path"]) != row["sha256"]:
            raise ValueError("Continuous review changed after audit")
    rows, baseline_counts = [], Counter()
    for entry in baseline["sequences"]:
        path = root / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Per-sequence baseline changed")
        data = read(path)
        uid, c = data["sequence_id"], data["counts"]
        if uid != entry["sequence_id"]:
            raise ValueError("Per-sequence baseline identity changed")
        baseline_counts.update(c)
        a = review_rows.get(uid, {}).get("counts", {})
        metadata_path = root / "dense_sequences" / uid / "metadata.json"
        metadata_sha = sha(metadata_path)
        if (metadata_sha != numeric_rows[uid]["metadata_sha256"]
                or metadata_sha != numeric_audit_rows[uid]["metadata_sha256"]):
            raise ValueError("Per-sequence numeric metadata changed after audit")
        meta = read(metadata_path)
        for key, filename in (("observables", "observables.npz"),
                              ("boundary_candidates", "boundary_candidates.json")):
            if sha(metadata_path.parent / filename) != meta["sha256"][key]:
                raise ValueError(f"Per-sequence numeric payload changed after audit: {uid}/{filename}")
        if (meta["sequence_id"] != uid or meta["split"] != data["split"]
                or meta["counts"]["frames_indexed"] != c["native_frames"]
                or any(numeric_rows[uid][key] != value for key, value in meta["counts"].items())):
            raise ValueError("Numeric metadata count/identity differs from audited report")
        if not 0 <= a.get("positive_negative_labeled_frames", 0) <= a.get("reviewed_frames", 0) <= c["native_frames"]:
            raise ValueError("Labeled/reviewed coverage exceeds native timeline")
        rows.append({"采集ID": uid, "划分": data["split"], "原生帧数": c["native_frames"],
                     "逐帧索引帧数": meta["counts"]["frames_indexed"],
                     "双手有效帧": c["hand_both_valid_frames"],
                     "至少一手缺失帧": c["native_frames"] - c["hand_both_valid_frames"],
                     "任意场景位姿候选帧": c["scene_any_pose_candidate_frames"],
                     "任意场景表面候选帧": c["scene_any_surface_candidate_frames"],
                     "完全缺少场景几何候选帧": c["native_frames"] - c["scene_any_geometry_candidate_frames"],
                     "旧抽样查看帧": c["v3_reviewed_sample_frames"],
                     "新增连续查看帧": a.get("reviewed_frames", 0),
                     "尚未连续查看帧": c["native_frames"] - a.get("reviewed_frames", 0),
                     "明确正值数": a.get("positive_state_values", 0),
                     "明确负值数": a.get("negative_state_values", 0),
                     "有明确状态的帧": a.get("positive_negative_labeled_frames", 0),
                     "物理边界范围数": a.get("physical_interaction_transition_uncertainty_ranges", 0),
                     "精确物理边界数": a.get("confirmed_physical_interaction_transition_bounds", 0),
                     "精确视觉信号变化数": a.get("confirmed_visual_signal_transition_bounds", 0),
                     "待审距离阈值边界数": meta["counts"]["candidate_boundary_count"],
                     "已完成全部层标注": False})
    if Counter(baseline["counts"]) != baseline_counts:
        raise ValueError("Baseline aggregate differs from hashed sequence counts")
    if sum(row["逐帧索引帧数"] for row in rows) != numeric["totals"]["frames_indexed"]:
        raise ValueError("Indexed frame aggregate differs from per-sequence rows")
    if sum(row["新增连续查看帧"] for row in rows) != continuous["counts"]["reviewed_frames"]:
        raise ValueError("Reviewed frame aggregate differs from per-sequence rows")
    if continuous["continuous_visual_review_complete"] != (continuous["counts"]["reviewed_frames"] == baseline_counts["native_frames"]):
        raise ValueError("Continuous completion flag conflicts with actual reviewed coverage")
    if any(data.get("training_allowed") is True or data.get("full_annotation_complete") is True
           for data in (baseline, numeric, numerical_audit, continuous, grounding, grounding_audit, window_audit, index, coverage, smoke)):
        raise ValueError("Candidate/partial annotation data promoted to completed training data")
    if check_only:
        return {"passed": True, "sequence_count": len(rows), "native_frame_count": baseline_counts["native_frames"],
                "continuous_review_counts": continuous["counts"], "audit_lock_sha256": sha(root / "audit_lock.json"),
                "full_annotation_complete": False, "outputs_written": False}
    with (root / "逐段连续标注缺口.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frame_count = baseline["counts"]["native_frames"]
    counts = continuous["counts"]
    class_values = Counter()
    for row in continuous["records"]:
        class_values.update(row["state_values_by_type"])
    report = {
        "schema_version": "assistant_annotation_v4_dense_delivery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sequence_count": len(rows), "native_frame_count": frame_count,
        "numeric_counts": numeric["totals"], "continuous_review_counts": counts,
        "sequences_with_continuous_review": len(review_rows),
        "sequences_fully_viewed": sum(r["新增连续查看帧"] == r["原生帧数"] for r in rows),
        "remaining_unreviewed_native_frames": frame_count - counts.get("reviewed_frames", 0),
        "remaining_sequences_without_complete_continuous_review": sum(r["尚未连续查看帧"] > 0 for r in rows),
        "state_value_counts_by_type": dict(sorted(class_values.items())),
        "entity_grounding_counts": grounding["totals"],
        "mamba_window_counts": {k: r["window_count"] for k, r in index["splits"].items()},
        "mamba_target_coverage": coverage["splits"],
        "mamba_forward": {k: smoke[k] for k in ("passed", "cuda_device", "output_shapes",
            "finite", "optimizer_steps", "training_executed", "graph_head_implemented")},
        "hand_frames_with_any_missing_side": frame_count - baseline["counts"]["hand_both_valid_frames"],
        "frames_without_any_scene_geometry": frame_count - baseline["counts"]["scene_any_geometry_candidate_frames"],
        "dense_index_complete": numeric["totals"]["frames_indexed"] == frame_count,
        "continuous_visual_review_complete": continuous["continuous_visual_review_complete"],
        "full_annotation_complete": False, "training_allowed": False,
        "human_label_valid": False,
        "scope_zh": "300段逐帧数值候选可读取；只有实际连续看过且明确判定的片段有正/负状态。距离、轨迹身份传播、文件齐全均不等于全标注完成。",
        "remaining_zh": ["绝大部分视频尚未连续逐帧视觉复核。",
                         "当前交互物体的稳定实例身份、整物体/部件对应及3D覆盖不足。",
                         "多数接触、抓持、支撑和脱离边界仍需查看；遮挡无法判定时必须保留范围。",
                         "至少一手缺测的帧保留mask；非刚体整物体6D不适用，铰接物需部件表示。",
                         "离线双向几何不等同已验证因果观测，当前单客体头不直接监督未映射视觉实体/双客体关系。"],
        "source_bindings": baseline["source_bindings"],
        "audit_lock_sha256": sha(root / "audit_lock.json"),
        "report_sha256": {name: sha(root / name) for name in ["audit_lock.json", "dense_gap_audit.json",
            "dense_observables_report.json", "dense_observables_audit.json", "continuous_review_audit.json",
            "entity_grounding_report.json", "entity_grounding_audit.json", "mamba_dense_index.json",
            "dense_window_audit.json", "dense_target_coverage.json", "dense_mamba_smoke.json",
            "review_web/build_report.json", "review_web_qa/browser_qa.json",
            "validation_receipt.json", "数据格式与使用说明.json", "逐段连续标注缺口.csv"]},
    }
    save(root / "dense_delivery_report.json", report)
    paths = sorted({p for pattern in ["src/annotation/*v4*.py", "src/dataset/*v4*.py", "src/mamba/*v4*.py", "tests/*v4*.py"]
                    for p in repo.glob(pattern)} | {repo / f for f in (
                        ".agent/memory/03_data_pipeline.md", ".agent/memory/04_resources.md",
                        ".agent/memory/05_progress_log.md", "docs/DELIVERABLES.md",
                        "docs/HAND_OBJECT_HANDOFF_SOP.md")})
    sources = []
    for path in paths:
        relative = path.relative_to(repo)
        target = root / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources.append({"path": str(relative), "sha256": sha(target)})
    save(root / "code_manifest.json", {"files": sources})
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "delivery_manifest.json" or path.suffix == ".log":
            continue
        if any(part in path.parts for part in ("__pycache__", "notes", "pilots", "dense_reviews_history")):
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)})
    save(root / "delivery_manifest.json", {"schema_version": "assistant_annotation_v4_manifest",
         "generated_at": datetime.now(timezone.utc).isoformat(), "files": files,
         "total_bytes": sum(r["bytes"] for r in files), "training_allowed": False,
         "scope_zh": "原始视频与基础V2/V3外部依赖不复制；其绑定SHA见各报告及Mamba索引。"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate frozen inputs without writing any delivery artifacts")
    args = parser.parse_args()
    report = deliver(args.root, args.repo, check_only=args.check_only)
    print(json.dumps(report if args.check_only else {k: report[k] for k in (
        "sequence_count", "native_frame_count", "continuous_review_counts", "full_annotation_complete")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
