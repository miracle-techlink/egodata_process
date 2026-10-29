"""Read-only automatic checks for the explicitly scoped V5 training subset.

Dependency discovery returns absolute FILE paths, including missing paths. It
does not return directories. Callers must fingerprint file contents and missing
state, then rediscover the list on every run (including resume).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback


REQUIRED_EVIDENCE = frozenset({
    "review_seed_audit.json", "training_index_audit.json", "delivery_integrity_checks_v5.json",
    "model_checks/real_graph_mamba_smoke.json", "model_checks/real_graph_target_coverage.json",
    "audit_sources/model_report_crosscheck_v2.json",
})
SPLITS = ("train", "val", "test")
SOURCE_KEYS = ("review", "observations", "observation_metadata", "seeds", "tracking_review", "sequence")


def _absolute(path) -> Path:
    # Keep the logical filename so a changed symlink is rediscovered as well.
    return Path(os.path.abspath(os.fspath(path)))


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _local(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"冻结包内路径必须是相对路径：{relative}")
    return root / p


def _referenced(root, value):
    p = Path(value)
    return _absolute(p if p.is_absolute() else root / p)


def training_dependencies(v5_root) -> list[Path]:
    """Enumerate numeric, annotation and nested evidence inputs, never just reports.

    Evidence manifests contribute page/crop files plus their source metadata,
    calibration, hands and video references. Those JSON metadata files are
    followed too. Missing manifests remain dependencies; once available their
    nested files are added on the next discovery. Malformed JSON is left to the
    actual check stage, while its bytes still invalidate cached results.
    """
    root = _absolute(v5_root)
    paths, inspected = set(), set()

    def add(path):
        path = _absolute(path)
        paths.add(path)
        return path

    def optional(path):
        add(path)
        try:
            value = _read(path)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def document(path, kind):
        """Follow declared evidence fields, never arbitrary *_path/debug values."""
        path = add(path)
        identity = (path, kind)
        if identity in inspected:
            return
        inspected.add(identity)
        doc = optional(path)

        def reference(value, next_kind=None):
            if isinstance(value, str) and value:
                target = add(_referenced(root, value))
                if next_kind and target.suffix.lower() == ".json":
                    document(target, next_kind)

        def pages(record):
            for value in record.get("reviewed_page_paths", []):
                reference(value)

        if kind == "review":
            for record in doc.get("evidence_manifests", []):
                reference(record.get("path"), "manifest")
                pages(record)
        elif kind == "seeds":
            for seed in doc.get("seeds", []):
                reference(seed.get("evidence_manifest_path"), "manifest")
                for key in ("evidence_frame_path", "evidence_crop_path", "evidence_page_path"):
                    reference(seed.get(key))
        elif kind == "tracking_review":
            reference(doc.get("evidence_manifest_path"), "manifest")
            pages(doc)
            for record in doc.get("additional_native_evidence", []):
                reference(record.get("path"))
        elif kind == "manifest":
            for page in doc.get("pages", []):
                reference(page.get("path"))
            for frame in doc.get("frames", []):
                reference(frame.get("crop_path"))
        if kind in {"manifest", "metadata"}:
            # K is normally inline in source_metadata_path; explicit external
            # calibration files remain dependencies when a manifest names them.
            declared = {
                "metadata_path": "metadata", "source_metadata_path": "metadata",
                "intrinsics_path": "metadata", "calibration_path": "metadata",
                "video_path": None, "source_video_path": None, "hands_path": None,
                "source_hands_path": None, "sequence_path": None, "base_sequence_path": None,
                "observations_path": None, "seeds_path": "seeds", "tracking_metadata_path": "metadata",
            }
            for record in (doc, doc.get("source_bindings", {})):
                for key, next_kind in declared.items():
                    reference(record.get(key), next_kind)

    index = optional(root / "training_index.json")
    release = optional(_referenced(root, index.get("admission_path", "training_release.json")))
    for name in REQUIRED_EVIDENCE | set(release.get("evidence_sha256", {}) if isinstance(release.get("evidence_sha256", {}), dict) else ()):
        add(_referenced(root, name))
    for name in ("base_dataset_index.json", "split_manifest.json", "vocab.json"):
        add(root / "sources" / name)
    for info in index.get("splits", {}).values():
        if isinstance(info, dict) and info.get("path"):
            add(_referenced(root, info["path"]))
    base = optional(root / "sources" / "base_dataset_index.json")
    originals = {r.get("sequence_id"): r for r in base.get("sequences", []) if isinstance(r, dict)}
    base_root = _referenced(root, index["base_root"]) if index.get("base_root") else None
    if base_root:
        add(base_root / "dataset_index.json")
    for row in index.get("sequences", []):
        if not isinstance(row, dict):
            continue
        for key in SOURCE_KEYS:
            if row.get(key + "_path"):
                path = add(_referenced(root, row[key + "_path"]))
                if key in {"review", "observation_metadata", "seeds", "tracking_review"}:
                    document(path, "metadata" if key == "observation_metadata" else key)
        original = originals.get(row.get("sequence_id"), {})
        if base_root and original.get("sequence_path"):
            add(_referenced(base_root, original["sequence_path"]))
    repo = Path(__file__).resolve().parents[2]
    for source in ("src/annotation/review_training.py", "src/dataset/audit_training_v5.py",
                   "src/dataset/audit_dense_annotation_v4.py", "src/mamba/dataset_v5.py",
                   "src/mamba/egodemo_v2.py"):
        add(repo / source)
    return sorted(paths, key=str)


def _save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _is_file(path):
    try:
        return path.is_file()
    except OSError:
        return False


def _check_output(root, output, index):
    roots = [root]
    if index.get("base_root"):
        roots.append(_referenced(root, index["base_root"]))
    for source in roots:
        source = source.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("复核输出目录必须与冻结数据及原数组目录完全分离")


def _make_overlay(source, target):
    """Only existing audit writes are top-level; every top-level file is copied."""
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            destination.symlink_to(child.absolute(), target_is_directory=True)
        elif child.is_file():
            shutil.copy2(child, destination, follow_symlinks=True)
        elif child.is_symlink():
            destination.symlink_to(child.absolute())
    report = target / "training_index_audit.json"
    if report.is_symlink():
        raise ValueError("审计写入目标不能链接到冻结源")


def _run_audit(root):
    from src.dataset.audit_training_v5 import audit_training_index
    return audit_training_index(root)


def _load_defaults(root, index):
    from src.mamba.dataset_v5 import EgoDemoGraphV5Dataset
    result = {}
    for split in SPLITS:
        # Intentionally no diagnostic=True: check the real admission gate.
        dataset = EgoDemoGraphV5Dataset(root, split=split)
        if len(dataset) != index["splits"][split]["windows"]:
            raise ValueError(f"{split} 默认 Dataset 窗口数量不一致")
        counts, ids, shapes = Counter(), set(), None
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["metadata"]["split"] != split:
                raise ValueError("默认 Dataset 返回跨划分样本")
            ids.add(sample["metadata"]["sequence_id"])
            mask, values = sample["graph_label_valid"], sample["graph_values"]
            positive, negative = int(((values == 1) & mask).sum()), int(((values == 0) & mask).sum())
            counts.update(windows=1, known_positive_cells=positive, known_negative_cells=negative,
                          windows_with_positive=int(positive > 0), windows_with_negative=int(negative > 0))
            if shapes is None:
                shapes = {k: list(sample[k].shape) for k in ("entity_features", "entity_valid", "graph_values", "graph_label_valid")}
        if dict(counts) != index["counts"][split]:
            raise ValueError(f"{split} 默认 Dataset 正负监督计数不一致")
        if len(ids) != index["splits"][split]["sequence_count"]:
            raise ValueError(f"{split} 默认 Dataset 采集数量不一致")
        result[split] = {"counts": dict(counts), "sequence_ids": sorted(ids), "sample_shapes": shapes,
                         "diagnostic": False}
    sets = [set(result[s]["sequence_ids"]) for s in SPLITS]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("同一采集出现在多个划分")
    return result


def check_training_subset(v5_root, output_dir) -> dict:
    """Run existing complete-window audit without changing frozen V5 artifacts.

    Missing source/evidence is pending, never pass. Malformed or inconsistent
    inputs fail. This stage creates reports only, with zero optimizer steps.
    The orchestrator owns full dependency fingerprints before/after execution.
    """
    root, output = _absolute(v5_root), Path(output_dir).resolve()
    try:
        preliminary = _read(root / "training_index.json")
    except (OSError, ValueError):
        preliminary = {}
    _check_output(root, output, preliminary)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "training_stage_report.json"
    result = {"schema_version": "automatic_training_subset_review_v1", "status": "pending",
              "created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
              "checks": [], "metrics": {"optimizer_steps": 0, "model_training_executed": False},
              "issues": [], "artifact_paths": [str(report_path)],
              "full_annotation_complete": False, "full_joint_training_allowed": False}

    def checked(code, message):
        result["checks"].append({"code": code, "status": "pass", "message_zh": message})

    try:
        index = _read(root / "training_index.json")
        if not isinstance(index, dict):
            raise ValueError("训练索引必须为 JSON 对象")
        dependencies = training_dependencies(root)
        result["metrics"]["dependency_file_count"] = len(dependencies)
        missing = [str(p) for p in dependencies if not _is_file(p)]
        if missing:
            result["metrics"]["missing_dependencies"] = missing
            raise FileNotFoundError(f"缺少或无法访问 {len(missing)} 个训练审计依赖；首项：{missing[0]}")
        checked("TRAINING_INPUTS_AVAILABLE", "训练数值、嵌套图像证据和原始数组依赖均可读取。")
        index_sha = _digest(root / "training_index.json")
        release_path = _local(root, index["admission_path"])
        release = _read(release_path)
        if (release.get("interaction_training_allowed") is not True
                or release.get("training_index_sha256") != index_sha
                or release.get("training_scope") != index["training_scope"]):
            raise ValueError("冻结训练放行记录与索引或限定范围不一致")
        for flag in ("full_joint_training_allowed", "full_annotation_complete", "whole_object_6d_certified"):
            if index.get(flag) is not False or release.get(flag) is not False:
                raise ValueError("交互子集放行不能替代完整联合标注或整物体6D认证")
        evidence = release.get("evidence_sha256")
        if not isinstance(evidence, dict) or not REQUIRED_EVIDENCE <= set(evidence):
            raise ValueError("训练放行缺少六份必需凭据")
        for name, expected in evidence.items():
            if not expected or _digest(_local(root, name)) != expected:
                raise ValueError(f"冻结训练放行凭据 SHA 不一致：{name}")
        for row in index["sequences"]:
            for key in SOURCE_KEYS:
                if _digest(_local(root, row[key + "_path"])) != row[key + "_sha256"]:
                    raise ValueError(f"冻结 {key} SHA 不一致：{row['sequence_id']}")
        result["metrics"].update(training_index_sha256=index_sha, training_release_sha256=_digest(release_path),
                                 source_interaction_training_allowed=True, training_scope=index["training_scope"])
        checked("TRAINING_FROZEN_ADMISSION", "核对限定范围放行、六份凭据及各采集六组数值/标注 SHA。")
        original_audit_sha = _digest(root / "training_index_audit.json")
        with tempfile.TemporaryDirectory(prefix="training_audit_overlay_", dir=output) as temporary:
            overlay = Path(temporary)
            _make_overlay(root, overlay)
            # Default admission must use the original frozen audit credential;
            # the independent auditor subsequently writes a NEW timestamped one.
            defaults = _load_defaults(overlay, index)
            default_path = output / "default_dataset_checks.json"
            _save(default_path, defaults)
            result["artifact_paths"].append(str(default_path))
            checked("TRAINING_DEFAULT_DATASETS", "三个固定划分均以默认正式准入加载并遍历全部窗口。")
            audit = _run_audit(overlay)
            if audit.get("failed") != 0:
                raise ValueError("独立训练窗口审计未通过")
            expected_count = sum(index["splits"][s]["windows"] for s in SPLITS)
            if audit.get("actual_dataset_windows_loaded") != expected_count or audit.get("counts") != index["counts"]:
                raise ValueError("独立审计未复算全部已发布窗口或汇总不一致")
            if audit.get("known_cells_by_type") != index["known_cells_by_type"]:
                raise ValueError("逐类型已知监督计数不一致")
            positive_sets = {s: set(audit["type_coverage"][s]["positive_cells_by_type"]) for s in SPLITS}
            absent = sorted((positive_sets["val"] | positive_sets["test"]) - positive_sets["train"])
            if absent != sorted(release.get("positive_types_absent_from_train", [])):
                raise ValueError("验证/测试中训练未见正例类型与冻结放行不一致")
            fresh_path = overlay / "training_index_audit.json"
            if fresh_path.is_symlink() or not fresh_path.is_file() or _read(fresh_path) != audit:
                raise ValueError("独立审计未在隔离目录生成匹配的新报告")
            audit_path = output / "training_index_audit_recomputed.json"
            shutil.copy2(fresh_path, audit_path)
            result["artifact_paths"].append(str(audit_path))
        if _digest(root / "training_index_audit.json") != original_audit_sha:
            raise ValueError("冻结源审计报告意外发生变化")
        checked("TRAINING_ALL_WINDOWS_RECOMPUTED", "独立复算全部窗口的固定划分、历史实体槽、120维输入与有向正负监督。")
        checked("TRAINING_TYPE_COVERAGE", "逐类型监督统计及训练未见的验证/测试正例类型一致。")
        checked("TRAINING_SOURCE_READ_ONLY", "新审计报告仅保存于独立输出目录，原准入凭据保持不变。")
        result["metrics"].update(actual_audited_windows=expected_count,
            default_dataset_windows_loaded=sum(d["counts"].get("windows", 0) for d in defaults.values()),
            split_counts=audit["counts"], known_cells_by_type=audit["known_cells_by_type"],
            type_coverage=audit["type_coverage"], positive_types_absent_from_train=absent,
            source_audit_sha256=original_audit_sha, recomputed_audit_sha256=_digest(audit_path))
        result["status"] = "pass"
    except (FileNotFoundError, PermissionError, ModuleNotFoundError) as error:
        result["status"] = "pending"
        result["checks"].append({"code": "TRAINING_PREREQUISITE_MISSING", "status": "pending", "message_zh": str(error)})
        result["issues"].append({"code": "TRAINING_PREREQUISITE_MISSING", "severity": "pending", "layer": "training",
            "start_frame": None, "end_frame": None, "entity_id": None, "message_zh": str(error),
            "next_action_zh": "恢复冻结外部来源、证据文件或审计运行环境，再重新执行；本次不能记为通过。"})
    except Exception as error:
        result["status"] = "fail"
        result["checks"].append({"code": "TRAINING_INTEGRITY_FAILED", "status": "fail", "message_zh": str(error)})
        result["issues"].append({"code": "TRAINING_INTEGRITY_FAILED", "severity": "error", "layer": "training",
            "start_frame": None, "end_frame": None, "entity_id": None, "message_zh": str(error),
            "next_action_zh": "核对失败文件及冻结 SHA；修订数据须新建版本并重新审计，不改旧放行标记。"})
        error_path = output / "training_check_error.txt"
        error_path.write_text(traceback.format_exc())
        result["artifact_paths"].append(str(error_path))
    _save(report_path, result)
    return result
