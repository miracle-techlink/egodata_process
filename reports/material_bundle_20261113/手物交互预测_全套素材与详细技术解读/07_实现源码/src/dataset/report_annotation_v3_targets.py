"""Count reviewed targets that actually survive the frozen window adapter."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from src.mamba.annotation_v3 import EgoDemoAnnotationV3Dataset, sha256, temporal_support


def report_targets(base_root, annotation_root, review_windows=False):
    annotation_root = Path(annotation_root)
    reports = {}
    for split in ("train", "val", "test"):
        kwargs = ({"windows_path": str((annotation_root / "review_windows" / f"{split}.jsonl").resolve())}
                  if review_windows else {})
        dataset = EgoDemoAnnotationV3Dataset(base_root, annotation_root, split=split, diagnostic=True, **kwargs)
        by_uid = {}
        for uid, entry in dataset.annotation_records.items():
            if entry["split"] == split:
                path = annotation_root / entry["annotation_path"]
                if sha256(path) != entry["sha256"]:
                    raise ValueError("Annotation SHA mismatch")
                by_uid[uid] = json.loads(path.read_text())["silver_event_records"]
        counts, reasons, label_types = Counter(), Counter(), Counter()
        graph_sequences, head_sequences = set(), set()
        examples = {"single_object_head": [], "graph_only": []}
        for i, window in enumerate(dataset.windows):
            uid = window["sequence_id"]
            if not any(temporal_support(r, window["anchor_frame"], window["target_frame"])
                       for r in by_uid[uid]):
                continue
            sample = dataset[i]
            if sample["event_label_valid"].any() or sample["event_weak_mask"].any():
                raise AssertionError("V3 enabled human or stale weak labels")
            positive = sample["silver_event_label_valid"]
            if not sample["graph_targets"]:
                raise AssertionError("Supported evidence disappeared from adapter")
            counts["windows_with_graph_targets"] += 1
            counts["graph_target_records"] += len(sample["graph_targets"])
            counts["windows_with_single_object_head_labels"] += int(positive.any())
            counts["single_object_head_positive_cells"] += int(positive.sum())
            counts["windows_with_only_graph_targets"] += int(not positive.any())
            counts["geometry_excluded_slots_in_loaded_windows"] += len(sample["metadata"]["geometry_exclusions_applied"])
            graph_sequences.add(uid)
            if positive.any():
                head_sequences.add(uid)
            category = "single_object_head" if positive.any() else "graph_only"
            if len(examples[category]) < 4 and uid not in {r["sequence_id"] for r in examples[category]}:
                examples[category].append({"window_index": i, "sequence_id": uid,
                                           "head_positive_cells": int(positive.sum()),
                                           "graph_records": len(sample["graph_targets"])})
            reasons.update(r["reason"] for r in sample["metadata"]["unmapped_silver_targets"])
            for _, _, kind in positive.nonzero().tolist():
                entry = dataset.event_types[kind]
                label_types[f"{entry['layer']}:{entry['code']}"] += 1
        reports[split] = {
            "total_frozen_windows": len(dataset), "windows_path": str(dataset.windows_path),
            "windows_sha256": sha256(dataset.windows_path), **dict(counts),
            "sequences_with_graph_targets": len(graph_sequences),
            "sequences_with_single_object_head_labels": len(head_sequences),
            "example_windows": examples,
            "unmapped_record_counts_by_reason": dict(sorted(reasons.items())),
            "single_object_head_positive_cells_by_type": dict(sorted(label_types.items()))}
    report = {
        "schema_version": "assistant_annotation_v3_target_coverage",
        "overlay_index_sha256": sha256(annotation_root / "mamba_overlay_index.json"),
        "window_mode": "review_aligned_positive_biased" if review_windows else "original_frozen_v2",
        "splits": reports, "training_executed": False, "training_allowed": False,
        "counting_scope_zh": "枚举全部冻结窗口；仅对时间目标命中明确证据的窗口实际调用v3加载器。图边未对齐旧3D槽位或双客体边不压入单客体头；没有标签的位置仍为未知。",
        "selection_warning_zh": "验证和测试画面已参与助手标注，不再是对本标注流程的盲测；原划分保留供后续模型实验，不能据此声称跨被试泛化。"}
    name = "review_window_target_coverage.json" if review_windows else "annotation_target_coverage.json"
    (annotation_root / name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--review-windows", action="store_true")
    args = parser.parse_args()
    print(json.dumps(report_targets(args.base_root, args.annotation_root, args.review_windows), ensure_ascii=False))


if __name__ == "__main__":
    main()
