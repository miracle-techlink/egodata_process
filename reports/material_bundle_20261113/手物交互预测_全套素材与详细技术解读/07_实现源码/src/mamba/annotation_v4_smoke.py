"""Measure actual V4 target availability and run untrained real Mamba forward."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import torch

from src.mamba.annotation_v4 import EgoDemoAnnotationV4Dataset, collate_annotation_v4
from src.mamba.annotation_v3 import sha256
from src.mamba.egodemo_v2 import EgoDemoV2Mamba, decode_events
from src.mamba.egodemo_v2_smoke import export_example, write_json


def run(base_root, v3_root, dense_root):
    root = Path(dense_root)
    dense_index = json.loads((root / "mamba_dense_index.json").read_text())
    reports, selected, selected_uids = {}, [], set()
    for split, info in dense_index["splits"].items():
        dataset = EgoDemoAnnotationV4Dataset(base_root, v3_root, root, split=split, diagnostic=True,
            windows_path=str((root / info["window_path"]).resolve()))
        counts = Counter({key: 0 for key in ("windows", "windows_with_graph_targets",
            "windows_with_positive_graph_states", "windows_with_negative_graph_states",
            "positive_graph_state_records", "negative_graph_state_records", "transition_graph_records",
            "windows_with_dense_single_object_head", "dense_single_object_positive_cells", "dense_single_object_negative_cells")})
        reasons = Counter()
        for i in range(len(dataset)):
            sample = dataset[i]
            values, valid, graph = sample["dense_event_values"], sample["dense_event_label_valid"], sample["graph_targets"]
            if sample["event_label_valid"].any() or sample["event_weak_mask"].any():
                raise AssertionError("Dense assistant data activated human or stale weak labels")
            if not graph:
                raise AssertionError("Selected dense target lost all actual reviewed states")
            positives = sum(r["layer"] != "transition" and r.get("value") == 1 for r in graph)
            negatives = sum(r["layer"] != "transition" and r.get("value") == 0 for r in graph)
            counts.update(windows=1, windows_with_graph_targets=int(bool(graph)),
                windows_with_positive_graph_states=int(positives > 0), windows_with_negative_graph_states=int(negatives > 0),
                positive_graph_state_records=positives, negative_graph_state_records=negatives,
                transition_graph_records=sum(r["layer"] == "transition" for r in graph),
                windows_with_dense_single_object_head=int(valid.any()),
                dense_single_object_positive_cells=int(((values == 1) & valid).sum()),
                dense_single_object_negative_cells=int(((values == 0) & valid).sum()))
            reasons.update(r["reason"] for r in sample["metadata"]["unmapped_dense_targets"])
            uid = sample["metadata"]["sequence_id"]
            if uid not in selected_uids and len(selected) < 4:
                selected.append(sample)
                selected_uids.add(uid)
        reports[split] = {**dict(counts), "unmapped_reasons": dict(reasons),
                          "windows_path": str(dataset.windows_path), "windows_sha256": sha256(dataset.windows_path)}
    coverage = {"schema_version": "assistant_annotation_v4_target_coverage", "splits": reports,
        "mamba_dense_index_sha256": sha256(root / "mamba_dense_index.json"),
        "training_executed": False, "training_allowed": False,
        "scope_zh": "实际加载全部逐帧复核窗口；图正负标签与旧单客体头可表达标签分开统计。未对齐整物体ID不使用部件ID替换。"}
    write_json(root / "dense_target_coverage.json", coverage)
    if not selected:
        raise ValueError("No actual dense window available")
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA Mamba is required; no substitute backbone")
    torch.manual_seed(42)
    cpu = collate_annotation_v4(selected)
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in cpu.items()}
    model = EgoDemoV2Mamba(len(dataset.event_types)).cuda().eval()
    modules = sorted({type(m).__module__ + "." + type(m).__name__ for m in model.modules()
                      if type(m).__module__.startswith("mamba_ssm")})
    if not modules:
        raise AssertionError("Actual Mamba modules missing")
    with torch.inference_mode():
        fp32 = model(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = model(batch)
    finite = {name: {k: bool(torch.isfinite(v).all()) for k, v in out.items()}
              for name, out in [("fp32", fp32), ("bf16", bf16)]}
    if not all(v for group in finite.values() for v in group.values()):
        raise AssertionError("Nonfinite real Mamba output")
    example = export_example(selected[0], root / "examples", dataset.event_types)
    write_json(root / "examples/sample_dense_graph_targets.json", selected[0]["graph_targets"])
    write_json(root / "examples/random_model_decoded_example.json", {
        "warning_zh": "随机初始化模型输出仅说明中文解码接口，不是标注或模型准确率。现有解码器不预测双客体图。",
        "predictions": decode_events(fp32["event_logits"], dataset.event_types, cpu["metadata"], top_k=2)})
    report = {"schema_version": "assistant_annotation_v4_mamba_smoke", "passed": True,
        "mamba_dense_index_sha256": sha256(root / "mamba_dense_index.json"),
        "sample_metadata": cpu["metadata"], "graph_target_counts": [len(x) for x in cpu["graph_targets"]],
        "dense_single_object_head_valid_cells": int(cpu["dense_event_label_valid"].sum()),
        "human_valid_cells": int(cpu["event_label_valid"].sum()), "stale_weak_valid_cells": int(cpu["event_weak_mask"].sum()),
        "real_mamba_classes": modules, "torch_version": torch.__version__, "mamba_ssm_version": version("mamba-ssm"),
        "cuda_device": torch.cuda.get_device_name(), "output_shapes": {k: list(v.shape) for k, v in fp32.items()},
        "finite": finite, "weights": "random_initialization", "optimizer_steps": 0,
        "training_executed": False, "training_allowed": False, "accuracy": None, "example_npz": example,
        "graph_head_implemented": False}
    write_json(root / "dense_mamba_smoke.json", report)
    return {"coverage": coverage, "smoke": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--v3-root", required=True)
    parser.add_argument("--dense-root", required=True)
    args = parser.parse_args()
    report = run(args.base_root, args.v3_root, args.dense_root)
    print(json.dumps({"splits": report["coverage"]["splits"], "passed": report["smoke"]["passed"]}))


if __name__ == "__main__":
    main()
