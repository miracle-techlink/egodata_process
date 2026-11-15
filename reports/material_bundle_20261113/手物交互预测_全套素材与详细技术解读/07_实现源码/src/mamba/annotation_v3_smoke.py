"""Run real Mamba on reviewed v3 overlay windows, without optimizing weights."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import torch

from src.mamba.annotation_v3 import EgoDemoAnnotationV3Dataset, collate_annotation_v3, sha256
from src.mamba.egodemo_v2 import EgoDemoV2Mamba, decode_events
from src.mamba.egodemo_v2_smoke import export_example, write_json


def run(base_root, annotation_root, diagnostic, review_windows=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA Mamba is required for this forward verification")
    kwargs = ({"windows_path": str((Path(annotation_root) / "review_windows" / "train.jsonl").resolve())}
              if review_windows else {})
    dataset = EgoDemoAnnotationV3Dataset(base_root, annotation_root, diagnostic=diagnostic, **kwargs)
    frames = {}
    for uid, entry in dataset.annotation_records.items():
        if entry["split"] != "train":
            continue
        annotation = json.loads((dataset.annotation_root / entry["annotation_path"]).read_text())
        frames[uid] = {f for r in annotation["silver_event_records"] for f in r["evidence_frames"]}
    indices, sequences = [], set()
    coverage_path = Path(annotation_root) / ("review_window_target_coverage.json" if review_windows else "annotation_target_coverage.json")
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        train = coverage["splits"]["train"]
        if (coverage["overlay_index_sha256"] == sha256(Path(annotation_root) / "mamba_overlay_index.json")
                and train.get("windows_sha256") == sha256(dataset.windows_path)):
            examples = train.get("example_windows", {})
            preferred = examples.get("single_object_head", [])[:2] + examples.get("graph_only", [])
            for row in preferred:
                if row["sequence_id"] not in sequences:
                    indices.append(row["window_index"])
                    sequences.add(row["sequence_id"])
                    if len(indices) == 4:
                        break
    for i, row in enumerate(dataset.windows):
        if len(indices) >= 4:
            break
        uid = row["sequence_id"]
        if uid not in sequences and row["target_frame"] in frames.get(uid, set()):
            indices.append(i)
            sequences.add(uid)
            if len(indices) == 4:
                break
    if not indices:
        raise ValueError("No native v2 train window lands on an explicitly reviewed target frame")
    samples = [dataset[i] for i in indices]
    cpu = collate_annotation_v3(samples)
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cpu.items()}
    torch.manual_seed(42)
    model = EgoDemoV2Mamba(len(dataset.event_types)).cuda().eval()
    real_modules = sorted({type(m).__module__ + "." + type(m).__name__ for m in model.modules()
                           if type(m).__module__.startswith("mamba_ssm")})
    if not real_modules:
        raise AssertionError("Real mamba_ssm module missing")
    with torch.inference_mode():
        outputs = model(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            amp = model(batch)
    finite = {mode: {key: bool(torch.isfinite(value).all()) for key, value in values.items()}
              for mode, values in [("fp32", outputs), ("bf16", amp)]}
    if not all(v for group in finite.values() for v in group.values()):
        raise AssertionError("Nonfinite model output")
    if cpu["event_label_valid"].any():
        raise AssertionError("Assistant labels were promoted into human supervision")
    if cpu["event_weak_mask"].any():
        raise AssertionError("Stale v2 weak supervision remained active in v3")
    if not any(cpu["graph_targets"]) and not cpu["silver_event_label_valid"].any():
        raise AssertionError("Selected windows did not retain reviewed annotation evidence")
    example = export_example(samples[0], Path(annotation_root) / "examples", dataset.event_types)
    summary_path = Path(annotation_root) / "examples" / "sample_window.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "assistant_annotation_v3_example"
    summary["字段说明"].update({
        "event_weak_mask": "v3禁用旧v2弱监督，全部false；旧候选值仅溯源",
        "silver_event_candidates": "实际证据点上的助手银标正例",
        "silver_event_label_valid": "仅已确认正例位置true，未观察位置unknown",
        "graph_targets": "双客体和无3D实体图边另存sample_graph_targets.json"})
    write_json(summary_path, summary)
    write_json(Path(annotation_root) / "examples" / "sample_graph_targets.json", samples[0]["graph_targets"])
    report = {"schema_version": "assistant_annotation_v3_mamba_smoke", "passed": True,
              "window_indices": indices, "sample_metadata": cpu["metadata"],
              "dataset_index_sha256": sha256(Path(base_root) / "dataset_index.json"),
              "overlay_index_sha256": sha256(Path(annotation_root) / "mamba_overlay_index.json"),
              "windows_path": str(dataset.windows_path), "windows_sha256": sha256(dataset.windows_path),
              "window_mode": "review_aligned_positive_biased" if review_windows else "original_frozen_v2",
              "torch_version": torch.__version__, "mamba_ssm_version": version("mamba-ssm"),
              "cuda_device": torch.cuda.get_device_name(), "real_mamba_classes": real_modules,
              "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
              "finite": finite, "silver_single_target_labels": int(cpu["silver_event_label_valid"].sum()),
              "supported_graph_targets": [len(r) for r in cpu["graph_targets"]],
              "human_valid_labels": int(cpu["event_label_valid"].sum()),
              "active_v2_weak_labels": int(cpu["event_weak_mask"].sum()),
              "geometry_exclusions_applied": [r["geometry_exclusions_applied"] for r in cpu["metadata"]],
              "training_executed": False, "optimizer_steps": 0, "weights": "random_initialization",
              "example_npz": example, "accuracy": None,
              "sampling_zh": "只在train选择已看证据帧，优先选择两类实际窗口（有单客体标签与仅图边）验证接口；不是代表性质量评测。"}
    write_json(Path(annotation_root) / "examples" / "random_model_decoded_example.json", {
        "warning_zh": "随机初始化模型仅演示中文解码；输出不是标签或预测准确率。",
        "predictions": decode_events(outputs["event_logits"], dataset.event_types,
                                     cpu["metadata"], top_k=2)})
    write_json(Path(annotation_root) / "annotation_mamba_smoke.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--review-windows", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.base_root, args.annotation_root, args.diagnostic, args.review_windows), ensure_ascii=False))


if __name__ == "__main__":
    main()
