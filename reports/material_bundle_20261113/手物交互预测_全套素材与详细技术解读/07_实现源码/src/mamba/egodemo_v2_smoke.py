"""Load real v2 windows and run real mamba_ssm CUDA forward, without training.

python -m src.mamba.egodemo_v2_smoke --config configs/egodemo_mamba_v2.yaml --diagnostic
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.mamba.egodemo_v2 import (
    EgoDemoV2Dataset, EgoDemoV2Mamba, collate_egodemo_v2, decode_events,
)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_example(sample, root, event_types):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.cpu().numpy() for key, value in sample.items() if torch.is_tensor(value)}
    arrays["metadata_json"] = np.asarray(json.dumps(sample["metadata"], ensure_ascii=False))
    arrays["event_types_json"] = np.asarray(json.dumps(event_types, ensure_ascii=False))
    np.savez_compressed(root / "sample_window.npz", **arrays)
    summary = {"source": "real_exported_sequence_window", "metadata": sample["metadata"],
               "training_executed": False, "event_types": event_types,
               "说明": "真实采集窗口；几何为最后观测左相机米制坐标。缺测保持NaN，人工有效事件数为0，不可用于正式联合训练。",
               "字段说明": {"hand_seq": "观测双手21个三维关节，左手在前",
                            "obj_pose_seq": "观测物体/刚性部件6D齐次位姿",
                            "surface_seq": "非刚体及其他物体的同键表面点轨迹",
                            "object_mask": "真实已观测实体槽位；false为填充",
                            "point_mask": "观测窗选出的点键；false为填充",
                            "target_hand": "约1秒后双手关节",
                            "target_obj_pose": "约1秒后同槽位物体/部件位姿",
                            "target_surface": "约1秒后同tracklet/point键表面点",
                            "event_candidates": "多层多标签候选；0且有效mask=false表示未知",
                            "event_weak_mask": "弱候选位置；不是人工金标准",
                            "event_label_valid": "人工监督有效性，当前全false"},
               "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                  **({"true_count": int(value.sum())} if value.dtype == bool else {})}
                          for key, value in arrays.items()}}
    write_json(root / "sample_window.json", summary)
    return str(root / "sample_window.npz")


def run(config, *, diagnostic=False, root=None, output=None):
    torch.manual_seed(int(config.get("seed", 42)))
    data = config["data"]
    dataset = EgoDemoV2Dataset(root or data["root"], data.get("split", "train"),
                              diagnostic=diagnostic, n_max=data.get("n_max", 8),
                              points_per_object=data.get("points_per_object", 16),
                              sequence_cache_size=data.get("sequence_cache_size", 1))
    if not len(dataset):
        raise ValueError("No exported windows in selected split")
    smoke = config["smoke"]
    if smoke.get("device", "cuda") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This verification requires real mamba_ssm on CUDA; no substitute backbone")
    batch_size = min(int(smoke.get("batch_size", 4)), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, batch_size, dtype=int).tolist()
    samples = [dataset[i] for i in indices]
    cpu_batch = collate_egodemo_v2(samples)
    batch = {key: value.cuda() if torch.is_tensor(value) else value
             for key, value in cpu_batch.items()}
    model = EgoDemoV2Mamba(len(dataset.event_types), **config["model"]).cuda().eval()
    mamba_modules = [module for module in model.modules()
                     if type(module).__module__.startswith("mamba_ssm")]
    if not mamba_modules:
        raise AssertionError("No real mamba_ssm modules present")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        fp32 = model(batch)
        if smoke.get("amp_bf16", True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bf16 = model(batch)
        else:
            bf16 = None
    torch.cuda.synchronize()
    finite = {"fp32": {key: bool(torch.isfinite(value).all()) for key, value in fp32.items()}}
    if bf16 is not None:
        finite["bf16"] = {key: bool(torch.isfinite(value).all()) for key, value in bf16.items()}
    if not all(value for mode in finite.values() for value in mode.values()):
        raise AssertionError(f"Nonfinite forward outputs: {finite}")
    target_path = Path(output or smoke["output_path"])
    # Pick a real observed-geometry example, so the inspectable artifact also
    # demonstrates object slots. This does not affect Dataset slot selection.
    example_position = max(range(len(samples)), key=lambda i: (
        int(samples[i]["obj_valid"].sum()), int(samples[i]["surface_valid"].sum())))
    example_path = export_example(samples[example_position], target_path.parent / "examples", dataset.event_types)
    coverage = []
    for i, sample in zip(indices, samples):
        coverage.append({"window_index": i, "sequence_id": sample["metadata"]["sequence_id"],
                         **{key: int(sample[key].sum()) for key in (
                             "hand_valid", "obj_valid", "surface_valid", "target_hand_valid",
                             "target_obj_valid", "target_surface_valid", "event_weak_mask", "event_label_valid")},
                         "any_object_input": bool(sample["obj_valid"].any() or sample["surface_valid"].any()),
                         "any_object_target": bool(sample["target_obj_valid"].any() or sample["target_surface_valid"].any())})
    report = {
        "schema_version": "egodemo_mamba_v2", "passed": True,
        "verification": "real_mamba_ssm_forward_only", "dataset_root": str(dataset.root),
        "dataset_index_sha256": hashlib.sha256((dataset.root / "dataset_index.json").read_bytes()).hexdigest(),
        "window_indices": indices, "sample_metadata": cpu_batch["metadata"],
        "torch_version": torch.__version__, "mamba_ssm_version": version("mamba-ssm"),
        "mamba_module_classes": sorted({f"{type(m).__module__}.{type(m).__name__}" for m in mamba_modules}),
        "cuda_device": torch.cuda.get_device_name(), "cuda_capability": list(torch.cuda.get_device_capability()),
        "input_shapes": {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)},
        "output_shapes": {key: list(value.shape) for key, value in fp32.items()},
        "finite": finite, "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "geometry_mask_counts": {key: int(cpu_batch[key].sum()) for key in
                                 ("hand_valid", "obj_valid", "surface_valid", "target_hand_valid",
                                  "target_obj_valid", "target_surface_valid")},
        "weak_event_positive_count": int(cpu_batch["event_weak_mask"].sum()),
        "selected_slot_coverage_sample_only": coverage,
        "human_valid_event_count": int(cpu_batch["event_label_valid"].sum()),
        "training_allowed": dataset.index.get("training_allowed") is True,
        "allowed_as_model_input": dataset.index.get("allowed_as_model_input") is True,
        "training_executed": False, "optimizer_steps": 0, "weights": "random_initialization",
        "accuracy": None, "example_path": example_path,
        "example_window_index": indices[example_position],
        "decode_example_untrained_do_not_use_as_annotations": decode_events(
            fp32["event_logits"], dataset.event_types, cpu_batch["metadata"], top_k=2),
    }
    write_json(target_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="真实 Mamba CUDA 前向验收；不执行训练")
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnostic", action="store_true", help="显式允许加载待审/非因果数据，仅用于结构诊断")
    parser.add_argument("--root", help="替换数据目录，例如独立 pilot")
    parser.add_argument("--output", help="前向验收 JSON 输出路径")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = run(config, diagnostic=args.diagnostic, root=args.root, output=args.output)
    print(json.dumps({key: report[key] for key in (
        "passed", "verification", "cuda_device", "input_shapes", "output_shapes", "finite",
        "human_valid_event_count", "training_executed", "example_path")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
