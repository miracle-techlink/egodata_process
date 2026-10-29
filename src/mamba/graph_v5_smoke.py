"""Validate real V5 data and real CUDA Mamba without an optimizer step."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from src.mamba.dataset_v5 import EgoDemoGraphV5Dataset, collate_graph_v5, sha
from src.mamba.graph_v5 import GraphV5Mamba, decode_graph, masked_graph_bce


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def scan(root):
    reports, selected, selected_sequences, transition_sample = {}, [], set(), None
    representative = {}  # per sequence: first window, upgraded to one with pos+neg
    event_types = None
    for split in ("train", "val", "test"):
        dataset = EgoDemoGraphV5Dataset(root, split, diagnostic=True)
        event_types = dataset.event_types
        counts = Counter({k: 0 for k in ("windows", "positive_cells", "negative_cells",
            "windows_with_positive", "windows_with_negative", "transition_positive_cells",
            "object_to_object_valid_cells", "unknown_cells")})
        layers, types, sequences = Counter(), Counter(), set()
        for i in range(len(dataset)):
            sample = dataset[i]
            values, mask = sample["graph_values"], sample["graph_label_valid"]
            if not mask.any():
                raise AssertionError("Published window has zero graph supervision")
            positive, negative = int(((values == 1) & mask).sum()), int(((values == 0) & mask).sum())
            counts.update(windows=1, positive_cells=positive, negative_cells=negative,
                windows_with_positive=int(positive > 0), windows_with_negative=int(negative > 0),
                unknown_cells=int((~mask).sum()), object_to_object_valid_cells=int(mask[3:, 3:].sum()))
            has_transition = False
            for _, _, k in mask.nonzero().tolist():
                event = event_types[k]
                layers[event["layer"]] += 1
                types[event["layer"] + ":" + event["code"]] += 1
                if event["layer"] == "transition":
                    counts["transition_positive_cells"] += 1
                    has_transition = True
            uid = sample["metadata"]["sequence_id"]
            sequences.add(uid)
            if split == "train":
                # Representative window per sequence: keep the first, but upgrade
                # to the first window that carries both positive and negative
                # supervision so the backward check exercises both cell kinds.
                if uid not in representative:
                    representative[uid] = sample
                current = representative[uid]
                if positive > 0 and negative > 0 and not (
                        ((current["graph_values"] == 1) & current["graph_label_valid"]).any()
                        and ((current["graph_values"] == 0) & current["graph_label_valid"]).any()):
                    representative[uid] = sample
                if has_transition and transition_sample is None:
                    transition_sample = sample
        reports[split] = {**dict(counts), "sequences": sorted(sequences),
            "valid_cells_by_layer": dict(layers), "valid_cells_by_type": dict(types)}
    selected = list(representative.values())
    if transition_sample is not None and not any(
            s["metadata"]["sequence_id"] == transition_sample["metadata"]["sequence_id"]
            and s["metadata"]["target_frame"] == transition_sample["metadata"]["target_frame"] for s in selected):
        selected.append(transition_sample)
    if not selected or event_types is None:
        raise ValueError("No real train window available for a backward check")
    return reports, selected, event_types


def run(root):
    root = Path(root)
    index_sha = sha(root / "training_index.json")
    coverage, selected, event_types = scan(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA Mamba required; no substitute implementation")
    torch.manual_seed(20261024)
    cpu = collate_graph_v5(selected)
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in cpu.items()}
    model = GraphV5Mamba(cpu["entity_features"].shape[-1], len(event_types)).cuda().eval()
    mamba_modules = sorted({type(m).__module__ + "." + type(m).__name__ for m in model.modules()
        if type(m).__module__.startswith("mamba_ssm")})
    if not mamba_modules:
        raise AssertionError("No real mamba_ssm module")
    original = {k: p.detach().clone() for k, p in model.named_parameters()}
    checks = {}
    fp32 = None
    for name, autocast in (("fp32", False), ("bf16", True)):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            output = model(batch)
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise AssertionError("Nonfinite real graph model output")
        logits = output["graph_logits"]
        logits.retain_grad()
        loss_info = masked_graph_bce(logits, batch["graph_values"], batch["graph_label_valid"],
            output["graph_prediction_mask"])
        if loss_info["valid_count"] <= 0 or not torch.isfinite(loss_info["loss"]):
            raise AssertionError("No real finite masked loss")
        loss_info["loss"].backward()
        gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        if not gradients or not all(torch.isfinite(g).all() for g in gradients.values()):
            raise AssertionError("Nonfinite/missing gradients")
        backbone_norm = sum(float(g.float().square().sum()) for n, g in gradients.items() if n.startswith("backbone.")) ** .5
        if backbone_norm <= 0:
            raise AssertionError("Real Mamba backbone received no gradient")
        if torch.count_nonzero(logits.grad[~batch["graph_label_valid"]]):
            raise AssertionError("Unknown graph cell acquired direct supervision")
        if not all(torch.equal(p.detach(), original[n]) for n, p in model.named_parameters()):
            raise AssertionError("Weights changed during zero-optimizer check")
        checks[name] = {k: v for k, v in loss_info.items() if k != "loss"}
        checks[name].update(loss=float(loss_info["loss"].detach()),
            finite_output=True, finite_gradients=True, backbone_gradient_l2=backbone_norm,
            parameters_with_gradient=len(gradients), unknown_logit_gradient_count=0,
            weights_unchanged=True, output_dtype=str(logits.dtype))
        if name == "fp32":
            fp32 = {k: v.detach() for k, v in output.items()}
    model.zero_grad(set_to_none=True)
    # Target values, metadata and arbitrary future arrays must not enter history.
    poisoned = dict(batch)
    poisoned["graph_values"] = torch.full_like(batch["graph_values"], float("nan"))
    poisoned["graph_label_valid"] = ~batch["graph_label_valid"]
    poisoned["metadata"] = [{"future_entity_id": "never_observed"}] * len(selected)
    poisoned["future_entity_features"] = torch.full_like(batch["entity_features"], 1e10)
    with torch.inference_mode():
        # Compare the same execution mode: inference attention may use a
        # different floating-point kernel from the gradient-enabled pass.
        inference_reference = model(batch)
        invariant = model(poisoned)
    torch.testing.assert_close(invariant["graph_logits"], inference_reference["graph_logits"], rtol=0, atol=0)
    # Permuting object identities permutes both directed graph axes, not scores.
    E = batch["entity_features"].shape[2]
    order = torch.tensor([0, 1, 2] + list(reversed(range(3, E))), device="cuda")
    permuted = dict(batch)
    permuted["entity_features"] = batch["entity_features"][:, :, order]
    permuted["entity_valid"] = batch["entity_valid"][:, :, order]
    permuted["entity_kind"] = batch["entity_kind"][:, order]
    with torch.inference_mode():
        permutation = model(permuted)
    expected = inference_reference["graph_logits"][:, order][:, :, order]
    torch.testing.assert_close(permutation["graph_logits"], expected, rtol=3e-4, atol=3e-5)
    if sha(root / "training_index.json") != index_sha:
        raise ValueError("Data index changed while smoke check ran")
    destination = root / "model_checks"
    destination.mkdir(exist_ok=True)
    example = destination / "real_graph_batch.npz"
    np.savez_compressed(example, **{k: v.numpy() for k, v in cpu.items() if torch.is_tensor(v)})
    write(destination / "real_graph_batch_metadata.json", cpu["metadata"])
    write(destination / "random_graph_decoding_zh.json", {
        "warning_zh": "随机初始化的模型输出，仅演示有向图中文解码；不是标注、已训练模型或准确率。",
        "predictions": decode_graph(fp32["graph_logits"], event_types, cpu["metadata"],
            prediction_mask=fp32["graph_prediction_mask"], top_k=3)})
    write(destination / "real_graph_target_coverage.json", {
        "schema_version": "v5_real_graph_target_coverage", "training_index_sha256": index_sha,
        "splits": coverage, "optimizer_steps": 0,
        "scope_zh": "完整读取冻结子集全部窗口；计数为明确银标单元，跨重叠窗口重复计数，不代表独立采集数。"})
    report = {"schema_version": "v5_real_graph_mamba_smoke", "passed": True,
        "training_index_sha256": index_sha, "sample_count": len(selected),
        "sample_metadata": cpu["metadata"], "real_mamba_classes": mamba_modules,
        "input_shape": list(cpu["entity_features"].shape), "output_shape": list(fp32["graph_logits"].shape),
        "precision_checks": checks, "future_labels_metadata_invariant": True,
        "object_slot_permutation_equivariant": True,
        "permutation_max_absolute_error": float((permutation["graph_logits"] - expected).abs().max()),
        "torch_version": torch.__version__, "mamba_ssm_version": version("mamba-ssm"),
        "cuda_device": torch.cuda.get_device_name(), "weights": "random_initialization",
        "optimizer_steps": 0, "training_executed": False, "accuracy": None,
        "backward_split": "train", "real_data_supervision_checked": True,
        "source_sha256": {str(Path(p)): sha(p) for p in [__file__,
            Path(__file__).with_name("graph_v5.py"), Path(__file__).with_name("dataset_v5.py")]},
        "example_npz": str(example), "example_sha256": sha(example),
        "scope_zh": "只验收真实银标数据可加载、真实Mamba前反向、非零梯度及未知掩码；没有参数更新，不表示300段全标或整物体6D已通过。"}
    write(destination / "real_graph_mamba_smoke.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    report = run(args.root)
    print(json.dumps({k: report[k] for k in ("passed", "sample_count", "input_shape", "output_shape", "precision_checks", "optimizer_steps")}))


if __name__ == "__main__":
    main()
