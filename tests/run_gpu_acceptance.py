"""L40 acceptance: multi-class synthetic memorization, causality, and reload checks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.mamba.train import DEFAULT_CONFIG, build_model, compute_losses, mask_object_logits


def run_case(backbone, precision, output_dir, steps):
    torch.manual_seed(20261022)
    torch.cuda.manual_seed_all(20261022)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"].update(backbone=backbone, d_model=32, n_layer=1, dropout=0.0)
    cfg["model"]["encoder"].update(n_max=2, n_heads=4, n_layers=1, dim_ff=64, max_frames=16, coordinate_mode="mixed")
    cfg["model"]["decoder"].update(n_max=2)
    model = build_model(cfg).cuda()
    batch_size, window = 14, 8
    batch = {
        "hand_seq": torch.rand(batch_size, window, 2, 21, 3, device="cuda"),
        "obj_seq": torch.rand(batch_size, window, 2, 13, device="cuda"),
        "obj_mask": torch.ones(batch_size, 2, device="cuda", dtype=torch.bool),
        "subject": torch.arange(batch_size, device="cuda") % 3,
        "object": torch.arange(batch_size, device="cuda") % 3,
        "interaction_type": torch.arange(batch_size, device="cuda") % 7,
        "label_valid": torch.ones(batch_size, device="cuda", dtype=torch.bool),
    }
    batch["obj_mask"][2::3] = False
    batch["obj_seq"][2::3] = float("nan")
    batch["hand_seq"][0, :, 1] = float("nan")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0)
    losses, gradient_norms = [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            logits = model(batch)
            loss, _ = compute_losses(logits, batch, cfg["train"]["loss_weights"])
        if not torch.isfinite(loss):
            raise AssertionError(f"{backbone}/{precision}: nonfinite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss))
        gradient_norms.append(float(gradient_norm))
    model.eval()
    with torch.no_grad():
        expected = model(batch)
        final_loss, _ = compute_losses(expected, batch, cfg["train"]["loss_weights"])
        temporal_input = torch.randn(3, 8, 32, device="cuda")
        changed_input = temporal_input.clone()
        changed_input[:, 4:] += 5
        causal_delta = float((model.backbone(temporal_input)[:, :4] - model.backbone(changed_input)[:, :4]).abs().max())
    checkpoint_path = output_dir / f"{backbone}_{precision}.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    restored = build_model(checkpoint["config"]).cuda().eval()
    restored.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        observed = restored(batch)
    reload_differences = []
    for head in expected:
        torch.testing.assert_close(expected[head], observed[head], rtol=0, atol=1e-5, equal_nan=False)
        matching_masked = torch.isneginf(expected[head]) & torch.isneginf(observed[head])
        difference = (expected[head] - observed[head]).masked_fill(matching_masked, 0.0)
        if not torch.isfinite(difference).all():
            raise AssertionError(f"{head}: nonfinite reload difference")
        reload_differences.append(float(difference.abs().max()))
    reload_delta = max(reload_differences)
    predictions = dict(expected)
    predictions["object"] = mask_object_logits(predictions["object"], batch["obj_mask"])
    correct = {head: predictions[head].argmax(-1).eq(batch[head]) for head in predictions}
    torch.cuda.synchronize()
    result = {
        "backbone": backbone, "precision": precision, "steps": steps,
        "initial_loss": losses[0], "final_loss": float(final_loss), "loss_history": losses,
        "maximum_gradient_norm": max(gradient_norms),
        "training_accuracy": {head: float(values.float().mean()) for head, values in correct.items()},
        "training_joint_accuracy": float(torch.stack(list(correct.values()), dim=1).all(1).float().mean()),
        "reload_max_absolute_difference": reload_delta,
        "backbone_causality_max_absolute_difference": causal_delta,
        "seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "checkpoint": str(checkpoint_path),
    }
    result["passed"] = bool(float(final_loss) < losses[0] * 0.5 and reload_delta <= 1e-5 and causal_delta <= 1e-5)
    del model, restored, optimizer, checkpoint
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1 or not torch.cuda.is_available():
        parser.error("Positive steps and an available CUDA GPU are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "data": "synthetic multiclass stress batch, not HOI effectiveness or real-data training",
        "acceptance": "finite loss/gradients; final loss < 50% of initial; reload and causal prefix deltas <= 1e-5",
        "interaction_class_counts": [2] * 7,
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "cases": [],
    }
    for backbone in ("lstm", "mamba"):
        for precision in ("fp32", "bf16"):
            result = run_case(backbone, precision, args.output_dir, args.steps)
            report["cases"].append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "loss_history"}), flush=True)
            (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if all(case["passed"] for case in report["cases"]) else 1)


if __name__ == "__main__":
    main()
