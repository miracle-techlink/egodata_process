"""World-wrist auxiliary prediction diagnostic, NOT an HOI benchmark.

Run: python -m src.mamba.trajectory_benchmark --staging-root STAGING --output-dir NEW
CPU smoke: add --models lstm --seeds 17 --epochs 1 --device cpu.
Errors retain the source coordinate units; their physical units are unverified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from uuid import UUID

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mamba.train import LSTMBackbone, MambaBackbone


WINDOW = 16
STRIDE = 5
HORIZON_SECONDS = 1.0
GAP_FACTOR = 1.5
CONTRACT_VERSION = "2026-10-22/round2-auxiliary-wrist"
LIMITATION = (
    "Auxiliary world-wrist trajectory diagnostic, NOT HOI accuracy. Ten task-balanced "
    "episodes with UUID-held-out tasks are a small cross-task diagnostic, not evidence "
    "of generalization or superiority of either backbone. Physical coordinate units "
    "and camera rotation conventions have not been independently verified."
)


@dataclass
class Samples:
    history: np.ndarray
    target: np.ndarray
    velocity_prediction: np.ndarray
    uuids: np.ndarray
    provenance: list[dict]
    audits: list[dict]
    split: str


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_splits(staging_root):
    """Read identity metadata, not held-out sequence values, before splitting."""
    episodes = {}
    for meta_path in sorted(Path(staging_root).glob("*/meta.json")):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        source_uuid = metadata.get("source_episode_uuid")
        if not isinstance(source_uuid, str) or str(UUID(source_uuid)) != source_uuid:
            raise ValueError(f"Invalid source_episode_uuid in {meta_path}")
        if source_uuid in episodes or meta_path.parent.name != source_uuid:
            raise ValueError(f"Duplicate or mismatched UUID directory: {meta_path}")
        if metadata.get("coordinate_system") != "world":
            raise ValueError(f"Expected untransformed world coordinates: {meta_path}")
        if metadata.get("world_to_camera_transform_applied", False):
            raise ValueError(f"Transformed coordinates are not this benchmark's input: {meta_path}")
        if metadata.get("hand_valid_order", ["left", "right"]) != ["left", "right"]:
            raise ValueError(f"Unexpected hand order in {meta_path}")
        if not (meta_path.parent / "sequence.npz").is_file():
            raise ValueError(f"Missing sequence.npz beside {meta_path}")
        episodes[source_uuid] = {"directory": meta_path.parent, "metadata": metadata}
    if len(episodes) != 10:
        raise ValueError(f"This fixed pilot protocol requires exactly 10 UUIDs, got {len(episodes)}")
    ordered = sorted(episodes)
    return episodes, {"train": ordered[:6], "val": ordered[6:8], "test": ordered[8:]}


def build_episode_samples(sequence, source_uuid, split):
    timestamps = np.asarray(sequence["timestamps"], dtype=np.float64)
    frames = np.asarray(sequence["frame_index"])
    valid = np.asarray(sequence["hand_valid"])
    hands = [np.asarray(sequence[name], dtype=np.float64)
             for name in ("left_hand_world", "right_hand_world")]
    count = len(timestamps)
    if timestamps.shape != (count,) or count < 2 or not np.isfinite(timestamps).all():
        raise ValueError(f"{source_uuid}: invalid timestamps")
    intervals = np.diff(timestamps)
    if (timestamps < 0).any() or (intervals <= 0).any():
        raise ValueError(f"{source_uuid}: timestamps must increase strictly")
    if (frames.shape != (count,) or frames.dtype.kind not in "iu"
            or (frames < 0).any() or (np.diff(frames) <= 0).any()):
        raise ValueError(f"{source_uuid}: invalid frame_index")
    if valid.shape != (count, 2) or valid.dtype.kind != "b":
        raise ValueError(f"{source_uuid}: hand_valid must be boolean (frames, 2)")
    if any(hand.shape != (count, 21, 3) for hand in hands):
        raise ValueError(f"{source_uuid}: expected hand arrays (frames, 21, 3)")
    wrists = np.stack([hand[:, 0] for hand in hands], axis=1)
    usable = valid.all(axis=1) & np.isfinite(wrists).all(axis=(1, 2))
    native_interval = float(np.median(intervals))
    gap_limit = native_interval * GAP_FACTOR
    tolerance = max(1e-9, native_interval * 1e-6)
    rejected = dict.fromkeys(
        ("no_future", "observation_time_gap", "target_time_gap",
         "invalid_observation", "invalid_target"), 0,
    )
    histories, targets, velocities, provenance = [], [], [], []
    candidates = 0
    for start in range(0, count - WINDOW + 1, STRIDE):
        candidates += 1
        end = start + WINDOW - 1
        requested_time = float(timestamps[end] + HORIZON_SECONDS)
        target_index = int(np.searchsorted(timestamps, requested_time, side="left"))
        if target_index >= count:
            rejected["no_future"] += 1
            continue
        if (intervals[start:end] > gap_limit + tolerance).any():
            rejected["observation_time_gap"] += 1
            continue
        overshoot = float(timestamps[target_index] - requested_time)
        if (overshoot > gap_limit + tolerance
                or intervals[target_index - 1] > gap_limit + tolerance):
            rejected["target_time_gap"] += 1
            continue
        if not usable[start:end + 1].all():
            rejected["invalid_observation"] += 1
            continue
        if not usable[target_index]:
            rejected["invalid_target"] += 1
            continue
        horizon = float(timestamps[target_index] - timestamps[end])
        anchor = wrists[end]
        histories.append((wrists[start:end + 1] - anchor).reshape(WINDOW, 6))
        targets.append((wrists[target_index] - anchor).reshape(6))
        velocity = (wrists[end] - wrists[end - 1]) / intervals[end - 1]
        velocities.append((velocity * horizon).reshape(6))
        provenance.append({
            "sample_id": f"{source_uuid}:{int(frames[start])}:{int(frames[target_index])}",
            "source_episode_uuid": source_uuid, "split": split,
            "start_index": start, "observation_end_index": end, "target_index": target_index,
            "observation_frame_indices": frames[start:end + 1].tolist(),
            "observation_timestamps": timestamps[start:end + 1].tolist(),
            "target_frame": int(frames[target_index]),
            "requested_target_timestamp": requested_time,
            "target_timestamp": float(timestamps[target_index]),
            "actual_horizon_seconds": horizon, "target_overshoot_seconds": overshoot,
        })
    audit = {
        "source_episode_uuid": source_uuid, "split": split, "frame_count": count,
        "native_interval_seconds": native_interval, "max_gap_seconds": gap_limit,
        "candidates": candidates, "accepted": len(provenance), "rejected": rejected,
    }
    return Samples(
        np.asarray(histories, dtype=np.float64).reshape(-1, WINDOW, 6),
        np.asarray(targets, dtype=np.float64).reshape(-1, 6),
        np.asarray(velocities, dtype=np.float64).reshape(-1, 6),
        np.full(len(provenance), source_uuid), provenance, [audit], split,
    )


def load_samples(episodes, uuids, split):
    packs = []
    for source_uuid in uuids:
        episode = episodes[source_uuid]
        directory = episode["directory"]
        with np.load(directory / "sequence.npz", allow_pickle=False) as sequence:
            pack = build_episode_samples(sequence, source_uuid, split)
        if pack.audits[0]["frame_count"] != episode["metadata"].get("frame_count"):
            raise ValueError(f"{source_uuid}: metadata frame_count mismatch")
        if len(pack.target) == 0:
            raise ValueError(f"{source_uuid}: no accepted {split} samples; audit={pack.audits[0]}")
        pack.audits[0].update({
            "sequence_path": str(directory / "sequence.npz"),
            "sequence_sha256": file_sha256(directory / "sequence.npz"),
            "meta_sha256": file_sha256(directory / "meta.json"),
            "task_name": episode["metadata"].get("task_name"),
        })
        packs.append(pack)
    return Samples(
        np.concatenate([pack.history for pack in packs]),
        np.concatenate([pack.target for pack in packs]),
        np.concatenate([pack.velocity_prediction for pack in packs]),
        np.concatenate([pack.uuids for pack in packs]),
        [record for pack in packs for record in pack.provenance],
        [audit for pack in packs for audit in pack.audits], split,
    )


def fit_normalization(samples):
    if samples.split != "train":
        raise ValueError("Normalization may only be fitted on train observations")
    if not samples.history.size or not np.isfinite(samples.history).all():
        raise ValueError("Cannot normalize empty or nonfinite train histories")
    standard_deviation = float(np.std(samples.history, dtype=np.float64))
    if not np.isfinite(standard_deviation):
        raise ValueError("Nonfinite train observation standard deviation")
    return {
        "scale": standard_deviation if standard_deviation >= 1e-8 else 1.0,
        "observed_std": standard_deviation, "fit_split": "train",
        "fit_uuids": sorted(set(samples.uuids.tolist())),
        "fit_sample_count": len(samples.target), "fit_scalar_count": samples.history.size,
        "method": "population scalar std of centered train histories only; divide without centering",
        "targets_used_for_fit": False, "minimum_std": 1e-8,
        "degenerate_scale_fallback": 1.0,
    }


def endpoint_metrics(prediction, samples):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != samples.target.shape or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite and match the target shape")
    errors = np.linalg.norm((prediction - samples.target).reshape(-1, 2, 3), axis=2)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Endpoint errors are empty or nonfinite")

    def averages(values):
        return {"left": float(values[:, 0].mean()), "right": float(values[:, 1].mean()),
                "combined": float(values.mean())}

    per_uuid = {}
    for source_uuid in sorted(set(samples.uuids.tolist())):
        mask = samples.uuids == source_uuid
        per_uuid[source_uuid] = {**averages(errors[mask]), "samples": int(mask.sum())}
    return {
        "sample_weighted": averages(errors),
        "episode_equal": {key: float(np.mean([value[key] for value in per_uuid.values()]))
                          for key in ("left", "right", "combined")},
        "per_uuid": per_uuid, "sample_count": len(errors), "episode_count": len(per_uuid),
    }


def persistence_improvement(metrics, persistence):
    return {
        weighting: {
            hand: (100.0 * (1.0 - metrics[weighting][hand] / persistence[weighting][hand])
                   if persistence[weighting][hand] > 0 else None)
            for hand in ("left", "right", "combined")
        }
        for weighting in ("sample_weighted", "episode_equal")
    }


class TrajectoryModel(nn.Module):
    def __init__(self, backbone, d_model=32):
        super().__init__()
        self.input_projection = nn.Linear(6, d_model)
        self.regression_head = nn.Linear(d_model, 6)
        if backbone == "lstm":
            self.backbone = LSTMBackbone(d_model, n_layer=1, dropout=0.0)
        elif backbone == "mamba":
            self.backbone = MambaBackbone(
                d_model, n_layer=1, d_state=16, d_conv=4, expand=2, dropout=0.0,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, history):
        return self.regression_head(self.backbone(self.input_projection(history))[:, -1])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device):
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def predict(model, samples, scale, device, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(samples.history), batch_size):
            history = torch.as_tensor(
                samples.history[start:start + batch_size] / scale,
                dtype=torch.float32, device=device,
            )
            prediction = model(history)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite inference prediction")
            predictions.append(prediction.cpu().numpy().astype(np.float64) * scale)
    return np.concatenate(predictions)


def reload_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TrajectoryModel(checkpoint["model"], checkpoint["d_model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def train_model(train_samples, val_samples, normalization, run_dir, config, model_name, seed):
    """Only train and validation data enter this function; no held-out loader exists here."""
    if train_samples.split != "train" or val_samples.split != "val":
        raise ValueError("Training requires train and validation splits only")
    device = torch.device(config["device"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    run_dir.mkdir()
    write_json(run_dir / "config.json", {**config, "model": model_name, "seed": seed})
    model = TrajectoryModel(model_name, config["d_model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    scale = normalization["scale"]
    loader = DataLoader(
        TensorDataset(torch.tensor(train_samples.history / scale, dtype=torch.float32),
                      torch.tensor(train_samples.target / scale, dtype=torch.float32)),
        batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    best_score = float("inf")
    best_prediction = None
    best_metrics = None
    best_epoch = None
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as history_file:
        for epoch in range(1, config["epochs"] + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            for history, target in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(history.to(device))
                loss = nn.functional.mse_loss(prediction, target.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss: {model_name}, seed={seed}")
                loss.backward()
                parameters = list(model.parameters())
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in parameters):
                    raise FloatingPointError("Missing or nonfinite model gradient")
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in parameters):
                    raise FloatingPointError("Nonfinite model parameter after optimizer update")
                total_loss += float(loss.detach()) * len(history)
            val_prediction = predict(model, val_samples, scale, device, config["batch_size"])
            val_metrics = endpoint_metrics(val_prediction, val_samples)
            score = val_metrics["episode_equal"]["combined"]
            selected = score < best_score
            if selected:
                best_score, best_epoch = score, epoch
                best_prediction, best_metrics = val_prediction.copy(), val_metrics
                torch.save({
                    "model_state": {name: value.detach().cpu().clone()
                                    for name, value in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                    "model": model_name, "d_model": config["d_model"], "seed": seed,
                    "normalization": normalization, "config": config,
                    "contract_version": CONTRACT_VERSION,
                    "selection_metric": "val.episode_equal.combined", "selection_value": score,
                }, run_dir / "best.pt")
            synchronize(device)
            entry = {
                "epoch": epoch, "train_normalized_mse": total_loss / len(train_samples.target),
                "val": val_metrics, "selected": selected,
                "seconds": time.perf_counter() - epoch_started,
                "loss_grad_parameters_finite": True,
            }
            history_file.write(json.dumps(entry, allow_nan=False) + "\n")
            history_file.flush()
            print(f"{model_name} seed={seed} epoch={epoch} val_episode_endpoint={score:.8g}", flush=True)
    del model, optimizer
    reloaded, checkpoint = reload_checkpoint(run_dir / "best.pt", device)
    repeated = predict(reloaded, val_samples, scale, device, config["batch_size"])
    reload_difference = float(np.max(np.abs(repeated - best_prediction)))
    if not np.allclose(repeated, best_prediction, atol=1e-7, rtol=1e-5):
        raise RuntimeError(f"Checkpoint reload prediction mismatch: {reload_difference}")
    repeated_metrics = endpoint_metrics(repeated, val_samples)
    synchronize(device)
    result = {
        "model": model_name, "seed": seed, "best_epoch": best_epoch,
        "checkpoint": str(run_dir / "best.pt"), "val": best_metrics,
        "selection_metric": "val.episode_equal.combined",
        "reload_check": {
            "split": "val", "passed": True, "prediction_max_abs_difference": reload_difference,
            "endpoint_abs_difference": abs(repeated_metrics["episode_equal"]["combined"] - best_score),
            "atol": 1e-7, "rtol": 1e-5, "checkpoint_epoch": checkpoint["epoch"],
        },
        "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
        "training_and_validation_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": peak_memory(device),
    }
    del reloaded, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_json(run_dir / "training_result.json", result)
    return result


def source_versions():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "git_head": revision,
        "source_sha256": {str(path.relative_to(root)): file_sha256(path) for path in (
            Path(__file__).resolve(), root / "src/mamba/train.py",
            root / ".agent/memory/03_data_pipeline.md",
        )},
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "cuda": torch.version.cuda,
    }


def run_benchmark(staging_root, output_dir, *, models=("lstm", "mamba"), seeds=(17, 29, 43),
                  epochs=20, batch_size=128, learning_rate=1e-3, d_model=32, device="cuda"):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output must be a new directory: {output_dir}")
    if (not models or len(set(models)) != len(models) or set(models) - {"lstm", "mamba"}
            or not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or not 0 <= seed < 2 ** 32 for seed in seeds)):
        raise ValueError("Specify distinct supported models and distinct nonnegative 32-bit seeds")
    if epochs < 1 or batch_size < 1 or d_model not in (32, 64):
        raise ValueError("epochs/batch_size must be positive; d_model must be 32 or 64")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    resolved_device = torch.device(device)
    if resolved_device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; CPU smoke requires --models lstm --device cpu")
    if resolved_device.type == "cpu" and "mamba" in models:
        raise ValueError("Existing Mamba backbone requires CUDA; use --models lstm for CPU")
    episodes, splits = discover_splits(staging_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = {
        "contract_version": CONTRACT_VERSION, "limitation": LIMITATION,
        "coordinate_units": "original source coordinate units (physical units unverified)",
        "staging_root": str(Path(staging_root).resolve()), "output_dir": str(output_dir),
        "models": list(models), "seeds": list(seeds), "epochs": epochs,
        "batch_size": batch_size, "learning_rate": learning_rate, "d_model": d_model,
        "n_layer": 1, "dropout": 0.0, "d_state": 16, "d_conv": 4, "expand": 2,
        "optimizer": "AdamW", "weight_decay": 0.01, "grad_clip_norm": 1.0,
        "window_frames": WINDOW, "stride": STRIDE, "horizon_seconds": HORIZON_SECONDS,
        "gap_factor": GAP_FACTOR, "gap_policy": "all observation intervals, target bracket and overshoot",
        "target_search": "searchsorted(timestamps, observation_end + 1.0, side='left')",
        "device": str(resolved_device), "dtype": "float32", "splits": splits,
        "selection_metric": "val.episode_equal.combined", "versions": source_versions(),
        "gpu_name": torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else None,
        "determinism": "seeded Python/NumPy/Torch/shuffle; CUDA bitwise determinism not guaranteed",
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "splits.json", {"rule": "sorted UUID, first 6 / next 2 / last 2", **splits})
    train_samples = load_samples(episodes, splits["train"], "train")
    val_samples = load_samples(episodes, splits["val"], "val")
    normalization = fit_normalization(train_samples)
    config["normalization"] = normalization
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "normalization.json", normalization)
    results = []
    for model_name in models:
        for seed in seeds:
            results.append(train_model(
                train_samples, val_samples, normalization, output_dir / f"{model_name}_seed{seed}",
                config, model_name, seed,
            ))
    test_samples = load_samples(episodes, splits["test"], "test")
    all_packs = (train_samples, val_samples, test_samples)
    with (output_dir / "sample_provenance.jsonl").open("w", encoding="utf-8") as handle:
        for pack in all_packs:
            for record in pack.provenance:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
    audits = [audit for pack in all_packs for audit in pack.audits]
    write_json(output_dir / "sample_summary.json", audits)
    baselines = {}
    for name in ("persistence", "constant_velocity"):
        baseline_started = time.perf_counter()
        baselines[name] = {
            pack.split: endpoint_metrics(
                np.zeros_like(pack.target) if name == "persistence" else pack.velocity_prediction, pack,
            ) for pack in (val_samples, test_samples)
        }
        baselines[name]["seconds"] = time.perf_counter() - baseline_started
        baselines[name]["peak_cuda_allocated_bytes"] = 0
    for baseline in baselines.values():
        for split in ("val", "test"):
            baseline[split]["persistence_improvement_percent"] = persistence_improvement(
                baseline[split], baselines["persistence"][split],
            )
    for result in results:
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
        synchronize(resolved_device)
        evaluation_started = time.perf_counter()
        model, checkpoint = reload_checkpoint(result["checkpoint"], resolved_device)
        test_prediction = predict(
            model, test_samples, checkpoint["normalization"]["scale"], resolved_device, batch_size,
        )
        result["test"] = endpoint_metrics(test_prediction, test_samples)
        synchronize(resolved_device)
        result["test_evaluation_seconds"] = time.perf_counter() - evaluation_started
        result["total_seconds"] = result["training_and_validation_seconds"] + result["test_evaluation_seconds"]
        result["peak_cuda_allocated_bytes"] = max(result["peak_cuda_allocated_bytes"], peak_memory(resolved_device))
        result["peak_cuda_allocated_mib"] = result["peak_cuda_allocated_bytes"] / (1024 ** 2)
        result["test_evaluations"] = 1
        result["coordinate_units"] = config["coordinate_units"]
        for split in ("val", "test"):
            result[split]["persistence_improvement_percent"] = persistence_improvement(
                result[split], baselines["persistence"][split],
            )
        run_dir = Path(result["checkpoint"]).parent
        np.savez_compressed(
            run_dir / "test_predictions.npz", prediction=test_prediction, target=test_samples.target,
            source_episode_uuid=test_samples.uuids,
            sample_id=np.asarray([record["sample_id"] for record in test_samples.provenance]),
        )
        write_json(run_dir / "metrics.json", result)
        del model, checkpoint
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "contract_version": CONTRACT_VERSION, "limitation": LIMITATION,
        "coordinate_units": config["coordinate_units"],
        "combined_definition": "arithmetic mean of left/right 3D Euclidean endpoint errors, not 6D norm",
        "episode_equal_definition": "mean of per-UUID sample means; each UUID has equal weight",
        "improvement_definition": "100 * (1 - error/persistence_error); null when persistence_error is zero",
        "normalization": normalization, "splits": splits, "sample_audits": audits,
        "baselines": baselines, "runs": results, "total_seconds": time.perf_counter() - started,
        "test_policy": "test sequences first loaded after every model/seed has completed validation-only selection",
    }
    write_json(output_dir / "metrics.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("lstm", "mamba"), default=["lstm", "mamba"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, choices=(32, 64), default=32)
    parser.add_argument("--device", default="cuda")
    arguments = vars(parser.parse_args(argv))
    report = run_benchmark(**arguments)
    print(json.dumps({"output_dir": str(arguments["output_dir"]), "runs": len(report["runs"]),
                      "total_seconds": report["total_seconds"], "limitation": LIMITATION}, ensure_ascii=False))


if __name__ == "__main__":
    main()
