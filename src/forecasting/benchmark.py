"""Acquisition-isolated pilot: persistence, velocity, LSTM and Mamba."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import build_samples, file_hash, load_manifest
from .evaluate import aggregate_reports, evaluate_sequence
from .online import OnlineForecaster, replay_sequence
from .train import fit_forecaster


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_benchmark(manifest_path, output_dir, models=("lstm", "mamba"), seeds=(17, 29, 43),
                  epochs=20, device="cuda", window_size=32, horizons=(0.5, 1.0)):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "started", "contract_version": "pose_forecast_v1",
              "scope": "small fixed-camera human sequence pilot; not production or egocentric accuracy",
              "reference": "future estimator pseudo-labels, not independent ground truth",
              "independent_accuracy": None, "window_size": window_size,
              "horizons_seconds": list(horizons), "models": list(models), "seeds": list(seeds),
              "epochs": epochs, "device": device, "manifest_sha256": file_hash(manifest_path),
              "checkpoint_selection": "validation masked loss only, including initial persistence epoch 0",
              "display_model_policy": "first requested model and seed, fixed before reading test scores",
              "rotation_caveat": "raw rotation consistency includes symmetry ambiguities and estimator noise"}
    _write(output / "protocol.json", report)
    try:
        manifest, splits = load_manifest(manifest_path)
        if not all(splits[name] for name in ("train", "val", "test")):
            raise ValueError("independent nonempty train/val/test acquisitions required")
        samples = {}
        report["data"] = {}
        for split in ("train", "val"):
            samples[split] = []
            details = []
            for sequence in splits[split]:
                extracted, rejected = build_samples(sequence, window_size, horizons)
                samples[split].extend(extracted)
                details.append({"sequence_id": sequence.sequence_id, "group_id": sequence.group_id,
                                "frames": len(sequence.timestamps), "samples": len(extracted),
                                "rejected": rejected, "hashes": sequence.metadata["observation_hashes"]})
            report["data"][split] = details
            if not samples[split]:
                raise ValueError(f"no valid joint samples in {split}; cannot claim learned prediction")
        checkpoints = []
        for model in models:
            for seed in seeds:
                name = f"{model}_seed{seed}"
                training = fit_forecaster(samples["train"], samples["val"], output / name,
                                          {"backbone": model, "device": device, "seed": seed,
                                           "epochs": epochs, "window_size": window_size,
                                           "horizons_seconds": list(horizons), "n_max": 1})
                checkpoints.append((name, training))
                print(f"[forecast-training] {name} best_epoch={training['best_epoch']} val={training['best_val_loss']:.6f}", flush=True)
        report["test_evaluation_after_all_training"] = True
        report["test_observations_loaded_with_manifest_but_never_used_for_training"] = True
        report["data"]["test"] = [{"sequence_id": sequence.sequence_id, "group_id": sequence.group_id,
                                    "frames": len(sequence.timestamps), "hashes": sequence.metadata["observation_hashes"]}
                                   for sequence in splits["test"]]
        methods = [("persistence", None), ("constant_velocity", None)] + checkpoints
        report["results"] = []
        for name, training in methods:
            method = "learned" if training else name
            checkpoint = training["checkpoint_path"] if training else None
            metrics = []
            for sequence in splits["test"]:
                forecaster = OnlineForecaster(method, window_size, horizons, checkpoint, device)
                predictions = replay_sequence(sequence, forecaster)
                directory = output / name / "test" / sequence.sequence_id
                directory.mkdir(parents=True, exist_ok=False)
                prediction_path = directory / "predictions.npz"
                np.savez_compressed(prediction_path, **predictions)
                evaluation = evaluate_sequence(sequence, predictions, horizons, window_size)
                evaluation["prediction_sha256"] = file_hash(prediction_path)
                metrics.append(evaluation)
                _write(directory / "metrics.json", evaluation)
            result = {"method": name, "test": aggregate_reports(metrics), "sequences": metrics}
            if training:
                result.update(best_epoch=training["best_epoch"], best_val_loss=training["best_val_loss"],
                              checkpoint=checkpoint, checkpoint_sha256=training["checkpoint_sha256"],
                              model_status="validation_selected_initial_persistence" if training["best_epoch"] == 0 else "trained")
            report["results"].append(result)
            print(f"[forecast-test] {name} saved", flush=True)
        report["source_hashes"] = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        report["status"] = "complete"
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", choices=("lstm", "mamba"), nargs="+", default=["lstm", "mamba"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    run_benchmark(args.manifest, args.out_dir, args.models, args.seeds, args.epochs, args.device)


if __name__ == "__main__":
    main()
