"""Replay saved real pose observations through a strictly causal forecaster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .data import file_hash, load_sequence
from .evaluate import evaluate_sequence
from .online import OnlineForecaster, replay_sequence
from .render import render_video, read_rgb


def run(args):
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "contract_version": "pose_forecast_v1", "method": args.method,
              "input_type": "causal_replay_of_saved_real_estimator_outputs", "live_camera_validated": False,
              "window_size": args.window_size, "horizons_seconds": args.horizons,
              "independent_accuracy": None, "training_performed": False,
              "source": {"module": str(Path(__file__).resolve()), "sha256": file_hash(__file__)}}
    started = time.perf_counter()
    try:
        sequence = load_sequence(args.observation_dir, args.sequence_id, args.group_id, args.scene_dir)
        forecaster = OnlineForecaster(args.method, args.window_size, args.horizons, args.checkpoint, args.device)
        predictions = replay_sequence(sequence, forecaster)
        predictions_path = output / "predictions.npz"
        np.savez_compressed(predictions_path, **predictions)
        image = read_rgb(sequence, 0)
        report["evaluation"] = evaluate_sequence(sequence, predictions, args.horizons, args.window_size,
                                                   (image.shape[1], image.shape[0]))
        ready_times = predictions["latency_ms"][predictions["ready"]]
        report["timing"] = {"scope": "forecast module only; excludes perception and video encoding",
                            "ready_frames": len(ready_times),
                            "mean_ms": float(np.mean(ready_times)) if len(ready_times) else None,
                            "p95_ms": float(np.percentile(ready_times, 95)) if len(ready_times) else None}
        report["source_observation_hashes"] = sequence.metadata["observation_hashes"]
        report["predictions_sha256"] = file_hash(predictions_path)
        if args.checkpoint:
            report["checkpoint"] = {"path": str(args.checkpoint), "sha256": file_hash(args.checkpoint),
                                    **{name: forecaster.model_metadata[name] for name in
                                       ("model_status", "epoch", "optimizer_steps", "selection", "checkpoint_role")}}
        report["display_method"] = forecaster.display_label
        if not args.no_video:
            report["video"] = render_video(sequence, predictions, output, forecaster.display_label)
        report["status"] = "complete"
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-dir", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--scene-dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=("persistence", "constant_velocity", "learned"), default="constant_velocity")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--horizons", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "out_dir": args.out_dir, "timing": report["timing"]}, indent=2))


if __name__ == "__main__":
    main()
