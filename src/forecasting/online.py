"""Stateful predictor receiving one observed frame at a time."""

from __future__ import annotations

from collections import deque
import time

import numpy as np

from .baselines import predict_baseline
from .data import history_features, validate_horizons, validate_transforms


class OnlineForecaster:
    def __init__(self, method="persistence", window_size=32, horizons=(0.5, 1.0), checkpoint=None, device="cpu"):
        self.method = method
        self.window_size = int(window_size)
        self.horizons = validate_horizons(horizons)
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.history = deque(maxlen=self.window_size)
        self.last_time = None
        self.model = None
        self.model_metadata = None
        self.device = device
        if method == "learned":
            from .train import load_forecaster
            if checkpoint is None:
                raise ValueError("learned forecasting requires a trained checkpoint")
            self.model, self.model_metadata = load_forecaster(checkpoint, device=device)
            config = self.model_metadata["config"]
            if config["window_size"] != self.window_size or not np.array_equal(config["horizons_seconds"], self.horizons):
                raise ValueError("checkpoint window/horizons mismatch")
            self.model.eval()
        elif method not in {"persistence", "constant_velocity"}:
            raise ValueError("unknown forecast method")

    @property
    def display_label(self):
        if self.model_metadata is None:
            return self.method
        backbone = self.model_metadata["config"]["backbone"]
        epoch = self.model_metadata["epoch"]
        return f"{backbone} trained epoch {epoch}" if epoch > 0 else f"{backbone} INIT persistence"

    def update(self, hands, objects, bboxes, timestamp):
        started = time.perf_counter()
        hands = np.asarray(hands, dtype=np.float32)
        objects = np.asarray(objects, dtype=np.float32)
        bboxes = np.asarray(bboxes)
        if not np.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("strictly increasing finite timestamps required")
        if hands.shape != (2, 21, 3) or objects.ndim != 3 or bboxes.shape != (len(objects), 4):
            raise ValueError("invalid streaming observation shape")
        if not (np.isfinite(hands).all(axis=(-2, -1)) | np.isnan(hands).all(axis=(-2, -1))).all():
            raise ValueError("each hand must be all finite or all NaN")
        validate_transforms(objects)
        if self.history and objects.shape != self.history[-1][1].shape:
            raise ValueError("object slots must remain stable")
        self.last_time = float(timestamp)
        self.history.append((hands.copy(), objects.copy(), bboxes.copy(), float(timestamp)))
        result = {"hand_future": np.full((len(self.horizons), 2, 21, 3), np.nan, np.float32),
                  "object_future": np.full((len(self.horizons), len(objects), 4, 4), np.nan, np.float32),
                  "ready": False, "observation_time": float(timestamp)}
        if len(self.history) == self.window_size:
            history_hands, history_objects, history_boxes, times = map(np.asarray, zip(*self.history))
            intervals = np.diff(times)
            if np.any(intervals > 1.5 * np.median(intervals) + 1e-9):
                result["reason"] = "history_timestamp_gap"
            elif self.model is None:
                result.update(predict_baseline(history_hands, history_objects, times, self.horizons, self.method))
                result["ready"] = True
            else:
                import torch
                features = history_features(history_objects, history_boxes, times)
                with torch.inference_mode():
                    output = self.model(torch.as_tensor(history_hands[None], device=self.device),
                                        torch.as_tensor(features[None], device=self.device),
                                        torch.ones((1, len(objects)), dtype=torch.bool, device=self.device))
                result.update({key: output[key][0].cpu().numpy() for key in ("hand_future", "object_future")})
                result["ready"] = True
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        return result


def replay_sequence(sequence, forecaster):
    hands, objects, timings, ready = [], [], [], []
    for frame_index, timestamp in enumerate(sequence.timestamps):
        result = forecaster.update(sequence.hands[frame_index], sequence.objects[frame_index],
                                   sequence.bboxes[frame_index], timestamp)
        hands.append(result["hand_future"])
        objects.append(result["object_future"])
        timings.append(result["latency_ms"])
        ready.append(result["ready"])
    return {"hand_future": np.asarray(hands), "object_future": np.asarray(objects),
            "latency_ms": np.asarray(timings), "ready": np.asarray(ready),
            "timestamps": sequence.timestamps.copy(), "horizons_seconds": forecaster.horizons.copy()}
