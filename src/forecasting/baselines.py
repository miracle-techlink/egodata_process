"""Causal persistence and timestamp-aware constant-velocity forecasting."""

from __future__ import annotations

import cv2
import numpy as np

from .data import validate_horizons, validate_transforms


def _slope(values, timestamps):
    centered = timestamps - timestamps.mean()
    denominator = float(np.square(centered).sum())
    if len(values) < 2 or denominator <= 0:
        return np.zeros(values.shape[1:], dtype=np.float64)
    return np.tensordot(centered, values - values.mean(axis=0), axes=(0, 0)) / denominator


def _tail_start(valid):
    invalid = np.flatnonzero(~valid)
    return int(invalid[-1] + 1) if len(invalid) else 0


def predict_baseline(hands, objects, timestamps, horizons=(0.5, 1.0), method="persistence", velocity_window=5):
    horizons = validate_horizons(horizons)
    hands = np.asarray(hands, dtype=np.float64)
    objects = np.asarray(objects, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if method not in {"persistence", "constant_velocity"}:
        raise ValueError("unsupported baseline")
    if (not len(timestamps) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0) or velocity_window < 2):
        raise ValueError("finite increasing times and velocity_window >= 2 required")
    if hands.shape != (len(timestamps), 2, 21, 3):
        raise ValueError("hands must be (W,2,21,3)")
    if not (np.isfinite(hands).all(axis=(-2, -1)) | np.isnan(hands).all(axis=(-2, -1))).all():
        raise ValueError("each hand must be all finite or all NaN")
    if objects.ndim != 4 or objects.shape[0] != len(timestamps):
        raise ValueError("objects must be (W,N,4,4)")
    validate_transforms(objects)
    hand_valid = np.isfinite(hands[-1]).all(axis=(-2, -1))
    object_valid = np.isfinite(objects[-1]).all(axis=(-2, -1))
    hand_future = np.repeat(hands[-1][None], len(horizons), axis=0)
    object_future = np.repeat(objects[-1][None], len(horizons), axis=0)
    hand_future[:, ~hand_valid] = np.nan
    object_future[:, ~object_valid] = np.nan
    if method == "constant_velocity":
        hands, objects, times = hands[-velocity_window:], objects[-velocity_window:], timestamps[-velocity_window:]
        if len(times) > 2:
            gaps = np.flatnonzero(np.diff(times) > 1.5 * np.median(np.diff(timestamps)) + 1e-9)
            if len(gaps):
                hands, objects, times = hands[gaps[-1] + 1:], objects[gaps[-1] + 1:], times[gaps[-1] + 1:]
        for hand_index in np.flatnonzero(hand_valid):
            track = hands[:, hand_index]
            start = _tail_start(np.isfinite(track).all(axis=(-2, -1)))
            speed = _slope(track[start:], times[start:])
            hand_future[:, hand_index] += horizons[:, None, None] * speed
        for object_index in np.flatnonzero(object_valid):
            track = objects[:, object_index]
            start = _tail_start(np.isfinite(track).all(axis=(-2, -1)))
            track, valid_times = track[start:], times[start:]
            speed = _slope(track[:, :3, 3], valid_times)
            object_future[:, object_index, :3, 3] += horizons[:, None] * speed
            last_rotation = track[-1, :3, :3]
            rotations = np.asarray([cv2.Rodrigues(pose[:3, :3] @ last_rotation.T)[0].ravel() for pose in track])
            angular_speed = _slope(rotations, valid_times)
            for horizon_index, horizon in enumerate(horizons):
                object_future[horizon_index, object_index, :3, :3] = cv2.Rodrigues(angular_speed * horizon)[0] @ last_rotation
    invalid_depth = object_future[..., 2, 3] <= 0
    object_future[invalid_depth] = np.nan
    return {"hand_future": hand_future.astype(np.float32), "object_future": object_future.astype(np.float32)}
