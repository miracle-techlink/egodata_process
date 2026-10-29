"""Clearly separated causal forecast overlays and offline future comparisons."""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np

from src.pose6d.run_rgbd_sequence import project_points
from .data import file_hash, target_indices


HAND_EDGES = tuple((start, start + 1) for start in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)) + (
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
BOX_EDGES = tuple((start, start ^ bit) for start in range(8) for bit in (1, 2, 4) if start < start ^ bit)
CURRENT_COLOR = (220, 230, 20)
FUTURE_COLORS = ((0, 165, 255), (235, 80, 235))


def _line(image, first, last, color, dashed=False):
    height, width = image.shape[:2]
    points = np.asarray([first, last], dtype=np.float64)
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 2**29:
        return
    points = np.rint(points).astype(np.int32)
    visible, first, last = cv2.clipLine((0, 0, width, height), tuple(points[0]), tuple(points[1]))
    if not visible:
        return
    first, last = np.asarray(first), np.asarray(last)
    if not dashed:
        cv2.line(image, tuple(first), tuple(last), color, 2, cv2.LINE_AA)
        return
    length = float(np.linalg.norm(last - first))
    distances = np.arange(0, length, 10)
    if len(distances):
        starts = first + (last - first) * distances[:, None] / max(length, 1)
        ends = first + (last - first) * np.minimum(distances + 6, length)[:, None] / max(length, 1)
        segments = np.stack([starts, ends], axis=1).astype(np.int32)
        cv2.polylines(image, list(segments), False, color, 2, cv2.LINE_AA)


def draw_geometry(image, hands, objects, intrinsics, corners, color, dashed=False):
    result = image.copy()
    height, width = result.shape[:2]
    for hand in hands:
        if not np.isfinite(hand).all():
            continue
        pixels = hand[:, :2] * [width, height]
        for first, last in HAND_EDGES:
            _line(result, pixels[first], pixels[last], color, dashed)
        if not dashed:
            for pixel in pixels:
                if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                    cv2.circle(result, tuple(np.rint(pixel).astype(int)), 2, color, -1)
    for pose in objects:
        if not np.isfinite(pose).all():
            continue
        try:
            pixels = project_points(corners, pose, intrinsics)
        except ValueError:
            continue
        for first, last in BOX_EDGES:
            _line(result, pixels[first], pixels[last], color, dashed)
    return result


def draw_forecast_overlay(image, hands, objects, forecast, intrinsics, corners, horizons, include_current=True):
    result = draw_geometry(image, hands, objects, intrinsics, corners, CURRENT_COLOR) if include_current else image.copy()
    if forecast.get("ready", True):
        for horizon_index, horizon in enumerate(horizons):
            color = FUTURE_COLORS[horizon_index % len(FUTURE_COLORS)]
            result = draw_geometry(result, forecast["hand_future"][horizon_index],
                                   forecast["object_future"][horizon_index], intrinsics, corners, color, True)
    return result


def _text(image, text, position, color=(240, 240, 240), scale=0.48):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose_frame(current_rgb, current_hands, current_objects, forecast, intrinsics, corners,
                  horizons, timestamp, method, future_rgb=None, future_hands=None, future_objects=None):
    height, width = current_rgb.shape[:2]
    header = 104
    left = draw_forecast_overlay(current_rgb, current_hands, current_objects, forecast, intrinsics, corners, horizons)
    right = np.zeros_like(current_rgb) if future_rgb is None else future_rgb.copy()
    if future_rgb is not None:
        right = draw_geometry(right, future_hands, future_objects, intrinsics, corners, CURRENT_COLOR)
        right = draw_geometry(right, forecast["hand_future"][-1], forecast["object_future"][-1],
                              intrinsics, corners, FUTURE_COLORS[(len(horizons) - 1) % len(FUTURE_COLORS)], True)
    canvas = np.full((height + header, width * 2, 3), (19, 20, 24), dtype=np.uint8)
    canvas[header:, :width] = left
    canvas[header:, width:] = right
    _text(canvas, f"CAUSAL FORECAST | {method} | t={timestamp:.2f}s", (12, 22), scale=0.55)
    _text(canvas, "Solid cyan: current estimates; dashed: future predictions", (12, 43))
    for horizon_index, horizon in enumerate(horizons):
        _text(canvas, f"+{horizon:.1f}s", (12 + horizon_index * 112, 64), FUTURE_COLORS[horizon_index % len(FUTURE_COLORS)])
    status = "history ready" if forecast.get("ready", True) else "warming up / invalid history"
    _text(canvas, f"DexYCB human RGB-D replay | {status}", (12, 87))
    _text(canvas, f"OFFLINE FUTURE CHECK | t+{horizons[-1]:.1f}s", (width + 12, 22), scale=0.55)
    _text(canvas, "Cyan: future estimates (NOT ground truth)", (width + 12, 43))
    _text(canvas, "Dashed: prediction made earlier; never fed back as input", (width + 12, 64))
    _text(canvas, "Mixed hand/image + object/camera coordinates; no texture needed", (width + 12, 87), scale=0.43)
    if future_rgb is None:
        _text(canvas, "Future comparison unavailable", (width + 100, header + height // 2), scale=0.6)
    return canvas


def scene_geometry(sequence):
    if sequence.objects.shape[1] != 1:
        raise ValueError("renderer currently supports one mesh/object per sequence")
    mesh_record = sequence.metadata["mesh"]
    path = Path(mesh_record["path"])
    if sequence.scene_dir is not None:
        candidate = sequence.scene_dir / "mesh" / path.name
        if candidate.exists():
            path = candidate
    if file_hash(path) != mesh_record["sha256"]:
        raise ValueError("mesh changed since observation extraction")
    vertices = []
    with path.open() as source:
        for line in source:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("OBJ has no finite vertices")
    corners = np.asarray(list(itertools.product(*zip(vertices.min(axis=0), vertices.max(axis=0)))))
    return np.asarray(sequence.metadata["intrinsics"]["matrix"]), corners


def read_rgb(sequence, index):
    record = sequence.records[index]
    path = Path(record["rgb"]["path"])
    if sequence.scene_dir is not None:
        candidate = sequence.scene_dir / "rgb" / path.name
        if candidate.exists():
            path = candidate
    if file_hash(path) != record["rgb"]["sha256"]:
        raise ValueError("RGB source changed since extraction")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def render_video(sequence, predictions, output_dir, method):
    output_dir = Path(output_dir)
    intrinsics, corners = scene_geometry(sequence)
    horizons = predictions["horizons_seconds"]
    first = read_rgb(sequence, 0)
    height, width = first.shape[:2]
    output = output_dir / "forecast_comparison.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), sequence.fps, (width * 2, height + 104))
    if not writer.isOpened():
        raise RuntimeError("forecast video writer failed")
    snapshots = []
    ready_indices = np.flatnonzero(predictions["ready"])
    eligible = [int(index) for index in ready_indices if target_indices(sequence.timestamps, int(index), [horizons[-1]]) is not None]
    preview_indices = set(np.linspace(0, len(eligible) - 1, min(3, len(eligible))).astype(int))
    preview_frames = {eligible[index] for index in preview_indices}
    try:
        for index, timestamp in enumerate(sequence.timestamps):
            current = first if index == 0 else read_rgb(sequence, index)
            if current.shape != first.shape:
                raise ValueError("source image dimensions changed")
            forecast = {name: predictions[name][index] for name in ("hand_future", "object_future", "ready")}
            targets = target_indices(sequence.timestamps, index, [horizons[-1]])
            target = int(targets[0]) if targets is not None and forecast["ready"] else None
            frame = compose_frame(current, sequence.hands[index], sequence.objects[index], forecast,
                                  intrinsics, corners, horizons, timestamp, method,
                                  read_rgb(sequence, target) if target is not None else None,
                                  sequence.hands[target] if target is not None else None,
                                  sequence.objects[target] if target is not None else None)
            writer.write(frame)
            if index in preview_frames:
                snapshots.append(frame)
                if not cv2.imwrite(str(output_dir / f"preview_{index:06d}.jpg"), frame):
                    raise RuntimeError("preview image write failed")
    finally:
        writer.release()
    reader = cv2.VideoCapture(str(output))
    decoded = 0
    while True:
        success, frame = reader.read()
        if not success:
            break
        decoded += 1
    reader.release()
    if decoded != len(sequence.timestamps):
        raise RuntimeError("forecast video frame count mismatch")
    if snapshots:
        if not cv2.imwrite(str(output_dir / "forecast_contact_sheet.jpg"), np.vstack(snapshots)):
            raise RuntimeError("contact sheet write failed")
    return {"path": str(output), "sha256": file_hash(output), "decoded_frames": decoded,
            "width": width * 2, "height": height + 104, "fps": sequence.fps,
            "preview_observation_indices": sorted(preview_frames),
            "left_panel_uses_future_images": False, "right_panel": "explicit offline future estimate comparison"}
