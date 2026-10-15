"""IoU-based multi-object tracker for cross-frame object-id consistency.

Greedy IoU matching (no scipy dependency): per frame, all (track, det)
pairs are sorted by IoU descending and matched greedily above
`iou_threshold`. Adequate for the few target objects per clip; known
limitation: crossing objects of similar size may swap ids.

Ordering rule (fixed for the whole clip, per skill manual): tracks are
sorted by (first_frame, -first_frame_area, track_id) and assigned
`obj_000`, `obj_001`, ... so ids are deterministic across runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def box_area_xyxy(box: np.ndarray) -> float:
    box = np.asarray(box, dtype=np.float64)
    w = max(float(box[2] - box[0]), 0.0)
    h = max(float(box[3] - box[1]), 0.0)
    return w * h


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two box sets: a (M, 4) xyxy, b (N, 4) xyxy -> (M, N)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # last observed xyxy
    first_frame: int
    last_frame: int
    first_area: float
    hits: int = 1
    active: bool = True


class IoUTracker:
    """Assigns stable track ids to per-frame detections.

    max_lost semantics: a track survives up to `max_lost` consecutive
    frames without a matching detection; on the (max_lost+1)-th missed
    frame it is retired and will never match again (a reappearing object
    gets a new id).
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self.iou_threshold = float(iou_threshold)
        self.max_lost = int(max_lost)
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, frame_idx: int, boxes: np.ndarray) -> Dict[int, int]:
        """Match detections to tracks. Returns {detection_index: track_id}."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        for t in self.tracks:
            if t.active and frame_idx - t.last_frame - 1 > self.max_lost:
                t.active = False
        active = [t for t in self.tracks if t.active]

        matches: Dict[int, int] = {}
        if active and len(boxes):
            ious = iou_matrix(np.stack([t.bbox for t in active]), boxes)
            order: List[Tuple[float, int, int]] = sorted(
                ((ious[ti, di], ti, di) for ti in range(len(active)) for di in range(len(boxes))),
                reverse=True,
            )
            used_t, used_d = set(), set()
            for v, ti, di in order:
                if v < self.iou_threshold:
                    break
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                t = active[ti]
                t.bbox = boxes[di].copy()
                t.last_frame = frame_idx
                t.hits += 1
                matches[di] = t.track_id

        for di in range(len(boxes)):
            if di in matches:
                continue
            t = Track(
                track_id=self._next_id,
                bbox=boxes[di].copy(),
                first_frame=frame_idx,
                last_frame=frame_idx,
                first_area=box_area_xyxy(boxes[di]),
            )
            self._next_id += 1
            self.tracks.append(t)
            matches[di] = t.track_id
        return matches

    def all_tracks(self) -> List[Track]:
        return list(self.tracks)


def track_order_key(t: Track) -> Tuple[int, float, int]:
    return (t.first_frame, -t.first_area, t.track_id)


def sorted_tracks(tracks: List[Track]) -> List[Track]:
    return sorted(tracks, key=track_order_key)


def assign_object_ids(tracks: List[Track]) -> Dict[int, str]:
    """Deterministic object ids: obj_000, obj_001, ... in sorted order."""
    return {t.track_id: f"obj_{i:03d}" for i, t in enumerate(sorted_tracks(tracks))}
