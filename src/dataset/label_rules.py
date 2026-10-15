"""Heuristic draft labels for interaction_type (task 4, D4).

The whole rule set lives here so it can be revised jointly with task 5 (arc)
and then written back into .agent/memory/03_data_pipeline.md.

Hand keypoints are normalized image coordinates (z = relative depth, not
metric), while object poses are metric camera-frame; the two spaces cannot be
mixed, so contact/reach are judged in normalized image space (bbox / image
size), and object speed in camera frame is used for hold/move/place/release.
"""

from dataclasses import dataclass

import numpy as np

INTERACTION_VOCAB = ["none", "reach", "grasp", "hold", "move", "place", "release"]
NONE, REACH, GRASP, HOLD, MOVE, PLACE, RELEASE = range(len(INTERACTION_VOCAB))
SUBJECT_LEFT, SUBJECT_RIGHT, SUBJECT_NONE = 0, 1, 2
WRIST_IDX = 0


@dataclass
class LabelParams:
    fps: float = 30.0
    contact_margin: float = 0.5   # bbox expanded by margin * bbox size => contact zone
    reach_margin: float = 1.5     # wider zone used for reach detection
    vel_thresh: float = 0.05      # m/s; object center speed above this = moving
    place_check_frames: int = 5   # lookahead frames to separate place from release


def _expanded_inside(pt, box, margin):
    # pt (T,2), box (T,4) xyxy, same coordinate space
    bw = box[:, 2] - box[:, 0]
    bh = box[:, 3] - box[:, 1]
    mx, my = margin * bw, margin * bh
    return (
        (pt[:, 0] >= box[:, 0] - mx) & (pt[:, 0] <= box[:, 2] + mx)
        & (pt[:, 1] >= box[:, 1] - my) & (pt[:, 1] <= box[:, 3] + my)
    )


def _pair_labels(wrist, box, valid, speed, p):
    """One (hand, object) track -> per-frame interaction ids + wrist-center dist."""
    T = wrist.shape[0]
    lab = np.full(T, NONE, dtype=np.int64)
    dist = np.full(T, np.inf)
    hand_ok = ~np.isnan(wrist).any(axis=1)
    box_ok = (
        valid & (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
        & ~np.isnan(box).any(axis=1)
    )
    ok = hand_ok & box_ok
    if not ok.any():
        return lab, dist
    cx = (box[:, 0] + box[:, 2]) / 2.0
    cy = (box[:, 1] + box[:, 3]) / 2.0
    dist = np.where(ok, np.hypot(wrist[:, 0] - cx, wrist[:, 1] - cy), np.inf)
    contact = ok & _expanded_inside(wrist, box, p.contact_margin)
    reach_zone = ok & _expanded_inside(wrist, box, p.reach_margin)
    approaching = np.zeros(T, bool)
    approaching[1:] = dist[1:] < dist[:-1]
    for t in range(T):
        c_now = contact[t]
        c_prev = contact[t - 1] if t > 0 else False
        if c_now and not c_prev:
            lab[t] = GRASP
        elif c_now and c_prev:
            lab[t] = MOVE if speed[t] > p.vel_thresh else HOLD
        elif c_prev and not c_now:
            future = speed[t + 1: t + 1 + p.place_check_frames]
            future = future[np.isfinite(future)]
            lab[t] = PLACE if future.size and np.all(future <= p.vel_thresh) else RELEASE
        elif reach_zone[t] and approaching[t]:
            lab[t] = REACH
    return lab, dist


def label_clip(wrist_xy, boxes_norm, obj_valid, obj_speed, params=None):
    """Frame-level interaction labels for one clip.

    wrist_xy   (T,2,2) normalized image xy of [left, right] wrists
    boxes_norm (N,T,4) xyxy normalized by image (W,H)
    obj_valid  (T,N)   1 where the object pose is present
    obj_speed  (N,T)   m/s, NaN where the pose is missing
    -> subject (T,) int64 in {0,1,2}, slot (T,) int64 (-1 = none), itype (T,) int64
    """
    p = params or LabelParams()
    T = wrist_xy.shape[0]
    N = boxes_norm.shape[0]
    subject = np.full(T, SUBJECT_NONE, np.int64)
    slot = np.full(T, -1, np.int64)
    itype = np.full(T, NONE, np.int64)
    if N == 0:
        return subject, slot, itype
    pair_lab = np.zeros((T, 2, N), np.int64)
    pair_dist = np.full((T, 2, N), np.inf)
    for h in range(2):
        for n in range(N):
            lab, dist = _pair_labels(
                wrist_xy[:, h], boxes_norm[n], obj_valid[:, n] > 0, obj_speed[n], p
            )
            pair_lab[:, h, n] = lab
            pair_dist[:, h, n] = dist
    active = pair_lab != NONE
    for t in range(T):
        if not active[t].any():
            continue
        d = np.where(active[t], pair_dist[t], np.inf)
        h, n = np.unravel_index(int(np.argmin(d)), d.shape)
        subject[t] = h
        slot[t] = n
        itype[t] = pair_lab[t, h, n]
    return subject, slot, itype
