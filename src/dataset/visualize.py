#!/usr/bin/env python3
"""Overlay hand skeletons + object bboxes + 6D pose axes on source clips (task 4, D1).

Usage (from repo root):
    python -m src.dataset.visualize --data-root /path/to/ego_hoi [--clip-id demo01_007 ...]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .build_samples import load_clip, object_features
    from .label_rules import INTERACTION_VOCAB, LabelParams, label_clip
except ImportError:  # direct script execution
    from build_samples import load_clip, object_features
    from label_rules import INTERACTION_VOCAB, LabelParams, label_clip

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
LEFT_COLOR = (0, 255, 0)
RIGHT_COLOR = (0, 0, 255)
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x, y, z in BGR
PALETTE = [(255, 128, 0), (0, 255, 255), (255, 0, 255),
           (128, 255, 0), (0, 128, 255), (255, 0, 128)]


def default_intrinsics(w, h):
    f = 1.2 * max(w, h)
    return f, f, w / 2.0, h / 2.0


def project(pts3, K):
    fx, fy, cx, cy = K
    z = np.where(np.abs(pts3[:, 2]) < 1e-6, np.nan, pts3[:, 2])
    return np.stack([fx * pts3[:, 0] / z + cx, fy * pts3[:, 1] / z + cy], axis=1)


def draw_hand(frame, kpts, w, h, color):
    if np.isnan(kpts).any():
        return
    pts = (kpts[:, :2] * np.array([w, h])).astype(int)
    for a, b in HAND_EDGES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 2, color, -1, cv2.LINE_AA)


def draw_object(frame, obj_id, pose, bbox, color, K, axis_len):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 > x1 and y2 > y1:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, obj_id, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if pose is None or np.isnan(pose).any():
        return
    origin = pose[:3, 3]
    ends = origin[None, :] + (pose[:3, :3] * axis_len).T
    px = project(np.concatenate([origin[None], ends]), K).astype(int)
    if np.isnan(px).any():
        return
    for i in range(3):
        cv2.arrowedLine(frame, tuple(px[0]), tuple(px[i + 1]), AXIS_COLORS[i], 2, cv2.LINE_AA)


def render_clip(clip_path, hand_npz, pose_npz, out_path, intrinsics=None,
                axis_len=0.05, show_labels=True):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = intrinsics or default_intrinsics(w, h)
    clip_id = Path(clip_path).stem
    clip = load_clip(Path(hand_npz).parent, Path(pose_npz).parent, clip_id,
                     default_fps=fps, image_size=f"{w}x{h}")
    labels = None
    if show_labels and clip["object_ids"]:
        _, valid, speed = object_features(clip["poses"], clip["bboxes"], clip["fps"])
        wrist = np.stack([clip["left"][:, 0, :2], clip["right"][:, 0, :2]], axis=1)
        labels = label_clip(wrist, clip["boxes_norm"], valid, speed,
                            LabelParams(fps=clip["fps"]))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t < clip["T"]:
            draw_hand(frame, clip["left"][t], w, h, LEFT_COLOR)
            draw_hand(frame, clip["right"][t], w, h, RIGHT_COLOR)
            for n, obj_id in enumerate(clip["object_ids"]):
                draw_object(frame, obj_id, clip["poses"][n, t], clip["bboxes"][n, t],
                            PALETTE[n % len(PALETTE)], K, axis_len)
        if labels is not None and t < len(labels[0]):
            subject = {0: "left", 1: "right", 2: "none"}[int(labels[0][t])]
            slot = int(labels[1][t])
            obj = clip["object_ids"][slot] if slot >= 0 else "-"
            text = f"t={t}  {subject} / {obj} / {INTERACTION_VOCAB[int(labels[2][t])]}"
            cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(frame)
        t += 1
    cap.release()
    writer.release()
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=".", help="contains clips/, hand_pose/, pose6d/")
    ap.add_argument("--clips-dir", default=None)
    ap.add_argument("--hand-pose-dir", default=None)
    ap.add_argument("--pose6d-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="default <data-root>/vis")
    ap.add_argument("--clip-id", nargs="*", default=None, help="render only these clips")
    ap.add_argument("--sample-ratio", type=float, default=1.0,
                    help="fraction of clips to render when --clip-id is absent")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--axis-len", type=float, default=0.05, help="object axis length in meters")
    ap.add_argument("--no-labels", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    clips_dir = Path(args.clips_dir) if args.clips_dir else root / "clips"
    hand_dir = Path(args.hand_pose_dir) if args.hand_pose_dir else root / "hand_pose"
    pose_dir = Path(args.pose6d_dir) if args.pose6d_dir else root / "pose6d"
    out_dir = Path(args.out_dir) if args.out_dir else root / "vis"

    if args.clip_id:
        clip_ids = list(args.clip_id)
    else:
        clip_ids = sorted(p.stem for p in clips_dir.glob("*.mp4"))
        rng = np.random.default_rng(args.seed)
        if args.sample_ratio < 1.0 and clip_ids:
            k = max(1, int(np.ceil(len(clip_ids) * args.sample_ratio)))
            clip_ids = sorted(rng.choice(clip_ids, size=k, replace=False).tolist())
        if args.max_clips > 0:
            clip_ids = clip_ids[:args.max_clips]
    if not clip_ids:
        sys.exit(f"no clips under {clips_dir}")

    intrinsics = None
    if args.fx is not None:
        intrinsics = (args.fx, args.fy or args.fx,
                      args.cx if args.cx is not None else 0.0,
                      args.cy if args.cy is not None else 0.0)
    n_done = 0
    for cid in clip_ids:
        clip_path = clips_dir / f"{cid}.mp4"
        if not clip_path.exists() or not (hand_dir / f"{cid}.npz").exists():
            print(f"[warn] {cid}: missing clip or hand_pose npz, skipped", file=sys.stderr)
            continue
        n_frames = render_clip(clip_path, hand_dir / f"{cid}.npz", pose_dir / f"{cid}.npz",
                               out_dir / f"{cid}.mp4", intrinsics, args.axis_len,
                               not args.no_labels)
        print(f"{cid}: {n_frames} frames -> {out_dir / (cid + '.mp4')}")
        n_done += 1
    print(f"done: {n_done}/{len(clip_ids)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
