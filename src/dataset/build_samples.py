#!/usr/bin/env python3
"""Build windowed training samples from hand_pose + pose6d npz files (task 4).

Usage (from repo root):
    python -m src.dataset.build_samples --data-root /path/to/ego_hoi
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .label_rules import INTERACTION_VOCAB, LabelParams, label_clip
    from .rot6d import matrix_to_rot6d
except ImportError:  # direct script execution
    from label_rules import INTERACTION_VOCAB, LabelParams, label_clip
    from rot6d import matrix_to_rot6d

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

CONTRACT_VERSION = "2026-10-15"


def parse_source_video(clip_id):
    return clip_id.rsplit("_", 1)[0] if "_" in clip_id else clip_id


def read_meta(npz_path):
    meta_path = npz_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def resolve_fps(hand_meta, pose_meta, default_fps):
    for m in (hand_meta, pose_meta):
        if m.get("fps"):
            return float(m["fps"])
    return float(default_fps)


def resolve_image_size(hand_meta, pose_meta, image_size_arg):
    if image_size_arg:
        w, h = image_size_arg.lower().split("x")
        return int(w), int(h)
    for m in (hand_meta, pose_meta):
        if m.get("width") and m.get("height"):
            return int(m["width"]), int(m["height"])
    return None


def load_clip(hand_dir, pose_dir, clip_id, default_fps=30.0, image_size=None):
    """Load one clip's raw arrays; truncates hand/object streams to a common T."""
    hand_npz = Path(hand_dir) / f"{clip_id}.npz"
    pose_npz = Path(pose_dir) / f"{clip_id}.npz"
    hand_meta = read_meta(hand_npz)
    pose_meta = read_meta(pose_npz)
    with np.load(hand_npz) as hp:
        left = np.asarray(hp["left_hand"], np.float32)
        right = np.asarray(hp["right_hand"], np.float32)
        conf = np.asarray(hp["conf"], np.float32) if "conf" in hp else None
    T_hand = left.shape[0]
    if pose_npz.exists():
        with np.load(pose_npz) as pd:
            object_ids = [str(x) for x in pd["object_ids"]]
            poses = np.asarray(pd["poses"], np.float32)
            bboxes = np.asarray(pd["bboxes"], np.int32)
        T_pose = poses.shape[1]
    else:
        object_ids = []
        poses = np.zeros((0, T_hand, 4, 4), np.float32)
        bboxes = np.zeros((0, T_hand, 4), np.int32)
        T_pose = T_hand
    T = min(T_hand, T_pose)
    fps = resolve_fps(hand_meta, pose_meta, default_fps)
    wh = resolve_image_size(hand_meta, pose_meta, image_size)
    if wh is None:
        raise ValueError(
            f"{clip_id}: image size unknown; pass --image-size WxH or add "
            f"width/height to the upstream meta.json"
        )
    scale = np.array([wh[0], wh[1], wh[0], wh[1]], np.float32)
    return {
        "clip_id": clip_id,
        "T": T,
        "T_mismatch": T_hand != T_pose,
        "fps": fps,
        "image_wh": wh,
        "left": left[:T],
        "right": right[:T],
        "object_ids": object_ids,
        "poses": poses[:, :T],
        "bboxes": bboxes[:, :T],
        "boxes_norm": bboxes[:, :T].astype(np.float32) / scale,
    }


def object_features(poses, bboxes, fps):
    """poses (N,T,4,4), bboxes (N,T,4) -> feats (T,N,13) f32, valid (T,N) f32, speed (N,T) f32.

    feat order: pos(3) + rot6d(6) + bbox area(1) + velocity(3); rows with a
    missing pose are all-NaN; velocity at t=0 is 0 by convention.
    """
    N, T = poses.shape[0], poses.shape[1]
    valid = ~np.isnan(poses).any(axis=(2, 3))          # (N,T)
    pos = poses[:, :, :3, 3].transpose(1, 0, 2).astype(np.float32)          # (T,N,3)
    rot = matrix_to_rot6d(poses[:, :, :3, :3]).transpose(1, 0, 2).astype(np.float32)  # (T,N,6)
    bb = bboxes.astype(np.float32)
    area = (np.clip(bb[..., 2] - bb[..., 0], 0, None)
            * np.clip(bb[..., 3] - bb[..., 1], 0, None)).T[:, :, None]      # (T,N,1)
    vel = np.zeros_like(pos)
    if T > 1:
        # velocity only where both adjacent frames have valid poses (else 0),
        # so a recovery frame never yields a partial-NaN feature row
        both = (valid[:, 1:] & valid[:, :-1]).T                          # (T-1,N)
        diff = (pos[1:] - pos[:-1]) * np.float32(fps)
        vel[1:] = np.where(both[..., None], diff, np.float32(0.0))
    feats = np.concatenate([pos, rot, area, vel], axis=-1).astype(np.float32)
    feats[~valid.T] = np.nan
    speed = np.linalg.norm(vel, axis=-1).T.astype(np.float32)               # (N,T)
    return feats, valid.T.astype(np.float32), speed


def split_videos(videos, val_ratio, test_ratio, seed, val_videos=None, test_videos=None):
    videos = sorted(videos)
    if val_videos or test_videos:
        val = set(val_videos or [])
        test = set(test_videos or [])
        unknown = (val | test) - set(videos)
        if unknown:
            raise ValueError(f"explicit split lists unknown videos: {sorted(unknown)}")
        if val & test:
            raise ValueError("a video cannot be in both val and test")
        train = [v for v in videos if v not in val | test]
        return {"train": train, "val": sorted(val), "test": sorted(test)}
    n = len(videos)
    if n < 3:
        print(f"[warn] only {n} source videos (<3); all samples go to train", file=sys.stderr)
        return {"train": videos, "val": [], "test": []}
    rng = np.random.default_rng(seed)
    order = [videos[i] for i in rng.permutation(n)]
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    return {
        "test": sorted(order[:n_test]),
        "val": sorted(order[n_test:n_test + n_val]),
        "train": sorted(order[n_test + n_val:]),
    }


def make_sample(clip, feats, valid, labels, start, window, n_max):
    s, f = start, start + window
    N = feats.shape[1]
    n_keep = min(N, n_max)
    hand_seq = np.stack([clip["left"][s:f], clip["right"][s:f]], axis=1).astype(np.float32)
    obj_seq = np.zeros((window, n_max, 13), np.float32)
    obj_seq[:, :n_keep] = feats[s:f, :n_keep]
    obj_mask = np.zeros(n_max, np.float32)
    obj_mask[:n_keep] = 1.0
    obj_valid = np.zeros((window, n_max), np.float32)
    obj_valid[:, :n_keep] = valid[s:f, :n_keep]
    object_ids = clip["object_ids"][:n_keep] + [""] * (n_max - n_keep)
    subject, slot, itype = (int(labels[k][f]) for k in range(3))
    return {
        "hand_seq": hand_seq,
        "obj_seq": obj_seq,
        "obj_mask": obj_mask,
        "obj_valid": obj_valid,
        "hand_valid": (~np.isnan(hand_seq).any(axis=(2, 3))).astype(np.float32),
        "object_ids": np.array(object_ids),
        "label_subject": np.int64(subject),
        "label_object_slot": np.int64(slot),
        "label_object_id": clip["object_ids"][slot] if slot >= 0 else "",
        "label_interaction": np.int64(itype),
        "label_interaction_name": INTERACTION_VOCAB[itype],
        "clip_id": clip["clip_id"],
        "source_video": parse_source_video(clip["clip_id"]),
        "start_frame": np.int64(s),
        "fps": np.float32(clip["fps"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=".", help="contains hand_pose/ and pose6d/")
    ap.add_argument("--hand-pose-dir", default=None)
    ap.add_argument("--pose6d-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="default <data-root>/samples")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--n-max", type=int, default=0,
                    help="object slots per sample; 0 = auto (ceil P95 of per-clip object count)")
    ap.add_argument("--fps", type=float, default=30.0, help="fallback when meta.json has no fps")
    ap.add_argument("--image-size", default=None, help="WxH fallback/override, e.g. 1280x720")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-videos", nargs="*", default=None)
    ap.add_argument("--test-videos", nargs="*", default=None)
    ap.add_argument("--contact-margin", type=float, default=LabelParams.contact_margin)
    ap.add_argument("--reach-margin", type=float, default=LabelParams.reach_margin)
    ap.add_argument("--vel-thresh", type=float, default=LabelParams.vel_thresh)
    ap.add_argument("--place-check-frames", type=int, default=LabelParams.place_check_frames)
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    hand_dir = Path(args.hand_pose_dir) if args.hand_pose_dir else root / "hand_pose"
    pose_dir = Path(args.pose6d_dir) if args.pose6d_dir else root / "pose6d"
    out_dir = Path(args.out_dir) if args.out_dir else root / "samples"

    hand_ids = {p.stem for p in hand_dir.glob("*.npz")}
    pose_ids = {p.stem for p in pose_dir.glob("*.npz")} if pose_dir.exists() else set()
    only_hand = sorted(hand_ids - pose_ids)
    if only_hand:
        print(f"[warn] clips without pose6d (treated as object-less): {only_hand}", file=sys.stderr)
    clip_ids = sorted(hand_ids | pose_ids)
    if not clip_ids:
        sys.exit(f"no clips found under {hand_dir} / {pose_dir}")

    clips = [load_clip(hand_dir, pose_dir, cid, args.fps, args.image_size)
             for cid in tqdm(clip_ids, desc="load")]
    for c in clips:
        if c["T_mismatch"]:
            print(f"[warn] {c['clip_id']}: hand/pose frame count differ, truncated to T={c['T']}",
                  file=sys.stderr)

    if args.n_max > 0:
        n_max = args.n_max
    else:
        counts = [len(c["object_ids"]) for c in clips]
        n_max = max(1, int(np.ceil(np.percentile(counts, 95))))
    dropped = sum(max(0, len(c["object_ids"]) - n_max) for c in clips)
    if dropped:
        print(f"[warn] {dropped} object track(s) dropped by N_max={n_max}", file=sys.stderr)

    splits = split_videos({parse_source_video(c["clip_id"]) for c in clips},
                          args.val_ratio, args.test_ratio, args.seed,
                          args.val_videos, args.test_videos)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[a]) & set(splits[b])
        assert not overlap, f"split leakage between {a} and {b}: {overlap}"
    video2split = {v: s for s, vs in splits.items() for v in vs}

    lp = LabelParams(contact_margin=args.contact_margin, reach_margin=args.reach_margin,
                     vel_thresh=args.vel_thresh, place_check_frames=args.place_check_frames)
    entries = []
    interaction_counts = {k: 0 for k in INTERACTION_VOCAB}
    skipped = []
    for c in tqdm(clips, desc="build"):
        split = video2split[parse_source_video(c["clip_id"])]
        n_keep = min(len(c["object_ids"]), n_max)
        c["object_ids"] = c["object_ids"][:n_keep]
        feats, valid, speed = object_features(
            c["poses"][:n_keep], c["bboxes"][:n_keep], c["fps"])
        lp.fps = c["fps"]
        wrist = np.stack([c["left"][:, 0, :2], c["right"][:, 0, :2]], axis=1)
        labels = label_clip(wrist, c["boxes_norm"][:n_keep], valid, speed, lp)
        # legal starts: label is frame s+window, so s+window <= T-1 -> s <= T-window-1
        starts = list(range(0, c["T"] - args.window, args.stride))
        if not starts:
            skipped.append(c["clip_id"])
            print(f"[warn] {c['clip_id']}: T={c['T']} too short for W={args.window}, skipped",
                  file=sys.stderr)
            continue
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        for s in starts:
            sample = make_sample(c, feats, valid, labels, s, args.window, n_max)
            sample_id = f"{c['clip_id']}_{s:06d}"
            np.savez_compressed(out_dir / split / f"{sample_id}.npz", **sample)
            interaction_counts[sample["label_interaction_name"]] += 1
            entries.append({
                "sample_id": sample_id,
                "clip_id": c["clip_id"],
                "source_video": sample["source_video"],
                "split": split,
                "start_frame": s,
                "label_subject": sample["label_subject"].item(),
                "label_object_id": sample["label_object_id"],
                "label_interaction": sample["label_interaction_name"],
            })
    for e in entries:
        v = parse_source_video(e["clip_id"])
        assert e["source_video"] == v and video2split[v] == e["split"], "split leakage"

    index = {s: [e for e in entries if e["split"] == s] for s in ("train", "val", "test")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    (out_dir / "meta.json").write_text(json.dumps({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "src/dataset/build_samples.py",
        "contract_version": CONTRACT_VERSION,
        "python": sys.version.split()[0],
        "config": vars(args),
        "label_params": vars(lp),
        "interaction_vocab": INTERACTION_VOCAB,
        "subject_vocab": {"left": 0, "right": 1, "none": 2},
        "window": args.window,
        "stride": args.stride,
        "n_max": n_max,
        "splits": splits,
        "stats": {
            "n_samples": len(entries),
            "per_split": {s: len(index[s]) for s in index},
            "interaction_counts": interaction_counts,
            "dropped_objects": dropped,
            "skipped_clips": skipped,
        },
    }, indent=2, ensure_ascii=False))
    print(f"done: {len(entries)} samples -> {out_dir} "
          f"(train/val/test = { {s: len(index[s]) for s in index} })")


if __name__ == "__main__":
    main()
