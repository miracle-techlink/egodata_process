#!/usr/bin/env python3
"""Synthetic end-to-end selftest for task 4 (not shipped to the training pipeline).

Generates synthetic hand_pose/pose6d npz + synthetic clips, then runs
build_samples.py and visualize.py as subprocesses and validates the outputs
against the contract (shapes, dtypes, NaN rules, split isolation, labels).

Run from repo root:  python -m src.dataset.selftest_synthetic
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.dataset.label_rules import HOLD, MOVE
from src.dataset.rot6d import matrix_to_rot6d, rot6d_to_matrix

W_IMG, H_IMG, FPS = 640, 480, 30.0
WINDOW, STRIDE = 32, 16
REPO_ROOT = Path(__file__).resolve().parents[2]


def rotz(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]], np.float64)


def make_hand(T, wrist_xy, keypoint_noise=0.002, seed=0):
    """wrist_xy: (T,2) normalized or None for an all-NaN track."""
    hand = np.full((T, 21, 3), np.nan, np.float32)
    if wrist_xy is None:
        return hand
    rng = np.random.default_rng(seed)
    wrist = np.asarray(wrist_xy, np.float32)
    for k in range(21):
        off = rng.uniform(-0.03, 0.03, size=2)
        hand[:, k, :2] = wrist + off + rng.normal(0, keypoint_noise, (T, 2)).astype(np.float32)
        hand[:, k, 2] = rng.normal(0, 0.01, T).astype(np.float32)
    return hand


def lin_interp(T, p0, p1, t_end):
    t = np.minimum(np.arange(T), t_end)
    return p0 + (p1 - p0) * (t / t_end)[:, None]


def write_clip(root, clip_id, T, left, right, conf, objects, video_draw):
    """objects: list of dicts with id, pos (T,3), rot (T,3,3), bbox (T,4) int."""
    hand_dir, pose_dir, clips_dir = root / "hand_pose", root / "pose6d", root / "clips"
    for d in (hand_dir, pose_dir, clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    np.savez(hand_dir / f"{clip_id}.npz", left_hand=left, right_hand=right, conf=conf)
    (hand_dir / f"{clip_id}.meta.json").write_text(json.dumps({
        "fps": FPS, "frames": T, "width": W_IMG, "height": H_IMG,
        "source_clip": clip_id, "created": "synthetic",
    }))
    N = len(objects)
    poses = np.full((N, T, 4, 4), np.nan, np.float32)
    bboxes = np.zeros((N, T, 4), np.int32)
    for i, o in enumerate(objects):
        for t in range(T):
            poses[i, t] = np.eye(4, dtype=np.float32)
            poses[i, t, :3, :3] = o["rot"][t]
            poses[i, t, :3, 3] = o["pos"][t]
        bboxes[i] = o["bbox"]
        if "nan_frames" in o:
            poses[i, o["nan_frames"]] = np.nan
    np.savez(pose_dir / f"{clip_id}.npz",
             object_ids=np.array([o["id"] for o in objects]),
             poses=poses, bboxes=bboxes)
    (pose_dir / f"{clip_id}.meta.json").write_text(json.dumps({
        "fps": FPS, "frames": T, "width": W_IMG, "height": H_IMG,
        "source_clip": clip_id, "created": "synthetic", "depth_source": "synthetic",
    }))
    vw = cv2.VideoWriter(str(clips_dir / f"{clip_id}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W_IMG, H_IMG))
    for t in range(T):
        frame = np.full((H_IMG, W_IMG, 3), 40, np.uint8)
        video_draw(frame, t)
        vw.write(frame)
    vw.release()


def gen_vidA_001(root):
    """Right hand reaches -> grasps -> carries (move) -> places the mug; left hand absent."""
    T = 90
    mug_c = np.array([0.5, 0.5])
    shift = np.where(np.arange(T) < 40, 0,
                     np.where(np.arange(T) < 70, np.arange(T) - 40, 30)) * 0.002
    mug_bbox = np.stack([ (mug_c[0] - 0.05 + shift) * W_IMG, np.full(T, (mug_c[1] - 0.05) * H_IMG),
                          (mug_c[0] + 0.05 + shift) * W_IMG, np.full(T, (mug_c[1] + 0.05) * H_IMG) ],
                        axis=1).astype(np.int32)
    mug_pos = np.stack([np.where(np.arange(T) < 40, 0.0,
                                 np.where(np.arange(T) < 70, (np.arange(T) - 40) * 0.0025, 0.075)),
                        np.zeros(T), np.full(T, 0.8)], axis=1)
    wrist = lin_interp(T, np.array([0.1, 0.1]), mug_c, 25)
    follow = np.stack([shift, np.zeros(T)], axis=1)
    wrist = wrist + np.where((np.arange(T) >= 19)[:, None], follow, 0)
    leave = np.clip(np.arange(T) - 71, 0, None) * 0.02
    wrist = wrist + np.stack([np.zeros(T), leave], axis=1)
    right = make_hand(T, wrist, seed=1)
    left = make_hand(T, None)  # all NaN
    conf = np.stack([np.zeros(T), np.ones(T)], axis=1).astype(np.float32)
    box_bbox = np.tile(np.array([0.12, 0.15, 0.28, 0.30]) * [W_IMG, H_IMG, W_IMG, H_IMG],
                       (T, 1)).astype(np.int32)
    box_pos = np.tile(np.array([-0.3, 0.1, 0.9]), (T, 1))
    objects = [
        {"id": "mug", "pos": mug_pos, "rot": np.stack([rotz(t) for t in range(T)]),
         "bbox": mug_bbox},
        {"id": "box", "pos": box_pos, "rot": np.stack([np.eye(3)] * T),
         "bbox": box_bbox, "nan_frames": np.arange(10, 15)},
    ]
    write_clip(root, "vidA_001", T, left, right, conf, objects,
               lambda f, t: cv2.circle(f, (int(0.5 * W_IMG), int(0.5 * H_IMG)), 20, (0, 200, 0), -1))
    return T


def gen_vidA_002(root):
    """Idle right hand far from the single static object -> all none labels."""
    T = 80
    wrist = np.tile(np.array([0.05, 0.9]), (T, 1))
    right = make_hand(T, wrist, seed=2)
    left = make_hand(T, wrist + np.array([0.05, 0.0]), seed=3)
    conf = np.ones((T, 2), np.float32)
    obj_bbox = np.tile(np.array([0.45, 0.45, 0.55, 0.55]) * [W_IMG, H_IMG, W_IMG, H_IMG],
                       (T, 1)).astype(np.int32)
    objects = [{"id": "mug", "pos": np.tile(np.array([0.0, 0.0, 0.8]), (T, 1)),
                "rot": np.stack([np.eye(3)] * T), "bbox": obj_bbox}]
    write_clip(root, "vidA_002", T, left, right, conf, objects, lambda f, t: None)
    return T


def gen_vidB_001(root):
    """Left hand approaches the mug but never touches -> reach; right hand NaN on 30-39."""
    T = 70
    wrist_l = lin_interp(T, np.array([0.05, 0.05]), np.array([0.35, 0.35]), 50)
    left = make_hand(T, wrist_l, seed=4)
    wrist_r = np.tile(np.array([0.9, 0.9]), (T, 1))
    right = make_hand(T, wrist_r, seed=5)
    right[30:40] = np.nan
    conf = np.ones((T, 2), np.float32)
    conf[30:40, 1] = 0.0
    obj_bbox = np.tile(np.array([0.45, 0.45, 0.55, 0.55]) * [W_IMG, H_IMG, W_IMG, H_IMG],
                       (T, 1)).astype(np.int32)
    objects = [{"id": "mug", "pos": np.tile(np.array([0.0, 0.0, 0.8]), (T, 1)),
                "rot": np.stack([np.eye(3)] * T), "bbox": obj_bbox}]
    write_clip(root, "vidB_001", T, left, right, conf, objects, lambda f, t: None)
    return T


def gen_vidC_001(root):
    """Both hands always NaN -> every label none; exercises hand_valid all-zero."""
    T = 65
    left = make_hand(T, None)
    right = make_hand(T, None)
    conf = np.zeros((T, 2), np.float32)
    obj_bbox = np.tile(np.array([0.4, 0.4, 0.5, 0.5]) * [W_IMG, H_IMG, W_IMG, H_IMG],
                       (T, 1)).astype(np.int32)
    objects = [{"id": "mug", "pos": np.tile(np.array([0.0, 0.0, 0.8]), (T, 1)),
                "rot": np.stack([np.eye(3)] * T), "bbox": obj_bbox}]
    write_clip(root, "vidC_001", T, left, right, conf, objects, lambda f, t: None)
    return T


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main():
    root = Path(tempfile.mkdtemp(prefix="ego_hoi_selftest_"))
    print(f"selftest root: {root}")
    lengths = {"vidA_001": gen_vidA_001(root), "vidA_002": gen_vidA_002(root),
               "vidB_001": gen_vidB_001(root), "vidC_001": gen_vidC_001(root)}

    print("[1] rot6d roundtrip")
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(256, 3, 3)))
    q *= np.sign(np.linalg.det(q))[:, None, None]
    err = np.abs(rot6d_to_matrix(matrix_to_rot6d(q)) - q).max()
    check(err < 1e-12, f"random rotation roundtrip err={err:.2e}")
    check(np.allclose(matrix_to_rot6d(np.eye(3)), [1, 0, 0, 0, 1, 0]), "identity -> [1,0,0,0,1,0]")

    print("[2] build_samples CLI")
    cmd = [sys.executable, "-m", "src.dataset.build_samples", "--data-root", str(root),
           "--window", str(WINDOW), "--stride", str(STRIDE), "--seed", "0"]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(r.stdout[-2000:])
    check(r.returncode == 0, f"build_samples exit 0\n{r.stderr[-2000:]}")

    samples_root = root / "samples"
    index = json.loads((samples_root / "dataset_index.json").read_text())
    meta = json.loads((samples_root / "meta.json").read_text())
    check(set(index) == {"train", "val", "test"}, "index top-level keys == {train,val,test}")
    check(meta["n_max"] == 2, f"n_max auto == 2 (got {meta['n_max']})")
    n_expected = sum(len(range(0, T - WINDOW, STRIDE)) for T in lengths.values())
    n_samples = sum(len(v) for v in index.values())
    check(n_samples == n_expected, f"sample count == {n_expected} (got {n_samples})")

    splits = meta["splits"]
    videos = set().union(*map(set, splits.values()))
    check(len(videos) == 3, f"3 source videos covered: {sorted(videos)}")
    check(not (set(splits["train"]) & set(splits["val"]))
          and not (set(splits["train"]) & set(splits["test"]))
          and not (set(splits["val"]) & set(splits["test"])), "splits disjoint")
    for split in ("train", "val", "test"):
        for e in index[split]:
            v = e["clip_id"].rsplit("_", 1)[0]
            assert v in splits[e["split"]], f"leakage: {e['sample_id']}"
            assert e["split"] == split, f"entry under wrong key: {e['sample_id']}"
    print("  ok: no video-level leakage for all samples")
    split_of = {c: next(s for s in ("train", "val", "test") if c.rsplit('_', 1)[0] in splits[s])
                for c in lengths}
    check(split_of["vidA_001"] == split_of["vidA_002"], "vidA clips share one split")

    print("[3] sample npz contract")
    for split in ("train", "val", "test"):
        for p in sorted((samples_root / split).glob("*.npz")):
            with np.load(p, allow_pickle=False) as d:
                check_shapes = (d["hand_seq"].shape == (WINDOW, 2, 21, 3)
                                and d["obj_seq"].shape == (WINDOW, 2, 13)
                                and d["hand_seq"].dtype == np.float32
                                and d["obj_seq"].dtype == np.float32)
                if not check_shapes:
                    raise AssertionError(f"{p}: bad shape/dtype "
                                         f"{d['hand_seq'].shape} {d['obj_seq'].shape}")
                assert d["obj_mask"].sum() in (1.0, 2.0)
                pad = d["obj_mask"] == 0
                if pad.any():
                    assert (d["obj_seq"][:, pad] == 0).all(), f"{p}: padding not zero"
                    assert (d["obj_valid"][:, pad] == 0).all()
                assert d["label_subject"] in (0, 1, 2)
                assert d["label_interaction_name"] in meta["interaction_vocab"]
    print(f"  ok: all {n_samples} samples shape/dtype/padding/label valid")

    print("[4] NaN + mask rules")
    a_samples = sorted((samples_root / split_of["vidA_001"]).glob("vidA_001_*.npz"))
    with np.load(a_samples[0]) as d:  # window 0..31, mug present, box NaN at 10-14
        check(np.isnan(d["hand_seq"][:, 0]).all(), "vidA_001 left hand all-NaN preserved")
        check((d["hand_valid"][:, 0] == 0).all(), "hand_valid left == 0")
        slot = list(d["object_ids"]).index("box")
        check((d["obj_valid"][10:15, slot] == 0).all(), "box obj_valid 0 on NaN frames")
        check(np.isnan(d["obj_seq"][10:15, slot]).all(), "box obj_seq NaN on NaN frames")
        check(np.isfinite(d["obj_seq"][:10, slot]).all(), "box obj_seq finite when present")
    c_samples = sorted((samples_root / split_of["vidC_001"]).glob("vidC_001_*.npz"))
    with np.load(c_samples[0]) as d:
        check((d["hand_valid"] == 0).all(), "vidC_001 hand_valid all 0")
        check(int(d["label_subject"]) == 2 and int(d["label_interaction"]) == 0,
              "vidC_001 label none/none")

    print("[5] heuristic labels")
    labs = {p.stem: str(np.load(p)["label_interaction_name"]) for p in a_samples}
    names = set(labs.values())
    check("hold" in names and "move" in names, f"vidA_001 has hold+move (got {sorted(names)})")
    with np.load(a_samples[0]) as d:  # next frame = 32: carrying not yet started -> hold
        check(d["label_interaction"].item() == HOLD, "vidA_001 w0 label == hold")
        check(str(d["label_object_id"]) == "mug" and int(d["label_subject"]) == 1,
              "vidA_001 w0 subject right, object mug")
    with np.load(a_samples[1]) as d:  # next frame = 48: being carried -> move
        check(d["label_interaction"].item() == MOVE, "vidA_002 w1 label == move")
    b_samples = sorted((samples_root / split_of["vidB_001"]).glob("vidB_001_*.npz"))
    b_names = {str(np.load(p)["label_interaction_name"]) for p in b_samples}
    check("reach" in b_names, f"vidB_001 has reach (got {sorted(b_names)})")

    print("[6] visualize CLI")
    cmd = [sys.executable, "-m", "src.dataset.visualize", "--data-root", str(root),
           "--clip-id", "vidA_001", "vidB_001"]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(r.stdout[-1000:])
    check(r.returncode == 0, f"visualize exit 0\n{r.stderr[-2000:]}")
    for cid in ("vidA_001", "vidB_001"):
        out = root / "vis" / f"{cid}.mp4"
        cap = cv2.VideoCapture(str(out))
        n = sum(1 for _ in iter(lambda: cap.read()[0], False))
        cap.release()
        check(out.exists() and n == lengths[cid], f"vis/{cid}.mp4 decodable, {n} frames")

    print("\nSELFTEST PASS")


if __name__ == "__main__":
    main()
