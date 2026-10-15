#!/usr/bin/env python3
"""Generate synthetic fixtures for tests/check_contracts.py.

Usage:
    python tests/make_fixtures.py <out_dir> [--seed 0]

Creates:
    <out_dir>/good/{clips,hand_pose,pose6d,samples}/...   contract-compliant
    <out_dir>/broken/<case>/...                           one violation each

Tiny scale: 2 clips, T=50, N=2, W=32. Pure numpy; clips mp4 are real videos
when ffmpeg is on PATH, otherwise placeholder files (ffprobe-based checks are
skipped by the checker when ffprobe is absent).

Follows the frozen contract in .agent/memory/03_data_pipeline.md (2026-10-15):
separated int64/str sample labels, obj_mask/obj_valid/hand_valid/object_ids,
clip_id/source_video/start_frame/fps provenance, conf NaN iff hand undetected,
bboxes -1 fill on undetected pose6d frames, fixed 6-column clips csv, and
clips/meta.json with dropped_residuals.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

T = 50
N_OBJ = 2
W = 32
FPS = 2.5
DUR_SEC = 20.0          # frames = 50, matches T; inside the 18-22 s target
CLIPS = (("demo01_001", "demo01"), ("demo02_001", "demo02"))
OBJECT_IDS = ("mug", "bottle")
INTERACTIONS = ("none", "reach", "grasp", "hold", "move", "place", "release")
SUBJECTS = ("left", "right", "none")
META = {"fps": FPS, "frames": T, "created_at": "2026-10-15T00:00:00Z",
        "env": "make_fixtures.py (synthetic)"}
SAMPLE_LABEL_KEYS = ("label_subject", "label_object_slot", "label_object_id",
                     "label_interaction", "label_interaction_name")


# ---------------------------------------------------------------- builders

def random_rot(rng, n):
    m = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(m)
    q *= np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    flip = np.linalg.det(q) < 0
    q[flip, :, 0] *= -1
    return q


def make_hand_npz(rng, nan_hand, nan_frames):
    left = rng.uniform(0, 1, (T, 21, 3)).astype(np.float32)
    right = rng.uniform(0, 1, (T, 21, 3)).astype(np.float32)
    left[..., 2] = rng.normal(0, 0.1, T).astype(np.float32)[:, None]
    right[..., 2] = rng.normal(0, 0.1, T).astype(np.float32)[:, None]
    conf = rng.uniform(0.5, 1.0, (T, 2)).astype(np.float32)
    if nan_hand == "left":
        left[nan_frames] = np.nan
        conf[nan_frames, 0] = np.nan    # contract: conf NaN iff hand undetected
    else:
        right[nan_frames] = np.nan
        conf[nan_frames, 1] = np.nan
    return {"left_hand": left, "right_hand": right, "conf": conf}


def make_pose_npz(rng, nan_obj, nan_frames):
    poses = np.zeros((N_OBJ, T, 4, 4), dtype=np.float32)
    poses[..., 3, 3] = 1.0
    rots = random_rot(rng, N_OBJ * T).reshape(N_OBJ, T, 3, 3)
    poses[..., :3, :3] = rots
    poses[..., :3, 3] = rng.uniform(-0.5, 0.5, (N_OBJ, T, 3))
    bboxes = np.zeros((N_OBJ, T, 4), dtype=np.int32)
    x1 = rng.integers(0, 100, (N_OBJ, T))
    y1 = rng.integers(0, 100, (N_OBJ, T))
    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2] = x1 + rng.integers(10, 50, (N_OBJ, T))
    bboxes[..., 3] = y1 + rng.integers(10, 50, (N_OBJ, T))
    poses[nan_obj, nan_frames] = np.nan
    bboxes[nan_obj, nan_frames] = -1    # contract: undetected frames fill -1 (int32)
    return {"object_ids": np.array(OBJECT_IDS), "poses": poses, "bboxes": bboxes}


def make_sample_npz(rng, n_max=N_OBJ, n_real=None, hand_nan=False,
                    interaction="grasp", clip_id="demo01_001", start_frame=0,
                    legacy_label=False, obj_nan_frames=(7, 8)):
    """One contract-compliant sample npz payload (frozen separated-label layout)."""
    if n_real is None:
        n_real = n_max
    hand_seq = rng.uniform(0, 1, (W, 2, 21, 3)).astype(np.float32)
    if hand_nan:
        hand_seq[5, 1] = np.nan  # right hand, whole frame -> compliant
    hand_valid = (~np.isnan(hand_seq).any(axis=(2, 3))).astype(np.float32)

    obj_seq = np.zeros((W, n_max, 13), dtype=np.float32)
    rots = random_rot(rng, n_real)
    pos0 = rng.uniform(-0.5, 0.5, (n_real, 3))
    for i in range(n_real):
        pos = pos0[i] + np.linspace(0, 0.05, W, dtype=np.float32)[:, None]
        obj_seq[:, i, 0:3] = pos
        obj_seq[:, i, 3:6] = rots[i][:, 0]
        obj_seq[:, i, 6:9] = rots[i][:, 1]
        obj_seq[:, i, 9] = float(rng.uniform(100, 2000))
        vel = np.zeros((W, 3), dtype=np.float32)
        vel[1:] = np.diff(pos, axis=0) * np.float32(FPS)  # frame diff x fps, t=0 is 0
        obj_seq[:, i, 10:13] = vel
    obj_mask = np.zeros(n_max, dtype=np.float32)
    obj_mask[:n_real] = 1.0
    obj_valid = np.zeros((W, n_max), dtype=np.float32)
    obj_valid[:, :n_real] = 1.0
    if n_real and obj_nan_frames:
        obj_seq[list(obj_nan_frames), 0] = np.nan   # pose missing -> whole row NaN
        obj_valid[list(obj_nan_frames), 0] = 0.0
    ids = list(OBJECT_IDS[:n_real]) + [""] * (n_max - n_real)

    itype = INTERACTIONS.index(interaction)
    if itype == 0 or n_real == 0:
        subject, slot, oid = 2, -1, ""
    else:
        subject, slot, oid = 0, 0, ids[0]
    out = {
        "hand_seq": hand_seq,
        "obj_seq": obj_seq,
        "obj_mask": obj_mask,
        "obj_valid": obj_valid,
        "hand_valid": hand_valid,
        "object_ids": np.array(ids),
        "clip_id": np.array(clip_id),
        "source_video": np.array(clip_id.rsplit("_", 1)[0]),
        "start_frame": np.int64(start_frame),
        "fps": np.float32(FPS),
    }
    if legacy_label:
        # pre-freeze draft layout; checker accepts with a warning
        out["label"] = np.array(json.dumps({
            "subject": SUBJECTS[subject],
            "object_id": oid if oid else "none",
            "interaction_type": INTERACTIONS[itype]}))
    else:
        out.update({
            "label_subject": np.int64(subject),
            "label_object_slot": np.int64(slot),
            "label_object_id": np.array(oid),
            "label_interaction": np.int64(itype),
            "label_interaction_name": np.array(INTERACTIONS[itype]),
        })
    return out


def write_meta(dir_path, clip_id):
    meta = dict(META, source_clip=clip_id)
    (dir_path / f"{clip_id}.meta.json").write_text(json.dumps(meta, indent=2))


def write_clips_meta(clips_dir):
    meta = dict(META, dropped_residuals=[
        {"source_video": "demo02", "start_sec": 20.0, "end_sec": 26.4, "frames": 16,
         "reason": "residual shorter than 18 s, dropped"}])
    (clips_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def write_samples_meta(samples_dir):
    meta = {"created_at": META["created_at"], "env": META["env"],
            "generator": "tests/make_fixtures.py", "contract_version": "2026-10-15",
            "window": W, "n_max": N_OBJ, "fps": FPS}
    (samples_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def write_clip_video(mp4_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-v", "error", "-y",
               "-f", "lavfi", "-i", f"color=c=black:s=64x64:r={FPS}:d={DUR_SEC}",
               "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(mp4_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    mp4_path.write_bytes(b"placeholder (ffmpeg not available)\n")
    return False


def write_clips_csv(clips_dir, rows, extra_column=False):
    header = "clip_id,source_video,start_sec,end_sec,fps,frames"
    if extra_column:
        header += ",note"
    lines = [header]
    for r in rows:
        line = ",".join(str(x) for x in r)
        if extra_column:
            line += ",synthetic"
        lines.append(line)
    (clips_dir / "clips_index.csv").write_text("\n".join(lines) + "\n")


def write_samples_case(case_dir, train, val=(), test=(), leak_video=None):
    """train/val/test: lists of (sample_id, npz_payload, source_video)."""
    base = case_dir / "samples"
    index = {"train": [], "val": [], "test": []}
    for split, spec in (("train", train), ("val", val), ("test", test)):
        (base / split).mkdir(parents=True, exist_ok=True)
        for sid, payload, src in spec:
            np.savez(base / split / f"{sid}.npz", **payload)
            index[split].append({"sample_id": sid,
                                 "source_video": leak_video or src,
                                 "clip_id": f"{src}_001", "start_frame": 0})
    (base / "dataset_index.json").write_text(json.dumps(index, indent=2))
    write_samples_meta(base)


# ---------------------------------------------------------------- good tree

def build_good(root):
    rng = np.random.default_rng(0)
    clips_dir = root / "clips"
    hand_dir = root / "hand_pose"
    pose_dir = root / "pose6d"
    samples_dir = root / "samples"
    for d in (clips_dir, hand_dir, pose_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    real_video = True
    rows = []
    for clip_id, src in CLIPS:
        real_video &= write_clip_video(clips_dir / f"{clip_id}.mp4")
        rows.append((clip_id, src, 0.0, DUR_SEC, FPS, T))
    write_clips_csv(clips_dir, rows)
    write_clips_meta(clips_dir)
    if not real_video:
        print("note: ffmpeg not found, clip mp4 are placeholders "
              "(checker skips ffprobe checks when ffprobe is absent)")

    for i, (clip_id, _src) in enumerate(CLIPS):
        np.savez(hand_dir / f"{clip_id}.npz",
                 **make_hand_npz(rng, "right" if i == 0 else "left",
                                 list(range(10, 13)) if i == 0 else list(range(30, 35))))
        write_meta(hand_dir, clip_id)
        np.savez(pose_dir / f"{clip_id}.npz",
                 **make_pose_npz(rng, 1, list(range(20, 25))))
        write_meta(pose_dir, clip_id)

    index = {"train": [], "val": [], "test": []}
    specs = [("train", "s000", "demo01_001", {}),
             ("train", "s001", "demo01_001",
              {"hand_nan": True, "interaction": "reach", "start_frame": 16}),
             ("val", "s002", "demo02_001",
              {"interaction": "none", "legacy_label": True}),
             ("test", "s003", "demo03_001",
              {"interaction": "hold", "n_real": 1})]
    for split, sid, clip_id, kw in specs:
        d = samples_dir / split
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{sid}.npz", **make_sample_npz(rng, clip_id=clip_id, **kw))
        index[split].append({"sample_id": sid, "source_video": clip_id.rsplit("_", 1)[0],
                             "clip_id": clip_id, "start_frame": kw.get("start_frame", 0)})
    (samples_dir / "dataset_index.json").write_text(json.dumps(index, indent=2))
    write_samples_meta(samples_dir)


# ---------------------------------------------------------------- broken tree

def build_broken(root):
    rng = np.random.default_rng(1)

    d = root / "handpose_partial_nan" / "hand_pose"
    d.mkdir(parents=True)
    kw = make_hand_npz(rng, "right", [])
    kw["right_hand"][7, :10] = np.nan
    np.savez(d / "bad_001.npz", **kw)

    d = root / "handpose_bad_dtype" / "hand_pose"
    d.mkdir(parents=True)
    kw = make_hand_npz(rng, "right", [])
    kw["left_hand"] = kw["left_hand"].astype(np.float64)
    np.savez(d / "bad_001.npz", **kw)

    d = root / "handpose_missing_conf" / "hand_pose"
    d.mkdir(parents=True)
    kw = make_hand_npz(rng, "right", [])
    del kw["conf"]
    np.savez(d / "bad_001.npz", **kw)

    d = root / "handpose_conf_mismatch" / "hand_pose"
    d.mkdir(parents=True)
    kw = make_hand_npz(rng, "right", [3, 4])
    kw["conf"][3:5, 1] = 0.9  # hand NaN but conf finite; rule: conf NaN iff undetected
    np.savez(d / "bad_001.npz", **kw)

    d = root / "pose6d_bad_rotation" / "pose6d"
    d.mkdir(parents=True)
    kw = make_pose_npz(rng, 1, [])
    kw["poses"][0, 3, :3, :3] *= 1.5
    np.savez(d / "bad_001.npz", **kw)

    d = root / "pose6d_partial_nan" / "pose6d"
    d.mkdir(parents=True)
    kw = make_pose_npz(rng, 1, [])
    kw["poses"][0, 5, 0, 0] = np.nan
    np.savez(d / "bad_001.npz", **kw)

    d = root / "pose6d_dup_ids" / "pose6d"
    d.mkdir(parents=True)
    kw = make_pose_npz(rng, 1, [])
    kw["object_ids"] = np.array(["mug", "mug"])
    np.savez(d / "bad_001.npz", **kw)

    kw = make_sample_npz(rng, clip_id="demo01_001")
    kw["label_interaction"] = np.int64(9)  # outside 0..6
    write_samples_case(root / "samples_bad_label",
                       [("b000", kw, "demo01")],
                       [("b001", make_sample_npz(rng, clip_id="demo02_001"), "demo02")])

    kw = make_sample_npz(rng, clip_id="demo01_001")
    kw["label_object_slot"] = np.int64(1)  # object_ids[1] == "bottle" != label_object_id "mug"
    write_samples_case(root / "samples_bad_slot",
                       [("b000", kw, "demo01")],
                       [("b001", make_sample_npz(rng, clip_id="demo02_001"), "demo02")])

    kw = make_sample_npz(rng, clip_id="demo01_001")
    del kw["obj_valid"]
    del kw["hand_valid"]
    write_samples_case(root / "samples_missing_keys",
                       [("b000", kw, "demo01")],
                       [("b001", make_sample_npz(rng, clip_id="demo02_001"), "demo02")])

    kw = make_sample_npz(rng, clip_id="demo01_001")
    for k in SAMPLE_LABEL_KEYS:
        del kw[k]
    write_samples_case(root / "samples_no_label",
                       [("b000", kw, "demo01")],
                       [("b001", make_sample_npz(rng, clip_id="demo02_001"), "demo02")])

    write_samples_case(root / "samples_split_leak",
                       [("b000", make_sample_npz(rng, clip_id="demo01_001"), "demo01")],
                       [("b001", make_sample_npz(rng, clip_id="demo02_001"), "demo02")],
                       leak_video="demo01")

    d = root / "samples_missing_index" / "samples" / "train"
    d.mkdir(parents=True)
    np.savez(d / "b000.npz", **make_sample_npz(rng, clip_id="demo01_001"))

    d = root / "clips_duplicate_id" / "clips"
    d.mkdir(parents=True)
    (d / "demo01_001.mp4").write_bytes(b"placeholder\n")
    write_clips_csv(d, [("demo01_001", "demo01", 0.0, DUR_SEC, FPS, T),
                        ("demo01_001", "demo01", 5.0, 25.0, FPS, T)])
    write_clips_meta(d)

    d = root / "clips_frames_mismatch" / "clips"
    d.mkdir(parents=True)
    (d / "demo01_001.mp4").write_bytes(b"placeholder\n")
    write_clips_csv(d, [("demo01_001", "demo01", 0.0, DUR_SEC, FPS, 999)])
    write_clips_meta(d)

    d = root / "clips_extra_column" / "clips"
    d.mkdir(parents=True)
    (d / "demo01_001.mp4").write_bytes(b"placeholder\n")
    write_clips_csv(d, [("demo01_001", "demo01", 0.0, DUR_SEC, FPS, T)], extra_column=True)
    write_clips_meta(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0, help="reserved; fixtures are deterministic")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"error: out_dir {out} exists and is not empty", file=sys.stderr)
        return 2
    build_good(out / "good")
    build_broken(out / "broken")
    print(f"fixtures written under {out}")


if __name__ == "__main__":
    sys.exit(main())
