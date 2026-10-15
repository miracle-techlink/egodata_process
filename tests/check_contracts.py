#!/usr/bin/env python3
"""Contract checker for the ego_hoi data pipeline (stages 1-4).

Usage:
    python tests/check_contracts.py clips     <clips_dir | clip.mp4>
    python tests/check_contracts.py hand_pose <npz | dir of npz>
    python tests/check_contracts.py pose6d    <npz | dir of npz>
    python tests/check_contracts.py samples   <samples root dir | sample.npz>

Source of truth: .agent/memory/03_data_pipeline.md (contract version 2026-10-15).
Exit code: 0 = pass (warnings allowed), 1 = any error, 2 = usage error.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

STAGES = ("clips", "hand_pose", "pose6d", "samples")

CLIPS_CSV_COLUMNS = ("clip_id", "source_video", "start_sec", "end_sec", "fps", "frames")
CLIP_ID_RE = re.compile(r"^.+_\d{3}$")
CLIP_MIN_SEC = 10.0          # deliverables: residual segments < 10 s must be dropped
CLIP_TARGET_SEC = (18.0, 22.0)  # "~20 s (18-22 acceptable)" -> warning outside

HAND_KEYS = ("left_hand", "right_hand", "conf")
POSE6D_KEYS = ("object_ids", "poses", "bboxes")

SPLITS = ("train", "val", "test")
INTERACTION_TYPES = ("none", "reach", "grasp", "hold", "move", "place", "release")
SUBJECTS = ("left", "right", "none")   # index order: 0=left, 1=right, 2=none
DEFAULT_W = 32               # contract default window length
ROT6D_TOL = 0.05
ROT_MAT_TOL = 1e-3
HOMO_ROW_TOL = 1e-4

META_REQUIRED_KEYS = ("fps", "frames", "source_clip", "created_at", "env")

# ④ sample npz: frozen separated-label layout (contract 2026-10-15)
SAMPLE_LABEL_KEYS = ("label_subject", "label_object_slot", "label_object_id",
                     "label_interaction", "label_interaction_name")
SAMPLE_STRUCT_KEYS = ("hand_seq", "obj_seq", "obj_mask", "obj_valid", "hand_valid",
                      "object_ids", "clip_id", "source_video", "start_frame", "fps")
SAMPLE_KEYS = SAMPLE_STRUCT_KEYS + SAMPLE_LABEL_KEYS


class Report:
    def __init__(self):
        self.entries = []   # (level, file, key, message)
        self.n_files = 0

    def error(self, file, key, msg):
        self.entries.append(("ERROR", str(file), str(key), msg))

    def warn(self, file, key, msg):
        self.entries.append(("WARN", str(file), str(key), msg))

    def info(self, file, key, msg):
        self.entries.append(("INFO", str(file), str(key), msg))

    @property
    def n_errors(self):
        return sum(1 for e in self.entries if e[0] == "ERROR")

    @property
    def n_warnings(self):
        return sum(1 for e in self.entries if e[0] == "WARN")

    def render(self, stage, path):
        lines = [f"== check_contracts :: stage={stage} path={path} ==",
                 f"checked {self.n_files} file(s)"]
        if not self.entries:
            lines.append("all checks passed, no issues found")
        for level, file, key, msg in self.entries:
            lines.append(f"[{level}] {file} :: {key} :: {msg}")
        verdict = "FAIL" if self.n_errors else "PASS"
        lines.append(f"== SUMMARY: {self.n_files} file(s), "
                     f"{self.n_errors} error(s), {self.n_warnings} warning(s) -> {verdict} ==")
        return "\n".join(lines)


# ---------------------------------------------------------------- helpers

def load_npz(path, rep):
    """Return dict of arrays, or None (report already filled)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            try:
                return {k: z[k] for k in z.files}
            except ValueError:
                pass
    except Exception as e:
        rep.error(path, "-", f"cannot read npz: {e}")
        return None
    rep.warn(path, "-", "contains pickled object arrays; loaded with allow_pickle=True")
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        rep.error(path, "-", f"cannot read npz: {e}")
        return None


def check_keys(data, required, path, rep):
    missing = [k for k in required if k not in data]
    extra = [k for k in data if k not in required]
    for k in missing:
        rep.error(path, k, "required key missing from npz")
    for k in extra:
        rep.warn(path, k, "extra key not in contract")
    return not missing


def check_dtype(arr, expect, path, key, rep):
    if arr.dtype != np.dtype(expect):
        rep.error(path, key, f"dtype {arr.dtype} != contract {expect}")
        return False
    return True


def check_whole_frame_nan(arr, frame_axes, path, key, rep):
    """NaN must fill whole frames; partial-frame NaN is an error. Inf is an error."""
    if np.isinf(arr).any():
        rep.error(path, key, "contains Inf values")
    nan_any = np.isnan(arr).any(axis=frame_axes)
    nan_all = np.isnan(arr).all(axis=frame_axes)
    partial = nan_any & ~nan_all
    if partial.any():
        idx = np.argwhere(partial)[:5]
        rep.error(path, key,
                  f"partial-frame NaN at {len(np.argwhere(partial))} frame(s), first indices "
                  f"{[tuple(i) for i in idx.tolist()]}; rule: whole frame NaN or none")


def scalar_str(arr):
    """Extract a python str from a 0-d / size-1 string or bytes array."""
    a = np.asarray(arr)
    if a.size != 1:
        raise ValueError(f"expected scalar string array, got shape {a.shape}")
    v = a.reshape(-1)[0]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, str):
        return v
    raise ValueError(f"expected string scalar, got {type(v).__name__}")


def check_meta_sidecar(npz_path, rep):
    """meta.json is contract-required; key names are not pinned by the contract,
    so issues are warnings (see task07 notes)."""
    meta_path = npz_path.with_suffix("").with_name(npz_path.stem + ".meta.json")
    if not meta_path.exists():
        rep.warn(npz_path, "meta.json",
                 f"sidecar {meta_path.name} missing; expected keys {list(META_REQUIRED_KEYS)}")
        return
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        rep.error(meta_path, "-", f"meta.json is not valid JSON: {e}")
        return
    for k in META_REQUIRED_KEYS:
        if k not in meta:
            rep.warn(meta_path, k, "recommended meta key missing")


def ffprobe_json(path):
    cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
           "-show_entries", "stream=nb_read_frames,nb_frames:format=duration",
           "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "ffprobe failed")
    return json.loads(out.stdout)


# ---------------------------------------------------------------- stage 1: clips

def _check_clip_row(row, lineno, csv_path, rep):
    cid = row.get("clip_id", "")
    fkey = f"row {lineno} ({cid or '?'})"
    if not CLIP_ID_RE.match(cid):
        rep.error(csv_path, fkey, f"clip_id '{cid}' does not match '<video>_<3-digit>'")
    for col in ("start_sec", "end_sec", "fps", "frames"):
        try:
            float(row[col])
        except (TypeError, ValueError):
            rep.error(csv_path, fkey, f"column '{col}' not numeric: {row.get(col)!r}")
            return None
    start, end, fps = float(row["start_sec"]), float(row["end_sec"]), float(row["fps"])
    frames = float(row["frames"])
    if end <= start:
        rep.error(csv_path, fkey, f"end_sec ({end}) <= start_sec ({start})")
    if fps <= 0:
        rep.error(csv_path, fkey, f"fps ({fps}) <= 0")
    if frames <= 0 or frames != int(frames):
        rep.error(csv_path, fkey, f"frames ({frames}) must be a positive integer")
    dur = end - start
    if dur < CLIP_MIN_SEC:
        rep.error(csv_path, fkey,
                  f"duration {dur:.2f}s < {CLIP_MIN_SEC:.0f}s; residual segments must be dropped")
    elif not (CLIP_TARGET_SEC[0] <= dur <= CLIP_TARGET_SEC[1]):
        rep.warn(csv_path, fkey,
                 f"duration {dur:.2f}s outside target {CLIP_TARGET_SEC[0]:.0f}-{CLIP_TARGET_SEC[1]:.0f}s")
    if fps > 0:
        expect_frames = dur * fps
        if abs(frames - expect_frames) > max(2.0, 0.02 * expect_frames):
            rep.error(csv_path, fkey,
                      f"frames={int(frames)} inconsistent with (end-start)*fps={expect_frames:.1f}")
    return cid


def _check_clips_meta(clips_dir, rep):
    """Contract: clips/meta.json must exist and record dropped residual segments."""
    meta_path = clips_dir / "meta.json"
    if not meta_path.exists():
        rep.error(clips_dir, "meta.json",
                  "clips/meta.json missing; contract requires it (with dropped_residuals)")
        return
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        rep.error(meta_path, "-", f"meta.json is not valid JSON: {e}")
        return
    if "dropped_residuals" not in meta:
        rep.error(meta_path, "dropped_residuals",
                  "key missing; dropped residual segments must be recorded here (not in csv)")
    elif not isinstance(meta["dropped_residuals"], list):
        rep.error(meta_path, "dropped_residuals",
                  f"must be a list, got {type(meta['dropped_residuals']).__name__}")


def _ffprobe_spot_check(clips_dir, rows, rep):
    probe = shutil.which("ffprobe")
    if probe is None:
        rep.info(clips_dir, "ffprobe", "ffprobe not found; duration/frame spot-check skipped")
        return
    idxs = sorted({0, len(rows) // 2, len(rows) - 1})
    for i in idxs:
        cid, csv_dur, csv_frames = rows[i]
        mp4 = clips_dir / f"{cid}.mp4"
        if not mp4.exists():
            continue  # already reported by existence check
        try:
            info = ffprobe_json(mp4)
        except Exception as e:
            rep.error(mp4, "ffprobe", f"unreadable by ffprobe: {e}")
            continue
        try:
            dur = float(info["format"]["duration"])
        except (KeyError, TypeError, ValueError):
            rep.error(mp4, "ffprobe", "no format duration in ffprobe output")
            continue
        if abs(dur - csv_dur) > 0.5:
            rep.error(mp4, "ffprobe",
                      f"probed duration {dur:.2f}s vs csv {csv_dur:.2f}s (tol 0.5s)")
        n = None
        for stream in info.get("streams", []):
            for k in ("nb_read_frames", "nb_frames"):
                v = stream.get(k)
                if v and str(v).isdigit():
                    n = int(v)
                    break
            if n is not None:
                break
        if n is None:
            rep.warn(mp4, "ffprobe", "frame count not reported; only duration checked")
        elif abs(n - csv_frames) > 2:
            rep.error(mp4, "ffprobe",
                      f"probed frames {n} vs csv {int(csv_frames)} (tol 2)")


def check_clips(path, rep):
    path = Path(path)
    if path.is_file():
        rep.n_files = 1
        if not CLIP_ID_RE.match(path.stem):
            rep.warn(path, "clip_id", f"filename '{path.stem}' does not match '<video>_<3-digit>'")
        if shutil.which("ffprobe"):
            try:
                info = ffprobe_json(path)
                dur = float(info["format"]["duration"])
                if dur < CLIP_MIN_SEC:
                    rep.error(path, "ffprobe", f"duration {dur:.2f}s < {CLIP_MIN_SEC:.0f}s")
            except Exception as e:
                rep.error(path, "ffprobe", f"unreadable by ffprobe: {e}")
        else:
            rep.warn(path, "ffprobe", "ffprobe not found; single-file check is name-only")
        return

    _check_clips_meta(path, rep)
    csv_path = path / "clips_index.csv"
    if not csv_path.exists():
        rep.error(path, "clips_index.csv", "index csv missing in clips dir")
        rep.n_files = 0
        return
    rep.n_files = 1
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        # contract: fixed 6 columns
        missing_cols = [c for c in CLIPS_CSV_COLUMNS if c not in header]
        extra_cols = [c for c in header if c not in CLIPS_CSV_COLUMNS]
        for c in missing_cols:
            rep.error(csv_path, c, "required column missing from clips_index.csv (fixed 6 columns)")
        for c in extra_cols:
            rep.error(csv_path, c, "extra column in clips_index.csv; contract fixes exactly "
                                   f"{len(CLIPS_CSV_COLUMNS)} columns {list(CLIPS_CSV_COLUMNS)}")
        if not missing_cols and not extra_cols and list(header) != list(CLIPS_CSV_COLUMNS):
            rep.warn(csv_path, "-", f"column order {list(header)} differs from contract "
                                    f"{list(CLIPS_CSV_COLUMNS)}")
        rows_raw = list(reader)
    if not rows_raw:
        rep.error(csv_path, "-", "clips_index.csv has no data rows")
        return
    if missing_cols:
        return

    seen = {}
    probe_rows = []
    for lineno, row in enumerate(rows_raw, start=2):
        cid = _check_clip_row(row, lineno, csv_path, rep)
        if cid is None:
            continue
        if cid in seen:
            rep.error(csv_path, f"row {lineno}",
                      f"duplicate clip_id '{cid}' (first seen row {seen[cid]})")
        seen[cid] = lineno
        mp4 = path / f"{cid}.mp4"
        if not mp4.exists():
            rep.error(csv_path, f"row {lineno}", f"clip file {cid}.mp4 missing in {path}")
        probe_rows.append((cid, float(row["end_sec"]) - float(row["start_sec"]),
                           float(row["frames"])))
    listed = set(seen)
    for mp4 in sorted(path.glob("*.mp4")):
        if mp4.stem not in listed:
            rep.warn(mp4, "clip_id", "mp4 file not listed in clips_index.csv")
    if probe_rows:
        _ffprobe_spot_check(path, probe_rows, rep)


# ---------------------------------------------------------------- stage 2: hand_pose

def check_hand_pose_file(npz_path, rep):
    data = load_npz(npz_path, rep)
    if data is None:
        return
    if not check_keys(data, HAND_KEYS, npz_path, rep):
        return
    left, right, conf = data["left_hand"], data["right_hand"], data["conf"]
    ok = True
    for key, arr in (("left_hand", left), ("right_hand", right)):
        ok &= check_dtype(arr, np.float32, npz_path, key, rep)
        if arr.ndim != 3 or arr.shape[1:] != (21, 3):
            rep.error(npz_path, key, f"shape {arr.shape} != (T, 21, 3)")
            ok = False
    ok &= check_dtype(conf, np.float32, npz_path, "conf", rep)
    if conf.ndim != 2 or conf.shape[1] != 2:
        rep.error(npz_path, "conf", f"shape {conf.shape} != (T, 2)")
        ok = False
    if not ok:
        return
    t_hand, t_conf = left.shape[0], conf.shape[0]
    if right.shape[0] != t_hand or t_conf != t_hand:
        rep.error(npz_path, "-",
                  f"T mismatch: left_hand {t_hand}, right_hand {right.shape[0]}, conf {t_conf}")
    if t_hand < 1:
        rep.error(npz_path, "-", "T == 0")
    # contract: undetected (incl. low-confidence) frames are whole-frame NaN, and
    # conf is NaN exactly on those frames (conf NaN iff that hand is undetected)
    hand_nan = {}
    for key, arr in (("left_hand", left), ("right_hand", right)):
        check_whole_frame_nan(arr, (1, 2), npz_path, key, rep)
        nan_frames = np.isnan(arr).all(axis=(1, 2))
        hand_nan[key] = nan_frames
        if nan_frames.all():
            rep.warn(npz_path, key, "hand never detected in any frame (all NaN)")
    if np.isinf(conf).any():
        rep.error(npz_path, "conf", "contains Inf values")
    for i, key in enumerate(("left_hand", "right_hand")):
        c = conf[:, i]
        c_nan = np.isnan(c)
        mismatch = c_nan != hand_nan[key]
        if mismatch.any():
            idx = np.argwhere(mismatch).reshape(-1)[:5].tolist()
            rep.error(npz_path, "conf",
                      f"conf[:, {i}] NaN does not coincide with {key} undetected frames at "
                      f"{int(mismatch.sum())} frame(s), first {idx}; rule: conf NaN iff hand "
                      f"undetected")
        finite = c[~c_nan]
        if finite.size and ((finite < 0) | (finite > 1)).any():
            rep.warn(npz_path, "conf", f"conf[:, {i}] has values outside [0, 1]")
    check_meta_sidecar(npz_path, rep)


def check_hand_pose(path, rep):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    if not files:
        rep.error(path, "-", "no .npz files found")
        return
    rep.n_files = len(files)
    for f in files:
        check_hand_pose_file(f, rep)


# ---------------------------------------------------------------- stage 3: pose6d

def check_pose6d_file(npz_path, rep):
    data = load_npz(npz_path, rep)
    if data is None:
        return
    if not check_keys(data, POSE6D_KEYS, npz_path, rep):
        return
    object_ids, poses, bboxes = data["object_ids"], data["poses"], data["bboxes"]
    ok = True
    if object_ids.ndim != 1 or object_ids.shape[0] < 1:
        rep.error(npz_path, "object_ids", f"shape {object_ids.shape} != (N,) with N>=1")
        ok = False
    elif object_ids.dtype.kind not in "USO":
        rep.error(npz_path, "object_ids", f"dtype {object_ids.dtype} is not a string dtype")
        ok = False
    else:
        ids = [str(x) for x in object_ids.tolist()]
        dup = sorted({x for x in ids if ids.count(x) > 1})
        if dup:
            rep.error(npz_path, "object_ids", f"duplicate object id(s): {dup}")
    ok &= check_dtype(poses, np.float32, npz_path, "poses", rep)
    ok &= check_dtype(bboxes, np.int32, npz_path, "bboxes", rep)
    if poses.ndim != 4 or poses.shape[2:] != (4, 4):
        rep.error(npz_path, "poses", f"shape {poses.shape} != (N, T, 4, 4)")
        ok = False
    if bboxes.ndim != 3 or bboxes.shape[2] != 4:
        rep.error(npz_path, "bboxes", f"shape {bboxes.shape} != (N, T, 4)")
        ok = False
    if not ok:
        return
    n_obj = object_ids.shape[0]
    if poses.shape[0] != n_obj or bboxes.shape[0] != n_obj:
        rep.error(npz_path, "-",
                  f"N mismatch: object_ids {n_obj}, poses {poses.shape[0]}, bboxes {bboxes.shape[0]}")
    if poses.shape[1] != bboxes.shape[1]:
        rep.error(npz_path, "-",
                  f"T mismatch: poses {poses.shape[1]}, bboxes {bboxes.shape[1]}")

    nan_any = np.isnan(poses).any(axis=(2, 3))
    nan_all = np.isnan(poses).all(axis=(2, 3))
    partial = nan_any & ~nan_all
    if partial.any():
        idx = np.argwhere(partial)[:5]
        rep.error(npz_path, "poses",
                  f"partial-frame NaN at {int(partial.sum())} (obj,frame), first "
                  f"{[tuple(i) for i in idx.tolist()]}; rule: whole 4x4 NaN or none")
    if np.isinf(poses).any():
        rep.error(npz_path, "poses", "contains Inf values")
    valid = ~nan_any
    if not valid.any():
        rep.warn(npz_path, "poses", "no valid (non-NaN) poses at all")
    else:
        mats = poses[valid].astype(np.float64)
        homo = mats[:, 3, :]
        if not np.allclose(homo, np.array([0, 0, 0, 1.0]), atol=HOMO_ROW_TOL):
            bad = int((np.abs(homo - np.array([0, 0, 0, 1.0])) > HOMO_ROW_TOL).any(axis=1).sum())
            rep.error(npz_path, "poses",
                      f"{bad} valid matrix/matrices have last row != [0,0,0,1] "
                      f"(tol {HOMO_ROW_TOL})")
        r = mats[:, :3, :3]
        rt_r = np.einsum("...ji,...jk->...ik", r, r)
        eye = np.eye(3)
        dev = np.abs(rt_r - eye).max(axis=(1, 2))
        if (dev > ROT_MAT_TOL).any():
            i = int(np.argmax(dev))
            rep.error(npz_path, "poses",
                      f"rotation not orthonormal: max |R^T R - I| = {dev[i]:.3e} "
                      f"(tol {ROT_MAT_TOL}), {int((dev > ROT_MAT_TOL).sum())} frame(s)")
        det = np.linalg.det(r)
        if (det <= 0).any():
            rep.error(npz_path, "poses",
                      f"{int((det <= 0).sum())} rotation(s) with det<=0 (reflection?)")

    # contract: bboxes are int32, undetected frames are filled with -1
    # (int32 cannot hold NaN) — that fill is explicitly allowed
    bb = bboxes.astype(np.int64)
    bbox_fill = (bb == -1).all(axis=2)          # (N, T) undetected-frame fill
    if nan_any.shape == bbox_fill.shape:
        missing_fill = nan_any & ~bbox_fill
        if missing_fill.any():
            rep.warn(npz_path, "bboxes",
                     f"{int(missing_fill.sum())} undetected (pose-NaN) (obj,frame) pair(s) "
                     f"without the all -1 fill required by the contract")
        ghost_fill = valid & bbox_fill
        if ghost_fill.any():
            rep.warn(npz_path, "bboxes",
                     f"all -1 bbox fill on {int(ghost_fill.sum())} (obj,frame) pair(s) "
                     f"whose pose is valid")
    detected = valid & ~bbox_fill
    ordered = (bb[..., 0] <= bb[..., 2]) & (bb[..., 1] <= bb[..., 3])
    bad_order = (~ordered) & detected
    if bad_order.any():
        idx = np.argwhere(bad_order)[:5]
        rep.error(npz_path, "bboxes",
                  f"x1>x2 or y1>y2 at {int(bad_order.sum())} detected (obj,frame), first "
                  f"{[tuple(i) for i in idx.tolist()]}")
    neg = (bb < 0).any(axis=2) & detected
    if neg.any():
        rep.warn(npz_path, "bboxes",
                 f"negative coordinates at {int(neg.sum())} detected (obj,frame) pairs "
                 f"(-1 fill on undetected frames is allowed)")
    check_meta_sidecar(npz_path, rep)


def check_pose6d(path, rep):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    if not files:
        rep.error(path, "-", "no .npz files found")
        return
    rep.n_files = len(files)
    for f in files:
        check_pose6d_file(f, rep)


# ---------------------------------------------------------------- stage 4: samples

def _label_int(data, key, path, rep):
    """Frozen contract: label_subject/label_object_slot/label_interaction are int64 scalars."""
    arr = np.asarray(data[key])
    if arr.dtype != np.dtype(np.int64):
        rep.error(path, key, f"dtype {arr.dtype} != contract int64")
        return None
    if arr.size != 1:
        rep.error(path, key, f"expected scalar, got shape {arr.shape}")
        return None
    return int(arr.reshape(-1)[0])


def _label_str(data, key, path, rep):
    try:
        return scalar_str(data[key])
    except ValueError as e:
        rep.error(path, key, str(e))
        return None


def _check_legacy_label(raw, path, rep):
    """Pre-freeze draft layout: single `label` key with a JSON string / pickled dict
    of {subject, object_id, interaction_type}. Accepted for compatibility but warned."""
    try:
        if raw.dtype == object:
            obj = raw.item() if raw.size == 1 else None
            if not isinstance(obj, dict):
                raise ValueError("label object is not a dict")
        else:
            obj = json.loads(scalar_str(raw))
        subject, object_id, itype = (str(obj["subject"]), str(obj["object_id"]),
                                     str(obj["interaction_type"]))
    except Exception as e:
        rep.error(path, "label", f"unparseable legacy label: {e}")
        return
    if subject not in SUBJECTS:
        rep.error(path, "label.subject", f"'{subject}' not in {list(SUBJECTS)}")
    if itype not in INTERACTION_TYPES:
        rep.error(path, "label.interaction_type",
                  f"'{itype}' not in {list(INTERACTION_TYPES)}")
    if itype == "none":
        if subject != "none" or object_id not in ("", "none"):
            rep.warn(path, "label",
                     "interaction_type 'none' but subject/object_id not none")
    elif subject == "none" or object_id in ("", "none"):
        rep.warn(path, "label",
                 f"subject/object_id none but interaction_type '{itype}'")


def _check_labels(data, ids, obj_mask, n_max, path, rep):
    """Canonical frozen layout: label_subject (int64: 0=left,1=right,2=none),
    label_object_slot (int64, -1=none), label_object_id (str, ""=none),
    label_interaction (int64, 0-6), label_interaction_name (str)."""
    present = [k for k in SAMPLE_LABEL_KEYS if k in data]
    if len(present) != len(SAMPLE_LABEL_KEYS):
        if present:
            rep.error(path, "label",
                      f"partial label keys {present}; contract requires all of "
                      f"{'/'.join(SAMPLE_LABEL_KEYS)}")
        elif "label" in data:
            rep.warn(path, "label",
                     "legacy single 'label' key; frozen contract requires separated keys "
                     + "/".join(SAMPLE_LABEL_KEYS))
            _check_legacy_label(data["label"], path, rep)
        else:
            rep.error(path, "label",
                      "label keys missing; contract requires " + "/".join(SAMPLE_LABEL_KEYS))
        return
    subj = _label_int(data, "label_subject", path, rep)
    slot = _label_int(data, "label_object_slot", path, rep)
    itype = _label_int(data, "label_interaction", path, rep)
    oid = _label_str(data, "label_object_id", path, rep)
    iname = _label_str(data, "label_interaction_name", path, rep)
    if None in (subj, slot, itype, oid, iname):
        return
    if subj not in (0, 1, 2):
        rep.error(path, "label_subject", f"{subj} not in {{0: left, 1: right, 2: none}}")
    if not 0 <= itype < len(INTERACTION_TYPES):
        rep.error(path, "label_interaction",
                  f"{itype} outside 0..{len(INTERACTION_TYPES) - 1} {list(INTERACTION_TYPES)}")
    if iname not in INTERACTION_TYPES:
        rep.error(path, "label_interaction_name", f"'{iname}' not in {list(INTERACTION_TYPES)}")
    elif 0 <= itype < len(INTERACTION_TYPES) and INTERACTION_TYPES[itype] != iname:
        rep.error(path, "label_interaction_name",
                  f"'{iname}' inconsistent with label_interaction={itype} "
                  f"('{INTERACTION_TYPES[itype]}')")
    if slot < -1 or slot >= n_max:
        rep.error(path, "label_object_slot", f"{slot} outside [-1, {n_max})")
    elif slot == -1:
        if oid != "":
            rep.error(path, "label_object_id",
                      f"must be \"\" when label_object_slot is -1, got '{oid}'")
    else:
        if obj_mask is not None and obj_mask[slot] != 1:
            rep.error(path, "label_object_slot",
                      f"slot {slot} is a padding slot (obj_mask=0)")
        if ids is not None and oid != ids[slot]:
            rep.error(path, "label_object_id",
                      f"'{oid}' != object_ids[{slot}] '{ids[slot]}'")
    if itype == 0:
        if subj != 2 or slot != -1:
            rep.warn(path, "label",
                     f"interaction none (0) but label_subject={subj}, label_object_slot={slot}")
    elif subj == 2 or slot == -1:
        rep.warn(path, "label",
                 f"interaction {itype} ('{INTERACTION_TYPES[itype]}') but subject/slot marks none")


def _check_binary(arr, path, key, rep):
    bad = ~((arr == 0) | (arr == 1))
    if bad.any():
        rep.error(path, key, f"values must be 0/1; {int(bad.sum())} entr(ies) otherwise")


def _check_provenance(data, path, rep):
    """clip_id / source_video / start_frame / fps traceability fields."""
    clip_id = None
    if "clip_id" in data:
        try:
            clip_id = scalar_str(data["clip_id"])
        except ValueError as e:
            rep.error(path, "clip_id", str(e))
        else:
            if not CLIP_ID_RE.match(clip_id):
                rep.error(path, "clip_id",
                          f"'{clip_id}' does not match '<video>_<3-digit>'")
    if "source_video" in data:
        try:
            sv = scalar_str(data["source_video"])
        except ValueError as e:
            rep.error(path, "source_video", str(e))
        else:
            if not sv:
                rep.error(path, "source_video", "empty source_video")
            elif clip_id and sv != clip_id.rsplit("_", 1)[0]:
                rep.error(path, "source_video",
                          f"'{sv}' != clip_id.rsplit('_', 1)[0] of '{clip_id}'")
    if "start_frame" in data:
        arr = np.asarray(data["start_frame"])
        if arr.dtype.kind not in "iu" or arr.size != 1:
            rep.error(path, "start_frame",
                      f"expected integer scalar, got dtype {arr.dtype} shape {arr.shape}")
        elif int(arr.reshape(-1)[0]) < 0:
            rep.error(path, "start_frame", f"negative start_frame {int(arr.reshape(-1)[0])}")
    if "fps" in data:
        arr = np.asarray(data["fps"])
        if arr.dtype.kind not in "fiu" or arr.size != 1:
            rep.error(path, "fps",
                      f"expected numeric scalar, got dtype {arr.dtype} shape {arr.shape}")
        elif float(arr.reshape(-1)[0]) <= 0:
            rep.error(path, "fps", f"fps {float(arr.reshape(-1)[0])} <= 0")


def check_sample_file(npz_path, rep, ws, nmaxs):
    data = load_npz(npz_path, rep)
    if data is None:
        return
    for k in SAMPLE_STRUCT_KEYS:
        if k not in data:
            rep.error(npz_path, k, "required key missing from sample npz")
    for k in data:
        if k not in SAMPLE_KEYS and k != "label":
            rep.warn(npz_path, k, "extra key not in contract")
    if "hand_seq" not in data or "obj_seq" not in data:
        return
    hand_seq, obj_seq = data["hand_seq"], data["obj_seq"]
    ok = check_dtype(hand_seq, np.float32, npz_path, "hand_seq", rep)
    ok &= check_dtype(obj_seq, np.float32, npz_path, "obj_seq", rep)
    if hand_seq.ndim != 4 or hand_seq.shape[1:] != (2, 21, 3):
        rep.error(npz_path, "hand_seq", f"shape {hand_seq.shape} != (W, 2, 21, 3)")
        ok = False
    if obj_seq.ndim != 3 or obj_seq.shape[2] != 13:
        rep.error(npz_path, "obj_seq", f"shape {obj_seq.shape} != (W, N_max, 13)")
        ok = False
    if not ok:
        return
    w, n_max = obj_seq.shape[0], obj_seq.shape[1]
    if hand_seq.shape[0] != w:
        rep.error(npz_path, "-", f"W mismatch: hand_seq {hand_seq.shape[0]}, obj_seq {w}")
    if w < 2:
        rep.error(npz_path, "-", f"W={w} too small (<2)")
    if n_max < 1:
        rep.error(npz_path, "obj_seq", "N_max == 0")
    ws.add(w)
    nmaxs.add(n_max)

    obj_mask = obj_valid = hand_valid = object_ids = None
    if "obj_mask" in data:
        arr = data["obj_mask"]
        good = check_dtype(arr, np.float32, npz_path, "obj_mask", rep)
        if arr.ndim != 1 or arr.shape[0] != n_max:
            rep.error(npz_path, "obj_mask", f"shape {arr.shape} != ({n_max},)")
            good = False
        if good:
            _check_binary(arr, npz_path, "obj_mask", rep)
            obj_mask = arr
    if "obj_valid" in data:
        arr = data["obj_valid"]
        good = check_dtype(arr, np.float32, npz_path, "obj_valid", rep)
        if arr.shape != (w, n_max):
            rep.error(npz_path, "obj_valid", f"shape {arr.shape} != ({w}, {n_max})")
            good = False
        if good:
            _check_binary(arr, npz_path, "obj_valid", rep)
            obj_valid = arr
    if "hand_valid" in data:
        arr = data["hand_valid"]
        good = check_dtype(arr, np.float32, npz_path, "hand_valid", rep)
        if arr.shape != (w, 2):
            rep.error(npz_path, "hand_valid", f"shape {arr.shape} != ({w}, 2)")
            good = False
        if good:
            _check_binary(arr, npz_path, "hand_valid", rep)
            hand_valid = arr
    if "object_ids" in data:
        arr = data["object_ids"]
        if arr.ndim != 1 or arr.shape[0] != n_max:
            rep.error(npz_path, "object_ids", f"shape {arr.shape} != ({n_max},)")
        elif arr.dtype.kind not in "US":
            rep.error(npz_path, "object_ids", f"dtype {arr.dtype} is not a string dtype")
        else:
            object_ids = arr

    # NaN rules: upstream NaN is preserved whole-frame; valid flags mirror NaN exactly
    check_whole_frame_nan(hand_seq, (2, 3), npz_path, "hand_seq", rep)
    if np.isinf(hand_seq).any():
        rep.error(npz_path, "hand_seq", "contains Inf values")
    hand_nan = np.isnan(hand_seq).all(axis=(2, 3))          # (W, 2)
    if hand_valid is not None:
        mm = (hand_valid == 0) != hand_nan
        if mm.any():
            rep.error(npz_path, "hand_valid",
                      f"hand_valid==0 must coincide with all-NaN hand frames; "
                      f"mismatch at {int(mm.sum())} (w, hand) pair(s)")

    nan_any = np.isnan(obj_seq).any(axis=2)
    nan_all = np.isnan(obj_seq).all(axis=2)
    partial = nan_any & ~nan_all
    if partial.any():
        rep.error(npz_path, "obj_seq",
                  f"partial NaN at {int(partial.sum())} (w,obj); rule: whole 13-dim row "
                  f"NaN or none")
    if np.isinf(obj_seq).any():
        rep.error(npz_path, "obj_seq", "contains Inf values")
    if obj_mask is not None:
        pad = obj_mask == 0
        if obj_valid is not None:
            if (obj_valid[:, pad] != 0).any():
                rep.error(npz_path, "obj_valid",
                          "obj_valid must be 0 on padding slots (obj_mask==0)")
            mm = (obj_valid[:, ~pad] == 0) != nan_all[:, ~pad]
            if mm.any():
                rep.error(npz_path, "obj_valid",
                          f"obj_valid==0 must coincide with all-NaN obj_seq rows on real "
                          f"slots; mismatch at {int(mm.sum())} (w, obj) pair(s)")
    ids = None
    if object_ids is not None:
        ids = [str(x) for x in object_ids.tolist()]
        non_empty = [s for s in ids if s != ""]
        dup = sorted({s for s in non_empty if non_empty.count(s) > 1})
        if dup:
            rep.error(npz_path, "object_ids", f"duplicate object id(s): {dup}")
        if obj_mask is not None:
            empty = np.array([s == "" for s in ids])
            if (empty != (obj_mask == 0)).any():
                rep.error(npz_path, "object_ids",
                          "padding slots (obj_mask==0) must have \"\" object id and real "
                          "slots a non-empty one")

    # rot6d orthonormality + bbox area on observed rows of real slots
    if obj_mask is not None and obj_valid is not None:
        sel = (obj_valid == 1) & (obj_mask == 1)[None, :]
        rows = obj_seq[sel]
    else:
        rows = obj_seq.reshape(-1, 13)
        rows = rows[~np.isnan(rows).any(axis=1)]
        rows = rows[(rows != 0).any(axis=1)]
    if rows.size:
        v = rows.astype(np.float64)
        c1, c2 = v[:, 3:6], v[:, 6:9]
        n1 = np.linalg.norm(c1, axis=1)
        n2 = np.linalg.norm(c2, axis=1)
        dot = np.abs((c1 * c2).sum(axis=1))
        bad = (np.abs(n1 - 1) > ROT6D_TOL) | (np.abs(n2 - 1) > ROT6D_TOL) | (dot > ROT6D_TOL)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            rep.error(npz_path, "obj_seq",
                      f"rot6d (dims 3:9) not orthonormal at {int(bad.sum())} observed row(s); "
                      f"e.g. |c1|={n1[i]:.3f} |c2|={n2[i]:.3f} |dot|={dot[i]:.3f} "
                      f"(tol {ROT6D_TOL})")
        if (v[:, 9] < 0).any():
            rep.error(npz_path, "obj_seq",
                      f"bbox area (dim 9) negative at {int((v[:, 9] < 0).sum())} row(s)")

    _check_provenance(data, npz_path, rep)
    _check_labels(data, ids, obj_mask, n_max, npz_path, rep)


def _check_samples_meta(root, rep):
    meta_path = root / "meta.json"
    if not meta_path.exists():
        rep.error(root, "meta.json",
                  "samples/meta.json missing (contract requires a meta.json per artifact dir)")
        return
    try:
        json.loads(meta_path.read_text())
    except Exception as e:
        rep.error(meta_path, "-", f"meta.json is not valid JSON: {e}")


def _normalize_index_entry(entry, split, index_path, rep):
    """Contract: dataset_index.json entries contain sample_id + source_video."""
    if not isinstance(entry, dict):
        rep.error(index_path, split,
                  f"index entry must be an object with sample_id + source_video, got {entry!r}")
        return None
    sid = entry.get("sample_id")
    if not isinstance(sid, str) or not sid:
        rep.error(index_path, split, f"index entry with missing/invalid sample_id: {entry!r}")
        return None
    if not isinstance(entry.get("source_video"), str) or not entry["source_video"]:
        rep.error(index_path, split,
                  f"sample '{sid}' missing source_video in dataset_index.json")
    return entry


def check_samples(path, rep):
    path = Path(path)
    if path.is_file():
        rep.n_files = 1
        check_sample_file(path, rep, set(), set())
        return
    _check_samples_meta(path, rep)
    index_path = path / "dataset_index.json"
    if not index_path.exists():
        rep.error(path, "dataset_index.json", "missing (expected at samples root)")
    else:
        try:
            index = json.loads(index_path.read_text())
        except Exception as e:
            rep.error(index_path, "-", f"not valid JSON: {e}")
            index = None
        if index is not None:
            if not isinstance(index, dict):
                rep.error(index_path, "-", "top level must be a JSON object")
                index = None
        if index is not None:
            usable = True
            for s in SPLITS:
                if s not in index:
                    rep.error(index_path, s,
                              "split missing; contract fixes top level as "
                              "{\"train\": [...], \"val\": [...], \"test\": [...]}")
                    usable = False
                elif not isinstance(index[s], list):
                    rep.error(index_path, s, "split entry must be a list")
                    usable = False
            for k in index:
                if k not in SPLITS:
                    rep.warn(index_path, k, "extra top-level key not in contract "
                                            "{train, val, test}")
            if usable:
                _check_samples_with_index(path, index, index_path, rep)
            return
    # no usable index: still check the npz files we can find
    files = sorted(p for s in SPLITS for p in (path / s).glob("*.npz"))
    if not files:
        files = sorted(path.glob("*.npz"))
    if not files:
        rep.error(path, "-", "no sample .npz files found")
        return
    rep.n_files = len(files)
    for f in files:
        check_sample_file(f, rep, set(), set())


def _check_samples_with_index(root, index, index_path, rep):
    listed = {}   # sample_id -> split
    video_split = {}
    n_listed = 0
    for split in SPLITS:
        entries = index.get(split) or []
        for entry in entries:
            e = _normalize_index_entry(entry, split, index_path, rep)
            if e is None:
                continue
            sid = e["sample_id"]
            n_listed += 1
            if sid in listed:
                rep.error(index_path, split,
                          f"sample '{sid}' listed in both '{listed[sid]}' and '{split}'")
            listed[sid] = split
            sv = e.get("source_video")
            if isinstance(sv, str) and sv:
                if sv in video_split and video_split[sv] != split:
                    rep.error(index_path, split,
                              f"source_video '{sv}' appears in both '{video_split[sv]}' and "
                              f"'{split}' (split must isolate source_video)")
                video_split[sv] = split
            fname = sid if sid.endswith(".npz") else sid + ".npz"
            if not (root / split / fname).exists():
                rep.error(index_path, split, f"listed sample file missing: {split}/{fname}")
    for split in SPLITS:
        for f in sorted((root / split).glob("*.npz")):
            if f.stem not in listed:
                rep.warn(f, "-", "sample file on disk but not listed in dataset_index.json")
    files = [root / listed[sid] / (sid if sid.endswith(".npz") else sid + ".npz")
             for sid in sorted(listed)]
    files = [f for f in files if f.exists()]
    rep.n_files = len(files)
    ws, nmaxs = set(), set()
    for f in files:
        check_sample_file(f, rep, ws, nmaxs)
    if len(ws) > 1:
        rep.error(index_path, "-", f"inconsistent window length W across samples: {sorted(ws)}")
    elif ws and ws != {DEFAULT_W}:
        rep.warn(index_path, "-", f"W={sorted(ws)[0]} != contract default {DEFAULT_W}")
    if len(nmaxs) > 1:
        rep.error(index_path, "-", f"inconsistent N_max across samples: {sorted(nmaxs)}")
    if n_listed == 0:
        rep.warn(index_path, "-", "dataset_index.json lists zero samples")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate ego_hoi pipeline artifacts against "
                    ".agent/memory/03_data_pipeline.md (contract 2026-10-15)")
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("path", help="file or directory, depending on stage")
    args = ap.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: path does not exist: {path}", file=sys.stderr)
        return 2

    rep = Report()
    checker = {"clips": check_clips, "hand_pose": check_hand_pose,
               "pose6d": check_pose6d, "samples": check_samples}[args.stage]
    checker(path, rep)
    print(rep.render(args.stage, path))
    return 1 if rep.n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
