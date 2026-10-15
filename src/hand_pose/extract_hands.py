"""Task 2: extract MediaPipe Hands 21-keypoint poses for every clip.

Per clip writes:
  <out_dir>/<clip_id>.npz       left_hand (T,21,3) float32, right_hand (T,21,3) float32,
                                conf (T,2) float32 (columns: left, right)
  <out_dir>/<clip_id>.meta.json fps / frames / source / params / detection stats / env
Batch writes <out_dir>/extract_summary.json (one record per clip, errors included).

Contract: .agent/memory/03_data_pipeline.md section 2.
Coordinates are stored raw: image-normalized x/y + relative depth z, exactly as
MediaPipe returns them. Undetected (or low-confidence) frames are all-NaN.

Usage:
  python src/hand_pose/extract_hands.py --clips_dir <clips dir or one .mp4> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NUM_KEYPOINTS = 21
NPZ_KEYS = ("left_hand", "right_hand", "conf")
LABEL_TO_INDEX = {"Left": 0, "Right": 1}  # conf column order: [left, right]
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi")

HANDEDNESS_NOTE = (
    "MediaPipe handedness assumes a mirrored (selfie) image. Egocentric footage "
    "matches that geometry (wearer's left hand appears on image left), so labels "
    "are stored as-is unless swap_applied is true (use --swap_handedness, e.g. for "
    "third-person non-mirrored footage)."
)

DEFAULT_PARAMS = {
    "max_num_hands": 2,
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "min_handedness_score": 0.5,
    "swap_handedness": False,
}


# ---------------------------------------------------------------------------
# pure-numpy core (no cv2/mediapipe needed; unit-tested locally)
# ---------------------------------------------------------------------------

def build_hand_arrays(frames, swap_handedness=False, min_handedness_score=0.5):
    """Pack per-frame detections into contract arrays.

    frames: list of length T; each element is None or a list of
        (label, score, landmarks) where label is "Left"/"Right", score is the
        handedness classification score, landmarks is array-like (21,3).
    Returns (left_hand, right_hand, conf, n_dropped): float32 arrays, NaN where
    a hand is undetected or below min_handedness_score; n_dropped counts
    discarded detections (bad label/shape or duplicate label in one frame,
    where the higher score wins).
    """
    T = len(frames)
    left_hand = np.full((T, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    right_hand = np.full((T, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    conf = np.full((T, 2), np.nan, dtype=np.float32)
    n_dropped = 0
    for t, dets in enumerate(frames):
        if not dets:
            continue
        best = {}  # side index -> (score, landmarks)
        for label, score, lm in dets:
            idx = LABEL_TO_INDEX.get(label)
            if idx is None:
                n_dropped += 1
                continue
            if swap_handedness:
                idx = 1 - idx
            score = float(score)
            if score < min_handedness_score:
                continue  # low confidence: treat as undetected, stays NaN
            arr = np.asarray(lm, dtype=np.float32)
            if arr.shape != (NUM_KEYPOINTS, 3):
                n_dropped += 1
                continue
            if idx not in best or score > best[idx][0]:
                n_dropped += int(idx in best)  # duplicate label: lower score discarded
                best[idx] = (score, arr)
            else:
                n_dropped += 1
        for idx, (score, arr) in best.items():
            (left_hand if idx == 0 else right_hand)[t] = arr
            conf[t, idx] = score
    return left_hand, right_hand, conf, n_dropped


def detection_stats(left_hand, right_hand):
    """Detection-rate stats per hand (for meta.json / summary)."""
    T = int(left_hand.shape[0])
    left_ok = ~np.isnan(left_hand[:, 0, 0])
    right_ok = ~np.isnan(right_hand[:, 0, 0])
    return {
        "frames": T,
        "left_detected_frames": int(left_ok.sum()),
        "right_detected_frames": int(right_ok.sum()),
        "left_detection_rate": (float(left_ok.mean()) if T else None),
        "right_detection_rate": (float(right_ok.mean()) if T else None),
        "both_detected_rate": (float((left_ok & right_ok).mean()) if T else None),
    }


def save_npz(path, left_hand, right_hand, conf):
    """Write one clip's npz, enforcing the contract dtypes/shapes."""
    left_hand = np.ascontiguousarray(left_hand, dtype=np.float32)
    right_hand = np.ascontiguousarray(right_hand, dtype=np.float32)
    conf = np.ascontiguousarray(conf, dtype=np.float32)
    T = left_hand.shape[0]
    assert left_hand.shape == (T, NUM_KEYPOINTS, 3), left_hand.shape
    assert right_hand.shape == (T, NUM_KEYPOINTS, 3), right_hand.shape
    assert conf.shape == (T, 2), conf.shape
    np.savez_compressed(
        str(path), left_hand=left_hand, right_hand=right_hand, conf=conf
    )


def validate_npz(path, expect_frames=None):
    """Contract check for one npz. Returns a list of problems ([] = ok)."""
    problems = []
    try:
        with np.load(str(path)) as data:
            keys = set(data.files)
            if keys != set(NPZ_KEYS):
                problems.append(f"keys {sorted(keys)} != {sorted(NPZ_KEYS)}")
                return problems
            arrays = {k: data[k] for k in NPZ_KEYS}
    except Exception as e:  # noqa: BLE001 - report any load failure as a problem
        return [f"cannot load: {type(e).__name__}: {e}"]

    T = None
    for name, shape_tail in (("left_hand", (NUM_KEYPOINTS, 3)),
                             ("right_hand", (NUM_KEYPOINTS, 3)),
                             ("conf", (2,))):
        arr = arrays[name]
        if arr.dtype != np.float32:
            problems.append(f"{name} dtype {arr.dtype} != float32")
        if arr.ndim != len(shape_tail) + 1 or arr.shape[1:] != shape_tail:
            problems.append(f"{name} shape {arr.shape} != (T,{','.join(map(str, shape_tail))})")
            continue
        if T is None:
            T = arr.shape[0]
        elif arr.shape[0] != T:
            problems.append(f"{name} T={arr.shape[0]} != {T}")
    if problems or T is None:
        return problems
    if T < 1:
        problems.append("T < 1")
    if expect_frames is not None and T != expect_frames:
        problems.append(f"T={T} != expect_frames={expect_frames}")

    for i, name in enumerate(("left_hand", "right_hand")):
        arr = arrays[name].reshape(T, -1)
        any_nan = np.isnan(arr).any(axis=1)
        all_nan = np.isnan(arr).all(axis=1)
        n_partial = int((any_nan & ~all_nan).sum())
        if n_partial:
            problems.append(f"{name}: {n_partial} frames partially NaN (must be all-or-nothing)")
        finite = np.isfinite(arr[~any_nan])
        if finite.size and not bool(finite.all()):
            problems.append(f"{name}: detected frames contain inf")
        conf_col = arrays["conf"][:, i]
        conf_nan = np.isnan(conf_col)
        if not bool((conf_nan == all_nan).all()):
            problems.append(f"conf[:,{i}] NaN pattern != {name} undetected pattern")
        valid = conf_col[~conf_nan]
        if valid.size and (bool((valid < 0).any()) or bool((valid > 1).any())):
            problems.append(f"conf[:,{i}] has values outside [0,1]")
    return problems


def _sanitize(obj):
    """Make an object strict-JSON safe (NaN/inf -> None, numpy scalars -> python)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.generic):
        return _sanitize(obj.item())
    return obj


def write_json(path, payload):
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# mediapipe/cv2 runtime (only used on the server; imported lazily)
# ---------------------------------------------------------------------------

def _require_runtime():
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "extract_hands.py needs opencv-python + mediapipe at runtime "
            f"(server env: conda activate ash_hands). Import failed: {e}"
        )


def collect_environment():
    env = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "numpy": np.__version__,
        "mediapipe": None,
        "opencv": None,
    }
    try:
        import mediapipe
        env["mediapipe"] = getattr(mediapipe, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2
        env["opencv"] = getattr(cv2, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return env


def detect_video(video_path, params):
    """Run MediaPipe Hands over every frame of a video.

    Returns (frames, fps, width, height) where frames is the per-frame
    detection list consumed by build_hand_arrays. Raises RuntimeError if the
    video cannot be opened or decodes zero frames.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,  # video stream mode: temporal tracking
        max_num_hands=int(params["max_num_hands"]),
        model_complexity=int(params["model_complexity"]),
        min_detection_confidence=float(params["min_detection_confidence"]),
        min_tracking_confidence=float(params["min_tracking_confidence"]),
    )
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            dets = []
            if res.multi_hand_landmarks and res.multi_handedness:
                for lm_list, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    if not handed.classification:
                        continue
                    cls = handed.classification[0]
                    dets.append((
                        str(cls.label),
                        float(cls.score),
                        [(lm.x, lm.y, lm.z) for lm in lm_list.landmark],
                    ))
            frames.append(dets or None)
    finally:
        hands.close()
        cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {video_path}")
    return frames, fps, width, height


# ---------------------------------------------------------------------------
# per-clip processing + batch driver
# ---------------------------------------------------------------------------

def process_clip(video_path, out_dir, params, overwrite=False):
    """Extract one clip. Returns its summary record (status ok/skipped)."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    clip_id = video_path.stem
    npz_path = out_dir / f"{clip_id}.npz"
    meta_path = out_dir / f"{clip_id}.meta.json"
    record = {
        "clip_id": clip_id,
        "source_video": video_path.name,
        "npz": str(npz_path),
        "meta": str(meta_path),
        "status": None,
        "error": None,
    }
    if npz_path.exists() and meta_path.exists() and not overwrite:
        record["status"] = "skipped"
        return record

    frames, fps, width, height = detect_video(video_path, params)
    left_hand, right_hand, conf, n_dropped = build_hand_arrays(
        frames,
        swap_handedness=bool(params["swap_handedness"]),
        min_handedness_score=float(params["min_handedness_score"]),
    )
    save_npz(npz_path, left_hand, right_hand, conf)
    problems = validate_npz(npz_path, expect_frames=len(frames))
    if problems:
        raise RuntimeError(f"self-validation failed: {problems}")

    stats = detection_stats(left_hand, right_hand)
    meta = {
        "clip_id": clip_id,
        "source_video": video_path.name,
        "source_path": str(video_path.resolve()),
        "fps": fps if fps and math.isfinite(fps) and fps > 0 else None,
        "frames": len(frames),
        "width": width,
        "height": height,
        "coordinates": "mediapipe image-normalized xy + relative depth z (raw)",
        "handedness": {
            "source": "mediapipe multi_handedness classification",
            "swap_applied": bool(params["swap_handedness"]),
            "note": HANDEDNESS_NOTE,
        },
        "params": {k: params[k] for k in DEFAULT_PARAMS},
        "stats": {**stats, "dropped_detections": int(n_dropped)},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "src/hand_pose/extract_hands.py",
        "environment": collect_environment(),
    }
    write_json(meta_path, meta)
    record.update({"status": "ok", "fps": meta["fps"], **stats,
                   "dropped_detections": int(n_dropped)})
    return record


def _worker(job):
    video_path, out_dir, params, overwrite = job
    try:
        return process_clip(video_path, out_dir, params, overwrite)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - one bad clip must not kill the batch
        return {
            "clip_id": Path(video_path).stem,
            "source_video": Path(video_path).name,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=5),
        }


def collect_clips(clips_dir):
    p = Path(clips_dir)
    if p.is_file():
        if p.suffix.lower() not in VIDEO_EXTS:
            raise SystemExit(f"unsupported video extension: {p}")
        return [p]
    if not p.is_dir():
        raise SystemExit(f"--clips_dir not found: {p}")
    clips = sorted(c for c in p.iterdir() if c.suffix.lower() == ".mp4")
    if not clips:
        raise SystemExit(f"no *.mp4 clips in {p}")
    return clips


def run_batch(clips, out_dir, params, num_workers=1, overwrite=False):
    try:
        from tqdm import tqdm
    except ImportError:  # tqdm is only a progress bar; degrade gracefully
        def tqdm(it, **kwargs):
            return it

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(c), str(out_dir), params, overwrite) for c in clips]
    if num_workers and num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            records = list(tqdm(pool.map(_worker, jobs), total=len(jobs), desc="clips"))
    else:
        records = [ _worker(job) for job in tqdm(jobs, desc="clips") ]

    counts = {"ok": 0, "skipped": 0, "error": 0}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "src/hand_pose/extract_hands.py",
        "out_dir": str(out_dir.resolve()),
        "num_clips": len(clips),
        "counts": counts,
        "params": {k: params[k] for k in DEFAULT_PARAMS},
        "environment": collect_environment(),
        "clips": records,
    }
    summary_path = out_dir / "extract_summary.json"
    write_json(summary_path, summary)
    return summary, summary_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe hand poses per clip -> <clip_id>.npz + .meta.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--clips_dir", required=True,
                        help="directory of <clip_id>.mp4 clips, or a single video file")
    ap.add_argument("--out_dir", required=True, help="output directory for npz/meta/summary")
    ap.add_argument("--num_workers", type=int, default=1,
                        help="parallel clips via processes (each owns a MediaPipe instance)")
    ap.add_argument("--overwrite", action="store_true",
                        help="re-run clips whose npz+meta already exist (default: skip)")
    ap.add_argument("--max_num_hands", type=int, default=DEFAULT_PARAMS["max_num_hands"])
    ap.add_argument("--model_complexity", type=int, default=DEFAULT_PARAMS["model_complexity"])
    ap.add_argument("--min_detection_confidence", type=float,
                    default=DEFAULT_PARAMS["min_detection_confidence"])
    ap.add_argument("--min_tracking_confidence", type=float,
                    default=DEFAULT_PARAMS["min_tracking_confidence"])
    ap.add_argument("--min_handedness_score", type=float,
                    default=DEFAULT_PARAMS["min_handedness_score"],
                    help="below this handedness score a detection is stored as NaN")
    ap.add_argument("--swap_handedness", action="store_true",
                    help="swap MediaPipe Left/Right labels (recorded in meta.json)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any clip failed (default: errors only recorded)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    clips = collect_clips(args.clips_dir)
    _require_runtime()
    params = {k: getattr(args, k) for k in DEFAULT_PARAMS}
    summary, summary_path = run_batch(
        clips, args.out_dir, params,
        num_workers=args.num_workers, overwrite=args.overwrite,
    )
    c = summary["counts"]
    print(f"done: {c.get('ok', 0)} ok, {c.get('skipped', 0)} skipped, "
          f"{c.get('error', 0)} error -> {summary_path}")
    if args.strict and c.get("error"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
