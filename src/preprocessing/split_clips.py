#!/usr/bin/env python3
"""Split long egocentric videos into ~20s clips (ego_hoi contract stage 1).

Outputs <output_dir>/<clip_id>.mp4 plus <output_dir>/clips_index.csv with
columns clip_id,source_video,start_sec,end_sec,fps,frames. Stream copy
(-c copy) is preferred; re-encode (libx264) is used for .webm inputs and as
fallback when stream copy fails. Every produced clip is re-probed with
ffprobe; the measured duration and frame count are written back to the csv,
so end_sec = start_sec + probed_duration always holds.

Example:
    python src/preprocessing/split_clips.py --input-dir data/ego_hoi/raw/EgoDemo \
        --output-dir data/ego_hoi/clips
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTS = {".mp4", ".webm"}
CSV_NAME = "clips_index.csv"
CSV_COLUMNS = ["clip_id", "source_video", "start_sec", "end_sec", "fps", "frames"]


class SplitError(RuntimeError):
    pass


class Segment:
    def __init__(self, index: int, start: float, end: float):
        self.index = index
        self.start = start
        self.end = end

    @property
    def duration(self) -> float:
        return self.end - self.start


def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise SplitError(
            f"command failed (rc={proc.returncode}): "
            + " ".join(str(c) for c in cmd)
            + ("\n" + "\n".join(tail) if tail else "")
        )
    return proc


def _parse_rate(text) -> float:
    if not text:
        return 0.0
    num, _, den = str(text).partition("/")
    try:
        n, d = float(num), (float(den) if den else 1.0)
    except ValueError:
        return 0.0
    return n / d if d else 0.0


def probe_source(path: Path, ffprobe: str) -> tuple[float, float]:
    out = run_cmd([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ])
    info = json.loads(out.stdout)
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError) as e:
        raise SplitError(f"cannot read duration of {path}: {e}")
    stream = (info.get("streams") or [{}])[0]
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if fps <= 0:
        raise SplitError(f"cannot determine fps of {path}")
    return duration, fps


def probe_clip(path: Path, ffprobe: str) -> tuple[float, int | None]:
    out = run_cmd([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames:format=duration",
        "-of", "json", str(path),
    ])
    info = json.loads(out.stdout)
    duration = float(info["format"]["duration"])
    raw = (info.get("streams") or [{}])[0].get("nb_read_frames")
    frames = int(raw) if raw and raw != "N/A" else None
    return duration, frames


def plan_segments(duration: float, seg_len: float, drop_threshold: float):
    n_full = int(duration // seg_len)
    segments = [
        Segment(i, i * seg_len, min((i + 1) * seg_len, duration))
        for i in range(n_full)
    ]
    remainder = duration - n_full * seg_len
    dropped = []
    if remainder >= drop_threshold:
        segments.append(Segment(n_full, n_full * seg_len, duration))
    elif remainder > 1e-3:
        dropped.append((n_full * seg_len, duration))
    return segments, dropped


def split_segment(ffmpeg: str, src: Path, seg: Segment, dst: Path, stream_copy: bool):
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", f"{seg.start:.3f}", "-i", str(src), "-t", f"{seg.duration:.3f}",
    ]
    if stream_copy:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac"]
    cmd += ["-y", str(dst)]
    run_cmd(cmd)


def process_video(video: Path, args, ffmpeg: str, ffprobe: str):
    duration, fps = probe_source(video, ffprobe)
    segments, dropped = plan_segments(duration, args.segment_len, args.drop_threshold)
    rows, warnings = [], []
    for seg in segments:
        clip_id = f"{video.stem}_{seg.index:03d}"
        dst = args.output_dir / f"{clip_id}.mp4"
        if args.dry_run:
            print(f"PLAN {clip_id} src={video.name} "
                  f"[{seg.start:.3f},{seg.end:.3f}] ({seg.duration:.2f}s)")
            rows.append({"clip_id": clip_id, "source_video": video.name,
                         "start_sec": f"{seg.start:.3f}", "end_sec": f"{seg.end:.3f}",
                         "fps": f"{fps:.3f}", "frames": ""})
            continue
        stream_copy = not args.reencode and video.suffix.lower() != ".webm"
        mode = "copy" if stream_copy else "reencode"
        try:
            split_segment(ffmpeg, video, seg, dst, stream_copy=stream_copy)
        except SplitError:
            if not stream_copy:
                dst.unlink(missing_ok=True)
                raise
            print(f"WARN {clip_id}: stream copy failed, retrying with re-encode",
                  file=sys.stderr)
            mode = "reencode"
            try:
                split_segment(ffmpeg, video, seg, dst, stream_copy=False)
            except SplitError:
                dst.unlink(missing_ok=True)
                raise
        actual_dur, frames = probe_clip(dst, ffprobe)
        if frames is None:
            frames = round(actual_dur * fps)
            warnings.append(f"{clip_id}: nb_read_frames unavailable, frames estimated as {frames}")
        if abs(seg.duration - args.segment_len) < 1e-6 and not (
            args.accept_min <= actual_dur <= args.accept_max
        ):
            warnings.append(
                f"{clip_id}: actual duration {actual_dur:.3f}s outside "
                f"[{args.accept_min}, {args.accept_max}] (keyframe drift)"
            )
        print(f"OK {clip_id} [{seg.start:.3f}->{seg.start + actual_dur:.3f}] "
              f"actual={actual_dur:.3f}s frames={frames} mode={mode}")
        rows.append({
            "clip_id": clip_id,
            "source_video": video.name,
            "start_sec": f"{seg.start:.3f}",
            "end_sec": f"{seg.start + actual_dur:.3f}",
            "fps": f"{fps:.3f}",
            "frames": frames,
        })
    return rows, dropped, warnings, duration


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="directory containing source .mp4/.webm videos")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="clips output directory; clips_index.csv is written inside it")
    parser.add_argument("--segment-len", type=float, default=20.0,
                        help="target segment length in seconds (default 20)")
    parser.add_argument("--accept-min", type=float, default=18.0,
                        help="warn if a full segment's probed duration is below this (default 18)")
    parser.add_argument("--accept-max", type=float, default=22.0,
                        help="warn if a full segment's probed duration is above this (default 22)")
    parser.add_argument("--drop-threshold", type=float, default=10.0,
                        help="drop the trailing remainder if shorter than this (default 10)")
    parser.add_argument("--reencode", action="store_true",
                        help="force re-encode for all videos instead of stream copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the split plan only; run no ffmpeg and write nothing")
    args = parser.parse_args(argv)

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or (ffmpeg is None and not args.dry_run):
        missing = [n for n, p in (("ffprobe", ffprobe), ("ffmpeg", ffmpeg))
                   if p is None and (n == "ffprobe" or not args.dry_run)]
        print(f"ERROR: required tool(s) not on PATH: {', '.join(missing)}", file=sys.stderr)
        return 2
    if not args.input_dir.is_dir():
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        return 2
    if args.segment_len <= 0 or args.drop_threshold < 0 or args.accept_min > args.accept_max:
        print("ERROR: invalid segment-len/drop-threshold/accept-range values", file=sys.stderr)
        return 2

    videos = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"ERROR: no .mp4/.webm videos in {args.input_dir}", file=sys.stderr)
        return 1
    stems = [v.stem for v in videos]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        print(f"ERROR: duplicate video stems would collide in clip_id: {dupes}",
              file=sys.stderr)
        return 2

    if not args.dry_run:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"ERROR: cannot create output dir {args.output_dir}: {e}", file=sys.stderr)
            return 2

    all_rows, all_dropped, all_warnings, failures = [], [], [], []
    for video in videos:
        try:
            rows, dropped, warnings, duration = process_video(video, args, ffmpeg, ffprobe)
        except SplitError as e:
            failures.append(video.name)
            print(f"ERROR {video.name}: {e}", file=sys.stderr)
            continue
        all_rows.extend(rows)
        all_warnings.extend(warnings)
        for start, end in dropped:
            print(f"DROP {video.name} [{start:.3f},{end:.3f}] "
                  f"({end - start:.2f}s < {args.drop_threshold:.1f}s threshold)")
            all_dropped.append({
                "source_video": video.name,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
            })
        if not rows and not dropped:
            all_warnings.append(f"{video.name}: duration {duration:.3f}s produced no clips")

    kind = "planned_clips" if args.dry_run else "clips"
    print(f"SUMMARY videos={len(videos)} {kind}={len(all_rows)} "
          f"dropped_residuals={len(all_dropped)} failures={len(failures)}")
    if args.dry_run:
        return 0 if not failures else 1

    csv_path = args.output_dir / CSV_NAME
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    ffmpeg_version = run_cmd([ffmpeg, "-version"]).stdout.splitlines()[0]
    meta = {
        "generator": "src/preprocessing/split_clips.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "ffmpeg": ffmpeg_version,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "input_dir": str(args.input_dir.resolve()),
        "params": {
            "segment_len": args.segment_len,
            "accept_range": [args.accept_min, args.accept_max],
            "drop_threshold": args.drop_threshold,
            "mode": "reencode" if args.reencode else "stream-copy preferred",
        },
        "num_source_videos": len(videos),
        "num_clips": len(all_rows),
        "dropped_residuals": all_dropped,
        "failures": failures,
        "warnings": all_warnings,
    }
    with open(args.output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"WROTE {csv_path} ({len(all_rows)} rows) + meta.json")
    for w in all_warnings:
        print(f"WARN {w}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
