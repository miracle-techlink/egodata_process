"""Render EgoDemo candidate or reviewed evidence without importing OpenCV at module load."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from src.annotation.schema import validate_event, validate_none_interval


class RenderError(ValueError):
    """The annotated video or annotation directory cannot be rendered safely."""


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _records(path, validator, frame_count):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validator(record, frame_count)
            except (json.JSONDecodeError, ValueError) as error:
                raise RenderError(f"{path}:{line_number}: {error}") from error
            result.append(record)
    return result


def _load(annotation_dir):
    annotation_dir = Path(annotation_dir).expanduser().resolve()
    timeline = _json(annotation_dir / "timeline.json")
    count = timeline.get("frame_count")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or timeline.get("frame_index") != list(range(count))):
        raise RenderError("Review rendering requires a native zero-based contiguous timeline")
    timestamps = timeline.get("timestamps")
    if not isinstance(timestamps, list) or len(timestamps) != count:
        raise RenderError("timeline.json timestamps do not match frame_count")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in timestamps):
        raise RenderError("timeline.json timestamps must be finite")
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise RenderError("timeline.json timestamps must be strictly increasing")
    events = _records(annotation_dir / "events.jsonl", validate_event, count)
    none_intervals = _records(annotation_dir / "none_intervals.jsonl", validate_none_interval, count)
    return annotation_dir, timeline, events, none_intervals


def _targets(events, none_intervals):
    targets = []
    for kind, records in (("event", events), ("none", none_intervals)):
        for record in records:
            identifier = record["event_id"] if kind == "event" else record["interval_id"]
            frames = []
            for frame in (record["start_frame"], record["end_frame"] - 1, *record["evidence_frames"]):
                if frame not in frames:
                    frames.append(frame)
            for frame in frames:
                targets.append({"kind": kind, "record": record, "identifier": identifier, "frame": frame})
    return targets


def _decode(video_path, needed, frame_count, cv2):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RenderError(f"Cannot open annotated video: {video_path}")
    frames = {}
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count in needed and count not in frames:
                frames[count] = frame.copy()
            count += 1
    finally:
        capture.release()
    if count != frame_count:
        raise RenderError(
            f"Annotated video decoded {count} frames but native timeline has {frame_count}; "
            "refusing raw/positional frame guessing"
        )
    missing = sorted(set(needed) - set(frames))
    if missing:
        raise RenderError(f"Video did not yield required native evidence frames: {missing}")
    return frames, count


def _text(image, text, y, color=(235, 235, 235), scale=0.43, thickness=1):
    import cv2
    cv2.putText(
        image, str(text), (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def _panel(frame, target, timeline, events, none_intervals, cv2):
    width, image_height, caption_height = 640, 360, 170
    height, width_source = frame.shape[:2]
    scale = min(width / width_source, image_height / height)
    resized = cv2.resize(frame, (max(1, round(width_source * scale)), max(1, round(height * scale))))
    panel = np.zeros((image_height + caption_height, width, 3), dtype=np.uint8)
    y_offset = (image_height - resized.shape[0]) // 2
    x_offset = (width - resized.shape[1]) // 2
    panel[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    record = target["record"]
    frame_index = target["frame"]
    roles = []
    if frame_index == record["start_frame"]:
        roles.append("START")
    if frame_index == record["end_frame"] - 1:
        roles.append("END")
    if frame_index in record["evidence_frames"]:
        roles.append("EVIDENCE")
    status = record["review_status"]
    color = {
        "pending": (40, 220, 255), "accepted": (80, 220, 80),
        "rejected": (80, 80, 240), "needs_revision": (40, 160, 255),
    }[status]
    timestamp = timeline["timestamps"][frame_index]
    _text(panel, f"native frame {frame_index}  t={timestamp:.6f}s  {'/'.join(roles)}", image_height + 22)
    _text(
        panel,
        f"{target['kind'].upper()} {target['identifier']}  [{record['start_frame']},{record['end_frame']})",
        image_height + 45, color=color, scale=0.47,
    )
    if target["kind"] == "event":
        object_text = record.get("object_id") or f"hint:{record['object_hint']}"
        _text(
            panel,
            f"{record['subject']}  {record['interaction_type']}  {object_text}",
            image_height + 68, color=color, scale=0.45,
        )
        _text(
            panel,
            f"review={status}  track={record.get('track_link_status', 'unknown')}  geometry={record['geometry_status']}",
            image_height + 91,
        )
    else:
        _text(panel, f"EXPLICIT NONE  review={status}", image_height + 68, color=color, scale=0.47)
        _text(panel, "Explicit none candidate; acceptance required", image_height + 91)
    active = [
        record["event_id"] for record in events
        if record["start_frame"] <= frame_index < record["end_frame"]
    ] + [
        record["interval_id"] for record in none_intervals
        if record["start_frame"] <= frame_index < record["end_frame"]
    ]
    active_text = ",".join(active)
    if len(active_text) > 82:
        active_text = active_text[:79] + "..."
    _text(panel, f"simultaneous records ({len(active)}): {active_text}", image_height + 114, scale=0.39)
    _text(panel, "2D GEOMETRY = REVIEW HEURISTIC ONLY; NOT 3D VERIFIED", image_height + 140,
          color=(40, 200, 255), scale=0.42)
    _text(panel, "PENDING/UNLABELLED IS NOT NONE; NO AUTOMATIC ACCEPTANCE", image_height + 161, scale=0.39)
    return panel


def _verify_video(path, expected, cv2):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RenderError(f"Cannot reopen rendered video for verification: {path}")
    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    if count != expected:
        raise RenderError(f"Rendered video verification decoded {count}/{expected} frames")
    return count


def _probe_video_pts(video_path, ffprobe_path=None):
    ffprobe = Path(ffprobe_path).expanduser().resolve() if ffprobe_path else shutil.which("ffprobe")
    if ffprobe is None:
        raise RenderError("ffprobe is required for native PTS verification; refusing FPS/frame-count inference")
    command = [
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_frames",
        "-show_entries",
        "stream=index,codec_type,time_base,avg_frame_rate,r_frame_rate:"
        "frame=best_effort_timestamp_time,pts_time,pkt_pts_time",
        "-of", "json", str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RenderError(f"ffprobe PTS verification failed for {video_path}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RenderError(f"ffprobe PTS verification failed for {video_path}: {detail}")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RenderError(f"ffprobe returned invalid JSON for {video_path}: {error}") from error
    streams = document.get("streams")
    frames = document.get("frames")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(frames, list):
        raise RenderError("ffprobe must report exactly one selected video stream and its frame list")
    pts_values = []
    pts_field = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RenderError(f"ffprobe frame {index} is not an object")
        fields = [
            name for name in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time")
            if frame.get(name) not in (None, "N/A")
        ]
        if not fields:
            raise RenderError(f"ffprobe frame {index} has no presentation timestamp")
        if pts_field is None:
            pts_field = fields[0]
        elif pts_field not in fields:
            raise RenderError(f"ffprobe frame {index} lacks the selected PTS field {pts_field}")
        try:
            value = float(frame[pts_field])
        except (TypeError, ValueError) as error:
            raise RenderError(f"ffprobe frame {index} has a nonnumeric PTS") from error
        if not math.isfinite(value):
            raise RenderError(f"ffprobe frame {index} has a nonfinite PTS")
        pts_values.append(value)
    return {
        "ffprobe_path": str(ffprobe),
        "stream": streams[0],
        "pts": pts_values,
        "pts_field": pts_field or "unavailable",
    }


def _verify_native_pts(probe, timeline, tolerance_seconds):
    pts = probe["pts"]
    expected = [float(value) for value in timeline["timestamps"]]
    if len(pts) != len(expected):
        raise RenderError(
            f"ffprobe found {len(pts)} video PTS values but native timeline has {len(expected)} frames"
        )
    presentation_pts = sorted(pts)
    if len(set(presentation_pts)) != len(presentation_pts) or any(
        later <= earlier for earlier, later in zip(presentation_pts, presentation_pts[1:])
    ):
        raise RenderError("Input video PTS values must be unique and strictly increasing in presentation order")
    errors = [abs(actual - wanted) for actual, wanted in zip(presentation_pts, expected)]
    max_error = max(errors, default=0.0)
    if max_error > tolerance_seconds:
        worst = max(range(len(errors)), key=errors.__getitem__)
        raise RenderError(
            f"Input video PTS does not match the native timeline at frame {worst}: "
            f"actual={presentation_pts[worst]:.9f}s timeline={expected[worst]:.9f}s "
            f"error={max_error:.9f}s exceeds {tolerance_seconds:.9f}s"
        )
    return {
        "status": "verified",
        "source": f"ffprobe {probe['pts_field']} sorted into presentation order",
        "frame_count": len(pts),
        "tolerance_seconds": tolerance_seconds,
        "max_abs_error_seconds": max_error,
        "mean_abs_error_seconds": sum(errors) / len(errors) if errors else 0.0,
        "first_pts_seconds": presentation_pts[0] if presentation_pts else None,
        "last_pts_seconds": presentation_pts[-1] if presentation_pts else None,
    }


def _active_records(events, none_intervals, frame_index):
    active = [
        ("event", record) for record in events
        if record["start_frame"] <= frame_index < record["end_frame"]
    ]
    active.extend(
        ("none", record) for record in none_intervals
        if record["start_frame"] <= frame_index < record["end_frame"]
    )
    return active


def _continuous_frame(
    frame, frame_index, actual_pts, timeline, events, none_intervals, cv2, caption_height
):
    width, image_height = 1280, 720
    height, width_source = frame.shape[:2]
    scale = min(width / width_source, image_height / height)
    resized = cv2.resize(frame, (max(1, round(width_source * scale)), max(1, round(height * scale))))
    canvas = np.zeros((image_height + caption_height, width, 3), dtype=np.uint8)
    y_offset = (image_height - resized.shape[0]) // 2
    x_offset = (width - resized.shape[1]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    timeline_pts = float(timeline["timestamps"][frame_index])
    _text(
        canvas,
        f"native frame {frame_index}/{timeline['frame_count'] - 1}  "
        f"input PTS={actual_pts:.9f}s  timeline={timeline_pts:.9f}s  "
        f"delta={actual_pts - timeline_pts:+.9f}s",
        image_height + 24, scale=0.52,
    )
    active = _active_records(events, none_intervals, frame_index)
    _text(canvas, f"active candidates at this frame: {len(active)}", image_height + 49, scale=0.48)
    colors = {
        "pending": (40, 220, 255), "accepted": (80, 220, 80),
        "rejected": (80, 80, 240), "needs_revision": (40, 160, 255),
    }
    y = image_height + 76
    if not active:
        _text(canvas, "NO ACTIVE CANDIDATE: unknown/unlabelled is NOT none", y,
              color=(40, 200, 255), scale=0.48)
    for kind, record in active:
        identifier = record["event_id"] if kind == "event" else record["interval_id"]
        roles = []
        if frame_index == record["start_frame"]:
            roles.append("START")
        if frame_index == record["end_frame"] - 1:
            roles.append("END")
        if frame_index in record["evidence_frames"]:
            roles.append("EVIDENCE")
        status = record["review_status"]
        if kind == "event":
            object_text = record.get("object_id") or f"hint:{record.get('object_hint')}"
            line = (
                f"EVENT {identifier} [{record['start_frame']},{record['end_frame']}) "
                f"{'/'.join(roles)} {record['subject']} {record['interaction_type']} "
                f"object={object_text} review={status} "
                f"track={record.get('track_link_status', 'not_provided')} "
                f"geometry={record.get('geometry_status', 'not_checked')}"
            )
        else:
            line = (
                f"NONE {identifier} [{record['start_frame']},{record['end_frame']}) "
                f"{'/'.join(roles)} review={status} "
                f"geometry={record.get('geometry_status', 'not_checked')}"
            )
        if len(line) > 175:
            line = line[:172] + "..."
        _text(canvas, line, y, color=colors[status], scale=0.43)
        y += 27
    footer = image_height + caption_height - 28
    _text(canvas, "FULL DECODE/DISPLAY IS NOT HUMAN REVIEW; pending is not accepted",
          footer, color=(40, 200, 255), scale=0.45)
    _text(canvas, "Container output PTS is independently probed; native PTS is shown in overlay",
          footer + 22, scale=0.40)
    return canvas


def _verify_output_pts(output_path, frame_count, fps, ffprobe_path, tolerance_seconds):
    probe = _probe_video_pts(output_path, ffprobe_path=ffprobe_path)
    pts = probe["pts"]
    if len(pts) != frame_count:
        raise RenderError(f"Output ffprobe found {len(pts)}/{frame_count} PTS values")
    if len(set(pts)) != len(pts) or any(
        later <= earlier for earlier, later in zip(pts, pts[1:])
    ):
        raise RenderError("Rendered output PTS values are missing, duplicated, or not strictly increasing")
    expected = [index / fps for index in range(frame_count)]
    errors = [abs(actual - wanted) for actual, wanted in zip(pts, expected)]
    max_error = max(errors, default=0.0)
    if max_error > tolerance_seconds:
        raise RenderError(
            f"Rendered output PTS does not match the requested CFR playback timeline; "
            f"max error {max_error:.9f}s exceeds {tolerance_seconds:.9f}s"
        )
    deltas = [later - earlier for earlier, later in zip(pts, pts[1:])]
    return {
        "status": "verified",
        "source": f"ffprobe {probe['pts_field']}",
        "frame_count": len(pts),
        "first_pts_seconds": pts[0] if pts else None,
        "last_pts_seconds": pts[-1] if pts else None,
        "min_delta_seconds": min(deltas) if deltas else None,
        "max_delta_seconds": max(deltas) if deltas else None,
        "max_abs_error_from_requested_cfr_seconds": max_error,
        "tolerance_seconds": tolerance_seconds,
    }


def _publish_temporary(temporary, output):
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite render output: {output}") from error
    finally:
        Path(temporary).unlink(missing_ok=True)


def render_review(annotation_dir, video_path, output_path, *, output_format=None, fps=2.0, max_panels=None):
    """Render event boundaries/evidence and fully verify the annotated input and output."""
    annotation_dir, timeline, events, none_intervals = _load(annotation_dir)
    video_path = Path(video_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite render output: {output}")
    if output_format is None:
        output_format = "mp4" if output.suffix.lower() == ".mp4" else "contact-sheet"
    if output_format not in {"mp4", "contact-sheet"}:
        raise RenderError("output_format must be mp4 or contact-sheet")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise RenderError("render fps must be positive")
    if max_panels is not None and (
        isinstance(max_panels, bool) or not isinstance(max_panels, int) or max_panels < 1
    ):
        raise RenderError("max_panels must be a positive integer")
    try:
        import cv2
    except ImportError as error:
        raise RenderError("Rendering requires optional OpenCV (cv2); annotation JSON/CSV does not") from error
    targets = _targets(events, none_intervals)
    if not targets:
        raise RenderError("No event or none boundary/evidence frames are available to render")
    total_targets = len(targets)
    truncated = max_panels is not None and len(targets) > max_panels
    if truncated:
        targets = targets[:max_panels]
    decoded, decoded_count = _decode(video_path, {target["frame"] for target in targets}, timeline["frame_count"], cv2)
    panels = [_panel(decoded[target["frame"]], target, timeline, events, none_intervals, cv2) for target in targets]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "contact-sheet":
        columns = min(3, len(panels))
        rows = math.ceil(len(panels) / columns)
        tile_height, tile_width = panels[0].shape[:2]
        sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
        for index, panel in enumerate(panels):
            row, column = divmod(index, columns)
            sheet[row * tile_height:(row + 1) * tile_height,
                  column * tile_width:(column + 1) * tile_width] = panel
        extension = ".jpg" if output.suffix.lower() in {".jpg", ".jpeg"} else ".png"
        ok, encoded = cv2.imencode(extension, sheet)
        if not ok:
            raise RenderError("OpenCV failed to encode contact sheet")
        with output.open("xb") as stream:
            stream.write(encoded.tobytes())
        verified = cv2.imread(str(output))
        if verified is None or verified.shape != sheet.shape:
            output.unlink(missing_ok=True)
            raise RenderError("Contact sheet verification failed")
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".mp4", dir=output.parent)
        os.close(descriptor)
        Path(temporary).unlink()
        writer = cv2.VideoWriter(
            temporary, cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
            (panels[0].shape[1], panels[0].shape[0]),
        )
        if not writer.isOpened():
            writer.release()
            Path(temporary).unlink(missing_ok=True)
            raise RenderError("OpenCV could not create an mp4v review video")
        try:
            for panel in panels:
                writer.write(panel)
        finally:
            writer.release()
        try:
            _verify_video(temporary, len(panels), cv2)
            _publish_temporary(temporary, output)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
    return {
        "status": "rendered",
        "format": output_format,
        "path": str(output),
        "panel_count": len(panels),
        "displayed_frame_count": len(panels),
        "human_reviewed_frames": 0,
        "human_reviewed_candidates": 0,
        "human_review_complete": False,
        "total_available_panels": total_targets,
        "truncated": truncated,
        "decoded_annotated_frames": decoded_count,
        "output_decoded_frames": decoded_count if output_format == "mp4" else None,
        "output_verified": True,
        "timeline_alignment": "native zero-based frame IDs and exact decoded frame count; PTS not independently verified",
        "video_role": "user-supplied annotated video, not raw",
        "3d_geometry_verified": False,
    }


def render_continuous_review(
    annotation_dir, video_path, output_path, *, fps=30.0, ffprobe_path=None,
    pts_tolerance_seconds=0.0005, output_pts_tolerance_seconds=0.001,
):
    """Render every native frame with candidate state and verify input/output PTS explicitly."""
    annotation_dir, timeline, events, none_intervals = _load(annotation_dir)
    video_path = Path(video_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite render output: {output}")
    if output.suffix.lower() != ".mp4":
        raise RenderError("Continuous review output must be an .mp4 file")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise RenderError("continuous output fps must be positive")
    for name, value in (
        ("pts_tolerance_seconds", pts_tolerance_seconds),
        ("output_pts_tolerance_seconds", output_pts_tolerance_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RenderError(f"{name} must be a nonnegative finite number")
    try:
        import cv2
    except ImportError as error:
        raise RenderError("Continuous rendering requires optional OpenCV (cv2)") from error
    probe = _probe_video_pts(video_path, ffprobe_path=ffprobe_path)
    input_pts_verification = _verify_native_pts(probe, timeline, float(pts_tolerance_seconds))
    presentation_pts = sorted(probe["pts"])
    max_active = max(
        len(_active_records(events, none_intervals, frame))
        for frame in range(timeline["frame_count"])
    )
    caption_height = max(140, 105 + 27 * max_active)
    if caption_height % 2:
        caption_height += 1
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RenderError(f"Cannot open annotated video: {video_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".mp4", dir=output.parent)
    os.close(descriptor)
    Path(temporary).unlink()
    writer = None
    decoded_count = 0
    written_count = 0
    expected_shape = None
    success = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_count >= timeline["frame_count"]:
                raise RenderError(
                    f"Annotated video decoded more than the native {timeline['frame_count']} frames"
                )
            canvas = _continuous_frame(
                frame, decoded_count, presentation_pts[decoded_count], timeline,
                events, none_intervals, cv2, caption_height,
            )
            if writer is None:
                expected_shape = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    temporary, cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
                    (canvas.shape[1], canvas.shape[0]),
                )
                if not writer.isOpened():
                    raise RenderError("OpenCV could not create an mp4v continuous review video")
            elif canvas.shape[:2] != expected_shape:
                raise RenderError("Decoded frame dimensions changed during continuous rendering")
            writer.write(canvas)
            written_count += 1
            decoded_count += 1
        if decoded_count != timeline["frame_count"]:
            raise RenderError(
                f"Annotated video decoded {decoded_count} frames but native timeline has "
                f"{timeline['frame_count']}; refusing raw/positional frame guessing"
            )
        if writer is None:
            raise RenderError("Annotated video produced no decodable frames")
        success = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not success:
            Path(temporary).unlink(missing_ok=True)
    try:
        output_decoded = _verify_video(temporary, written_count, cv2)
        output_pts_verification = _verify_output_pts(
            temporary, written_count, float(fps), ffprobe_path, float(output_pts_tolerance_seconds)
        )
        _publish_temporary(temporary, output)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return {
        "status": "rendered",
        "format": "continuous",
        "path": str(output),
        "panel_count": 0,
        "displayed_frame_count": written_count,
        "human_reviewed_frames": 0,
        "human_reviewed_candidates": 0,
        "human_review_complete": False,
        "decoded_annotated_frames": decoded_count,
        "output_decoded_frames": output_decoded,
        "output_verified": True,
        "input_pts": input_pts_verification,
        "output_pts": output_pts_verification,
        "output_timebase": {
            "mode": "constant_frame_rate_playback",
            "fps": float(fps),
            "claim": "output container PTS is verified playback timing only; it is not asserted to equal VFR/native input PTS",
        },
        "timeline_alignment": "ffprobe PTS matched native timeline for every frame before rendering",
        "video_role": "user-supplied annotated video, not raw",
        "3d_geometry_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path, help="matching annotated video, not raw")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("mp4", "contact-sheet", "continuous"), dest="output_format")
    parser.add_argument("--fps", type=float,
                        help="panel playback fps (default 2.0); continuous output fps (default 30.0)")
    parser.add_argument("--max-panels", type=int)
    parser.add_argument("--ffprobe-path",
                        help="explicit ffprobe binary for continuous PTS verification")
    parser.add_argument("--pts-tolerance-seconds", type=float, default=0.0005,
                        help="max |input PTS - native timeline| per frame for continuous rendering")
    parser.add_argument("--output-pts-tolerance-seconds", type=float, default=0.001,
                        help="max |output PTS - requested CFR| per frame for continuous rendering")
    args = parser.parse_args(argv)
    try:
        if args.output_format == "continuous":
            result = render_continuous_review(
                args.annotation_dir, args.video, args.output,
                fps=30.0 if args.fps is None else args.fps,
                ffprobe_path=args.ffprobe_path,
                pts_tolerance_seconds=args.pts_tolerance_seconds,
                output_pts_tolerance_seconds=args.output_pts_tolerance_seconds,
            )
        else:
            result = render_review(
                args.annotation_dir, args.video, args.output, output_format=args.output_format,
                fps=2.0 if args.fps is None else args.fps, max_panels=args.max_panels,
            )
    except (RenderError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
