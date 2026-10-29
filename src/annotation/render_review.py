"""Render EgoDemo candidate or reviewed evidence without importing OpenCV at module load."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
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
        raise RenderError(f"Rendered video verification decoded {count}/{expected} panels")


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
        "total_available_panels": total_targets,
        "truncated": truncated,
        "decoded_annotated_frames": decoded_count,
        "output_verified": True,
        "timeline_alignment": "native zero-based frame IDs and exact decoded frame count; PTS not independently verified",
        "video_role": "user-supplied annotated video, not raw",
        "3d_geometry_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path, help="matching annotated video, not raw")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("mp4", "contact-sheet"), dest="output_format")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-panels", type=int)
    args = parser.parse_args(argv)
    try:
        result = render_review(
            args.annotation_dir, args.video, args.output, output_format=args.output_format,
            fps=args.fps, max_panels=args.max_panels,
        )
    except (RenderError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
