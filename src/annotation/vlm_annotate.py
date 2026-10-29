from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from . import schema
from .qwen_backend import (
    DEFAULT_MAX_CONTINUATION_ROUNDS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MODEL_PATH,
    BackendError,
    GenerationResult,
    QwenBackend,
    environment_metadata,
)

MIN_NUM_FRAMES = 4
MAX_NUM_FRAMES = 32
MAX_NEW_TOKENS_LIMIT = 16384
MAX_CONTINUATION_ROUNDS_LIMIT = 16
EVIDENCE_TIME_EXACT_TOLERANCE_SECONDS = 1e-4
EVIDENCE_TIME_SNAP_TOLERANCE_SECONDS = 0.050001
SUPPORTED_INTERACTIONS = ("reach", "grasp", "hold", "move", "place", "release")
EVENT_FIELDS = {
    "subject", "object_hint", "interaction_type", "start_sec", "end_sec",
    "evidence_times_sec", "confidence", "notes", "source_event_ids",
}
NONE_FIELDS = {
    "start_sec", "end_sec", "evidence_times_sec", "confidence", "notes",
    "source_event_ids",
}


class AnnotationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoTimeline:
    pts_seconds: tuple
    duration_seconds: float
    stream: dict
    format_duration_seconds: Optional[float]
    duration_source: str

    @property
    def frame_count(self) -> int:
        return len(self.pts_seconds)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _probe_number(value, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"ffprobe returned invalid {name}: {value!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise AnnotationError(f"ffprobe returned invalid {name}: {value!r}")
    return result


def probe_video(video: Path, ffprobe_path: str = "ffprobe") -> VideoTimeline:
    video = Path(video)
    command = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_duration_time:"
        "stream=index,codec_name,width,height,time_base,avg_frame_rate,r_frame_rate,duration:"
        "format=duration",
        "-of", "json",
        str(video),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise AnnotationError(f"failed to execute ffprobe: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-5:]
        raise AnnotationError(
            f"ffprobe failed for {video} (exit {completed.returncode}): " + "\n".join(detail)
        )
    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise AnnotationError("ffprobe did not return valid JSON") from exc
    streams = payload.get("streams")
    frames = payload.get("frames")
    if not isinstance(streams, list) or len(streams) != 1:
        raise AnnotationError("video must contain exactly one selected v:0 stream")
    if not isinstance(frames, list) or not frames:
        raise AnnotationError("video has no decodable PTS frames")

    timed_frames = []
    for position, frame in enumerate(frames):
        if not isinstance(frame, dict) or "best_effort_timestamp_time" not in frame:
            raise AnnotationError(f"frame {position} has no best_effort_timestamp_time")
        pts = _probe_number(frame["best_effort_timestamp_time"], f"frame {position} PTS")
        timed_frames.append((pts, frame))
    timed_frames.sort(key=lambda item: item[0])
    pts_seconds = tuple(item[0] for item in timed_frames)
    if any(current <= previous for previous, current in zip(pts_seconds, pts_seconds[1:])):
        raise AnnotationError("video PTS must be unique and strictly increasing")

    deltas = [current - previous for previous, current in zip(pts_seconds, pts_seconds[1:])]
    last_frame = timed_frames[-1][1]
    last_duration = None
    if last_frame.get("pkt_duration_time") not in (None, "N/A"):
        last_duration = _probe_number(
            last_frame["pkt_duration_time"], "last frame duration", positive=True
        )
    elif deltas:
        last_duration = deltas[-1]
    observed_end = pts_seconds[-1] + last_duration if last_duration is not None else None

    stream = streams[0]
    stream_duration = None
    if stream.get("duration") not in (None, "N/A"):
        stream_duration = _probe_number(stream["duration"], "stream duration", positive=True)
    format_duration = None
    if payload.get("format", {}).get("duration") not in (None, "N/A"):
        format_duration = _probe_number(
            payload["format"]["duration"], "format duration", positive=True
        )
    candidates = [value for value in (observed_end, stream_duration) if value is not None]
    if not candidates:
        candidates = [value for value in (format_duration,) if value is not None]
    if not candidates:
        raise AnnotationError("cannot determine video end time from PTS or stream duration")
    duration = max(candidates)
    if not math.isfinite(duration) or duration <= pts_seconds[-1]:
        raise AnnotationError("video duration must extend beyond the last frame PTS")
    duration_source = "frame_pts_and_stream_duration" if observed_end is not None else "container_duration"
    return VideoTimeline(
        pts_seconds=pts_seconds,
        duration_seconds=duration,
        stream=stream,
        format_duration_seconds=format_duration,
        duration_source=duration_source,
    )


def sample_frame_indices(frame_count: int, num_frames: int) -> list:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if (isinstance(num_frames, bool) or not isinstance(num_frames, int)
            or not MIN_NUM_FRAMES <= num_frames <= MAX_NUM_FRAMES):
        raise ValueError(f"num_frames must be an integer in [{MIN_NUM_FRAMES}, {MAX_NUM_FRAMES}]")
    if frame_count < num_frames:
        raise AnnotationError(
            f"video is too short: {frame_count} frames available, {num_frames} requested"
        )
    return [index * (frame_count - 1) // (num_frames - 1) for index in range(num_frames)]


def extract_evidence_frames(
    video: Path,
    timeline: VideoTimeline,
    frame_indices: Sequence[int],
    evidence_dir: Path,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list:
    try:
        import cv2
    except ImportError as exc:
        raise AnnotationError("evidence extraction requires opencv-python-headless") from exc

    selected = set(frame_indices)
    if len(selected) != len(frame_indices) or not selected:
        raise AnnotationError("sampled frame indices must be nonempty and unique")
    if any(index < 0 or index >= timeline.frame_count for index in selected):
        raise AnnotationError("sampled frame index is outside the native timeline")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise AnnotationError(f"OpenCV cannot open video: {video}")
    records = []
    decoded_count = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if decoded_count >= timeline.frame_count:
                raise AnnotationError("OpenCV decoded more frames than ffprobe PTS entries")
            if decoded_count in selected:
                if image is None or image.size == 0 or len(image.shape) < 2:
                    raise AnnotationError(f"decoded frame {decoded_count} is empty")
                original_height, original_width = int(image.shape[0]), int(image.shape[1])
                if original_height < 1 or original_width < 1:
                    raise AnnotationError(f"decoded frame {decoded_count} has invalid dimensions")
                pixel_count = original_height * original_width
                if pixel_count > max_pixels:
                    scale = math.sqrt(max_pixels / pixel_count)
                    width = max(1, int(original_width * scale))
                    height = max(1, int(original_height * scale))
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if not encoded_ok:
                    raise AnnotationError(f"failed to JPEG-encode frame {decoded_count}")
                data = encoded.tobytes()
                filename = f"frame_{decoded_count:08d}.jpg"
                path = evidence_dir / filename
                with path.open("xb") as stream:
                    stream.write(data)
                records.append({
                    "frame_index": decoded_count,
                    "pts_seconds": timeline.pts_seconds[decoded_count],
                    "path": f"evidence/{filename}",
                    "image_path": str(path.resolve()),
                    "sha256": _sha256_bytes(data),
                    "sha256_basis": "encoded_jpeg",
                    "original_width": original_width,
                    "original_height": original_height,
                    "input_width": int(image.shape[1]),
                    "input_height": int(image.shape[0]),
                })
            decoded_count += 1
    except AnnotationError:
        raise
    except Exception as exc:
        raise AnnotationError(f"video decoding failed: {exc}") from exc
    finally:
        capture.release()
    if decoded_count != timeline.frame_count:
        raise AnnotationError(
            f"video is damaged or timeline mismatch: OpenCV decoded {decoded_count} frames, "
            f"ffprobe returned {timeline.frame_count} PTS entries"
        )
    if len(records) != len(frame_indices):
        raise AnnotationError("not every requested evidence frame could be decoded")
    records.sort(key=lambda item: item["frame_index"])
    return records


def build_prompt(
    sequence_id: str,
    timeline: VideoTimeline,
    evidence_frames: Sequence[dict],
    source_events=None,
) -> str:
    frame_list = "\n".join(
        f"  - native_frame={frame['frame_index']}, PTS={frame['pts_seconds']:.9f}"
        for frame in evidence_frames
    )
    earliest = min(0.0, timeline.pts_seconds[0])
    source_text = (
        json.dumps(source_events, ensure_ascii=False, indent=2, allow_nan=False)
        if source_events is not None else "null"
    )
    return f"""You are performing offline, non-causal semantic pre-annotation of a human-object interaction video. This is not frame interpolation and not a causal prediction input.

Sequence: {sequence_id}
Native timeline: {timeline.frame_count} frames; usable boundary range [{earliest:.9f}, {timeline.duration_seconds:.9f}] seconds. The end time is exclusive.
Evidence images follow this prompt in the exact order below. Times are actual video PTS, not an assumed 30 fps timeline:
{frame_list}

Task:
- Identify actual interactions, proposing separate events for left and right hands. Handedness is from the image viewpoint and must be reviewed; do not assume egocentric means mirrored.
- Use only these six interaction types: reach (hand approaches an object), grasp (hand establishes a grip), hold (controlled holding with little transport), move (hand transports or reorients the object), place (object is set at a destination before letting go), release (hand relinquishes control).
- Describe each object by visible appearance and location in object_hint (for example color, shape, material). Do not invent a stable track ID.
- Multiple events may overlap. Do not merge distinct hands or distinct interaction phases (for example grasp then move then release are separate events).
- Report each continuous interaction phase as ONE event covering its whole duration. Never emit one event per evidence image, fixed-length time slices, or consecutive same-hand same-object same-type events with touching boundaries: a sustained left-hand grasp supported by PTS 0.0, 3.600000000, 7.200000000 and 10.833333000 is ONE event, for example {{"subject": "left", "object_hint": "white paper with black text", "interaction_type": "grasp", "start_sec": 0.0, "end_sec": 14.433333, "evidence_times_sec": [0.0, 3.6, 7.2, 10.833333], "confidence": 0.8, "notes": "sustained grip; boundaries rough", "source_event_ids": []}}. Its start_sec is the first supporting PTS, its end_sec is the next supplied PTS after the last supporting one, and every supporting PTS goes into evidence_times_sec. The example numbers are illustrative only; use the actual supplied PTS.
- Every event needs rough start_sec and exclusive end_sec, plus one or more evidence_times_sec copied verbatim from the supplied PTS values and lying inside [start_sec, end_sec). Preserve all printed decimal digits; do not round, shorten, or invent evidence times. Copy start_sec and end_sec verbatim from the supplied PTS values whenever a boundary coincides with an evidence frame. Because end_sec is exclusive, an event whose last supporting PTS is X must have end_sec later than X (for example the next supplied PTS). Boundaries require later human correction.
- confidence is your uncalibrated self-assessment in [0, 1], not a verified probability. notes must mention uncertainty, occlusion, and boundary limitations when relevant, and stay under 20 words.
- none is not an interaction event. Add a none_interval only when the evidence explicitly supports no interaction for that whole interval. Missing labels, occlusion, unknown periods, and unobserved gaps are not none intervals.
- Do not output review_status, accepted, object_id, geometry_status, or model-input eligibility. Human review and geometry checks are separate.

Optional source events below are untrusted coarse time context only. Do not copy or map source descriptions, subjects, or labels into the six classes; every proposal must be supported by the images. Use source_event_ids only to cite a directly relevant supplied source interval, otherwise use [].
Source events JSON:
{source_text}

Return only one strict JSON object, without Markdown or commentary, in exactly this structure:
{{
  "events": [
    {{"subject": "left|right", "object_hint": "visual description", "interaction_type": "reach|grasp|hold|move|place|release", "start_sec": 0.0, "end_sec": 1.0, "evidence_times_sec": [0.0], "confidence": 0.5, "notes": "rough boundary; review required", "source_event_ids": []}}
  ],
  "none_intervals": [
    {{"start_sec": 0.0, "end_sec": 1.0, "evidence_times_sec": [0.0], "confidence": 0.5, "notes": "explicit visual support", "source_event_ids": []}}
  ]
}}
Use empty arrays when there are no proposals. Every numeric example above is illustrative; use the actual supplied PTS and video range."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"JSON float must be finite: {value}")
    return result


def _load_raw_json(raw_text: str):
    return json.loads(
        raw_text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_json_float,
    )


def _check_fields(item, allowed, required, name: str) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(required - set(item))
    unknown = sorted(set(item) - allowed)
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(unknown)}")


def _confidence(item) -> Optional[float]:
    value = item["confidence"]
    if value is None:
        return None
    result = _finite_number(value, "confidence")
    if not 0 <= result <= 1:
        raise ValueError("confidence must be in [0, 1]")
    return result


def _source_ids(item, has_source_events: bool) -> list:
    value = item.get("source_event_ids", [])
    if not isinstance(value, list) or any(not isinstance(entry, str) for entry in value):
        raise ValueError("source_event_ids must be a list of strings")
    if len(set(value)) != len(value):
        raise ValueError("source_event_ids must be unique")
    if value and not has_source_events:
        raise ValueError("source_event_ids require --source-events input")
    return list(value)


def _evidence_frames(item, evidence_frames: Sequence[dict], corrections: list, record_id: str) -> list:
    times = item["evidence_times_sec"]
    if not isinstance(times, list) or not times:
        raise ValueError("evidence_times_sec must be a nonempty array")
    sampled = [(frame["frame_index"], float(frame["pts_seconds"])) for frame in evidence_frames]
    indices = []
    candidate_corrections = []
    for position, value in enumerate(times):
        target = _finite_number(value, f"evidence_times_sec[{position}]")
        index, pts = min(sampled, key=lambda pair: abs(pair[1] - target))
        error = abs(pts - target)
        if error > EVIDENCE_TIME_SNAP_TOLERANCE_SECONDS:
            raise ValueError(
                f"evidence time {target} exceeds the "
                f"{EVIDENCE_TIME_SNAP_TOLERANCE_SECONDS}-second snap budget"
            )
        if error > EVIDENCE_TIME_EXACT_TOLERANCE_SECONDS:
            candidate_corrections.append({
                "record_id": record_id,
                "evidence_time_index": position,
                "requested_time_seconds": target,
                "snapped_frame_index": index,
                "snapped_pts_seconds": pts,
                "absolute_error_seconds": error,
            })
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("evidence times must map to unique native frames")
    corrections.extend(candidate_corrections)
    return indices


def _snap_boundary_seconds(value: float, sampled, record_id: str, field: str, corrections: list) -> float:
    """Snap a rounded boundary to the evidence PTS the model clearly meant.

    Only values within the audited snap tolerance of a sampled evidence PTS are
    changed; every change is recorded so reviewers see the requested value.
    """
    frame_index, pts = min(sampled, key=lambda pair: abs(pair[1] - value))
    error = abs(pts - value)
    if error <= EVIDENCE_TIME_EXACT_TOLERANCE_SECONDS:
        return value
    if error > EVIDENCE_TIME_SNAP_TOLERANCE_SECONDS:
        return value
    corrections.append({
        "record_id": record_id,
        "field": field,
        "requested_time_seconds": value,
        "snapped_frame_index": frame_index,
        "snapped_pts_seconds": pts,
        "absolute_error_seconds": error,
    })
    return pts


def _extend_end_past_cited_evidence(end: float, item, timeline: VideoTimeline, record_id: str,
                                    corrections: list) -> float:
    """Repair an event whose evidence list cites end_sec itself.

    Models treat [start_sec, end_sec] as closed and list both boundary PTS as
    evidence. The contract is half-open and requires evidence strictly inside,
    so a cited end boundary is extended past the exclusive edge by half a
    native frame, keeping the cited frame inside the interval. The change is
    recorded for review.
    """
    times = item.get("evidence_times_sec")
    if not isinstance(times, list):
        return end
    cited = any(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and abs(float(value) - end) <= EVIDENCE_TIME_EXACT_TOLERANCE_SECONDS
        for value in times
    )
    if not cited:
        return end
    gaps = [later - earlier for earlier, later in zip(timeline.pts_seconds, timeline.pts_seconds[1:])]
    offset = (min(gaps) / 2.0) if gaps else 0.0
    extended = min(end + offset, timeline.duration_seconds)
    if extended <= end:
        return end
    corrections.append({
        "record_id": record_id,
        "field": "end_sec",
        "requested_time_seconds": end,
        "snapped_pts_seconds": extended,
        "snapped_frame_index": bisect.bisect_left(timeline.pts_seconds, extended),
        "absolute_error_seconds": extended - end,
        "reason": "end_sec was cited as evidence; extended past the exclusive boundary by half a frame",
    })
    return extended


def _boundaries(item, timeline: VideoTimeline, evidence_frames, corrections: list, record_id: str) -> tuple:
    start = _finite_number(item["start_sec"], "start_sec")
    end = _finite_number(item["end_sec"], "end_sec")
    sampled = [(frame["frame_index"], float(frame["pts_seconds"])) for frame in evidence_frames]
    start = _snap_boundary_seconds(start, sampled, record_id, "start_sec", corrections)
    end = _snap_boundary_seconds(end, sampled, record_id, "end_sec", corrections)
    end = _extend_end_past_cited_evidence(end, item, timeline, record_id, corrections)
    earliest = min(0.0, timeline.pts_seconds[0])
    if start < earliest or end > timeline.duration_seconds or start >= end:
        raise ValueError(
            f"require {earliest} <= start_sec < end_sec <= {timeline.duration_seconds}"
        )
    start_frame = bisect.bisect_left(timeline.pts_seconds, start)
    end_frame = bisect.bisect_left(timeline.pts_seconds, end)
    if start_frame >= end_frame or end_frame > timeline.frame_count:
        raise ValueError("seconds do not define a nonempty in-range native frame interval")
    return start_frame, end_frame


def _common_candidate(
    item, timeline, evidence_frames, has_source_events: bool, corrections: list,
    boundary_corrections: list, record_id: str
) -> dict:
    start_frame, end_frame = _boundaries(item, timeline, evidence_frames, boundary_corrections, record_id)
    notes = item["notes"]
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "evidence_frames": _evidence_frames(item, evidence_frames, corrections, record_id),
        "confidence": _confidence(item),
        "label_source": "vlm",
        "review_status": "pending",
        "geometry_status": "not_checked",
        "source_event_ids": _source_ids(item, has_source_events),
        "notes": notes,
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
    }


def _convert_event(
    item, index, sequence_id, run_id, timeline, evidence_frames, has_source, corrections,
    boundary_corrections
):
    _check_fields(item, EVENT_FIELDS, EVENT_FIELDS - {"source_event_ids"}, "event")
    if item["subject"] not in ("left", "right"):
        raise ValueError("subject must be left or right")
    if item["interaction_type"] not in SUPPORTED_INTERACTIONS:
        raise ValueError(f"interaction_type must be one of {SUPPORTED_INTERACTIONS}")
    if not isinstance(item["object_hint"], str) or not item["object_hint"].strip():
        raise ValueError("object_hint must be a nonempty visual description")
    event_id = f"{run_id}:event:{index:04d}"
    result = {
        "event_id": event_id,
        "sequence_id": sequence_id,
        "subject": item["subject"],
        "object_id": None,
        "object_hint": item["object_hint"],
        "interaction_type": item["interaction_type"],
    }
    result.update(_common_candidate(
        item, timeline, evidence_frames, has_source, corrections, boundary_corrections, event_id
    ))
    schema.validate_event(result, frame_count=timeline.frame_count)
    return result


def _convert_none(
    item, index, sequence_id, run_id, timeline, evidence_frames, has_source, corrections,
    boundary_corrections
):
    _check_fields(item, NONE_FIELDS, NONE_FIELDS - {"source_event_ids"}, "none interval")
    interval_id = f"{run_id}:none:{index:04d}"
    result = {
        "interval_id": interval_id,
        "sequence_id": sequence_id,
    }
    result.update(_common_candidate(
        item, timeline, evidence_frames, has_source, corrections, boundary_corrections, interval_id
    ))
    schema.validate_none_interval(result, frame_count=timeline.frame_count)
    return result


def parse_model_response(
    raw_text: str,
    sequence_id: str,
    run_id: str,
    timeline: VideoTimeline,
    evidence_frames: Sequence[dict],
    source_events=None,
) -> tuple:
    corrections = []
    boundary_corrections = []
    record = {
        "response_id": f"{run_id}:response:0000",
        "sequence_id": sequence_id,
        "raw_text": raw_text,
        "raw_text_sha256": _sha256_bytes(raw_text.encode("utf-8")),
        "parsed_json": None,
        "status": "invalid",
        "errors": [],
        "evidence_time_snap_tolerance_seconds": EVIDENCE_TIME_SNAP_TOLERANCE_SECONDS,
        "evidence_time_corrections": corrections,
        "boundary_time_corrections": boundary_corrections,
    }
    try:
        payload = _load_raw_json(raw_text)
        record["parsed_json"] = payload
        if not isinstance(payload, dict):
            raise ValueError("top-level response must be an object")
        _check_fields(payload, {"events", "none_intervals"}, {"events", "none_intervals"}, "response")
        if not isinstance(payload["events"], list):
            raise ValueError("events must be an array")
        if not isinstance(payload["none_intervals"], list):
            raise ValueError("none_intervals must be an array")
    except (ValueError, TypeError) as exc:
        record["errors"].append(str(exc))
        return record, [], []

    events = []
    none_intervals = []
    errors = []
    has_source = source_events is not None
    for index, item in enumerate(payload["events"]):
        try:
            events.append(_convert_event(
                item, index, sequence_id, run_id, timeline, evidence_frames, has_source,
                corrections, boundary_corrections
            ))
        except (ValueError, TypeError) as exc:
            errors.append(f"events[{index}]: {exc}")
    for index, item in enumerate(payload["none_intervals"]):
        try:
            none_intervals.append(_convert_none(
                item, index, sequence_id, run_id, timeline, evidence_frames, has_source,
                corrections, boundary_corrections
            ))
        except (ValueError, TypeError) as exc:
            errors.append(f"none_intervals[{index}]: {exc}")
    record["candidate_counts"] = {
        "events": len(payload["events"]),
        "none_intervals": len(payload["none_intervals"]),
    }
    record["event_ids"] = [event["event_id"] for event in events]
    record["none_interval_ids"] = [interval["interval_id"] for interval in none_intervals]
    record["rejected_candidate_count"] = len(errors)
    if errors:
        record["errors"] = errors
        record["status"] = "partial" if events or none_intervals else "invalid"
    else:
        record["status"] = "valid"
    return record, events, none_intervals


def _load_source_events(path: Optional[Path]):
    if path is None:
        return None, None
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
        payload = _load_raw_json(raw)
    except (OSError, ValueError) as exc:
        raise AnnotationError(f"invalid source-events JSON {path}: {exc}") from exc
    return payload, {"path": str(path.resolve()), "sha256": schema.sha256_file(path)}


def _write_json_exclusive(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_response_exclusive(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _code_metadata() -> list:
    files = []
    for path in (Path(__file__), Path(schema.__file__), Path(sys.modules[QwenBackend.__module__].__file__)):
        resolved = path.resolve()
        files.append({"path": str(resolved), "sha256": schema.sha256_file(resolved)})
    return files


def run_annotation(
    video: Path,
    sequence_id: str,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    num_frames: int = 12,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
    source_events: Optional[Path] = None,
    backend=None,
    ffprobe_path: Optional[str] = None,
    max_continuation_rounds: int = DEFAULT_MAX_CONTINUATION_ROUNDS,
) -> dict:
    started = time.perf_counter()
    video = Path(video).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("sequence_id must be a nonempty string")
    if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT):
        raise ValueError(f"max_new_tokens must be an integer in [1, {MAX_NEW_TOKENS_LIMIT}]")
    if (isinstance(max_continuation_rounds, bool) or not isinstance(max_continuation_rounds, int)
            or not 0 <= max_continuation_rounds <= MAX_CONTINUATION_ROUNDS_LIMIT):
        raise ValueError(
            f"max_continuation_rounds must be an integer in [0, {MAX_CONTINUATION_ROUNDS_LIMIT}]"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(num_frames, bool) or not isinstance(num_frames, int)
            or not MIN_NUM_FRAMES <= num_frames <= MAX_NUM_FRAMES):
        raise ValueError(f"num_frames must be an integer in [{MIN_NUM_FRAMES}, {MAX_NUM_FRAMES}]")
    if os.path.lexists(str(output_dir)):
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if not video.is_file():
        raise AnnotationError(f"video not found: {video}")
    source_payload, source_metadata = _load_source_events(source_events)
    if ffprobe_path is None:
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            raise AnnotationError("ffprobe is required on PATH")
    timeline = probe_video(video, ffprobe_path)
    indices = sample_frame_indices(timeline.frame_count, num_frames)
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir()
    evidence = extract_evidence_frames(
        video, timeline, indices, evidence_dir, max_pixels=DEFAULT_MAX_PIXELS
    )
    prompt = build_prompt(sequence_id, timeline, evidence, source_payload)
    if backend is None:
        backend = QwenBackend(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            seed=seed,
            max_pixels=DEFAULT_MAX_PIXELS,
            max_continuation_rounds=max_continuation_rounds,
        )
    generation = backend.generate(prompt, evidence)
    if not isinstance(generation, GenerationResult):
        raise AnnotationError("backend returned an invalid GenerationResult")
    response, events, none_intervals = parse_model_response(
        generation.raw_text,
        sequence_id,
        output_dir.name,
        timeline,
        evidence,
        source_payload,
    )
    response["input_frames"] = evidence
    response["generation"] = {
        "elapsed_seconds": generation.elapsed_seconds,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "peak_gpu_memory_bytes": generation.peak_gpu_memory_bytes,
        "continuation_rounds": generation.continuation_rounds,
        "token_cap_rounds": generation.token_cap_rounds,
        "hit_length_limit": generation.hit_length_limit,
    }
    _write_response_exclusive(output_dir / "responses.jsonl", response)
    schema.write_jsonl(output_dir / "events.jsonl", events, frame_count=timeline.frame_count)
    schema.write_jsonl(
        output_dir / "none_intervals.jsonl", none_intervals, frame_count=timeline.frame_count
    )
    manifest = {
        "schema_version": "hoi_annotation_v1",
        "run_id": output_dir.name,
        "sequence_id": sequence_id,
        "created_at": _utc_now(),
        "run_status": "completed",
        "response_status": response["status"],
        "annotation_mode": "offline_noncausal",
        "allowed_as_model_input": False,
        "model": backend.model_metadata,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "prompt_text": prompt,
        "code": _code_metadata(),
        "parameters": {
            "num_frames": num_frames,
            "max_new_tokens": max_new_tokens,
            "max_continuation_rounds": max_continuation_rounds,
            "seed": seed,
            "do_sample": False,
            "max_pixels_per_frame": DEFAULT_MAX_PIXELS,
            "ffprobe_path": str(ffprobe_path),
            "evidence_time_snap_tolerance_seconds": EVIDENCE_TIME_SNAP_TOLERANCE_SECONDS,
        },
        "environment": environment_metadata(),
        "video": {
            "path": str(video),
            "sha256": schema.sha256_file(video),
            "native_frame_count": timeline.frame_count,
            "first_pts_seconds": timeline.pts_seconds[0],
            "last_pts_seconds": timeline.pts_seconds[-1],
            "duration_seconds": timeline.duration_seconds,
            "duration_source": timeline.duration_source,
            "format_duration_seconds": timeline.format_duration_seconds,
            "stream": timeline.stream,
        },
        "source_events": source_metadata,
        "input_frames": evidence,
        "raw_response": {
            "path": "responses.jsonl",
            "response_id": response["response_id"],
            "sha256": response["raw_text_sha256"],
        },
        "timing": {
            "generation_seconds": generation.elapsed_seconds,
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": generation.peak_gpu_memory_bytes,
        },
        "outputs": {
            "events": "events.jsonl",
            "none_intervals": "none_intervals.jsonl",
            "event_count": len(events),
            "none_interval_count": len(none_intervals),
            "evidence_time_correction_count": len(response["evidence_time_corrections"]),
            "boundary_time_correction_count": len(response["boundary_time_corrections"]),
            "rejected_candidate_count": response.get("rejected_candidate_count", 0),
            "invalid_errors": response["errors"],
        },
        "quality_flags": [
            "pending_human_review",
            "rough_boundaries",
            "vlm_confidence_not_calibrated",
            "offline_noncausal_not_model_input",
        ],
    }
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline non-causal Qwen2.5-VL semantic pre-annotation. "
            "Outputs are pending review and are not allowed as model inputs."
        )
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory; never overwritten")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--max-continuation-rounds", type=int, default=DEFAULT_MAX_CONTINUATION_ROUNDS,
        help="extra generation rounds appended when output is cut at the token limit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-events", type=Path, default=None, help="optional coarse-interval JSON context")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_annotation(
            video=args.video,
            sequence_id=args.sequence_id,
            output_dir=args.output_dir,
            model_path=args.model_path,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            source_events=args.source_events,
            max_continuation_rounds=args.max_continuation_rounds,
        )
    except (AnnotationError, BackendError, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "response_status": manifest["response_status"],
        "event_count": manifest["outputs"]["event_count"],
        "none_interval_count": manifest["outputs"]["none_interval_count"],
    }, ensure_ascii=False))
    return 0 if manifest["response_status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
