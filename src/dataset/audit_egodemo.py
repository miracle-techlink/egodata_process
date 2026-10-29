"""Read-only audit of annotated EgoDemo payloads and their imported staging data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from uuid import UUID
import zipfile

import numpy as np


class AuditError(ValueError):
    """An asset cannot be safely interpreted or fully decoded."""


def _issue(report, level, code, message, **details):
    report[level].append({"code": code, "message": message, **details})


def _integer(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and np.isfinite(value) and value >= 0 and int(value) == value)
    except (TypeError, ValueError, OverflowError):
        return False


def _path(root, relative):
    if not isinstance(relative, str) or not relative or any(ord(char) < 32 for char in relative):
        raise AuditError("Expected a nonempty local relative path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or "\\" in relative or ":" in relative or any(
        part in ("", ".", "..") for part in relative.split("/")
    ):
        raise AuditError("Unsafe relative asset path")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise AuditError("Asset path escapes its root")
    return result


def _fingerprint(path, expected=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        prefix = stream.read(8192)
        if not prefix or prefix.lstrip().startswith((b"version https://git-lfs", b"oid sha256:")):
            raise AuditError("Empty asset or Git LFS pointer instead of real payload")
        digest.update(prefix)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}
    for field, actual in (("expected_size", result["size"]), ("expected_sha256", result["sha256"])):
        if expected and expected.get(field) is not None and expected[field] != actual:
            raise AuditError(f"Payload {field} mismatch (actual={actual})")
    return result


def _json(path):
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise AuditError("JSON asset must be an object")
    return result


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {"count": int(values.size), "finite_count": int(finite.size), **dict(zip(
        ("min", "p50", "p95", "p99", "max"),
        np.quantile(finite, [0, .5, .95, .99, 1]).tolist() if finite.size else [None] * 5,
    ))}


def probe_video(path):
    """Decode every frame with ffprobe; never rely on a container's frame header."""
    command = [
        "ffprobe", "-v", "warning", "-err_detect", "explode", "-threads", "1",
        "-select_streams", "v:0", "-count_frames", "-show_frames", "-show_streams",
        "-show_entries", "frame=pts_time,pkt_pts_time:stream=nb_read_frames,time_base,avg_frame_rate",
        "-of", "json", str(path),
    ]
    try:
        process = subprocess.run(command, capture_output=True, timeout=1200, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AuditError(f"ffprobe could not complete: {type(error).__name__}") from error
    if process.returncode or process.stderr.strip():
        digest = hashlib.sha256(process.stderr).hexdigest()
        raise AuditError(f"ffprobe decoding failed: exit={process.returncode}, stderr_sha256={digest}")
    try:
        payload = json.loads(process.stdout)
        streams, frames = payload["streams"], payload["frames"]
        if len(streams) != 1 or not frames:
            raise ValueError("expected one video stream and nonempty decoded frames")
        count = int(streams[0]["nb_read_frames"])
        timestamps = np.asarray([
            float(frame["pts_time"] if "pts_time" in frame else frame["pkt_pts_time"])
            for frame in frames
        ], dtype=np.float64)
        if count != len(timestamps):
            raise ValueError("decoded frame count disagrees with frame records")
        if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
            raise ValueError("decoded PTS must be finite and strictly increasing")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise AuditError("ffprobe returned invalid frame counts or presentation timestamps") from error
    return {"decoded_frame_count": count, "pts": timestamps,
            "stream": streams[0], "method": "ffprobe full -count_frames -show_frames; stderr must be empty"}


def _timing(report, native, video):
    presentation = np.asarray(video.pop("pts"), dtype=np.float64)
    video["pts_range_seconds"] = [float(presentation[0]), float(presentation[-1])]
    video["interval_seconds"] = _distribution(np.diff(presentation))
    if video["decoded_frame_count"] != len(native):
        _issue(report, "errors", "video_frame_count", "Decoded frames differ from native sequence length")
        return
    interval = float(np.median(np.diff(native))) if len(native) > 1 else 0.0
    precision = float(np.finfo(np.float32).eps * np.max(np.abs(native)))
    tolerance = max(2e-5, 4 * precision, interval * .001)
    offset = float(presentation[0] - native[0])
    relative_error = np.abs((presentation - presentation[0]) - (native - native[0]))
    interval_error = np.abs(np.diff(presentation) - np.diff(native))
    video.update({"native_comparison_tolerance_seconds": tolerance,
                  "start_offset_seconds": offset,
                  "span_error_seconds": float(relative_error[-1]),
                  "max_relative_pts_error_seconds": float(relative_error.max()),
                  "max_interval_error_seconds": float(interval_error.max()) if interval_error.size else 0.0})
    if relative_error.max() > tolerance or (interval_error.size and interval_error.max() > tolerance):
        _issue(report, "errors", "video_timing", "Video PTS span/intervals differ from native timestamps")
    if abs(offset) > tolerance:
        _issue(report, "warnings", "video_start_offset", "Video and native timestamps have different origins",
               offset_seconds=offset)


def _sequence(report, sequence, meta, info, quality, episode):
    native = sequence.get("timestamps")
    if native is None or native.ndim != 1 or not native.size or native.dtype.kind != "f":
        raise AuditError("timestamps must be a nonempty floating point vector")
    if not np.isfinite(native).all() or (native < 0).any() or not (np.diff(native) > 0).all():
        raise AuditError("Native timestamps must be finite, nonnegative and strictly increasing")
    count = len(native)
    for label, declared in (("manifest", episode.get("frame_count")), ("meta", meta.get("frame_count")),
                            ("info", info.get("total_frames")), ("bad_frame_ratio", quality.get("total_frames"))):
        if not _integer(declared) or declared != count:
            _issue(report, "errors", "frame_count", f"{label} frame count disagrees with sequence")
    for key in ("frame_index", "index", "episode_index", "task_index"):
        values = sequence.get(key)
        if values is None or values.shape != (count,) or values.dtype.kind not in "iu" or (values < 0).any():
            raise AuditError(f"{key} must be a nonnegative integer vector of sequence length")
        if key in ("frame_index", "index") and not (values[1:] > values[:-1]).all():
            raise AuditError(f"{key} must be strictly increasing")
        if key == "episode_index" and len(np.unique(values)) != 1:
            raise AuditError("Multiple episode_index values within one UUID")
    mask = sequence.get("hand_valid")
    if mask is None or mask.shape != (count, 2) or mask.dtype != np.bool_:
        raise AuditError("hand_valid must be a boolean (T,2) array")
    details = quality.get("hand_bad_frame_details")
    if not isinstance(details, list):
        raise AuditError("Missing hand_bad_frame_details list")
    positions = {int(frame): position for position, frame in enumerate(sequence["frame_index"])}
    bad = np.zeros((count, 2), dtype=bool)
    seen = set()
    for detail in details:
        if not isinstance(detail, dict) or not {"frame", "left_bad_reason", "right_bad_reason"} <= detail.keys():
            raise AuditError("Malformed bad-frame reason record")
        frame = detail["frame"]
        if not _integer(frame) or frame not in positions or frame in seen:
            raise AuditError("Bad-frame reference absent or duplicated")
        seen.add(frame)
        bad[positions[frame]] = [bool(detail["left_bad_reason"]), bool(detail["right_bad_reason"])]
    summary = {"frame_count": count, "timestamp_range_seconds": [float(native[0]), float(native[-1])],
               "interval_seconds": _distribution(np.diff(native)), "valid_hand_counts": mask.sum(axis=0).tolist(),
               "wrist_speed_units": "source coordinate units / second; not calibrated", "wrists": {}}
    for hand_index, side in enumerate(("left", "right")):
        raw, masked = sequence.get(f"{side}_hand_world_raw"), sequence.get(f"{side}_hand_world")
        if any(value is None or value.shape != (count, 21, 3) or value.dtype.kind != "f" for value in (raw, masked)):
            raise AuditError(f"{side} raw/masked hands must be floating point (T,21,3) arrays")
        expected = np.isfinite(raw).all(axis=(1, 2)) & ~bad[:, hand_index]
        valid = mask[:, hand_index]
        if not np.array_equal(expected, valid):
            _issue(report, "errors", "hand_mask", f"{side} validity differs from raw finite values / bad-frame reasons")
        if not np.array_equal(masked[valid], raw[valid]) or not np.isnan(masked[~valid]).all():
            _issue(report, "errors", "masked_hand", f"{side} masked hands do not preserve raw valid rows / NaN invalid rows")
        adjacent = valid[:-1] & valid[1:]
        with np.errstate(invalid="ignore", over="ignore"):
            speed = np.linalg.norm(np.diff(raw[:, 0].astype(np.float64), axis=0), axis=1)[adjacent] / np.diff(native)[adjacent]
        finite = speed[np.isfinite(speed)]
        median = float(np.median(finite)) if finite.size else 0.0
        deviation = float(np.median(np.abs(finite - median))) if finite.size else 0.0
        threshold = max(10 * median, median + 20 * deviation, 1e-12)
        flagged = np.flatnonzero(adjacent)[(speed > threshold) | ~np.isfinite(speed)] + 1
        summary["wrists"][side] = {**_distribution(speed), "high_jump_threshold": threshold,
                                    "high_jump_count": len(flagged),
                                    "high_jump_frame_indices": sequence["frame_index"][flagged].tolist()}
        if flagged.size:
            _issue(report, "warnings", "wrist_jump", f"{side} wrist speed outliers; diagnostic only, not a physical verdict",
                   count=len(flagged))
    report["sequence"] = summary
    shapes = {"head_world": (1, 3), "head_world_rotation": (1, 4),
              "head_left_camera_position": (3,), "head_left_camera_rotation": (4,),
              "head_right_camera_position": (3,), "head_right_camera_rotation": (4,)}
    report["quaternions"] = {}
    for key, shape in shapes.items():
        required = key.startswith("head_left_camera") or f"observation.state.{key}" in info.get("features", {})
        if key not in sequence and not required:
            continue
        values = sequence.get(key)
        if values is None or values.shape != (count, *shape) or values.dtype.kind != "f":
            _issue(report, "errors", "spatial_shape", f"{key} has missing/invalid shape or dtype")
            continue
        if not np.isfinite(values).all():
            _issue(report, "errors", "spatial_nonfinite", f"{key} contains nonfinite values")
        if key.endswith("rotation"):
            norms = np.linalg.norm(values.astype(np.float64), axis=-1)
            report["quaternions"][key] = {"norm": _distribution(norms),
                                          "nonunit_count": int((np.abs(norms - 1) > 1e-3).sum())}
            if (norms < 1e-8).any():
                _issue(report, "errors", "quaternion_zero", f"{key} contains zero-length quaternions")
            elif (np.abs(norms - 1) > 1e-3).any():
                _issue(report, "warnings", "quaternion_norm", f"{key} quaternion norm differs from one")
    return native


def _events(report, annotation, sequence, timeline_path):
    events = annotation.get("subtask_annotations")
    if not isinstance(annotation.get("episode_description"), str) or not isinstance(events, list):
        raise AuditError("Invalid semantic event annotation schema")
    native, frame_indices = sequence["timestamps"], sequence["frame_index"]
    coverage = np.zeros(len(native), dtype=np.int64)
    identifiers, previous_start, previous_end = set(), -np.inf, -np.inf
    expected_ids = [[] for _ in native]
    interval = float(np.median(np.diff(native))) if len(native) > 1 else 0.0
    domain_end = float(native[-1] + interval)
    outside = 0
    for event in events:
        if not isinstance(event, dict):
            raise AuditError("Event records must be objects")
        start, end, identifier = event.get("start_time"), event.get("end_time"), event.get("subtask_id")
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) for value in (start, end)):
            raise AuditError("Event endpoints must be finite numbers")
        if start < 0 or end <= start or start < previous_start or end < previous_end:
            raise AuditError("Event intervals must be positive and monotone in source order")
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or identifier == "" or identifier in identifiers:
            raise AuditError("Event IDs must be unique nonempty strings or integers")
        if not isinstance(event.get("description"), str):
            raise AuditError("Event descriptions must be strings")
        identifiers.add(identifier)
        previous_start, previous_end = start, end
        selected = (native >= start) & (native < end)
        coverage += selected
        for position in np.flatnonzero(selected):
            expected_ids[position].append(identifier)
        outside += int(start < native[0] - 2e-5 or end > domain_end + 2e-5)
    report["events"] = {"count": len(events), "interval_convention": "[start_time,end_time)",
                        "covered_frames": int((coverage > 0).sum()), "unlabelled_frames": int((coverage == 0).sum()),
                        "overlap_frames": int((coverage > 1).sum()), "max_simultaneous_events": int(coverage.max()),
                        "out_of_range_events": outside, "estimated_native_end_seconds": domain_end}
    for code, value in (("event_unlabelled", (coverage == 0).sum()), ("event_overlap", (coverage > 1).sum()),
                        ("event_range", outside)):
        if value:
            _issue(report, "warnings", code, "Semantic event coverage diagnostic; not an HOI label", count=int(value))
    with timeline_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(native):
        raise AuditError("event_timeline.csv frame count mismatch")
    for position, row in enumerate(rows):
        timestamp = float(row["timestamp"])
        if (int(row["frame_index"]) != frame_indices[position] or not np.isfinite(timestamp)
                or abs(timestamp - native[position]) > 1e-9 or json.loads(row["subtask_ids"]) != expected_ids[position]):
            raise AuditError("event_timeline.csv disagrees with native half-open event membership")


def audit_episode(episode, dataset_root, staging_root, manifest_digest):
    report = {"source_episode_uuid": episode.get("source_episode_uuid"), "errors": [], "warnings": [],
              "sources": [], "videos": []}
    try:
        identifier = report["source_episode_uuid"]
        if not isinstance(identifier, str) or str(UUID(identifier)) != identifier.lower():
            raise AuditError("Invalid canonical episode UUID")
        directory = _path(staging_root, identifier.lower())
        base = _path(dataset_root, episode.get("lerobot_path"))
        if (episode.get("variant") not in ("EgoProStandard", "EgoProStandard-body")
                or PurePosixPath(episode["lerobot_path"]).parts[0] != episode["variant"]
                or base.name.lower() != identifier.lower()):
            raise AuditError("Manifest variant/UUID disagrees with lerobot_path")
        grouped, seen_paths = {}, set()
        if not isinstance(episode.get("files"), list):
            raise AuditError("Episode files must be a list")
        for record in episode["files"]:
            path = None
            try:
                if not isinstance(record, dict) or not isinstance(record.get("role"), str):
                    raise AuditError("Invalid manifest file record")
                path = _path(dataset_root, record.get("path"))
                if not path.is_relative_to(base) or path == base:
                    raise AuditError("Source file is outside this episode's lerobot_path")
                if path in seen_paths:
                    raise AuditError("Duplicate manifest asset path")
                seen_paths.add(path)
                provenance = {**_fingerprint(path, record), "role": record["role"], "relative_path": record["path"]}
                report["sources"].append(provenance)
                grouped.setdefault(record["role"], []).append(path)
            except (AuditError, OSError, ValueError) as error:
                _issue(report, "errors", "source_asset", str(error), path=str(path) if path else None)
        for role in ("info", "annotation", "bad_frame_ratio", "data", "head_left_video"):
            if not grouped.get(role) or (role != "data" and len(grouped[role]) != 1):
                _issue(report, "errors", "missing_source", f"Missing or ambiguous real {role} payload")
        for path in grouped.get("head_left_video", []):
            try:
                report["videos"].append({"path": str(path), **probe_video(path)})
            except AuditError as error:
                _issue(report, "errors", "video_decode", str(error), path=str(path))
        staged = {}
        for filename in ("sequence.npz", "meta.json", "events.json", "bad_frame_ratio.json", "event_timeline.csv"):
            path = _path(staging_root, f"{identifier.lower()}/{filename}")
            staged[filename] = _fingerprint(path)
            report["sources"].append({**staged[filename], "role": f"staged_{filename}"})
        meta, annotation, quality = (_json(directory / name) for name in ("meta.json", "events.json", "bad_frame_ratio.json"))
        info = _json(grouped["info"][0]) if grouped.get("info") else {}
        for role, filename in (("annotation", "events.json"), ("bad_frame_ratio", "bad_frame_ratio.json")):
            source = next((item for item in report["sources"] if item["role"] == role), None)
            if source and source["sha256"] != staged[filename]["sha256"]:
                _issue(report, "errors", "staged_source_mismatch", f"{filename} differs from official source bytes")
        if meta.get("source_episode_uuid") != identifier.lower() or meta.get("coordinate_system") != "world":
            _issue(report, "errors", "staged_metadata", "Staged UUID or coordinate_system disagrees with native world staging")
        provenance = meta.get("provenance", {})
        if not isinstance(provenance, dict) or not isinstance(provenance.get("manifest"), dict):
            raise AuditError("Staging provenance must contain a manifest object")
        if provenance.get("manifest", {}).get("sha256") != manifest_digest:
            _issue(report, "errors", "manifest_provenance", "Staging manifest SHA256 differs from audited manifest")
        saved_sources = provenance.get("verified_source_files")
        if not isinstance(saved_sources, list) or any(not isinstance(saved, dict) for saved in saved_sources):
            raise AuditError("Staging provenance requires verified_source_files records")
        for source in report["sources"]:
            if source["role"] not in ("info", "annotation", "bad_frame_ratio", "data"):
                continue
            saved = [item for item in saved_sources if item.get("path") == source["relative_path"]]
            if len(saved) != 1 or source["sha256"] != saved[0].get("sha256"):
                _issue(report, "errors", "source_provenance", "Staged source digest no longer matches current payload")
        with np.load(directory / "sequence.npz", allow_pickle=False) as archive:
            sequence = {key: archive[key] for key in archive.files}
        native = _sequence(report, sequence, meta, info, quality, episode)
        for video in report["videos"]:
            _timing(report, native, video)
        _events(report, annotation, sequence, directory / "event_timeline.csv")
    except (AuditError, OSError, ValueError, TypeError, KeyError, OverflowError, EOFError, zipfile.BadZipFile) as error:
        _issue(report, "errors", "staging_invalid", f"{type(error).__name__}: {error}")
    for video in report["videos"]:
        video.pop("pts", None)
    report["summary"] = {"error_count": len(report["errors"]), "warning_count": len(report["warnings"]),
                         "passed": not report["errors"]}
    return report


def audit_manifest(manifest_path, dataset_root, staging_root):
    manifest_path, dataset_root, staging_root = [Path(value).expanduser().resolve() for value in (
        manifest_path, dataset_root, staging_root,
    )]
    report = {"audit_version": 1, "errors": [], "warnings": [], "episodes": [],
              "limitations": ["Parquet payload hashes verified; raw coordinate comparisons use staged raw arrays, not a second parquet read.",
                              "No coordinate unit, extrinsic direction, object trajectory or HOI label certification.",
                              "Event range, overlap and wrist-speed outliers are diagnostics, not automatic label rejection."],
              "dataset_root": str(dataset_root), "staging_root": str(staging_root), "numpy_version": np.__version__,
              "auditor": _fingerprint(Path(__file__).resolve())}
    try:
        report["manifest"] = _fingerprint(manifest_path)
        episodes = _json(manifest_path).get("selected_episodes")
        if not isinstance(episodes, list) or not episodes:
            raise AuditError("Manifest requires a nonempty selected_episodes list")
        seen = set()
        for episode in episodes:
            if not isinstance(episode, dict):
                raise AuditError("Manifest episodes must be objects")
            identifier = episode.get("source_episode_uuid")
            if not isinstance(identifier, str) or identifier.lower() in seen:
                raise AuditError("Duplicate or missing episode UUID")
            seen.add(identifier.lower())
            result = audit_episode(episode, dataset_root, staging_root, report["manifest"]["sha256"])
            report["episodes"].append(result)
            for level in ("errors", "warnings"):
                report[level].extend({"source_episode_uuid": identifier, **item} for item in result[level])
    except (AuditError, OSError, ValueError, TypeError) as error:
        _issue(report, "errors", "manifest_invalid", f"{type(error).__name__}: {error}")
    report["summary"] = {"episode_count": len(report["episodes"]), "error_count": len(report["errors"]),
                         "warning_count": len(report["warnings"]), "passed": not report["errors"],
                         "decoded_frames": sum(video["decoded_frame_count"] for episode in report["episodes"] for video in episode["videos"]),
                         "native_frames": sum(episode.get("sequence", {}).get("frame_count", 0) for episode in report["episodes"])}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "dataset-root", "staging-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.exit(2, "error: output already exists; choose a NEW report path\n")
    report = audit_manifest(args.manifest, args.dataset_root, args.staging_root)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as error:
        parser.exit(2, f"error: cannot create report ({type(error).__name__})\n")
    print(json.dumps(report["summary"], ensure_ascii=False, allow_nan=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
