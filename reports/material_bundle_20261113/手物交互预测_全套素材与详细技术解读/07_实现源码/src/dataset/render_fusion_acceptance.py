"""Render calibrated hand/camera evidence while blocking missing object 6D acceptance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from src.dataset.check_camera_rig import quaternion_matrices
from src.dataset.audit_egodemo import probe_video
from src.dataset.fusion_geometry import HAND_EDGES, camera_to_world, project_pinhole, world_to_camera
from src.dataset.import_egodemo import verify_payload


HAND_COLORS = ((90, 225, 100), (40, 180, 255))
AXIS_COLORS = ((85, 85, 250), (80, 210, 90), (240, 155, 70))
BACKGROUND = (27, 23, 20)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, contents):
    Path(path).write_text(json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verified_source_file(path, record):
    return {**verify_payload(path, record), "relative_path": record["path"], "path": str(path.resolve())}


def validated_episode_payload(episode, record, dataset_root):
    base = PurePosixPath(episode["lerobot_path"])
    task = PurePosixPath(episode["task_name"])
    if task.is_absolute() or not task.parts or ".." in task.parts:
        raise ValueError("Task path must be a nonempty relative path without parent traversal")
    expected = (episode["variant"], "lerobot", *task.parts, episode["source_episode_uuid"])
    if episode["variant"] not in ("EgoProStandard", "EgoProStandard-body") or base.parts != expected:
        raise ValueError("Manifest variant/task/UUID disagrees with lerobot_path")
    suffixes = {
        "episodes": "meta/episodes/chunk-000/file-000.parquet",
        "head_left_video": "videos/observation.images.head_left/chunk-000/file-000.mp4",
    }
    if record["role"] not in suffixes or record["path"] != str(base / suffixes[record["role"]]):
        raise ValueError("Payload must belong to the requested annotated episode and head_left camera")
    root = Path(dataset_root).resolve()
    path = root / record["path"]
    if not path.resolve().is_relative_to(root):
        raise ValueError("Payload path escapes dataset root")
    return path.resolve()


def validate_video_alignment(sequence, video):
    native = np.asarray(sequence["timestamps"], dtype=np.float64)
    indices = np.asarray(sequence["frame_index"])
    presentation = np.asarray(video["pts"], dtype=np.float64)
    if native.ndim != 1 or not len(native) or not np.isfinite(native).all() or np.any(np.diff(native) <= 0):
        raise ValueError("Native timestamps must be finite and strictly increasing")
    if indices.dtype.kind not in "iu" or not np.array_equal(indices, np.arange(len(native))):
        raise ValueError("Native frame_index must be exactly zero-based sequential video indices")
    if presentation.shape != native.shape or video["decoded_frame_count"] != len(native):
        raise ValueError("Decoded video PTS/frame count differs from staged sequence")
    if not np.isfinite(presentation).all() or np.any(np.diff(presentation) <= 0):
        raise ValueError("Decoded video PTS must be finite and strictly increasing")
    error = float(np.max(np.abs(native - presentation)))
    if abs(float(presentation[0])) > 1e-4 or abs(float(native[0])) > 1e-4 or error > 1e-4:
        raise ValueError("Native timestamps disagree with zero-origin video PTS")
    return {"decoded_frame_count": len(native), "max_absolute_pts_error_seconds": error,
            "tolerance_seconds": 1e-4, "frame_index_zero_based_sequential": True,
            "method": "ffprobe full decode with per-frame PTS; absolute timestamp comparison"}


def calibrated_intrinsics(row, source_uuid, frame_count):
    if row.get("episode_uuid") != source_uuid or row.get("length") != frame_count:
        raise ValueError("Calibration episode UUID or frame count differs from staged sequence")
    values = np.asarray(row.get("camera_intrinsics/head_left"), dtype=np.float64)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("Expected eight finite annotated head_left intrinsic coefficients")
    if np.any(values[4:] != 0):
        raise ValueError("Nonzero distortion requires an explicitly supported model; no silent pinhole fallback")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    return dict(zip(("fx", "fy", "cx", "cy"), values[:4].tolist()))


def load_episode(episode, dataset_root, staging_root):
    import pyarrow.parquet as parquet

    identifier = episode["source_episode_uuid"]
    directory = Path(staging_root) / identifier
    meta = json.loads((directory / "meta.json").read_text())
    if (meta.get("source_episode_uuid") != identifier or meta.get("coordinate_system") != "world"
            or meta.get("quaternion_order") != ["qw", "qx", "qy", "qz"]
            or meta.get("world_to_camera_transform_applied") is not False):
        raise ValueError("Staged input is not the expected untransformed world sequence")
    with np.load(directory / "sequence.npz", allow_pickle=False) as archive:
        sequence = {key: archive[key] for key in archive.files}
    source_files = {}
    for role in ("episodes", "head_left_video"):
        records = [record for record in episode["files"] if record["role"] == role]
        if len(records) != 1:
            raise ValueError(f"Expected exactly one {role} payload")
        path = validated_episode_payload(episode, records[0], dataset_root)
        source_files[role] = verified_source_file(path, records[0])
    rows = parquet.read_table(source_files["episodes"]["path"]).to_pylist()
    if len(rows) != 1:
        raise ValueError("Calibration table must describe one episode")
    count = len(sequence["timestamps"])
    intrinsics = calibrated_intrinsics(rows[0], identifier, count)
    shape = meta["provenance"]["source_info"]["features"]["observation.images.head_left"]["shape"]
    height, width = int(shape[0]), int(shape[1])
    if rows[0]["videos/observation.images.head_left/from_timestamp"] != 0:
        raise ValueError("Nonzero annotated video offset requires explicit alignment")
    capture = cv2.VideoCapture(source_files["head_left_video"]["path"])
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open annotated video")
        observed = tuple(int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_COUNT))
        if observed != (height, width, count):
            raise ValueError(f"Annotated video dimensions/frame count disagree: {observed}")
    finally:
        capture.release()
    timing = validate_video_alignment(sequence, probe_video(source_files["head_left_video"]["path"]))
    return {
        "uuid": identifier, "task_name": episode["task_name"], "sequence": sequence,
        "intrinsics": intrinsics, "width": width, "height": height,
        "video_path": source_files["head_left_video"]["path"],
        "provenance": {
            "source_files": source_files,
            "sequence_path": str(directory / "sequence.npz"),
            "sequence_sha256": digest(directory / "sequence.npz"),
            "meta_sha256": digest(directory / "meta.json"),
            "video_alignment": timing,
            "calibration_field": "camera_intrinsics/head_left",
            "calibration_source": "same annotated episode meta/episodes parquet; not Raw",
            "camera_frame": "head_left OpenCV: x right, y down, z forward",
            "transform": "p_camera = R_c2w.T @ (p_world - camera_position_world)",
            "units": "meters according to publisher; independent scale measurement unavailable",
        },
    }


def put_text(canvas, text, position, scale=0.58, color=(225, 225, 225)):
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_skeleton(canvas, points, color, offset=(0, 0)):
    coordinates = np.asarray(points, dtype=np.float64) + np.asarray(offset)
    safe = np.isfinite(coordinates).all(axis=1) & (np.abs(coordinates) < 100000).all(axis=1)
    for start, end in HAND_EDGES:
        if safe[start] and safe[end]:
            cv2.line(canvas, tuple(np.rint(coordinates[start]).astype(int)), tuple(np.rint(coordinates[end]).astype(int)), color, 2, cv2.LINE_AA)
    for point in coordinates[safe]:
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)


def draw_geometry_panel(canvas, bounds, hands, camera_position, camera_rotation, center, span, basis, title):
    left, top, width, height = bounds
    panel = np.full((height, width, 3), (38, 33, 29), dtype=np.uint8)
    put_text(panel, title, (12, 25), 0.55)
    scale = min(width - 60, height - 80) / span

    def project(values):
        projected = (np.asarray(values) - center) @ basis.T
        return projected * [scale, -scale] + [width / 2, height / 2 + 12]

    for hand_index, hand in enumerate(hands):
        draw_skeleton(panel, project(hand), HAND_COLORS[hand_index])
    center_pixel = project(camera_position)
    if np.isfinite(center_pixel).all():
        center_tuple = tuple(np.rint(center_pixel).astype(int))
        cv2.circle(panel, center_tuple, 5, (240, 240, 240), -1, cv2.LINE_AA)
        for axis_index, axis_name in enumerate(("X", "Y", "Z")):
            endpoint = project(camera_position + camera_rotation[:, axis_index] * .08)
            end_tuple = tuple(np.rint(endpoint).astype(int))
            cv2.arrowedLine(panel, center_tuple, end_tuple, AXIS_COLORS[axis_index], 2, cv2.LINE_AA, tipLength=.2)
            put_text(panel, axis_name, (end_tuple[0] + 3, end_tuple[1]), .4, AXIS_COLORS[axis_index])
    ruler = min(.1, span / 3)
    cv2.line(panel, (16, height - 28), (16 + int(ruler * scale), height - 28), (210, 210, 210), 2)
    put_text(panel, f"{ruler:.2f} m | camera axes, NOT object axes", (16, height - 8), .35)
    canvas[top:top + height, left:left + width] = panel


def build_panel(frame, data, frame_index, points_uv, points_camera, center, span):
    canvas = np.full((1080, 1600, 3), BACKGROUND, dtype=np.uint8)
    sequence = data["sequence"]
    put_text(canvas, "REAL HAND / CAMERA GEOMETRY", (22, 32), .85, (230, 235, 235))
    put_text(canvas, "NOT ACCEPTED: OBJECT 6D MISSING", (920, 32), .65, (90, 100, 255))
    put_text(canvas, f"{data['uuid']}  frame {frame_index}  t={sequence['timestamps'][frame_index]:.3f}s", (22, 65), .55)
    put_text(canvas, "Green = left hand | Orange = right hand | white origin + XYZ = CAMERA", (22, 90), .51)
    resized = cv2.resize(frame, (960, 728), interpolation=cv2.INTER_AREA)
    for hand_index in range(2):
        scaled = points_uv[frame_index, hand_index] * [960 / data["width"], 728 / data["height"]]
        draw_skeleton(resized, scaled, HAND_COLORS[hand_index])
    canvas[110:838, 20:980] = resized
    hands_world = np.stack([sequence["left_hand_world"][frame_index], sequence["right_hand_world"][frame_index]])
    rotation = quaternion_matrices(sequence["head_left_camera_rotation"][frame_index:frame_index + 1])[0]
    position = sequence["head_left_camera_position"][frame_index]
    frame_points = np.concatenate([hands_world.reshape(-1, 3), position[None]])
    frame_points = frame_points[np.isfinite(frame_points).all(axis=1)]
    lower, upper = frame_points.min(axis=0), frame_points.max(axis=0)
    center, span = (lower + upper) / 2, max(float(np.max(upper - lower)) * 1.4, .65)
    isometric = np.array([[.70710678, -.70710678, 0], [.40824829, .40824829, .81649658]])
    draw_geometry_panel(canvas, (1000, 110, 580, 350), hands_world, position, rotation, center, span, isometric, "WORLD 3D: fixed axes / auto-fit viewport")
    draw_geometry_panel(canvas, (1000, 480, 580, 350), points_camera[frame_index], np.zeros(3), np.eye(3), np.array([0, .12, .3]), .8,
                        np.array([[1, 0, 0], [0, 0, 1]]), "CAMERA X/Z: optical depth forward")
    draw_geometry_panel(canvas, (20, 855, 470, 210), hands_world, position, rotation, center, span,
                        np.array([[1, 0, 0], [0, 1, 0]]), "WORLD XY: top")
    draw_geometry_panel(canvas, (510, 855, 470, 210), hands_world, position, rotation, center, span,
                        np.array([[1, 0, 0], [0, 0, 1]]), "WORLD XZ: side")
    put_text(canvas, "ANNOTATED HANDS + SAME-EPISODE CALIBRATION", (1000, 875), .49)
    put_text(canvas, "No mock object, no guessed camera K.", (1000, 905), .5)
    put_text(canvas, "Reprojection accuracy: NOT EVALUATED (no GT 2D)", (1000, 935), .46)
    put_text(canvas, "Joint hand-object acceptance: BLOCKED", (1000, 965), .5, (90, 100, 255))
    put_text(canvas, "This is a hand/camera diagnostic, not full fusion.", (1000, 995), .45)
    put_text(canvas, "Camera world xyz (m): " + ", ".join(f"{value:.3f}" for value in position), (1000, 1025), .43)
    return canvas


def evaluate_episode(data, output_dir, preview_start=8.0, preview_seconds=10.0):
    output_dir.mkdir(parents=True, exist_ok=False)
    sequence = data["sequence"]
    hands = np.stack([sequence["left_hand_world"], sequence["right_hand_world"]], axis=1)
    positions, rotations = sequence["head_left_camera_position"], sequence["head_left_camera_rotation"]
    camera = world_to_camera(hands, positions, rotations)
    projection = project_pinhole(camera, data["intrinsics"], data["width"], data["height"])
    recovered = camera_to_world(camera, positions, rotations)
    difference = np.linalg.norm(recovered - hands, axis=-1)
    count = len(hands)
    valid = sequence["hand_valid"]
    if not np.array_equal(valid, np.isfinite(hands).all(axis=(2, 3))):
        raise ValueError("Staged validity and finite hand rows disagree")
    positive = projection["positive_depth"].all(axis=-1) & valid
    inside = projection["inside_image"].all(axis=-1) & valid
    header = ["frame_index", "timestamp_seconds", "left_valid", "right_valid", "left_positive_depth", "right_positive_depth",
              "left_inside_joint_count", "right_inside_joint_count", "object_real_pose_available", "joint_eligible", "joint_pass",
              "independent_reprojection_error_px", "world_camera_roundtrip_max_m"]
    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for frame_index in range(count):
            finite_difference = difference[frame_index][np.isfinite(difference[frame_index])]
            writer.writerow({
                "frame_index": int(sequence["frame_index"][frame_index]), "timestamp_seconds": float(sequence["timestamps"][frame_index]),
                "left_valid": int(valid[frame_index, 0]), "right_valid": int(valid[frame_index, 1]),
                "left_positive_depth": int(positive[frame_index, 0]), "right_positive_depth": int(positive[frame_index, 1]),
                "left_inside_joint_count": int(projection["inside_image"][frame_index, 0].sum()),
                "right_inside_joint_count": int(projection["inside_image"][frame_index, 1].sum()),
                "object_real_pose_available": 0, "joint_eligible": 0, "joint_pass": "", "independent_reprojection_error_px": "",
                "world_camera_roundtrip_max_m": float(finite_difference.max()) if finite_difference.size else "",
            })
    world_points = np.concatenate([hands.reshape(-1, 3), positions])
    world_points = world_points[np.isfinite(world_points).all(axis=1)]
    lower, upper = np.min(world_points, axis=0), np.max(world_points, axis=0)
    center, span = (lower + upper) / 2, max(float(np.max(upper - lower)) * 1.15, .8)
    selected = sorted(set(int(value) for value in np.linspace(0, count - 1, 6)) | {
        min(int(np.searchsorted(sequence["timestamps"], moment)), count - 1) for moment in (9.0, 9.5, 21.9667)
    })
    capture = cv2.VideoCapture(data["video_path"])
    pngs = []
    try:
        for frame_index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index + 1:
                raise ValueError(f"Failed exact video-frame read {frame_index}")
            panel = build_panel(frame, data, frame_index, projection["points_uv"], camera, center, span)
            filename = f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(output_dir / filename), panel):
                raise OSError("Could not save evidence PNG")
            pngs.append({"file": filename, "frame_index": frame_index, "timestamp": float(sequence["timestamps"][frame_index])})
        start = int(np.searchsorted(sequence["timestamps"], preview_start))
        end = min(int(np.searchsorted(sequence["timestamps"], preview_start + preview_seconds)), count)
        if start >= end:
            raise ValueError("Preview interval outside episode")
        native_fps = 1.0 / float(np.median(np.diff(sequence["timestamps"])))
        stride = max(1, round(native_fps / 10))
        writer = cv2.VideoWriter(str(output_dir / "hand_camera_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), native_fps / stride, (1280, 864))
        if not writer.isOpened():
            raise OSError("Cannot open preview video writer")
        rendered_frames = 0
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for frame_index in range(start, end):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Preview decoding ended early")
                if (frame_index - start) % stride == 0:
                    panel = build_panel(frame, data, frame_index, projection["points_uv"], camera, center, span)
                    writer.write(cv2.resize(panel, (1280, 864), interpolation=cv2.INTER_AREA))
                    rendered_frames += 1
        finally:
            writer.release()
    finally:
        capture.release()
    summary = {
        "source_episode_uuid": data["uuid"], "task_name": data["task_name"], "frames": count,
        "status": "blocked", "overall_accepted": False,
        "blockers": ["real_object_6d_missing", "independent_hand_reprojection_reference_missing", "independent_object_pose_reference_missing"],
        "intrinsics": data["intrinsics"], "provenance": data["provenance"],
        "valid_hand_frames": {side: int(valid[:, index].sum()) for index, side in enumerate(("left", "right"))},
        "both_hands_valid_frames": int(valid.all(axis=1).sum()),
        "positive_depth_hand_frames": {side: int(positive[:, index].sum()) for index, side in enumerate(("left", "right"))},
        "whole_hand_inside_image_frames": {side: int(inside[:, index].sum()) for index, side in enumerate(("left", "right"))},
        "roundtrip_max_m": float(np.nanmax(difference)), "roundtrip_is_accuracy_metric": False,
        "real_object_pose_frames": 0, "joint_eligible_frames": 0, "joint_evaluated_frames": 0,
        "joint_success_rate": None, "pose_accuracy_success_rate": None, "independent_reprojection_error_px": None,
        "images": pngs, "preview": {"file": "hand_camera_preview.mp4", "start_frame": start, "end_frame_exclusive": end,
                                     "rendered_frames": rendered_frames, "fps": native_fps / stride},
        "limitation": "Hand/camera-only diagnostic. Finite points, visibility, in-frame coverage and inverse consistency are not pose accuracy or fused acceptance.",
    }
    write_json(output_dir / "acceptance.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "dataset-root", "staging-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--episode", action="append", required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory must be new")
    manifest = json.loads(args.manifest.read_text())
    episodes = {episode["source_episode_uuid"]: episode for episode in manifest["selected_episodes"]}
    if len(args.episode) != len(set(args.episode)) or any(identifier not in episodes for identifier in args.episode):
        parser.error("Episodes must be unique UUIDs present in the manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for identifier in args.episode:
        data = load_episode(episodes[identifier], args.dataset_root, args.staging_root)
        summaries.append(evaluate_episode(data, args.output_dir / identifier))
        print(json.dumps({"episode": identifier, "status": summaries[-1]["status"], "frames": summaries[-1]["frames"]}), flush=True)
    report = {
        "status": "blocked", "overall_accepted": False, "episodes": summaries,
        "manifest_sha256": digest(args.manifest), "script_sha256": digest(__file__),
        "joint_success_rate": None, "reason": "No real object 6D inputs or independent accuracy references; diagnostic hand/camera evidence only.",
    }
    write_json(args.output_dir / "acceptance.json", report)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
