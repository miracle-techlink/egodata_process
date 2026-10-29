"""Strict single-object benchmark_demo RGB-D runner, NOT an EgoDemo backend.

Run with ``python -m src.pose6d.run_rgbd_sequence --help``. Scene layout is
rgb/<stem>.png, depth/<stem>.png, masks/<stem>.png, cam_K.txt. Later masks may
be absent; the first real segmentation mask is mandatory. Mesh coordinates
must already be in meters. No resizing, mesh centering, GT initialization,
alignment, or model fallback is performed here. References are <stem>.txt
4x4 original-mesh object-to-camera matrices, read only after ALL inference.
The caller must explicitly confirm their frame/units and supply thresholds.
An injected estimator is exclusively a CPU code-test seam, never model evidence.
"""

import argparse
import csv
import hashlib
import importlib
import inspect
import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .extract_pose6d import write_pose6d_npz
else:
    from extract_pose6d import write_pose6d_npz


def file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite (4,4)")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3, rtol=0)):
        raise ValueError("pose must be rigid with SO(3) rotation")
    if pose[2, 3] <= 0:
        raise ValueError("pose must have positive camera z")
    return pose.copy()


def read_frame(record):
    """Decode one frame, detecting source changes since preflight; bounded RAM."""
    import cv2

    images = {}
    stem = record["stem"]
    for kind, source in record["sources"].items():
        contents = Path(source["path"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != source["sha256"]:
            raise ValueError(f"source changed after preflight: {source['path']}")
        images[kind] = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if any(image is None for image in images.values()):
        raise ValueError(f"cannot decode PNG for frame {stem}")
    color, depth, mask = images["rgb"], images["depth"], images.get("masks")
    if color.ndim != 3 or color.shape[2] not in (3, 4) or color.dtype != np.uint8:
        raise ValueError(f"RGB must be uint8 three/four-channel PNG: {stem}")
    image_shape = record.get("image_shape") or color.shape[:2]
    if (color.shape[:2] != image_shape or min(image_shape) < 2
            or depth.shape != image_shape or (mask is not None and mask.shape[:2] != image_shape)):
        raise ValueError(f"RGB/depth/mask dimensions differ: {stem}")
    depth = depth.astype(np.float32) * record["depth_scale"]
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError(f"depth must be finite and nonnegative: {stem}")
    depth[depth < 0.001] = 0
    if not np.any(depth > 0):
        raise ValueError(f"no positive metric depth: {stem}")
    if mask is not None:
        mask = np.any(mask[..., :3] > 0, axis=2) if mask.ndim == 3 else mask > 0
    if record["first"] and (mask is None or np.count_nonzero(mask & (depth > 0)) < 4):
        raise ValueError("first-frame mask needs at least four valid depth pixels")
    return {"stem": stem, "rgb": np.ascontiguousarray(color[..., :3][..., ::-1]), "depth": depth, "mask": mask}


def load_scene(scene_dir, limit_frames=None, depth_scale=0.001):
    """Preflight selected frames; retain paths/hashes only, not full-sequence images."""
    if (limit_frames is not None and limit_frames < 1) or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("limit_frames and depth_scale must be positive")
    scene_dir = Path(scene_dir)
    paths = {kind: {path.stem: path for path in (scene_dir / kind).glob("*.png")}
             for kind in ("rgb", "depth", "masks")}
    stems = sorted(paths["rgb"])
    if not stems or not paths["rgb"].keys() <= paths["depth"].keys():
        raise ValueError("every RGB frame stem must have a matching depth PNG")
    if not paths["masks"].keys() <= paths["rgb"].keys() or stems[0] not in paths["masks"]:
        raise ValueError("mask stems must match RGB; first-frame real mask is required")
    intrinsics_path = scene_dir / "cam_K.txt"
    intrinsics = np.loadtxt(intrinsics_path)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-8, rtol=0)
            or not np.isclose(intrinsics[1, 0], 0, atol=1e-8)):
        raise ValueError("cam_K.txt must be a finite calibrated pinhole (3,3) matrix")
    frames, image_shape = [], None
    for stem in stems[:limit_frames]:
        files = {kind: mapping[stem] for kind, mapping in paths.items() if stem in mapping}
        record = {"stem": stem, "depth_scale": depth_scale, "first": not frames, "image_shape": image_shape,
                  "sources": {kind: file_record(path) for kind, path in files.items()}}
        image_shape = read_frame(record)["rgb"].shape[:2]
        record["image_shape"] = image_shape
        frames.append(record)
    return intrinsics, frames, file_record(intrinsics_path)


def create_estimator(repo, mesh, debug_dir):
    """Lazy-load the actual official models and CUDA rasterizer; fail closed."""
    repo = Path(repo).expanduser().resolve(strict=True)
    sys.path.insert(0, str(repo))
    import torch
    import nvdiffrast.torch as rasterizer

    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose requires working CUDA; no CPU/mock fallback")
    module = importlib.import_module("estimater")
    if Path(module.__file__).resolve() != repo / "estimater.py":
        raise RuntimeError("estimater was imported from a different FoundationPose repository")
    module.set_seed(0)
    scorer, refiner = module.ScorePredictor(), module.PoseRefinePredictor()
    weights = {}
    for name, predictor in (("scorer", scorer), ("refiner", refiner)):
        checkpoint = Path(predictor.cfg["ckpt_dir"]).resolve(strict=True)
        weights[name] = {"checkpoint": file_record(checkpoint),
                         "config": file_record(checkpoint.parent / "config.yml"),
                         "implementation": file_record(inspect.getfile(type(predictor)))}
    context = rasterizer.RasterizeCudaContext()
    estimator = module.FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                      mesh=mesh, scorer=scorer, refiner=refiner,
                                      glctx=context, debug=0, debug_dir=str(debug_dir))
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return estimator, {"name": "foundationpose", "real_model": True, "repo": str(repo),
                       "seed": 0, "seed_source": "official_estimater.set_seed_before_predictor_initialization",
                       "git_commit": revision, "git_dirty": bool(dirty), "weights": weights,
                       "code": [file_record(repo / name) for name in ("estimater.py", "Utils.py")],
                       "torch": torch.__version__, "cuda": torch.version.cuda,
                       "device": torch.cuda.get_device_name()}, torch.cuda.synchronize


def infer_sequence(frames, intrinsics, estimator, est_refine_iter=5, track_refine_iter=2,
                   synchronize=None):
    """Yield validated snapshots. No reference path or pose enters this API."""
    for index, record in enumerate(frames):
        try:
            frame = read_frame(record)
            inputs = {"K": intrinsics.copy(), "rgb": frame["rgb"].copy(), "depth": frame["depth"].copy()}
            if index == 0:
                pose = estimator.register(**inputs, ob_mask=frame["mask"].copy(), iteration=est_refine_iter)
                if hasattr(estimator, "pose_last") and estimator.pose_last is None:
                    raise RuntimeError("registration did not initialize tracking; refusing upstream placeholder")
            else:
                pose = estimator.track_one(**inputs, iteration=track_refine_iter)
            if synchronize is not None:
                synchronize()
            yield validate_pose(pose)
        except Exception as error:
            raise RuntimeError(f"inference failed at frame {index} ({record['stem']}): {error}") from error


def evaluate_reference(reference_dir, stems, poses, translation_threshold_m, rotation_threshold_deg):
    """Read stem-matched original-mesh references, never align or initialize."""
    references, sources = [], []
    for stem in stems:
        path = Path(reference_dir) / f"{stem}.txt"
        references.append(validate_pose(np.loadtxt(path)))
        sources.append(file_record(path))
    references = np.stack(references).astype(np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    if references.shape != poses.shape:
        raise ValueError("reference/prediction frame count mismatch")
    translations = np.linalg.norm(poses[:, :3, 3] - references[:, :3, 3], axis=1)
    relative = poses[:, :3, :3] @ references[:, :3, :3].transpose(0, 2, 1)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    skew = np.stack((relative[:, 2, 1] - relative[:, 1, 2], relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]), axis=1)
    rotations = np.degrees(np.arctan2(np.linalg.norm(skew, axis=1) / 2, cosine))
    translation_pass = translations <= translation_threshold_m
    rotation_pass = rotations <= rotation_threshold_deg
    return {"evaluated": True, "reference_frame": "original_mesh_object_to_camera_m",
            "alignment": "none", "threshold_kind": "caller_declared_engineering_not_official_benchmark",
            "translation_threshold_m": translation_threshold_m, "rotation_threshold_deg": rotation_threshold_deg,
            "translation_accuracy": float(translation_pass.mean()), "rotation_accuracy": float(rotation_pass.mean()),
            "joint_accuracy": float((translation_pass & rotation_pass).mean()),
            "translation_error_m": translations.tolist(), "rotation_error_deg": rotations.tolist(),
            "reference_files": sources}


def project_points(points, pose, intrinsics):
    camera = np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        raise ValueError("projected geometry must have finite positive camera z")
    pixels = camera @ intrinsics.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 2**30:
        raise ValueError("nonfinite/overflowing image projection")
    return pixels


def render_frame(frame, pose, mesh, intrinsics):
    """Draw the original-mesh AABB and original XYZ axes, not centered demo axes."""
    import cv2

    height, width = frame["rgb"].shape[:2]
    image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
    corners = np.array(list(itertools.product(*mesh.bounds.T)))
    axis_length = float(np.min(mesh.extents) / 2)
    points = np.vstack((corners, np.zeros((1, 3)), np.eye(3) * axis_length))
    pixels = np.rint(project_points(points, pose, intrinsics)).astype(np.int32)
    edges = [(start, start ^ bit) for start in range(8) for bit in (1, 2, 4) if start < (start ^ bit)]
    lines = [(start, end, (0, 255, 255)) for start, end in edges]
    lines += [(8, 9 + axis, color) for axis, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)))]
    for start, end, color in lines:
        visible, first, last = cv2.clipLine((0, 0, width, height), tuple(pixels[start]), tuple(pixels[end]))
        if visible:
            cv2.line(image, first, last, color, 2, cv2.LINE_AA)
    mask = frame["mask"]
    if mask is not None and mask.any():
        rows, columns = np.nonzero(mask)
        bbox, source = [columns.min(), rows.min(), columns.max(), rows.max()], "real_mask"
    else:
        projected = project_points(mesh.vertices, pose, intrinsics)
        lower, upper = np.floor(projected.min(axis=0)), np.ceil(projected.max(axis=0))
        if np.any(upper < 0) or lower[0] > width - 1 or lower[1] > height - 1:
            raise ValueError("mesh projection does not intersect the image")
        bbox = [max(0, lower[0]), max(0, lower[1]), min(width - 1, upper[0]), min(height - 1, upper[1])]
        source = "mesh_projection_missing_or_empty_mask"
    return image, np.asarray(bbox, dtype=np.int32), source


def export_visuals(out_dir, frames, poses, mesh, intrinsics, preview_fps):
    """Headless PNG/MP4 export; fully decode MP4 to reject truncated multi-frame output."""
    import cv2

    (out_dir / "overlays").mkdir()
    height, width = frames[0]["image_shape"]
    size = (width + width % 2, height + height % 2)
    video_path = out_dir / "preview.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), preview_fps, size)
    boxes, sources = [], []
    try:
        if not writer.isOpened():
            raise RuntimeError("MP4 encoder could not open")
        for frame, pose in zip(frames, poses):
            image, bbox, source = render_frame(read_frame(frame), pose, mesh, intrinsics)
            if not cv2.imwrite(str(out_dir / "overlays" / f"{frame['stem']}.png"), image):
                raise RuntimeError(f"PNG export failed: {frame['stem']}")
            writer.write(cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT))
            boxes.append(bbox)
            sources.append(source)
    finally:
        writer.release()
    capture, decoded = cv2.VideoCapture(str(video_path)), 0
    try:
        if not capture.isOpened():
            raise RuntimeError("cannot decode exported MP4")
        while True:
            available, image = capture.read()
            if not available:
                break
            if image.shape[:2] != size[::-1]:
                raise RuntimeError("exported MP4 dimensions mismatch")
            decoded += 1
    finally:
        capture.release()
    if decoded != len(frames) or len(boxes) != len(frames):
        raise RuntimeError(f"multi-frame export mismatch: {decoded}/{len(frames)} decoded")
    return np.stack(boxes), sources, {"decoded_frames": decoded, "size": list(size), "fps": preview_fps,
                                    "fps_source": "preview_only_not_acquisition", "padding": "right_bottom_to_even"}


def validate_options(args):
    if ((args.limit_frames is not None and args.limit_frames < 1)
            or min(args.est_refine_iter, args.track_refine_iter) < 1):
        raise ValueError("frame limit and refinement iterations must be positive")
    if not all(np.isfinite(value) and value > 0 for value in (args.depth_scale, args.preview_fps)):
        raise ValueError("depth scale and preview fps must be finite and positive")
    thresholds = (args.translation_threshold_m, args.rotation_threshold_deg)
    if args.reference_dir:
        if not args.reference_frame_confirmed or any(value is None for value in thresholds):
            raise ValueError("references require --reference-frame-confirmed and both explicit thresholds")
        if not all(np.isfinite(value) and value > 0 for value in thresholds) or thresholds[1] > 180:
            raise ValueError("reference thresholds must be finite, positive, and rotation <= 180 degrees")
    elif args.reference_frame_confirmed or any(value is not None for value in thresholds):
        raise ValueError("reference confirmation/thresholds require --reference-dir")


def run_sequence(args, estimator=None):
    """Run once into a NEW directory. Injected estimators are always marked test-only."""
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "dataset_role": "benchmark_demo", "is_egodemo": False,
              "real_model": estimator is None, "inference_frames_completed": 0,
              "inference_success_rate": 0.0, "inference_rate_meaning": "valid returned poses, NOT GT accuracy",
              "accuracy": {"evaluated": False, "translation_accuracy": None, "rotation_accuracy": None,
                           "joint_accuracy": None, "reason": "no_reference_requested" if not args.reference_dir else "not_evaluated"},
              "parameters": {key: str(value) if isinstance(value, Path) or
                             (isinstance(value, float) and not np.isfinite(value)) else value
                             for key, value in vars(args).items()},
              "generated_at": datetime.now(timezone.utc).isoformat(), "phase": "validation"}
    try:
        import trimesh

        validate_options(args)
        intrinsics, frames, calibration = load_scene(Path(args.scene_dir).expanduser(), args.limit_frames, args.depth_scale)
        report["inference_frames_requested"] = len(frames)
        mesh_path = Path(args.mesh_file).expanduser().resolve(strict=True)
        mesh = trimesh.load(str(mesh_path), process=False)
        if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces)
                or not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.vertex_normals).all()
                or np.any(mesh.extents <= 0)):
            raise ValueError("mesh must be a finite, nondegenerate single mesh in original meter coordinates")
        report["phase"] = "backend_initialization"
        backend, synchronize = {"name": "injected_cpu_test", "real_model": False, "weights": None}, None
        if estimator is None:
            estimator, backend, synchronize = create_estimator(args.foundationpose_repo, mesh, out_dir / "backend_debug")
        report["backend"] = backend
        report["phase"] = "inference"
        inference_started = time.perf_counter()
        poses = []
        for pose in infer_sequence(frames, intrinsics, estimator, args.est_refine_iter, args.track_refine_iter, synchronize):
            poses.append(pose)
            report["inference_frames_completed"] = len(poses)
            report["inference_success_rate"] = len(poses) / len(frames)
            report["inference_elapsed_seconds"] = time.perf_counter() - inference_started
            if len(poses) % 50 == 0 or len(poses) == len(frames):
                print(f"[rgbd] inference {len(poses)}/{len(frames)} frames; "
                      f"elapsed={report['inference_elapsed_seconds']:.1f}s", file=sys.stderr, flush=True)
        poses = np.stack(poses)
        report["phase"] = "post_inference_evaluation"
        if args.reference_dir:
            report["accuracy"] = evaluate_reference(Path(args.reference_dir).expanduser(), [frame["stem"] for frame in frames],
                                                    poses, args.translation_threshold_m, args.rotation_threshold_deg)
        report["phase"] = "export"
        boxes, bbox_sources, video = export_visuals(out_dir, frames, poses, mesh, intrinsics, args.preview_fps)
        with (out_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            table = csv.writer(stream)
            table.writerow(["frame_index", "stem", "object_id", "bbox_source", "x1", "y1", "x2", "y2"]
                           + [f"pose_{row}{column}" for row in range(4) for column in range(4)]
                           + ["translation_error_m", "rotation_error_deg"])
            for index, frame in enumerate(frames):
                errors = [report["accuracy"][key][index] if report["accuracy"]["evaluated"] else None
                          for key in ("translation_error_m", "rotation_error_deg")]
                table.writerow([index, frame["stem"], "obj_000", bbox_sources[index], *boxes[index], *poses[index].flat, *errors])
        metadata = {"dataset_role": "benchmark_demo", "is_egodemo": False,
                    "warning": "Benchmark demo is NOT EgoDemo; injected tests are NOT real-model validation; coverage is NOT accuracy.",
                    "clip_id": Path(args.scene_dir).name, "source_scene": str(Path(args.scene_dir).expanduser().resolve()),
                    "num_frames": len(frames), "num_objects": 1, "object_ids": ["obj_000"],
                    "frame_selection": "sorted_RGB_stems; matching_depth_required; extra_depth_files_ignored",
                    "height": frames[0]["image_shape"][0], "width": frames[0]["image_shape"][1],
                    "fps": None, "fps_source": "not_provided_by_scene", "video": video,
                    "pose_estimator": backend["name"], "backend": backend, "pose_convention": "object_to_camera",
                    "coordinate_space": "camera_opencv", "mesh_frame": "original_unmodified", "mesh_units": "m_caller_supplied",
                    "depth_source": "scene_depth_png", "depth_scale": args.depth_scale, "depth_units": "m",
                    "depth_scale_source": "CLI_or_documented_default_0.001", "depth_below_0.001m": "zeroed_like_official_reader",
                    "detector": "supplied_real_segmentation_mask", "mask_rule": "nonzero_RGB_channels_or_grayscale_ignore_alpha",
                    "bboxes_source_per_frame": bbox_sources, "bbox_convention": "xyxy_inclusive_clipped_pixels",
                    "visualization": "original_mesh_AABB_and_XYZ_axes_length_half_min_extent", "bbox_undetected_value": -1,
                    "intrinsics": {"matrix": intrinsics.tolist(), "source": calibration}, "mesh": file_record(mesh_path),
                    "mesh_sibling_material_files": [file_record(path) for path in sorted(mesh_path.parent.iterdir())
                                                    if path.suffix.lower() in (".mtl", ".png", ".jpg", ".jpeg")],
                    "frames": [{"stem": frame["stem"], **frame["sources"]} for frame in frames],
                    "reference_files": report["accuracy"].get("reference_files", []), "reference_alignment": "none",
                    "runner": file_record(__file__), "npz_writer": file_record(inspect.getfile(write_pose6d_npz)),
                    "parameters": report["parameters"], "generated_at": report["generated_at"],
                    "environment": {"python": platform.python_version(), "numpy": np.__version__, "trimesh": trimesh.__version__},
                    "quality_flags": ["benchmark_demo_not_egodemo", "acquisition_fps_unknown"] + ([] if backend["real_model"] else ["injected_test_only"])}
        write_pose6d_npz(str(out_dir / "poses.npz"), ["obj_000"], poses[None], boxes[None])
        (out_dir / "poses.meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        report.update(status="success" if backend["real_model"] else "test_only_success", phase="complete", video=video,
                      artifacts={"poses": "poses.npz", "metadata": "poses.meta.json", "trajectory": "trajectory.csv",
                                 "overlays": "overlays/", "video": "preview.mp4"})
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["total_elapsed_seconds"] = time.perf_counter() - started
        (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("foundationpose-repo", "scene-dir", "mesh-file", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--limit-frames", type=int, help="positive prefix length; omit for the complete sequence")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="meters per raw depth unit; recorded in metadata")
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--preview-fps", type=float, default=10, help="visualization playback only, NOT acquisition FPS")
    parser.add_argument("--reference-dir", help="optional <stem>.txt references; no automatic scene GT discovery")
    parser.add_argument("--reference-frame-confirmed", action="store_true", help="caller confirms original mesh -> same camera, meters, matching stems")
    parser.add_argument("--translation-threshold-m", type=float)
    parser.add_argument("--rotation-threshold-deg", type=float)
    return parser


if __name__ == "__main__":
    print(json.dumps(run_sequence(build_parser().parse_args()), indent=2, allow_nan=False))
