"""Causal human-hand + FoundationPose RGB-D replay with measured latency.

Both real estimators consume the same frame before it is rendered. A recorded
sequence is not a live camera connection. Image-normalized hand coordinates
are never relabeled as metric camera-space joints. No reference poses are read.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import inspect
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from src.hand_pose.extract_hands import save_npz
from src.pose6d.extract_pose6d import write_pose6d_npz
from src.pose6d.run_rgbd_sequence import (
    create_estimator, file_record, load_scene, read_frame, render_frame, validate_pose,
)


def summarize_timing(rows, source_fps):
    warm = rows[1:]
    if not warm:
        return {"steady_frames": 0, "steady_processing_fps": None,
                "steady_latency_p50_ms": None, "steady_latency_p95_ms": None,
                "keeps_up_with_source_steady": None}
    durations = np.array([row["total_ms"] for row in warm], dtype=np.float64)
    rate = float(1000.0 / durations.mean())
    percentile = float(np.percentile(durations, 95))
    return {"steady_frames": len(warm), "steady_processing_fps": rate,
            "steady_latency_p50_ms": float(np.median(durations)),
            "steady_latency_p95_ms": percentile,
            "keeps_up_with_source_steady": bool(rate >= source_fps and percentile <= 1000 / source_fps),
            "criterion": "mean throughput >= source FPS AND p95 frame work <= frame period; excludes registration, pacing sleep, model loading, preflight",
            "registration_frame_ms": rows[0]["total_ms"],
            "source_read_to_output_p95_ms": float(np.percentile(
                [row.get("source_read_to_output_ms", row["total_ms"]) for row in warm], 95)),
            "source_latency_meaning": "file-read start to rendered video write, including bounded prefetch queue; NOT camera capture-to-display latency",
            "live_camera_validated": False}


def checked_hands(result):
    packed = {}
    for name in ("left_hand", "right_hand"):
        values = np.asarray(result[name], dtype=np.float32)
        if values.shape != (21, 3):
            raise ValueError(f"{name} must be (21,3)")
        if not (np.isfinite(values).all() or np.isnan(values).all()):
            raise ValueError(f"{name} must be fully finite or fully NaN")
        packed[name] = values.copy()
    scores = np.asarray(result["conf"], dtype=np.float32)
    if scores.shape != (2,):
        raise ValueError("hand confidence must be (2,)")
    for index, name in enumerate(("left_hand", "right_hand")):
        detected = np.isfinite(packed[name]).all()
        if detected and (not np.isfinite(scores[index]) or not 0 <= scores[index] <= 1):
            raise ValueError("detected hands require finite scores in [0,1]")
        if not detected and not np.isnan(scores[index]):
            raise ValueError("missing hand scores must be NaN")
    packed["conf"] = scores.copy()
    return packed


def infer_hands(hand_backend, rgb, timestamp_ms):
    started = time.perf_counter()
    result = checked_hands(hand_backend.process(rgb, timestamp_ms))
    return result, (time.perf_counter() - started) * 1000


def decode_input(record):
    started = time.perf_counter()
    frame = read_frame(record)
    return frame, started, (time.perf_counter() - started) * 1000


def run_pipeline(args, hand_backend=None, pose_estimator=None):
    from src.hand_pose.online_hands import OnlineHands, draw_hands
    import cv2
    import trimesh

    output = Path(args.out_dir).expanduser()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    injected = hand_backend is not None or pose_estimator is not None
    report = {"status": "running", "input_type": "recorded_rgbd_replay",
              "dataset_name": args.dataset_name, "viewpoint": args.viewpoint,
              "is_egodemo": args.dataset_name == "EgoDemo", "live_camera_validated": False,
              "real_models": not injected, "frames_completed": 0,
              "accuracy": {"hand": None, "object_translation": None, "object_rotation": None, "joint": None},
              "coordinate_warning": "Hands: normalized image XY and relative Z. Object: original mesh -> camera, metres. Shared image overlay is NOT calibrated 3D hand-object fusion.",
              "parameters": {name: str(value) if isinstance(value, Path) or
                             (isinstance(value, float) and not np.isfinite(value)) else value
                             for name, value in vars(args).items()}, "phase": "input_validation"}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="human_hands")
    input_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgbd_decode")
    writer = None
    timings, poses, boxes, hands, bbox_sources = [], [], [], [], []
    forecaster, forecast_outputs, forecast_corners = None, [], None
    previous_opencv_threads = cv2.getNumThreads()
    try:
        if not np.isfinite(args.source_fps) or args.source_fps <= 0:
            raise ValueError("source_fps must be finite and positive with a stated source")
        if not args.fps_source.strip() or not args.human_evidence.strip():
            raise ValueError("source FPS and human input need explicit evidence")
        if min(args.est_refine_iter, args.track_refine_iter, args.torch_threads, args.opencv_threads, args.save_every) < 1:
            raise ValueError("iterations, threads and save_every must be positive")
        cv2.setNumThreads(args.opencv_threads)
        if args.display and os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("--display requires a desktop display; headless replay still saves video")
        intrinsics, frames, calibration = load_scene(args.scene_dir, args.limit_frames, args.depth_scale)
        report["frames_requested"] = len(frames)
        mesh_path = Path(args.mesh_file).resolve(strict=True)
        mesh = trimesh.load(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
            raise ValueError("mesh must be a finite nondegenerate original metre-scale mesh")
        if args.forecast_method != "none":
            from src.forecasting.online import OnlineForecaster
            from src.forecasting.render import draw_forecast_overlay

            if not injected and args.viewpoint != "third_person_fixed_camera":
                raise ValueError("forecast prototype requires fixed-camera inputs, not unstabilized egocentric video")
            forecaster = OnlineForecaster(args.forecast_method, args.forecast_window, args.forecast_horizons,
                                          args.forecast_checkpoint, args.forecast_device)
            forecast_corners = np.asarray(list(itertools.product(*mesh.bounds.T)))
            report["forecast"] = {"method": args.forecast_method, "window_size": args.forecast_window,
                                  "horizons_seconds": args.forecast_horizons, "future_frame_inference": False,
                                  "coordinate_mode": "mixed", "independent_accuracy": None,
                                  "predictor_source": file_record(inspect.getfile(OnlineForecaster)),
                                  "checkpoint": file_record(args.forecast_checkpoint) if args.forecast_checkpoint else None}
            report["forecast"]["display_method"] = forecaster.display_label
            if forecaster.model_metadata is not None:
                report["forecast"]["checkpoint_status"] = {
                    name: forecaster.model_metadata[name] for name in
                    ("model_status", "epoch", "optimizer_steps", "selection", "checkpoint_role")}
        report["phase"] = "model_initialization"
        if hand_backend is None:
            hand_backend = executor.submit(
                OnlineHands, input_mirrored=args.input_mirrored,
                model_complexity=args.hand_model_complexity,
                min_detection_confidence=args.hand_detection_threshold,
                min_tracking_confidence=args.hand_tracking_threshold,
            ).result()
        hand_metadata = executor.submit(hand_backend.metadata).result()
        synchronize = lambda: None
        pose_metadata = {"name": "injected_test_only", "real_model": False}
        if pose_estimator is None:
            import torch

            torch.set_num_threads(args.torch_threads)
            pose_estimator, pose_metadata, synchronize = create_estimator(
                args.foundationpose_repo, mesh, output / "backend_debug",
            )
        report["hand_backend"] = hand_metadata
        report["object_backend"] = pose_metadata
        height, width = frames[0]["image_shape"]
        video_size = (width + width % 2, height + height % 2)
        writer = cv2.VideoWriter(str(output / "human_object_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.source_fps, video_size)
        if not writer.isOpened():
            raise RuntimeError("video encoder did not open")
        (output / "overlays").mkdir()
        report["phase"] = "online_inference_and_rendering"
        stream_start = time.perf_counter()
        playback_start = None
        input_future = input_executor.submit(decode_input, frames[0]) if args.prefetch_input else None
        for index, record in enumerate(frames):
            frame_start = time.perf_counter()
            frame, source_read_start, input_decode_ms = input_future.result() if input_future is not None else decode_input(record)
            input_read_ms = (time.perf_counter() - frame_start) * 1000
            if args.prefetch_input and index + 1 < len(frames):
                input_future = input_executor.submit(decode_input, frames[index + 1])
            timestamp_ms = index * 1000.0 / args.source_fps
            hand_future = executor.submit(infer_hands, hand_backend, frame["rgb"].copy(), timestamp_ms)
            object_start = time.perf_counter()
            object_inputs = {"K": intrinsics.copy(), "rgb": frame["rgb"].copy(), "depth": frame["depth"].copy()}
            if index == 0:
                predicted = pose_estimator.register(**object_inputs, ob_mask=frame["mask"].copy(), iteration=args.est_refine_iter)
                if hasattr(pose_estimator, "pose_last") and pose_estimator.pose_last is None:
                    raise RuntimeError("FoundationPose registration returned without initializing tracking")
            else:
                predicted = pose_estimator.track_one(**object_inputs, iteration=args.track_refine_iter)
            synchronize()
            object_ms = (time.perf_counter() - object_start) * 1000
            pose = validate_pose(predicted)
            hand_result, hand_ms = hand_future.result()
            joint_inference_ms = (time.perf_counter() - object_start) * 1000
            render_start = time.perf_counter()
            image, bbox, bbox_source = render_frame(frame, pose, mesh, intrinsics)
            image = draw_hands(image, hand_result)
            forecast_ms = 0.0
            if forecaster is not None:
                hand_values = np.stack([hand_result["left_hand"], hand_result["right_hand"]])
                forecast = forecaster.update(hand_values, pose[None], bbox[None], timestamp_ms / 1000)
                forecast_outputs.append(forecast)
                forecast_ms = forecast["latency_ms"]
                image = draw_forecast_overlay(image, hand_values, pose[None], forecast, intrinsics,
                                              forecast_corners, args.forecast_horizons, include_current=False)
                cv2.rectangle(image, (0, height - 42), (width, height), (18, 18, 18), -1)
                state = "ready" if forecast["ready"] else "history warmup"
                cv2.putText(image, f"FORECAST {forecaster.display_label} | {state} | {forecast_ms:.1f}ms",
                            (8, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
                horizon_label = ", ".join(f"+{value:g}s" for value in args.forecast_horizons)
                cv2.putText(image, f"Dashed: {horizon_label} future estimates | no future input",
                            (8, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            present = [bool(np.isfinite(hand_result[name]).all()) for name in ("left_hand", "right_hand")]
            cv2.rectangle(image, (0, 0), (width, 54), (18, 18, 18), -1)
            cv2.putText(image, f"HUMAN HANDS + FoundationPose | {args.dataset_name} | {index + 1}/{len(frames)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"L:{int(present[0])} R:{int(present[1])} | hand {hand_ms:.1f}ms | object {object_ms:.1f}ms | RGB-D replay", (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)
            render_ms = (time.perf_counter() - render_start) * 1000
            write_start = time.perf_counter()
            writer.write(cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT))
            if index % args.save_every == 0 or index == len(frames) - 1:
                if not cv2.imwrite(str(output / "overlays" / f"{index:06d}_{record['stem']}.png"), image):
                    raise RuntimeError("overlay PNG export failed")
            if args.display:
                cv2.imshow("Human hands + FoundationPose", image)
                if cv2.waitKey(1) & 0xFF == 27:
                    raise InterruptedError("display stopped by user before full sequence completion")
            output_write_ms = (time.perf_counter() - write_start) * 1000
            total_ms = (time.perf_counter() - frame_start) * 1000
            timings.append({"frame_index": index, "stem": record["stem"], "timestamp_ms": timestamp_ms,
                            "hand_ms": hand_ms, "object_ms": object_ms, "total_ms": total_ms, "forecast_ms": forecast_ms,
                            "input_read_ms": input_read_ms, "joint_inference_ms": joint_inference_ms,
                            "input_decode_ms": input_decode_ms,
                            "source_read_to_output_ms": (time.perf_counter() - source_read_start) * 1000,
                            "render_ms": render_ms, "output_write_ms": output_write_ms,
                            "left_detected": int(present[0]), "right_detected": int(present[1]),
                            "object_valid": 1, "joint_covered": int(any(present)), "bbox_source": bbox_source})
            poses.append(pose)
            boxes.append(bbox)
            hands.append(hand_result)
            bbox_sources.append(bbox_source)
            report["frames_completed"] = len(poses)
            if (index + 1) % 25 == 0 or index == len(frames) - 1:
                print(f"[human-rgbd] {index + 1}/{len(frames)} | hand {hand_ms:.1f}ms object {object_ms:.1f}ms frame {total_ms:.1f}ms", flush=True)
            if args.pace_replay:
                if playback_start is None:
                    playback_start = time.perf_counter()
                time.sleep(max(0, playback_start + (index + 1) / args.source_fps - time.perf_counter()))
        writer.release()
        writer = None
        report["stream_elapsed_seconds"] = time.perf_counter() - stream_start
        report["timing"] = summarize_timing(timings, args.source_fps)
        hand_any = np.array([row["joint_covered"] for row in timings], dtype=bool)
        report["coverage"] = {"object_frames": len(poses), "human_hand_frames": int(hand_any.sum()),
                              "joint_frames": int(hand_any.sum()), "joint_rate": float(hand_any.mean()),
                              "meaning": "frames with valid predictions, NOT accuracy"}
        report["phase"] = "export_validation"
        capture = cv2.VideoCapture(str(output / "human_object_overlay.mp4"))
        decoded = 0
        try:
            while True:
                available, image = capture.read()
                if not available:
                    break
                if image.shape[:2] != (video_size[1], video_size[0]):
                    raise RuntimeError("video decode shape mismatch")
                decoded += 1
        finally:
            capture.release()
        if decoded != len(frames):
            raise RuntimeError("video decode frame count mismatch")
        report["video"] = {"decoded_frames": decoded, "fps": args.source_fps, "fps_source": args.fps_source}
        if forecast_outputs:
            forecast_path = output / "forecast_predictions.npz"
            np.savez_compressed(forecast_path, **{
                name: np.asarray([result[name] for result in forecast_outputs])
                for name in ("hand_future", "object_future", "ready", "observation_time", "latency_ms")
            }, horizons_seconds=np.asarray(args.forecast_horizons),
                timestamps=np.asarray([result["observation_time"] for result in forecast_outputs]))
            report["forecast"]["output"] = file_record(forecast_path)
            report["forecast"]["ready_frames"] = sum(bool(result["ready"]) for result in forecast_outputs)
        write_pose6d_npz(str(output / "object_poses.npz"), ["obj_000"], np.stack(poses)[None], np.stack(boxes)[None])
        save_npz(output / "hand_poses.npz", np.stack([result["left_hand"] for result in hands]),
                 np.stack([result["right_hand"] for result in hands]), np.stack([result["conf"] for result in hands]))
        with (output / "frames.csv").open("w", newline="") as stream:
            table = csv.DictWriter(stream, fieldnames=list(timings[0]))
            table.writeheader()
            table.writerows(timings)
        hand_metadata = executor.submit(hand_backend.metadata).result()
        report["hand_backend"] = hand_metadata
        metadata = {"dataset_name": args.dataset_name, "viewpoint": args.viewpoint,
                    "human_evidence": args.human_evidence, "input_type": "recorded_rgbd_replay", "live_camera_validated": False,
                    "num_frames": len(frames), "fps": args.source_fps, "fps_source": args.fps_source,
                    "frame_timestamp_source": "frame_index / declared_source_fps; no claim of hardware timestamps",
                    "input_prefetch": {"enabled": args.prefetch_input, "max_future_decodes": 1,
                                       "future_frame_inference": False, "role": "bounded file decoding producer, not a connected camera"},
                    "pacing_origin": "first rendered registration frame; initialization excluded from playback clock",
                    "hand_backend": hand_metadata, "object_backend": pose_metadata,
                    "hand_coordinate_space": "image_normalized_xy_relative_z", "pose_convention": "original_mesh_object_to_camera",
                    "mesh_units": "m_source_declared", "depth_scale": args.depth_scale,
                    "intrinsics": {"matrix": intrinsics.tolist(), "source": calibration}, "mesh": file_record(mesh_path),
                    "frames": [{"stem": record["stem"], **record["sources"]} for record in frames],
                    "object_ids": ["obj_000"], "bbox_sources": bbox_sources,
                    "runner": file_record(__file__), "hand_npz_writer": file_record(inspect.getfile(save_npz)),
                    "quality_flags": ["no_independent_accuracy", "not_calibrated_3d_hand_fusion"] + (["injected_test_only"] if injected else [])}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if not hand_any.any():
            raise RuntimeError("no human hand detected: this output cannot fulfill human hand-object delivery")
        report.update(status="test_only_success" if injected else "success", phase="complete")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        for cleanup in (
            lambda: writer.release() if writer is not None else None,
            lambda: executor.submit(hand_backend.close).result() if hand_backend is not None else None,
            lambda: executor.shutdown(wait=True),
            lambda: input_executor.shutdown(wait=True),
            lambda: cv2.destroyAllWindows() if args.display else None,
            lambda: cv2.setNumThreads(previous_opencv_threads),
        ):
            try:
                cleanup()
            except Exception as error:
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            if report["status"] != "failed":
                report.update(status="failed", error="resource cleanup failed")
        report["total_wall_seconds"] = time.perf_counter() - start
        (output / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if cleanup_errors and not active_error:
            raise RuntimeError("resource cleanup failed: " + "; ".join(cleanup_errors))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("foundationpose-repo", "scene-dir", "mesh-file", "out-dir", "dataset-name", "viewpoint", "human-evidence", "fps-source"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--hand-model-complexity", type=int, choices=(0, 1), default=1)
    parser.add_argument("--hand-detection-threshold", type=float, default=0.5)
    parser.add_argument("--hand-tracking-threshold", type=float, default=0.5)
    parser.add_argument("--input-mirrored", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--pace-replay", action="store_true")
    parser.add_argument("--prefetch-input", action="store_true", help="decode at most one subsequent recorded frame while current frame runs; no future-frame inference")
    parser.add_argument("--forecast-method", choices=("none", "persistence", "constant_velocity", "learned"), default="none")
    parser.add_argument("--forecast-window", type=int, default=32)
    parser.add_argument("--forecast-horizons", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--forecast-checkpoint")
    parser.add_argument("--forecast-device", default="cpu")
    return parser


if __name__ == "__main__":
    print(json.dumps(run_pipeline(build_parser().parse_args()), indent=2, allow_nan=False))
