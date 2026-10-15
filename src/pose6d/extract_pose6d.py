#!/usr/bin/env python3
"""Extract per-object 6D poses from a video clip (task 3).

Pipeline with three pluggable stages:
  1. Detector      -> per-frame bboxes (xyxy)
  2. DepthProvider -> per-frame metric depth (optional)
  3. PoseBackend   -> per-object 4x4 object-to-camera pose (optional)

Output (contract: .agent/memory/03_data_pipeline.md section 3):
  <out_dir>/<clip_id>.npz
    object_ids : (N,) str
    poses      : (N, T, 4, 4) float32, object-to-camera; NaN on undetected frames
    bboxes     : (N, T, 4) int32 xyxy; -1 on undetected frames
                 (int32 cannot hold NaN; sentinel documented in meta.json)
  <out_dir>/<clip_id>.meta.json   (includes depth_source)

bbox-from-file format (--bbox-from-file, detector "file"):
  JSON, either {"<frame_idx>": [{"bbox": [x1,y1,x2,y2], "score": 0.9,
  "label": "mug"}, ...], ...} or a list of per-frame lists. Missing
  frame entries mean no detections that frame. The IoU tracker always
  runs on top, so per-entry track ids are not required.

Examples
--------
Chain test without GPU deps (poses all NaN):
  python src/pose6d/extract_pose6d.py --clip clips/demo01_007.mp4 \
      --out-dir pose6d --bbox-from-file boxes.json --pose-estimator none

Mock pose (deterministic, for pipeline tests):
  ... --pose-estimator mock

Full run on the server (repos pre-pulled under ~/refs):
  ... --depth-source depth_anything \
      --depth-anything-repo ~/refs/Depth-Anything-V2 \
      --depth-anything-ckpt <ckpt>.pth --depth-anything-metric \
      --pose-estimator foundationpose --foundationpose-repo ~/refs/FoundationPose \
      --mesh-dir meshes/ --intrinsics 735 735 640 360
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tracker import IoUTracker, assign_object_ids, sorted_tracks
    from rot_utils import make_transform, nan_poses
else:
    from .tracker import IoUTracker, assign_object_ids, sorted_tracks
    from .rot_utils import make_transform, nan_poses

log = logging.getLogger("pose6d")

BBOX_UNDETECTED = -1  # int32 fill value; contract's NaN rule applies to poses

# ---------------------------------------------------------------------------
# stage 1: detectors
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    bbox: np.ndarray  # xyxy
    score: float = 1.0
    label: Optional[str] = None


class BBoxDetector:
    name = "base"

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> List[Detection]:
        raise NotImplementedError


class FileBBoxDetector(BBoxDetector):
    """Placeholder / manual boxes from a JSON file (--bbox-from-file)."""

    name = "file"

    def __init__(self, path: str):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("frames", data)
            if not isinstance(data, dict):
                raise ValueError("bbox file dict form must map frame_idx -> detections")
            self._by_frame = {int(k): v for k, v in data.items()}
            self._list = None
        elif isinstance(data, list):
            self._by_frame = None
            self._list = data
        else:
            raise ValueError("bbox file must be a dict or a list of frames")

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> List[Detection]:
        if self._list is not None:
            entries = self._list[frame_idx] if frame_idx < len(self._list) else []
        else:
            entries = self._by_frame.get(frame_idx, [])
        out = []
        for e in entries:
            out.append(Detection(
                bbox=np.asarray(e["bbox"], dtype=np.float64).reshape(4),
                score=float(e.get("score", 1.0)),
                label=e.get("label"),
            ))
        return out


class EmptyDetector(BBoxDetector):
    name = "empty"

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> List[Detection]:
        return []


class GroundedSAMDetector(BBoxDetector):
    """Grounded-SAM bbox detector (server only; lazy heavy imports).

    Expects the pre-pulled Grounded-SAM repo layout (GroundingDINO under
    <repo>/GroundingDINO). Masks are not produced here; the pose stage
    falls back to rectangle masks from bboxes until SAM is wired in.
    UNVERIFIED on the dev Mac - validate on the server.
    """

    name = "grounded_sam"

    def __init__(self, repo: str, text_prompt: str,
                 groundingdino_config: str, groundingdino_ckpt: str,
                 box_threshold: float = 0.3, text_threshold: float = 0.25):
        if not repo:
            raise RuntimeError("pass --grounded-sam-repo or set GROUNDED_SAM_REPO")
        repo = str(Path(repo).expanduser())
        sys.path.insert(0, repo)
        sys.path.insert(0, str(Path(repo) / "GroundingDINO"))
        try:
            import torch  # noqa: F401
            from groundingdino.util.inference import load_model, predict
        except Exception as e:  # pragma: no cover - server only
            raise RuntimeError(
                f"Grounded-SAM import failed from {repo}: {e}. "
                "Build it on the server (see .agent/skills/pose6d/SKILL.md); "
                "meanwhile use --bbox-from-file."
            ) from e
        self._predict = predict
        self._torch = torch
        self.model = load_model(groundingdino_config, groundingdino_ckpt)
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> List[Detection]:
        import cv2
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, logits, _phrases = self._predict(
            model=self.model, image=self._torch.tensor(image_rgb).permute(2, 0, 1),
            caption=self.text_prompt, box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        h, w = frame_bgr.shape[:2]
        dets = []
        for box, score in zip(boxes.cpu().numpy(), logits.cpu().numpy()):
            cx, cy, bw, bh = box * np.array([w, h, w, h])
            dets.append(Detection(
                bbox=np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]),
                score=float(score),
            ))
        return dets


# ---------------------------------------------------------------------------
# stage 2: depth providers
# ---------------------------------------------------------------------------


class DepthProvider:
    name = "base"

    def depth(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
        """Return (H, W) float32 depth in meters, or None if unavailable."""
        raise NotImplementedError


class NullDepthProvider(DepthProvider):
    name = "none"

    def depth(self, frame_bgr, frame_idx):
        return None


class NpyDepthProvider(DepthProvider):
    """Per-frame depth from <depth_dir>/<frame_idx:06d>.npy (meters)."""

    name = "file"

    def __init__(self, depth_dir: str):
        if not depth_dir:
            raise RuntimeError("--depth-source file requires --depth-dir")
        self.depth_dir = Path(depth_dir)
        self.missed = 0

    def depth(self, frame_bgr, frame_idx):
        p = self.depth_dir / f"{frame_idx:06d}.npy"
        if not p.exists():
            self.missed += 1
            return None
        d = np.load(p).astype(np.float32)
        if d.shape != frame_bgr.shape[:2]:
            import cv2
            d = cv2.resize(d, (frame_bgr.shape[1], frame_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        return d


_DA2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class DepthAnythingProvider(DepthProvider):
    """Monocular metric depth via Depth-Anything-V2 (server + CUDA).

    Lazy-imports the repo given by --depth-anything-repo (or env
    DEPTH_ANYTHING_REPO). Metric checkpoints (metric_depth/, Hypersim
    indoor, max_depth=20) output meters - pass --depth-anything-metric.
    UNVERIFIED on the dev Mac - validate on the server.
    """

    name = "depth_anything_v2"

    def __init__(self, repo: str, ckpt: str, encoder: str = "vitl",
                 input_size: int = 518, metric: bool = False, max_depth: float = 20.0):
        if not repo:
            raise RuntimeError("pass --depth-anything-repo or set DEPTH_ANYTHING_REPO")
        if not ckpt:
            raise RuntimeError("--depth-source depth_anything requires --depth-anything-ckpt")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; DepthAnything inference needs the GPU")
        repo_path = Path(repo).expanduser()
        sys.path.insert(0, str(repo_path))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            sys.path.insert(0, str(repo_path / "metric_depth"))
            from depth_anything_v2.dpt import DepthAnythingV2
        cfg = dict(_DA2_CONFIGS[encoder])
        if metric:
            cfg["max_depth"] = max_depth
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(str(Path(ckpt).expanduser()), map_location="cpu"))
        self._torch = torch
        self.model = model.cuda().eval()
        self.input_size = int(input_size)

    def depth(self, frame_bgr, frame_idx):
        import cv2
        with self._torch.no_grad():
            d = self.model.infer_image(frame_bgr, self.input_size)
        d = np.asarray(d, dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        return d


# ---------------------------------------------------------------------------
# stage 3: pose estimators
# ---------------------------------------------------------------------------


class ObjectPoseTracker:
    """One instance per tracked object."""

    def estimate(self, frame_bgr: np.ndarray, depth: Optional[np.ndarray],
                 K: np.ndarray, bbox: np.ndarray, is_new: bool) -> Optional[np.ndarray]:
        """Return the 4x4 object-to-camera pose, or None on failure."""
        raise NotImplementedError


class PoseBackend:
    name = "base"
    requires_depth = True

    def create(self, track_id: int) -> ObjectPoseTracker:
        raise NotImplementedError


class _NullObjectTracker(ObjectPoseTracker):
    def estimate(self, frame_bgr, depth, K, bbox, is_new):
        return None


class NullPoseBackend(PoseBackend):
    """Chain-test mode: no pose estimation, poses stay NaN."""

    name = "none"
    requires_depth = False

    def create(self, track_id):
        return _NullObjectTracker()


class _MockObjectTracker(ObjectPoseTracker):
    """Deterministic fake pose: identity rotation, translation = bbox
    center back-projected with the frame depth (or a fixed z)."""

    def __init__(self, fallback_z: float):
        self.fallback_z = float(fallback_z)

    def estimate(self, frame_bgr, depth, K, bbox, is_new):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        z = self.fallback_z
        if depth is not None:
            h, w = depth.shape
            ix = int(np.clip(round(cx), 0, w - 1))
            iy = int(np.clip(round(cy), 0, h - 1))
            d = float(depth[iy, ix])
            if np.isfinite(d) and d > 0:
                z = d
        fx, fy, ppx, ppy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        t = np.array([(cx - ppx) * z / fx, (cy - ppy) * z / fy, z])
        return make_transform(np.eye(3), t).astype(np.float32)


class MockPoseBackend(PoseBackend):
    name = "mock"
    requires_depth = False

    def __init__(self, fallback_z: float = 1.0):
        self.fallback_z = fallback_z

    def create(self, track_id):
        return _MockObjectTracker(self.fallback_z)


class _FoundationPoseObjectTracker(ObjectPoseTracker):
    def __init__(self, est, est_refine_iter: int, track_refine_iter: int):
        self.est = est
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter

    @staticmethod
    def _rect_mask(depth: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
        y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
        mask = np.zeros((h, w), dtype=bool)
        if x2 >= x1 and y2 >= y1:
            mask[y1:y2 + 1, x1:x2 + 1] = True
        return mask

    def estimate(self, frame_bgr, depth, K, bbox, is_new):
        if depth is None:
            return None
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        mask = self._rect_mask(depth, bbox)
        if is_new:
            pose = self.est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                     iteration=self.est_refine_iter)
        else:
            pose = self.est.track_one(rgb=rgb, depth=depth, K=K,
                                      iteration=self.track_refine_iter)
        return np.asarray(pose, dtype=np.float32)


class FoundationPoseBackend(PoseBackend):
    """FoundationPose wrapper; heavy deps imported lazily from
    --foundationpose-repo (or env FOUNDATIONPOSE_REPO), i.e. the
    pre-pulled refs/FoundationPose on the server.

    Per-object mesh lookup in --mesh-dir: track_{track_id:03d}.ply/.obj,
    or a single --mesh-file shared by all objects. FoundationPose's
    native output is object-to-camera (OpenCV camera convention),
    matching the contract. UNVERIFIED on the dev Mac - validate on the
    server. Model-free (bundlesdf) mesh reconstruction is not wired in
    yet; provide meshes for now.
    """

    name = "foundationpose"
    requires_depth = True

    def __init__(self, repo: str, mesh_dir: Optional[str] = None,
                 mesh_file: Optional[str] = None,
                 est_refine_iter: int = 5, track_refine_iter: int = 2,
                 debug_dir: Optional[str] = None):
        if not repo:
            raise RuntimeError("pass --foundationpose-repo or set FOUNDATIONPOSE_REPO")
        repo_path = Path(repo).expanduser()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        try:
            import torch  # noqa: F401
            import trimesh  # noqa: F401
            from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        except Exception as e:
            raise RuntimeError(
                f"FoundationPose import failed from {repo_path}: {e}. "
                "Build the CUDA extensions first (nvdiffrast/kaolin/bundlesdf; "
                "see .agent/skills/pose6d/SKILL.md)."
            ) from e
        self._trimesh = trimesh
        self._fp_cls = FoundationPose
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.mesh_dir = Path(mesh_dir).expanduser() if mesh_dir else None
        self.mesh_file = Path(mesh_file).expanduser() if mesh_file else None
        self.est_refine_iter = int(est_refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.debug_dir = debug_dir or os.devnull

    def _mesh_for(self, track_id: int):
        candidates = []
        if self.mesh_file:
            candidates.append(self.mesh_file)
        if self.mesh_dir:
            for ext in ("ply", "obj"):
                candidates.append(self.mesh_dir / f"track_{track_id:03d}.{ext}")
        for p in candidates:
            if p.exists():
                return self._trimesh.load(str(p), force="mesh")
        raise FileNotFoundError(
            f"no mesh for track {track_id}; looked for: "
            + ", ".join(str(c) for c in candidates))

    def create(self, track_id):
        mesh = self._mesh_for(track_id)
        est = self._fp_cls(
            model_pts=np.asarray(mesh.vertices, dtype=np.float32),
            model_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
            mesh=mesh, scorer=self.scorer, refiner=self.refiner,
            debug=0, debug_dir=self.debug_dir)
        return _FoundationPoseObjectTracker(est, self.est_refine_iter,
                                            self.track_refine_iter)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


class ClipReader:
    def __init__(self, path: str):
        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def frames(self):
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
        self.cap.release()


def resolve_intrinsics(args, width: int, height: int,
                       warnings: List[str], flags: List[str]) -> Tuple[np.ndarray, str]:
    if args.intrinsics:
        fx, fy, cx, cy = [float(v) for v in args.intrinsics]
        source = "cli"
    elif args.intrinsics_json:
        with open(args.intrinsics_json) as f:
            data = json.load(f)
        if "K" in data:
            K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
            return K, str(args.intrinsics_json)
        fx, fy, cx, cy = (float(data[k]) for k in ("fx", "fy", "cx", "cy"))
        source = str(args.intrinsics_json)
    else:
        fx = fy = float(max(width, height))
        cx, cy = width / 2.0, height / 2.0
        source = "heuristic"
        warnings.append("camera intrinsics not provided; using heuristic "
                        f"fx=fy=max(W,H)={fx:.1f}")
        flags.append("heuristic_intrinsics")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, source


def write_pose6d_npz(path: str, object_ids: List[str],
                     poses: np.ndarray, bboxes: np.ndarray) -> None:
    object_ids = np.asarray(object_ids)
    if object_ids.dtype.kind not in ("U", "S", "O"):
        object_ids = object_ids.astype(str)
    poses = np.asarray(poses, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.int32)
    n = len(object_ids)
    if poses.shape[0] != n or poses.ndim != 4 or poses.shape[2:] != (4, 4):
        raise ValueError(f"poses must be (N, T, 4, 4) with N={n}, got {poses.shape}")
    if bboxes.shape[0] != n or bboxes.ndim != 3 or bboxes.shape[2] != 4:
        raise ValueError(f"bboxes must be (N, T, 4) with N={n}, got {bboxes.shape}")
    if bboxes.shape[1] != poses.shape[1]:
        raise ValueError("poses/bboxes T mismatch")
    np.savez(path, object_ids=object_ids, poses=poses, bboxes=bboxes)


def collect_environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cv2": None,
        "torch": None,
        "cuda_available": False,
    }
    try:
        import cv2
        env["cv2"] = cv2.__version__
    except Exception:
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return env


# ---------------------------------------------------------------------------
# stage factories (graceful degradation unless --strict)
# ---------------------------------------------------------------------------


def build_detector(args, warnings, flags) -> BBoxDetector:
    name = args.detector
    if name is None:
        name = "file" if args.bbox_from_file else None
    if name is None:
        raise SystemExit("no detector: pass --bbox-from-file or --detector grounded_sam")
    if name == "file":
        if not args.bbox_from_file:
            raise SystemExit("--detector file requires --bbox-from-file")
        return FileBBoxDetector(args.bbox_from_file)
    if name == "grounded_sam":
        try:
            return GroundedSAMDetector(
                repo=args.grounded_sam_repo, text_prompt=args.text_prompt,
                groundingdino_config=args.groundingdino_config,
                groundingdino_ckpt=args.groundingdino_ckpt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold)
        except Exception as e:
            if args.strict:
                raise
            warnings.append(f"grounded_sam_unavailable: {e}")
            flags.append("detector_unavailable")
            return EmptyDetector()
    raise SystemExit(f"unknown detector: {name}")


def build_depth_provider(args, warnings, flags) -> DepthProvider:
    if args.depth_source == "none":
        return NullDepthProvider()
    if args.depth_source == "file":
        return NpyDepthProvider(args.depth_dir)
    if args.depth_source == "depth_anything":
        try:
            return DepthAnythingProvider(
                repo=args.depth_anything_repo, ckpt=args.depth_anything_ckpt,
                encoder=args.depth_anything_encoder,
                input_size=args.depth_anything_input_size,
                metric=args.depth_anything_metric,
                max_depth=args.depth_anything_max_depth)
        except Exception as e:
            if args.strict:
                raise
            warnings.append(f"depth_anything_unavailable: {e}")
            flags.append("depth_anything_unavailable")
            return NullDepthProvider()
    raise SystemExit(f"unknown depth source: {args.depth_source}")


def build_pose_backend(args, warnings, flags) -> PoseBackend:
    if args.pose_estimator == "none":
        return NullPoseBackend()
    if args.pose_estimator == "mock":
        return MockPoseBackend(fallback_z=args.mock_z)
    if args.pose_estimator == "foundationpose":
        try:
            return FoundationPoseBackend(
                repo=args.foundationpose_repo, mesh_dir=args.mesh_dir,
                mesh_file=args.mesh_file,
                est_refine_iter=args.est_refine_iter,
                track_refine_iter=args.track_refine_iter,
                debug_dir=args.debug_dir)
        except Exception as e:
            if args.strict:
                raise
            warnings.append(f"foundationpose_unavailable: {e}")
            flags.append("pose_backend_unavailable")
            return NullPoseBackend()
    raise SystemExit(f"unknown pose estimator: {args.pose_estimator}")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def run(args) -> dict:
    warnings: List[str] = []
    flags: List[str] = []

    detector = build_detector(args, warnings, flags)
    depth_provider = build_depth_provider(args, warnings, flags)
    pose_backend = build_pose_backend(args, warnings, flags)

    clip_path = Path(args.clip)
    clip_id = args.clip_id or clip_path.stem
    reader = ClipReader(str(clip_path))
    fps = args.fps if args.fps else reader.fps
    K, K_source = resolve_intrinsics(args, reader.width, reader.height, warnings, flags)

    tracker = IoUTracker(iou_threshold=args.iou_threshold, max_lost=args.max_lost)
    obj_trackers: Dict[int, ObjectPoseTracker] = {}
    # track_id -> frame_idx -> (pose or None, bbox)
    records: Dict[int, Dict[int, Tuple[Optional[np.ndarray], np.ndarray]]] = {}
    pose_failures = 0
    frames_done = 0

    for idx, frame in reader.frames():
        if args.limit_frames and idx >= args.limit_frames:
            break
        dets = detector.detect(frame, idx)
        boxes = np.array([d.bbox for d in dets], dtype=np.float64).reshape(-1, 4)
        matches = tracker.update(idx, boxes)

        depth: Optional[np.ndarray] = None
        depth_done = False
        for det_idx, track_id in matches.items():
            bbox = boxes[det_idx]
            is_new = track_id not in obj_trackers
            if is_new:
                try:
                    obj_trackers[track_id] = pose_backend.create(track_id)
                except Exception as e:
                    warnings.append(f"track {track_id}: pose backend create failed: {e}")
                    obj_trackers[track_id] = _NullObjectTracker()
            ot = obj_trackers[track_id]

            pose = None
            if not isinstance(ot, _NullObjectTracker):
                if not depth_done and (pose_backend.requires_depth
                                       or depth_provider.name != "none"):
                    depth = depth_provider.depth(frame, idx)
                    depth_done = True
                if pose_backend.requires_depth and depth is None:
                    pose = None
                else:
                    try:
                        pose = ot.estimate(frame, depth, K, bbox, is_new)
                    except Exception as e:
                        pose_failures += 1
                        if pose_failures <= 5:
                            warnings.append(f"frame {idx} track {track_id}: pose failed: {e}")
                        pose = None
            records.setdefault(track_id, {})[idx] = (pose, bbox)
        frames_done = idx + 1

        if idx % 50 == 0:
            log.info("frame %d: %d detections, %d tracks", idx, len(dets), len(tracker.tracks))

    T = frames_done
    if T == 0:
        raise SystemExit(f"no frames read from {clip_path}")

    tracks = [t for t in tracker.all_tracks() if t.hits >= args.min_hits]
    ordered = sorted_tracks(tracks)  # first frame, then first-frame area desc
    id_map = assign_object_ids(tracks)
    ordered_ids = [id_map[t.track_id] for t in ordered]

    N = len(ordered)
    poses = nan_poses(N, T)
    bboxes = np.full((N, T, 4), BBOX_UNDETECTED, dtype=np.int32)
    W, H = reader.width, reader.height
    for i, t in enumerate(ordered):
        for fidx, (pose, bbox) in records.get(t.track_id, {}).items():
            if pose is not None:
                poses[i, fidx] = pose
            x1, y1, x2, y2 = bbox
            bboxes[i, fidx] = [
                int(np.clip(round(x1), 0, W - 1)), int(np.clip(round(y1), 0, H - 1)),
                int(np.clip(round(x2), 0, W - 1)), int(np.clip(round(y2), 0, H - 1)),
            ]

    if N == 0:
        flags.append("no_detections")
    if pose_backend.requires_depth and depth_provider.name == "none" and N > 0:
        flags.append("no_depth_for_estimator")
        warnings.append("pose estimator requires depth but depth_source=none; poses are NaN")
    if pose_failures:
        flags.append("pose_failures")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{clip_id}.npz"
    write_pose6d_npz(str(npz_path), ordered_ids, poses, bboxes)

    meta = {
        "clip_id": clip_id,
        "source_clip": str(clip_path),
        "fps": float(fps),
        "num_frames": int(T),
        "width": int(W),
        "height": int(H),
        "num_objects": int(N),
        "object_ids": ordered_ids,
        "depth_source": depth_provider.name,
        "detector": detector.name,
        "pose_estimator": pose_backend.name,
        "intrinsics": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "source": K_source,
        },
        "bbox_undetected_value": BBOX_UNDETECTED,
        "tracker": {
            "type": "iou",
            "iou_threshold": args.iou_threshold,
            "max_lost": args.max_lost,
            "min_hits": args.min_hits,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(),
        "quality_flags": sorted(set(flags)),
        "warnings": warnings,
    }
    meta_path = out_dir / f"{clip_id}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {"clip_id": clip_id, "npz": str(npz_path), "meta": str(meta_path),
            "num_objects": N, "num_frames": T, "object_ids": ordered_ids,
            "quality_flags": meta["quality_flags"]}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="input clip path (clips/<clip_id>.mp4)")
    p.add_argument("--out-dir", required=True, help="output dir for <clip_id>.npz/.meta.json")
    p.add_argument("--clip-id", default=None, help="default: video filename stem")
    p.add_argument("--fps", type=float, default=None, help="override video fps")
    p.add_argument("--limit-frames", type=int, default=None)

    g = p.add_argument_group("intrinsics")
    g.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    g.add_argument("--intrinsics-json", default=None,
                   help='json with {"fx","fy","cx","cy"} or {"K": [[...]]}')

    g = p.add_argument_group("detector")
    g.add_argument("--detector", choices=["file", "grounded_sam"], default=None,
                   help="default: file when --bbox-from-file is given")
    g.add_argument("--bbox-from-file", default=None,
                   help="json with per-frame bboxes (placeholder/manual mode)")
    g.add_argument("--grounded-sam-repo",
                   default=os.environ.get("GROUNDED_SAM_REPO"))
    g.add_argument("--groundingdino-config", default=None)
    g.add_argument("--groundingdino-ckpt", default=None)
    g.add_argument("--text-prompt", default="object .")
    g.add_argument("--box-threshold", type=float, default=0.3)
    g.add_argument("--text-threshold", type=float, default=0.25)

    g = p.add_argument_group("depth")
    g.add_argument("--depth-source", choices=["none", "file", "depth_anything"],
                   default="none")
    g.add_argument("--depth-dir", default=None,
                   help="for depth-source file: <dir>/<frame_idx:06d>.npy in meters")
    g.add_argument("--depth-anything-repo",
                   default=os.environ.get("DEPTH_ANYTHING_REPO"))
    g.add_argument("--depth-anything-ckpt", default=None)
    g.add_argument("--depth-anything-encoder", choices=["vits", "vitb", "vitl"],
                   default="vitl")
    g.add_argument("--depth-anything-input-size", type=int, default=518)
    g.add_argument("--depth-anything-metric", action="store_true",
                   help="metric checkpoint (meters); e.g. Hypersim indoor model")
    g.add_argument("--depth-anything-max-depth", type=float, default=20.0)

    g = p.add_argument_group("pose estimator")
    g.add_argument("--pose-estimator", choices=["none", "mock", "foundationpose"],
                   default="none")
    g.add_argument("--mock-z", type=float, default=1.0,
                   help="mock estimator fallback depth (m)")
    g.add_argument("--foundationpose-repo",
                   default=os.environ.get("FOUNDATIONPOSE_REPO"))
    g.add_argument("--mesh-dir", default=None,
                   help="per-object meshes: track_<track_id:03d>.ply/.obj")
    g.add_argument("--mesh-file", default=None, help="single mesh shared by all objects")
    g.add_argument("--est-refine-iter", type=int, default=5)
    g.add_argument("--track-refine-iter", type=int, default=2)
    g.add_argument("--debug-dir", default=None)

    g = p.add_argument_group("tracker")
    g.add_argument("--iou-threshold", type=float, default=0.3)
    g.add_argument("--max-lost", type=int, default=10,
                   help="frames a track may be missing before retirement")
    g.add_argument("--min-hits", type=int, default=1,
                   help="drop tracks with fewer matched frames")

    p.add_argument("--strict", action="store_true",
                   help="fail instead of degrading when optional deps are missing")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> dict:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="[%(levelname)s] %(message)s", stream=sys.stderr)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
