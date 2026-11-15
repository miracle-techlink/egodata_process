"""Causal single-frame MediaPipe 0.10.21 legacy Hands, never metric camera XYZ.

Use ``with OnlineHands(input_mirrored=False) as hands:`` then
``result = hands.process(rgb_uint8, timestamp_ms)``. Results contain only
left_hand/right_hand (21,3) and conf (2,), all float32. Missing or rejected
hands are entirely NaN, including confidence. Timestamp milliseconds must
strictly increase; legacy Hands receives images, not these timestamps.
Injected ``backend=`` objects are exclusively for CPU code tests.
"""

import copy
import hashlib
import importlib
import math
import threading
from numbers import Integral, Real
from pathlib import Path

import numpy as np


MEDIAPIPE_VERSION = "0.10.21"
HANDEDNESS_SOURCE = "https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/docs/solutions/hands.md#multi_handedness"
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
HAND_COLORS_BGR = {"left_hand": (0, 220, 0), "right_hand": (255, 120, 0)}


def _file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _create_backend(options):
    """Require the selected packaged models before constructing the real graph."""
    try:
        mediapipe = importlib.import_module("mediapipe")
    except ImportError as error:
        raise RuntimeError("OnlineHands requires mediapipe==0.10.21; no mock fallback") from error
    version = getattr(mediapipe, "__version__", None)
    if version != MEDIAPIPE_VERSION:
        raise RuntimeError(f"OnlineHands requires mediapipe=={MEDIAPIPE_VERSION}, found {version!r}")
    package = Path(mediapipe.__file__).resolve().parent
    variant = "full" if options["model_complexity"] == 1 else "lite"
    model_files = [_file_record(package / "modules" / folder / filename) for folder, filename in (
        ("palm_detection", f"palm_detection_{variant}.tflite"),
        ("hand_landmark", f"hand_landmark_{variant}.tflite"),
    )]
    graph = _file_record(package / "modules/hand_landmark/hand_landmark_tracking_cpu.binarypb")
    implementation = _file_record(package / "python/solutions/hands.py")
    backend = mediapipe.solutions.hands.Hands(**options)
    return backend, {"backend": "mediapipe.solutions.hands.Hands", "backend_version": version,
                     "test_only": False, "model_files": model_files, "graph": graph,
                     "implementation": implementation}


def _validate_image(image, name):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError(f"{name} must be a nonempty uint8 (H,W,3) array")


def _probability(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be finite in [0,1]")
    return float(value)


class OnlineHands:
    """One ordered stream, with caller-owned mirroring semantics and no fallback.

    ``backend`` must implement ``process(image=...)`` and ``close()``; injecting
    it always marks metadata test_only. Low handedness scores mean missing, not
    low joint precision. Duplicate labels retain the highest score, first on a
    tie. Malformed model outputs fail explicitly and poison this stream, as do
    backend exceptions; construct a new instance rather than silently reset.
    """

    def __init__(self, input_mirrored=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, min_handedness_score=0.5,
                 max_num_hands=2, model_complexity=1, *, backend=None):
        if not isinstance(input_mirrored, (bool, np.bool_)):
            raise ValueError("input_mirrored must be explicitly boolean")
        if isinstance(max_num_hands, (bool, np.bool_)) or not isinstance(max_num_hands, Integral) or max_num_hands not in (1, 2):
            raise ValueError("max_num_hands must be 1 or 2 for the two-slot contract")
        if isinstance(model_complexity, (bool, np.bool_)) or not isinstance(model_complexity, Integral) or model_complexity not in (0, 1):
            raise ValueError("model_complexity must be 0 (lite) or 1 (full)")
        thresholds = {name: _probability(value, name) for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
            ("min_handedness_score", min_handedness_score),
        )}
        self._input_mirrored = bool(input_mirrored)
        self._max_num_hands = int(max_num_hands)
        self._min_handedness_score = thresholds["min_handedness_score"]
        self._closed, self._failed = False, False
        self._owner_thread = threading.get_ident()
        self._last_timestamp_ms, self._frames_processed = None, 0
        options = {"static_image_mode": False, "model_complexity": int(model_complexity), "max_num_hands": self._max_num_hands,
                   "min_detection_confidence": thresholds["min_detection_confidence"],
                   "min_tracking_confidence": thresholds["min_tracking_confidence"]}
        provenance = {"backend": "injected_test_backend", "backend_version": None, "test_only": True,
                      "model_files": [], "graph": None, "implementation": None}
        if backend is None:
            backend, provenance = _create_backend(options)
        if not callable(getattr(backend, "process", None)) or not callable(getattr(backend, "close", None)):
            raise TypeError("backend must implement process(image=...) and close()")
        self._backend = backend
        self._metadata = {
            **provenance, "thresholds": thresholds, "backend_options": options,
            "handedness": {"input_mirrored": self._input_mirrored,
                           "raw_label_assumption": "mirrored_selfie_input",
                           "label_mapping": {"Left": "left_hand" if self._input_mirrored else "right_hand",
                                             "Right": "right_hand" if self._input_mirrored else "left_hand"},
                           "rule": "keep_labels" if self._input_mirrored else "swap_left_right_labels",
                           "image_or_coordinates_flipped": False, "inferred_from_ego_view": False,
                           "source": HANDEDNESS_SOURCE},
            "coordinate_semantics": {"x": "normalized_by_input_image_width",
                                     "y": "normalized_by_input_image_height",
                                     "z": "wrist_relative_depth_roughly_x_scale_smaller_is_closer_NOT_meters",
                                     "camera_metric_xyz": False, "world_local": "omitted_not_camera_or_global_frame",
                                     "landmark_order": "MediaPipe_Hands_21_wrist_index_0"},
            "confidence_semantics": "handedness_classification_score_NOT_detection_score_or_joint_accuracy",
            "conf_slot_order": ["left_hand", "right_hand"], "missing": "whole_hand_and_corresponding_conf_NaN",
            "duplicate_label_policy": "highest_score_first_on_tie", "interpolation": "none",
            "thread_policy": "construct_process_close_on_the_same_thread; metadata_after_join_is_allowed",
            "timestamp_policy": {"units": "ms", "strictly_increasing": True,
                                 "passed_to_legacy_graph": False, "legacy_graph_clock": "internal_simulated_33333us_per_call",
                                 "real_time_performance_claimed": False},
            "drawing_colors_bgr": {name: list(color) for name, color in HAND_COLORS_BGR.items()},
        }

    def process(self, rgb_uint8, timestamp_ms):
        """Return current-frame image-normalized landmarks only, never interpolated."""
        self._check_thread()
        if self._closed or self._failed:
            raise RuntimeError("OnlineHands is closed or failed; create a new stream")
        _validate_image(rgb_uint8, "rgb_uint8")
        if (isinstance(timestamp_ms, (bool, np.bool_)) or not isinstance(timestamp_ms, Real)
                or not math.isfinite(timestamp_ms) or timestamp_ms < 0):
            raise ValueError("timestamp_ms must be finite, nonnegative milliseconds")
        timestamp_ms = timestamp_ms.item() if isinstance(timestamp_ms, np.generic) else timestamp_ms
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must be strictly increasing")
        image = np.array(rgb_uint8, copy=True, order="C")
        image.flags.writeable = False
        self._last_timestamp_ms = timestamp_ms
        result = {name: np.full((21, 3), np.nan, dtype=np.float32) for name in ("left_hand", "right_hand")}
        result["conf"] = np.full(2, np.nan, dtype=np.float32)
        try:
            predictions = self._backend.process(image=image)
            landmarks = predictions.multi_hand_landmarks or []
            handedness = predictions.multi_handedness or []
            if len(landmarks) != len(handedness) or len(landmarks) > self._max_num_hands:
                raise ValueError("landmark/handedness count mismatch or too many hands")
            best_scores = [-math.inf, -math.inf]
            for hand, classification in zip(landmarks, handedness):
                if len(classification.classification) != 1:
                    raise ValueError("each hand must have exactly one handedness classification")
                category = classification.classification[0]
                if category.label not in ("Left", "Right"):
                    raise ValueError(f"unknown handedness label: {category.label!r}")
                score = _probability(category.score, "handedness score")
                points = np.asarray([(point.x, point.y, point.z) for point in hand.landmark], dtype=np.float32)
                if points.shape != (21, 3) or not np.isfinite(points).all():
                    raise ValueError("hand landmarks must be finite (21,3)")
                slot = 0 if (category.label == "Left") == self._input_mirrored else 1
                if score >= self._min_handedness_score and score > best_scores[slot]:
                    result[("left_hand", "right_hand")[slot]] = points.copy()
                    result["conf"][slot] = score
                    best_scores[slot] = score
        except Exception as error:
            self._failed = True
            raise RuntimeError(f"OnlineHands inference failed at timestamp_ms={timestamp_ms}: {error}") from error
        self._frames_processed += 1
        return result

    def metadata(self):
        """Return a detached, strict-JSON-safe provenance/configuration snapshot."""
        return {**copy.deepcopy(self._metadata), "frames_processed": self._frames_processed,
                "last_input_timestamp_ms": self._last_timestamp_ms, "closed": self._closed, "failed": self._failed}

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("OnlineHands process/close must use the same thread that created it")

    def close(self):
        self._check_thread()
        if not self._closed:
            self._closed = True
            self._backend.close()

    def __enter__(self):
        self._check_thread()
        if self._closed or self._failed:
            raise RuntimeError("OnlineHands is closed or failed")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False


def draw_hands(image_bgr, result):
    """Copy BGR image and draw only finite, in-image observed joints and edges."""
    import cv2

    _validate_image(image_bgr, "image_bgr")
    image = image_bgr.copy()
    height, width = image.shape[:2]
    for name, color in HAND_COLORS_BGR.items():
        if result.get(name) is None:
            continue
        points = np.asarray(result[name], dtype=np.float64)
        if points.shape != (21, 3):
            raise ValueError(f"{name} must have shape (21,3)")
        valid = np.isfinite(points).all(axis=1) & (points[:, :2] >= 0).all(axis=1) & (points[:, :2] <= 1).all(axis=1)
        pixels = {int(index): (min(int(points[index, 0] * width), width - 1),
                               min(int(points[index, 1] * height), height - 1)) for index in np.flatnonzero(valid)}
        for start, end in HAND_CONNECTIONS:
            if start in pixels and end in pixels:
                cv2.line(image, pixels[start], pixels[end], color, 2, cv2.LINE_AA)
        for point in pixels.values():
            cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)
    return image
