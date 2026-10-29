"""CPU injection tests validate wrapper logic only, NOT real hand-model quality."""

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.hand_pose import online_hands as online


def points(offset=0):
    values = np.linspace(0.1, 0.8, 21, dtype=np.float32)
    return np.column_stack((values + offset, values[::-1], np.linspace(-0.1, 0.1, 21))).astype(np.float32)


def prediction(*hands):
    return SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=[SimpleNamespace(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                                                        for point in coordinates]) for _, _, coordinates in hands] or None,
        multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label=label, score=score)])
                         for label, score, _ in hands] or None,
        multi_hand_world_landmarks="must never be read or emitted as camera XYZ",
    )


class FakeBackend:
    def __init__(self, *outputs):
        self.outputs = list(outputs or [prediction()])
        self.images = []
        self.close_calls = 0

    def process(self, image):
        self.images.append(image.copy())
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    def close(self):
        self.close_calls += 1


@pytest.fixture
def rgb():
    return np.full((60, 80, 3), [17, 41, 93], dtype=np.uint8)


def test_import_and_injected_backend_do_not_import_mediapipe():
    script = "from src.hand_pose.online_hands import OnlineHands, draw_hands; import sys; assert 'mediapipe' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", script], check=True)


@pytest.mark.parametrize("mirrored", [False, True])
def test_handedness_rule_confidence_and_coordinates(rgb, mirrored):
    left_raw, right_raw = points(), points(0.05)
    backend = FakeBackend(prediction(("Left", 0.9, left_raw), ("Right", 0.8, right_raw)))
    with online.OnlineHands(input_mirrored=mirrored, backend=backend) as hands:
        result = hands.process(rgb, 0)
        assert set(result) == {"left_hand", "right_hand", "conf"}
        np.testing.assert_array_equal(result["left_hand"], left_raw if mirrored else right_raw)
        np.testing.assert_array_equal(result["right_hand"], right_raw if mirrored else left_raw)
        np.testing.assert_allclose(result["conf"], [0.9, 0.8] if mirrored else [0.8, 0.9])
        for key, shape in (("left_hand", (21, 3)), ("right_hand", (21, 3)), ("conf", (2,))):
            assert result[key].shape == shape and result[key].dtype == np.float32
        np.testing.assert_array_equal(backend.images[0], rgb)
        meta = hands.metadata()
        assert meta["handedness"]["rule"] == ("keep_labels" if mirrored else "swap_left_right_labels")
        assert meta["handedness"]["image_or_coordinates_flipped"] is False
        assert meta["handedness"]["inferred_from_ego_view"] is False
    assert backend.close_calls == 1


def test_missing_and_low_score_hands_remain_nan_without_interpolation(rgb):
    backend = FakeBackend(prediction(("Right", 0.7, points())), prediction(), prediction(("Right", 0.49, points())))
    with online.OnlineHands(backend=backend) as hands:
        first = hands.process(rgb, 0)
        assert np.isfinite(first["left_hand"]).all()
        assert np.isnan(first["right_hand"]).all() and np.isnan(first["conf"][1])
        for timestamp in (1, 2):
            result = hands.process(rgb, timestamp)
            assert all(np.isnan(array).all() for array in result.values())
        assert hands.metadata()["frames_processed"] == 3


@pytest.mark.parametrize("scores,expected", [((0.7, 0.9), 1), ((0.9, 0.7), 0), ((0.8, 0.8), 0)])
def test_duplicate_handedness_best_score_and_stable_tie(rgb, scores, expected):
    candidates = [points(), points(0.03)]
    backend = FakeBackend(prediction(*[("Right", score, candidates[index]) for index, score in enumerate(scores)]))
    with online.OnlineHands(backend=backend) as hands:
        result = hands.process(rgb, 0)
    np.testing.assert_array_equal(result["left_hand"], candidates[expected])
    assert result["conf"][0] == np.float32(scores[expected])
    assert np.isnan(result["right_hand"]).all()


def test_threshold_inclusive_and_one_hand_configuration(rgb):
    with online.OnlineHands(min_handedness_score=0.8, max_num_hands=1,
                            backend=FakeBackend(prediction(("Left", 0.8, points())))) as hands:
        result = hands.process(rgb, 0)
        assert result["conf"][1] == np.float32(0.8)
        assert hands.metadata()["backend_options"]["max_num_hands"] == 1


def test_injected_metadata_is_json_safe_detached_and_explicitly_test_only(rgb):
    with online.OnlineHands(backend=FakeBackend()) as hands:
        metadata = hands.metadata()
        assert metadata["backend"] == "injected_test_backend" and metadata["test_only"] is True
        assert metadata["backend_version"] is None and metadata["model_files"] == []
        assert metadata["coordinate_semantics"]["camera_metric_xyz"] is False
        assert "NOT_meters" in metadata["coordinate_semantics"]["z"]
        assert metadata["timestamp_policy"]["passed_to_legacy_graph"] is False
        assert metadata["timestamp_policy"]["real_time_performance_claimed"] is False
        metadata["thresholds"]["min_handedness_score"] = 0
        assert hands.metadata()["thresholds"]["min_handedness_score"] == 0.5
        hands.process(rgb, np.int64(1234567890123))
        json.dumps(hands.metadata(), allow_nan=False)
        assert hands.metadata()["last_input_timestamp_ms"] == 1234567890123


@pytest.mark.parametrize("timestamp", [0, -1, float("nan"), float("inf"), True, "1", None])
def test_timestamp_rejected_without_backend_call(rgb, timestamp):
    backend = FakeBackend(prediction(), prediction())
    with online.OnlineHands(backend=backend) as hands:
        hands.process(rgb, 0)
        with pytest.raises(ValueError, match="timestamp_ms"):
            hands.process(rgb, timestamp)
        assert len(backend.images) == 1
        hands.process(rgb, 0.125)
        assert hands.metadata()["frames_processed"] == 2


def test_timestamp_decrease_rejected_and_large_integer_precision_preserved(rgb):
    backend = FakeBackend(prediction(), prediction())
    with online.OnlineHands(backend=backend) as hands:
        hands.process(rgb, 2**54)
        with pytest.raises(ValueError, match="increasing"):
            hands.process(rgb, 2**54 - 1)
        hands.process(rgb, 2**54 + 1)
        assert hands.metadata()["last_input_timestamp_ms"] == 2**54 + 1


@pytest.mark.parametrize("image", [None, [[[0, 0, 0]]], np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8),
                                       np.zeros((4, 4, 3), np.float32), np.zeros((0, 4, 3), np.uint8)])
def test_bad_rgb_rejected_before_predictor(image):
    backend = FakeBackend()
    with online.OnlineHands(backend=backend) as hands:
        with pytest.raises(ValueError, match="uint8"):
            hands.process(image, 0)
    assert backend.images == []


def test_noncontiguous_rgb_is_copied_without_channel_or_horizontal_flip(rgb):
    sliced = rgb[:, ::-1].copy()
    sliced[:, 0] = [1, 2, 3]
    noncontiguous = sliced[:, ::-1]
    backend = FakeBackend()
    with online.OnlineHands(backend=backend) as hands:
        hands.process(noncontiguous, 0)
    np.testing.assert_array_equal(backend.images[0], noncontiguous)
    np.testing.assert_array_equal(sliced[:, 0], np.tile([1, 2, 3], (60, 1)))


@pytest.mark.parametrize("options", [
    {"input_mirrored": "False"}, {"max_num_hands": 0}, {"max_num_hands": 3}, {"max_num_hands": True}, {"max_num_hands": 1.5},
    {"min_detection_confidence": -0.1}, {"min_tracking_confidence": 1.1}, {"min_handedness_score": float("nan")},
    {"min_detection_confidence": float("inf")}, {"min_handedness_score": True}, {"min_handedness_score": "0.5"},
    {"model_complexity": -1}, {"model_complexity": 2}, {"model_complexity": True}, {"model_complexity": 0.5},
])
def test_configuration_validation_precedes_backend_creation(options, monkeypatch):
    def forbidden_create(*arguments):
        pytest.fail("bad configuration must not create a backend")

    monkeypatch.setattr(online, "_create_backend", forbidden_create)
    with pytest.raises(ValueError):
        online.OnlineHands(**options)


@pytest.mark.parametrize("fault", ["mismatched_count", "too_many", "label", "empty_classification", "shape", "nan", "inf", "score"])
def test_malformed_model_output_fails_without_fake_missing_hand(rgb, fault):
    output = prediction(("Right", 0.9, points()))
    if fault == "mismatched_count":
        output.multi_handedness = None
    elif fault == "too_many":
        output = prediction(*[("Right", 0.9, points())] * 3)
    elif fault == "label":
        output.multi_handedness[0].classification[0].label = "Unknown"
    elif fault == "empty_classification":
        output.multi_handedness[0].classification = []
    elif fault == "shape":
        output.multi_hand_landmarks[0].landmark.pop()
    elif fault in ("nan", "inf"):
        output.multi_hand_landmarks[0].landmark[0].x = float(fault)
    elif fault == "score":
        output.multi_handedness[0].classification[0].score = float("nan")
    backend = FakeBackend(output)
    with online.OnlineHands(backend=backend) as hands:
        with pytest.raises(RuntimeError, match="inference failed"):
            hands.process(rgb, 0)
        assert hands.metadata()["failed"] is True and hands.metadata()["frames_processed"] == 0
        with pytest.raises(RuntimeError, match="new stream"):
            hands.process(rgb, 1)
    assert len(backend.images) == 1 and backend.close_calls == 1


def test_backend_exception_propagates_and_context_closes(rgb):
    backend = FakeBackend(RuntimeError("real graph failure"))
    with pytest.raises(RuntimeError, match="real graph failure"):
        with online.OnlineHands(backend=backend) as hands:
            hands.process(rgb, 0)
    assert backend.close_calls == 1


def test_idempotent_close_and_closed_context_rejected(rgb):
    backend = FakeBackend()
    hands = online.OnlineHands(backend=backend)
    hands.close()
    hands.close()
    assert backend.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        hands.process(rgb, 0)
    with pytest.raises(RuntimeError, match="closed"):
        with hands:
            pytest.fail("closed context was entered")


def test_result_arrays_do_not_alias_other_frames_or_slots(rgb):
    output = prediction(("Right", 0.9, points()))
    with online.OnlineHands(backend=FakeBackend(output, prediction(), output)) as hands:
        first = hands.process(rgb, 0)
        missing = hands.process(rgb, 1)
        first["left_hand"][:] = 10
        missing["left_hand"][:] = 0
        third = hands.process(rgb, 2)
        np.testing.assert_array_equal(third["left_hand"], points())
        assert np.isnan(missing["right_hand"]).all()


@pytest.fixture
def packaged_mediapipe(tmp_path, monkeypatch):
    package = tmp_path / "mediapipe"
    for relative in ("__init__.py", "modules/palm_detection/palm_detection_full.tflite",
                     "modules/hand_landmark/hand_landmark_full.tflite",
                     "modules/palm_detection/palm_detection_lite.tflite", "modules/hand_landmark/hand_landmark_lite.tflite",
                     "modules/hand_landmark/hand_landmark_tracking_cpu.binarypb", "python/solutions/hands.py"):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    backend, calls = FakeBackend(), []

    def constructor(**options):
        calls.append(options)
        return backend

    module = SimpleNamespace(__version__="0.10.21", __file__=str(package / "__init__.py"),
                             solutions=SimpleNamespace(hands=SimpleNamespace(Hands=constructor)))
    monkeypatch.setattr(online.importlib, "import_module", lambda name: module)
    return package, module, backend, calls


def test_packaged_model_hashes_and_legacy_constructor_plumbing(packaged_mediapipe):
    """Stub package files verify plumbing only, never a real MediaPipe execution."""
    _, _, backend, calls = packaged_mediapipe
    with online.OnlineHands(min_detection_confidence=0.6, min_tracking_confidence=0.7) as hands:
        metadata = hands.metadata()
        assert calls == [{"static_image_mode": False, "model_complexity": 1, "max_num_hands": 2,
                          "min_detection_confidence": 0.6, "min_tracking_confidence": 0.7}]
        assert metadata["backend_version"] == "0.10.21"
        assert len(metadata["model_files"]) == 2
        for record in [*metadata["model_files"], metadata["graph"], metadata["implementation"]]:
            assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        json.dumps(metadata, allow_nan=False)
    assert backend.close_calls == 1


def test_missing_packaged_model_fails_before_graph_creation(packaged_mediapipe):
    package, _, _, calls = packaged_mediapipe
    (package / "modules/hand_landmark/hand_landmark_full.tflite").unlink()
    with pytest.raises(FileNotFoundError):
        online.OnlineHands()
    assert calls == []


def test_lite_model_selection_hashes_and_options(packaged_mediapipe):
    package, _, _, calls = packaged_mediapipe
    (package / "modules/hand_landmark/hand_landmark_full.tflite").unlink()
    with online.OnlineHands(model_complexity=0) as hands:
        metadata = hands.metadata()
        assert calls[0]["model_complexity"] == 0
        assert {Path(record["path"]).name for record in metadata["model_files"]} == {
            "palm_detection_lite.tflite", "hand_landmark_lite.tflite"}
        for record in metadata["model_files"]:
            assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()


def test_cross_thread_calls_rejected_and_worker_lifecycle_supported(rgb):
    backend = FakeBackend(prediction(), prediction())
    with online.OnlineHands(backend=backend) as hands:
        with ThreadPoolExecutor(max_workers=1) as worker:
            with pytest.raises(RuntimeError, match="same thread"):
                worker.submit(hands.process, rgb, 0).result()
            with pytest.raises(RuntimeError, match="same thread"):
                worker.submit(hands.close).result()
        assert backend.images == [] and backend.close_calls == 0
        hands.process(rgb, 0)

    def worker_owned_stream():
        with online.OnlineHands(backend=FakeBackend(prediction(), prediction())) as tracker:
            tracker.process(rgb, 0.125)
            result = tracker.process(rgb, 33.458333333)
        return tracker, result

    with ThreadPoolExecutor(max_workers=1) as worker:
        tracker, result = worker.submit(worker_owned_stream).result()
    assert tracker.metadata()["closed"] is True
    assert tracker.metadata()["last_input_timestamp_ms"] == 33.458333333
    assert tracker.metadata()["frames_processed"] == 2
    assert np.isnan(result["conf"]).all()


def test_wrong_mediapipe_version_fails_without_fallback(packaged_mediapipe):
    _, module, _, calls = packaged_mediapipe
    module.__version__ = "0.10.31"
    with pytest.raises(RuntimeError, match="requires mediapipe==0.10.21"):
        online.OnlineHands()
    assert calls == []


def test_mediapipe_import_error_fails_without_fallback(monkeypatch):
    def missing_module(name):
        raise ImportError("not installed")

    monkeypatch.setattr(online.importlib, "import_module", missing_module)
    with pytest.raises(RuntimeError, match="no mock fallback"):
        online.OnlineHands()


def test_draw_hands_copy_colors_and_twenty_one_points_each(rgb, monkeypatch):
    result = {"left_hand": points(), "right_hand": points(0.05), "conf": np.array([0.9, 0.8], np.float32)}
    original = rgb.copy()
    circles, lines = [], []
    original_circle, original_line = cv2.circle, cv2.line

    def record_circle(image, center, radius, color, thickness, line_type):
        circles.append(color)
        return original_circle(image, center, radius, color, thickness, line_type)

    def record_line(image, first, last, color, thickness, line_type):
        lines.append(color)
        return original_line(image, first, last, color, thickness, line_type)

    monkeypatch.setattr(cv2, "circle", record_circle)
    monkeypatch.setattr(cv2, "line", record_line)
    rendered = online.draw_hands(rgb, result)
    np.testing.assert_array_equal(rgb, original)
    assert not np.shares_memory(rendered, rgb) and np.any(rendered != rgb)
    assert circles.count(online.HAND_COLORS_BGR["left_hand"]) == 21
    assert circles.count(online.HAND_COLORS_BGR["right_hand"]) == 21
    assert len(lines) == 42 and len(set(circles)) == 2


def test_draw_missing_hands_unchanged_and_partial_joints_not_fabricated(rgb, monkeypatch):
    missing = np.full((21, 3), np.nan, np.float32)
    rendered = online.draw_hands(rgb, {"left_hand": missing, "right_hand": missing})
    np.testing.assert_array_equal(rendered, rgb)
    assert not np.shares_memory(rendered, rgb)
    partial = missing.copy()
    partial[0] = [0.2, 0.3, 0]
    partial[2] = [0.4, 0.3, 0]
    partial[3] = [-0.1, 0.3, 0]
    partial[4] = [0.5, 0.3, np.nan]
    circles, lines = [], []
    monkeypatch.setattr(cv2, "circle", lambda *arguments: circles.append(arguments[1]))
    monkeypatch.setattr(cv2, "line", lambda *arguments: lines.append(arguments[1:3]))
    online.draw_hands(rgb, {"left_hand": partial})
    assert circles == [(16, 18), (32, 18)] and lines == []


def test_draw_invalid_shape_is_explicit(rgb):
    with pytest.raises(ValueError, match="21,3"):
        online.draw_hands(rgb, {"left_hand": np.zeros((20, 3))})
