import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from src.pose6d.run_human_rgbd import build_parser, checked_hands, run_pipeline, summarize_timing


class TestHands:
    __test__ = False

    def __init__(self, events, missing=False, fail_close=False):
        self.events = events
        self.missing = missing
        self.closed = False
        self.fail_close = fail_close

    def process(self, image, timestamp_ms):
        self.events.append(("hands", int(image[0, 0, 0])))
        return {
            "left_hand": np.full((21, 3), np.nan, dtype=np.float32),
            "right_hand": np.full((21, 3), np.nan if self.missing else 0.5, dtype=np.float32),
            "conf": np.array([np.nan, np.nan if self.missing else 0.9], dtype=np.float32),
        }

    def metadata(self):
        return {"backend": "injected_test_only", "coordinate_space": "image_normalized_xy_relative_z"}

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failure")


class TestPose:
    __test__ = False

    def __init__(self, events, fail_at=None):
        self.events = events
        self.fail_at = fail_at
        self.count = 0
        self.masks = []

    def register(self, rgb, ob_mask, **kwargs):
        self.masks.append(ob_mask.copy())
        return self.estimate(rgb)

    def track_one(self, rgb, **kwargs):
        assert "ob_mask" not in kwargs
        return self.estimate(rgb)

    def estimate(self, image):
        self.events.append(("pose", int(image[0, 0, 0])))
        if self.count == self.fail_at:
            raise RuntimeError("pose inference failure")
        self.count += 1
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 1
        return pose


def test_online_forecast_is_saved_with_current_frame_inference(sequence_args):
    sequence_args.forecast_method = "constant_velocity"
    sequence_args.forecast_window = 2
    events = []
    report = run_pipeline(sequence_args, TestHands(events), TestPose(events))
    assert report["status"] == "test_only_success"
    assert report["forecast"]["ready_frames"] == 2
    assert report["forecast"]["future_frame_inference"] is False
    with np.load(Path(sequence_args.out_dir) / "forecast_predictions.npz", allow_pickle=False) as data:
        assert data["hand_future"].shape == (3, 2, 2, 21, 3)
        assert data["object_future"].shape == (3, 2, 1, 4, 4)
        np.testing.assert_equal(data["ready"], [False, True, True])
        assert np.isnan(data["hand_future"][0]).all()
        assert np.isnan(data["hand_future"][:, :, 0]).all()
        np.testing.assert_allclose(data["observation_time"], np.arange(3) / 30)
    assert report["video"]["decoded_frames"] == 3


@pytest.fixture
def sequence_args(tmp_path):
    scene = tmp_path / "scene"
    for folder in ("rgb", "depth", "masks"):
        (scene / folder).mkdir(parents=True)
    for index in range(3):
        cv2.imwrite(str(scene / "rgb" / f"{index:03d}.png"), np.full((64, 80, 3), index + 10, dtype=np.uint8))
        cv2.imwrite(str(scene / "depth" / f"{index:03d}.png"), np.full((64, 80), 1000, dtype=np.uint16))
    mask = np.zeros((64, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    cv2.imwrite(str(scene / "masks/000.png"), mask)
    np.savetxt(scene / "cam_K.txt", [[50, 0, 40], [0, 50, 32], [0, 0, 1]])
    mesh = tmp_path / "mesh.obj"
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(mesh)
    return build_parser().parse_args([
        "--foundationpose-repo", str(tmp_path / "unused"), "--scene-dir", str(scene),
        "--mesh-file", str(mesh), "--out-dir", str(tmp_path / "output"),
        "--dataset-name", "synthetic_test_only", "--viewpoint", "test_only",
        "--human-evidence", "test_fixture_not_real_human_evidence", "--source-fps", "30",
        "--fps-source", "test_fixture_only", "--save-every", "1",
    ])


@pytest.mark.parametrize("prefetch", [False, True])
def test_same_frame_causal_export_and_honest_metadata(sequence_args, prefetch):
    sequence_args.prefetch_input = prefetch
    events = []
    hands, pose = TestHands(events), TestPose(events)
    report = run_pipeline(sequence_args, hands, pose)
    assert report["status"] == "test_only_success"
    assert report["real_models"] is False
    assert report["live_camera_validated"] is False
    assert report["coverage"]["joint_frames"] == 3
    assert all(value is None for value in report["accuracy"].values())
    assert report["video"]["decoded_frames"] == 3
    assert hands.closed
    for index in range(3):
        assert set(events[index * 2:index * 2 + 2]) == {("hands", index + 10), ("pose", index + 10)}
    assert len(pose.masks) == 1
    output = Path(sequence_args.out_dir)
    with np.load(output / "hand_poses.npz", allow_pickle=False) as archive:
        assert archive["right_hand"].shape == (3, 21, 3)
        assert np.isnan(archive["left_hand"]).all()
    with np.load(output / "object_poses.npz", allow_pickle=False) as archive:
        assert archive["poses"].shape == (1, 3, 4, 4)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["hand_coordinate_space"] == "image_normalized_xy_relative_z"
    assert "injected_test_only" in metadata["quality_flags"]
    assert metadata["input_prefetch"]["future_frame_inference"] is False
    assert metadata["input_prefetch"]["enabled"] is prefetch
    with (output / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [float(row["timestamp_ms"]) for row in rows] == pytest.approx([0, 1000 / 30, 2000 / 30])


def test_pacing_clock_starts_after_registration(sequence_args, monkeypatch):
    from src.pose6d import run_human_rgbd

    sleeps = []
    sequence_args.pace_replay = True
    monkeypatch.setattr(run_human_rgbd.time, "sleep", sleeps.append)
    run_pipeline(sequence_args, TestHands([]), TestPose([]))
    assert len(sleeps) == 3
    assert 0 < sleeps[0] <= 1 / sequence_args.source_fps


def test_no_hand_cannot_be_successful_human_delivery(sequence_args):
    with pytest.raises(RuntimeError, match="no human hand detected"):
        run_pipeline(sequence_args, TestHands([], missing=True), TestPose([]))
    report = json.loads((Path(sequence_args.out_dir) / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["coverage"]["joint_rate"] == 0


def test_pose_failure_does_not_fall_back_or_lose_failure_report(sequence_args):
    hand = TestHands([])
    with pytest.raises(RuntimeError, match="pose inference failure"):
        run_pipeline(sequence_args, hand, TestPose([], fail_at=1))
    report = json.loads((Path(sequence_args.out_dir) / "run_report.json").read_text())
    assert report["frames_completed"] == 1
    assert report["status"] == "failed"
    assert not (Path(sequence_args.out_dir) / "object_poses.npz").exists()
    assert hand.closed


def test_new_output_directory_required(sequence_args):
    Path(sequence_args.out_dir).mkdir()
    with pytest.raises(FileExistsError):
        run_pipeline(sequence_args, TestHands([]), TestPose([]))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_source_fps_rejected(sequence_args, value):
    sequence_args.source_fps = value
    with pytest.raises(ValueError, match="source_fps"):
        run_pipeline(sequence_args, TestHands([]), TestPose([]))
    report = json.loads((Path(sequence_args.out_dir) / "run_report.json").read_text())
    assert report["status"] == "failed"


def test_cleanup_error_does_not_replace_inference_error(sequence_args):
    with pytest.raises(RuntimeError, match="pose inference failure"):
        run_pipeline(sequence_args, TestHands([], fail_close=True), TestPose([], fail_at=1))
    report = json.loads((Path(sequence_args.out_dir) / "run_report.json").read_text())
    assert "pose inference failure" in report["error"]
    assert report["cleanup_errors"] == ["RuntimeError: close failure"]


def test_cleanup_error_cannot_leave_success_report(sequence_args):
    with pytest.raises(RuntimeError, match="resource cleanup failed"):
        run_pipeline(sequence_args, TestHands([], fail_close=True), TestPose([]))
    report = json.loads((Path(sequence_args.out_dir) / "run_report.json").read_text())
    assert report["status"] == "failed"


@pytest.mark.parametrize("name", ["fps_source", "human_evidence"])
def test_missing_provenance_rejected(sequence_args, name):
    setattr(sequence_args, name, " ")
    with pytest.raises(ValueError, match="evidence"):
        run_pipeline(sequence_args, TestHands([]), TestPose([]))


def test_p95_and_average_required_for_realtime_claim():
    rows = [{"total_ms": 1000}, {"total_ms": 10}, {"total_ms": 10}, {"total_ms": 60}]
    report = summarize_timing(rows, 30)
    assert report["steady_processing_fps"] > 30
    assert report["keeps_up_with_source_steady"] is False
    assert report["registration_frame_ms"] == 1000
    assert report["live_camera_validated"] is False
    assert summarize_timing(rows[:1], 30)["keeps_up_with_source_steady"] is None


@pytest.mark.parametrize("bad", ["shape", "partial_nan", "inf", "score", "missing_score"])
def test_hand_contract_validation(bad):
    result = TestHands([]).process(np.zeros((1, 1, 3), dtype=np.uint8), 0)
    if bad == "shape":
        result["right_hand"] = np.zeros((20, 3))
    elif bad == "partial_nan":
        result["right_hand"][0, 0] = np.nan
    elif bad == "inf":
        result["right_hand"][0, 0] = np.inf
    elif bad == "score":
        result["conf"][1] = 1.1
    else:
        result["conf"][0] = 0.5
    with pytest.raises(ValueError):
        checked_hands(result)
