"""Independent adversarial checks; synthetic fixtures are never delivery evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import trimesh

from src.forecasting.baselines import predict_baseline
from src.forecasting.benchmark import run_benchmark
from src.forecasting.data import PoseSequence, build_samples, load_manifest, load_sequence
from src.forecasting.evaluate import evaluate_sequence
from src.forecasting.model import PoseForecastModel
from src.forecasting.online import OnlineForecaster, replay_sequence
from src.forecasting.render import compose_frame
from src.forecasting.run import run
from src.forecasting.train import fit_forecaster, load_forecaster
from src.pose6d.run_human_rgbd import build_parser, run_pipeline


@pytest.fixture(autouse=True)
def cpu_only_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def review_sequence(count=110, sequence_id="qa_synthetic", group_id="qa_synthetic_capture"):
    timestamps = np.arange(count, dtype=np.float64) / 30
    hands = np.full((count, 2, 21, 3), np.nan, dtype=np.float32)
    hands[:, 1, :, 0] = 0.25 + timestamps[:, None] * 0.06
    hands[:, 1, :, 1] = 0.45
    hands[:, 1, :, 2] = 0
    objects = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1, 1))
    objects[:, 0, 0, 3] = 0.1 + timestamps * 0.12
    objects[:, 0, 2, 3] = 1.2
    boxes = np.tile([100, 110, 200, 210], (count, 1, 1))
    records = [{"stem": f"{index:06d}"} for index in range(count)]
    return PoseSequence(sequence_id, group_id, hands, objects, boxes, timestamps,
                        {"test_only": True}, records, Path("qa_synthetic_only"))


def sample_at(sequence, observation_index=40):
    samples, _ = build_samples(sequence)
    return next(sample for sample in samples if sample["observation_index"] == observation_index)


def test_future_timestamps_cannot_change_historical_model_features():
    original = review_sequence()
    changed = copy.deepcopy(original)
    changed.timestamps[32:] = original.timestamps[31] + np.arange(1, len(changed.timestamps) - 31) / 10
    original_sample = sample_at(original, 31)
    changed_sample = sample_at(changed, 31)
    np.testing.assert_array_equal(original_sample["hand_seq"], changed_sample["hand_seq"])
    np.testing.assert_allclose(original_sample["obj_seq"], changed_sample["obj_seq"], atol=1e-7)


def test_irregular_timestamp_velocity_uses_each_elapsed_interval():
    sequence = review_sequence()
    intervals = np.resize([0.03, 0.04], len(sequence.timestamps) - 1)
    sequence.timestamps = np.concatenate(([0.0], np.cumsum(intervals)))
    sequence.objects[:, 0, 0, 3] = 0.1 + sequence.timestamps * 0.12
    sample = sample_at(sequence)
    np.testing.assert_allclose(sample["obj_seq"][1:, 0, 10], 0.12, atol=2e-6)


def test_training_and_online_windows_have_identical_model_inputs():
    sequence = review_sequence()
    forecaster = OnlineForecaster("persistence")
    captured = []

    class CaptureInputs:
        def __call__(self, hand_seq, obj_seq, obj_mask):
            captured.append((hand_seq.detach().cpu().numpy().copy(), obj_seq.detach().cpu().numpy().copy()))
            return {"hand_future": torch.full((1, 2, 2, 21, 3), torch.nan),
                    "object_future": torch.full((1, 2, 1, 4, 4), torch.nan)}

    forecaster.model = CaptureInputs()
    for index in range(41):
        forecaster.update(sequence.hands[index], sequence.objects[index], sequence.bboxes[index],
                          sequence.timestamps[index])
    sample = sample_at(sequence)
    np.testing.assert_array_equal(captured[-1][0][0], sample["hand_seq"])
    np.testing.assert_allclose(captured[-1][1][0], sample["obj_seq"], atol=1e-7)


def test_renaming_same_observation_directory_cannot_bypass_split_isolation(tmp_path, monkeypatch):
    def synthetic_loader(observation_dir, sequence_id, group_id, scene_dir=None):
        sequence = review_sequence(sequence_id=sequence_id, group_id=group_id)
        sequence.observation_dir = Path(observation_dir)
        return sequence

    monkeypatch.setattr("src.forecasting.data.load_sequence", synthetic_loader)
    observation_dir = tmp_path / "same_observations"
    observation_dir.mkdir()
    manifest = {"contract_version": "pose_forecast_v1", "sequences": [
        {"sequence_id": "renamed_train", "group_id": "group_train", "split": "train",
         "observation_dir": str(observation_dir)},
        {"sequence_id": "renamed_test", "group_id": "group_test", "split": "test",
         "observation_dir": str(observation_dir / ".")},
    ]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="(?i)(duplicate|repeat|leak|observation)"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("damage", ["timestamps", "horizons_seconds"])
def test_evaluation_rejects_forecast_time_alignment_mismatch(damage):
    sequence = review_sequence()
    predictions = replay_sequence(sequence, OnlineForecaster())
    predictions[damage] = predictions[damage] + 0.25
    with pytest.raises(ValueError, match="(?i)(time|horizon|align)"):
        evaluate_sequence(sequence, predictions)


def test_not_ready_frames_cannot_count_as_successful_predictions():
    sequence = review_sequence()
    predictions = replay_sequence(sequence, OnlineForecaster())
    predictions["ready"][:] = False
    report = evaluate_sequence(sequence, predictions)
    for horizon in report["horizons"]:
        assert horizon["eligible_joint_frames"] > 0
        assert horizon["predicted_joint_frames"] == 0
        assert horizon["joint_coverage"] == 0
        assert horizon["hand_xy_pixels"]["count"] == 0
        assert horizon["object_translation_mm"]["count"] == 0


def test_missing_prediction_is_not_zero_error_or_hidden_coverage():
    sequence = review_sequence()
    predictions = replay_sequence(sequence, OnlineForecaster("constant_velocity"))
    predictions["hand_future"][40, :, 1] = np.nan
    report = evaluate_sequence(sequence, predictions)
    for horizon in report["horizons"]:
        assert horizon["predicted_joint_frames"] == horizon["eligible_joint_frames"] - 1
        assert horizon["hand_xy_pixels"]["count"] == horizon["eligible_hand_targets"] - 1
        assert horizon["object_translation_mm"]["count"] == horizon["eligible_object_targets"]
        assert horizon["hand_xy_pixels"]["mean"] < 0.001
    assert report["independent_accuracy"] is None
    json.dumps(report, allow_nan=False)


def test_full_output_coverage_does_not_imply_small_pose_error():
    sequence = review_sequence()
    predictions = replay_sequence(sequence, OnlineForecaster("constant_velocity"))
    predictions["hand_future"][..., 0] += 3
    predictions["object_future"][..., 0, 3] += 10
    report = evaluate_sequence(sequence, predictions)
    for horizon in report["horizons"]:
        assert horizon["joint_coverage"] == 1
        assert horizon["hand_xy_pixels"]["mean"] == pytest.approx(1920, abs=0.001)
        assert horizon["object_translation_mm"]["mean"] == pytest.approx(10000, abs=0.001)
    assert report["independent_accuracy"] is None


def test_stream_rejects_partial_hand_nan_before_mutating_history():
    sequence = review_sequence()
    forecaster = OnlineForecaster()
    broken_hands = sequence.hands[0].copy()
    broken_hands[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="(?i)(hand|NaN|finite)"):
        forecaster.update(broken_hands, sequence.objects[0], sequence.bboxes[0], sequence.timestamps[0])
    assert not forecaster.history
    assert forecaster.last_time is None


def test_rotation_crossing_pi_remains_correct_and_proper():
    sequence = review_sequence()
    angular_rate = 0.35
    for index, timestamp in enumerate(sequence.timestamps):
        sequence.objects[index, 0, :3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, 3.0 + angular_rate * timestamp]))[0]
    result = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32],
                              method="constant_velocity")
    for horizon_index, horizon in enumerate([0.5, 1.0]):
        rotation = result["object_future"][horizon_index, 0, :3, :3]
        expected = cv2.Rodrigues(np.array([0.0, 0.0, 3.0 + angular_rate * (sequence.timestamps[31] + horizon)]))[0]
        np.testing.assert_allclose(rotation, expected, atol=2e-5)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-6)


def test_left_video_panel_is_independent_of_future_pose_labels():
    sequence = review_sequence()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    forecast = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32])
    intrinsics = np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1]])
    corners = np.array([[horizontal, vertical, depth] for horizontal in (-0.1, 0.1)
                        for vertical in (-0.1, 0.1) for depth in (-0.1, 0.1)])
    first = compose_frame(image, sequence.hands[31], sequence.objects[31], forecast, intrinsics,
                          corners, [0.5, 1.0], sequence.timestamps[31], "qa_synthetic_only", image,
                          sequence.hands[61], sequence.objects[61])
    future_hands = sequence.hands[61].copy()
    future_hands[1, :, 0] += 0.5
    future_objects = sequence.objects[61].copy()
    future_objects[:, 0, 3] += 0.5
    second = compose_frame(image, sequence.hands[31], sequence.objects[31], forecast, intrinsics,
                           corners, [0.5, 1.0], sequence.timestamps[31], "qa_synthetic_only", image,
                           future_hands, future_objects)
    np.testing.assert_array_equal(first[:, :160], second[:, :160])
    assert not np.array_equal(first[:, 160:], second[:, 160:])


def test_random_initialized_checkpoint_is_not_authorized_as_trained(tmp_path):
    model = PoseForecastModel({"d_model": 8, "hidden_dim": 8, "n_heads": 2, "dim_ff": 16})
    checkpoint_path = tmp_path / "random_initialization_only.pt"
    torch.save({"contract_version": "pose_forecast_v1", "config": model.config,
                "model_state_dict": model.state_dict()}, checkpoint_path)
    with pytest.raises(ValueError, match="(?i)(train|provenance|checkpoint|selection)"):
        load_forecaster(checkpoint_path, device="cpu")


def test_best_checkpoint_selected_by_validation_not_last_epoch(tmp_path, monkeypatch):
    train_sample = sample_at(review_sequence(sequence_id="qa_train", group_id="qa_train_capture"))
    val_sample = sample_at(review_sequence(sequence_id="qa_val", group_id="qa_val_capture"))
    validations = iter([3.0, 1.0, 2.0, 1.0])
    calls = []

    def synthetic_epoch(model, loader, config, optimizer=None):
        calls.append((optimizer is not None, tuple(sorted(loader.dataset.group_ids))))
        loss = 0.0 if optimizer is not None else next(validations)
        return {"loss": loss, "hand_loss": loss, "translation_loss": 0.0,
                "rotation_loss": 0.0, "hand_count": 2, "object_count": 2, "skipped_batches": 0,
                "optimizer_steps": int(optimizer is not None)}

    monkeypatch.setattr("src.forecasting.train._run_epoch", synthetic_epoch)
    report = fit_forecaster([train_sample], [val_sample], tmp_path / "synthetic_selection_test",
                            {"d_model": 8, "hidden_dim": 8, "n_heads": 2, "dim_ff": 16,
                             "epochs": 2, "batch_size": 1, "device": "cpu",
                             "source_metadata": {"test_only": True}})
    checkpoint = torch.load(report["checkpoint_path"], weights_only=True, map_location="cpu")
    assert report["best_epoch"] == checkpoint["epoch"] == 1
    assert report["best_val_loss"] == 1.0
    assert checkpoint["selection"] == "minimum_validation_masked_loss_only"
    assert calls == [(False, ("qa_val_capture",)), (True, ("qa_train_capture",)),
                     (False, ("qa_val_capture",)), (True, ("qa_train_capture",)),
                     (False, ("qa_val_capture",)), (False, ("qa_val_capture",))]
    assert report["provenance"]["test_accessed"] is False


def test_delivery_report_exposes_untrained_selected_initialization(tmp_path, monkeypatch):
    sequence = review_sequence()
    sequence.metadata["observation_hashes"] = {}
    forecaster = OnlineForecaster()
    forecaster.model_metadata = {
        "model_status": "validation_selected_untrained_persistence_initialization",
        "epoch": 0, "optimizer_steps": 0, "selection": "minimum_validation_masked_loss_only",
        "checkpoint_role": "best",
        "config": {"backbone": "lstm"},
    }
    monkeypatch.setattr("src.forecasting.run.load_sequence", lambda *args: sequence)
    monkeypatch.setattr("src.forecasting.run.OnlineForecaster", lambda *args: forecaster)
    monkeypatch.setattr("src.forecasting.run.read_rgb", lambda *args: np.zeros((480, 640, 3), np.uint8))
    checkpoint_path = tmp_path / "qa_not_a_real_checkpoint.txt"
    checkpoint_path.write_text("Synthetic report propagation fixture, never a trained checkpoint.")
    args = SimpleNamespace(out_dir=tmp_path / "qa_report_fixture", method="learned", window_size=32,
                           horizons=[0.5, 1.0], observation_dir="qa_only", sequence_id=sequence.sequence_id,
                           group_id=sequence.group_id, scene_dir=None, checkpoint=checkpoint_path,
                           device="cpu", no_video=True)
    report = run(args)
    assert report["checkpoint"]["model_status"] == forecaster.model_metadata["model_status"]
    assert report["checkpoint"]["epoch"] == 0
    assert report["checkpoint"]["optimizer_steps"] == 0
    assert report["checkpoint"]["selection"] == "minimum_validation_masked_loss_only"
    assert report["checkpoint"]["checkpoint_role"] == "best"
    assert "INIT" in report["display_method"]


def test_observation_loader_requires_explicit_real_hand_provenance(tmp_path):
    sequence = review_sequence(count=3)
    metadata = {"hand_coordinate_space": "image_normalized_xy_relative_z",
                "viewpoint": "third_person_fixed_camera", "pose_convention": "original_mesh_object_to_camera",
                "object_backend": {"real_model": True}, "mesh_units": "m_source_declared",
                "num_frames": 3, "frames": sequence.records}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    np.savez(tmp_path / "hand_poses.npz", left_hand=sequence.hands[:, 0], right_hand=sequence.hands[:, 1])
    np.savez(tmp_path / "object_poses.npz", poses=sequence.objects.transpose(1, 0, 2, 3),
             bboxes=sequence.bboxes.transpose(1, 0, 2))
    rows = ["frame_index,stem,timestamp_ms"]
    rows.extend(f"{index},{record['stem']},{sequence.timestamps[index] * 1000}"
                for index, record in enumerate(sequence.records))
    (tmp_path / "frames.csv").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="(?i)(hand|provenance|real)"):
        load_sequence(tmp_path, "qa_synthetic", "qa_synthetic_capture")


def test_online_saved_forecasts_are_directly_usable_by_common_evaluator(tmp_path):
    scene = tmp_path / "qa_synthetic_scene"
    for folder in ("rgb", "depth", "masks"):
        (scene / folder).mkdir(parents=True)
    for index in range(6):
        cv2.imwrite(str(scene / "rgb" / f"{index:03d}.png"), np.full((64, 80, 3), index + 10, np.uint8))
        cv2.imwrite(str(scene / "depth" / f"{index:03d}.png"), np.full((64, 80), 1000, np.uint16))
    mask = np.zeros((64, 80), np.uint8)
    mask[20:40, 30:50] = 255
    cv2.imwrite(str(scene / "masks/000.png"), mask)
    np.savetxt(scene / "cam_K.txt", [[50, 0, 40], [0, 50, 32], [0, 0, 1]])
    mesh_path = tmp_path / "qa_cube.obj"
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(mesh_path)
    args = build_parser().parse_args([
        "--foundationpose-repo", str(tmp_path / "qa_unused"), "--scene-dir", str(scene),
        "--mesh-file", str(mesh_path), "--out-dir", str(tmp_path / "qa_online_output"),
        "--dataset-name", "qa_synthetic_only", "--viewpoint", "test_only",
        "--human-evidence", "qa_synthetic_not_real", "--source-fps", "30", "--fps-source", "qa_only",
        "--forecast-method", "constant_velocity", "--forecast-window", "2",
        "--forecast-horizons", str(1 / 30), str(2 / 30),
    ])

    class SyntheticHands:
        def process(self, image, timestamp_ms):
            return {"left_hand": np.full((21, 3), np.nan, np.float32),
                    "right_hand": np.full((21, 3), 0.4, np.float32),
                    "conf": np.array([np.nan, 0.9], np.float32)}

        def metadata(self):
            return {"backend": "injected_test_only", "test_only": True}

        def close(self):
            return None

    class SyntheticPose:
        def register(self, **inputs):
            pose = np.eye(4, dtype=np.float32)
            pose[2, 3] = 1
            return pose

        def track_one(self, **inputs):
            return self.register(**inputs)

    report = run_pipeline(args, SyntheticHands(), SyntheticPose())
    assert report["status"] == "test_only_success"
    output = Path(args.out_dir)
    with np.load(output / "forecast_predictions.npz", allow_pickle=False) as archive:
        predictions = dict(archive)
    assert "timestamps" in predictions, "online archive must use the same timestamp contract as replay"
    np.testing.assert_array_equal(predictions["timestamps"], predictions["observation_time"])
    sequence = review_sequence(count=6)
    sequence.timestamps = predictions["timestamps"]
    with np.load(output / "hand_poses.npz", allow_pickle=False) as archive:
        sequence.hands = np.stack([archive["left_hand"], archive["right_hand"]], axis=1)
    with np.load(output / "object_poses.npz", allow_pickle=False) as archive:
        sequence.objects = archive["poses"].transpose(1, 0, 2, 3)
    evaluation = evaluate_sequence(sequence, predictions, horizons=(1 / 30, 2 / 30), window_size=2,
                                   image_size=(80, 64))
    assert evaluation["horizons"][0]["joint_coverage"] == 1
    assert evaluation["horizons"][1]["joint_coverage"] == 1


def test_benchmark_evaluation_order_is_not_misreported_as_loading_order(tmp_path, monkeypatch):
    events = []
    splits = {name: [review_sequence(sequence_id="qa_" + name, group_id="qa_capture_" + name)]
              for name in ("train", "val", "test")}
    for sequences in splits.values():
        sequences[0].metadata["observation_hashes"] = {}

    def synthetic_manifest_loader(path):
        events.append("all_splits_loaded_including_test")
        return {}, splits

    def synthetic_fit(train_samples, val_samples, destination, config):
        events.append("train_fit")
        return {"best_epoch": 0, "best_val_loss": 0.0, "checkpoint_path": "qa_no_real_checkpoint",
                "checkpoint_sha256": "0" * 64}

    def audited_evaluation(*args, **kwargs):
        events.append("test_evaluation")
        return evaluate_sequence(*args, **kwargs)

    monkeypatch.setattr("src.forecasting.benchmark.load_manifest", synthetic_manifest_loader)
    monkeypatch.setattr("src.forecasting.benchmark.fit_forecaster", synthetic_fit)
    monkeypatch.setattr("src.forecasting.benchmark.OnlineForecaster", lambda *args: OnlineForecaster())
    monkeypatch.setattr("src.forecasting.benchmark.evaluate_sequence", audited_evaluation)
    manifest_path = tmp_path / "qa_stub_manifest.json"
    manifest_path.write_text("{}")
    report = run_benchmark(manifest_path, tmp_path / "qa_benchmark", models=("lstm",), seeds=(7,),
                           epochs=1, device="cpu")
    assert events.index("all_splits_loaded_including_test") < events.index("train_fit")
    assert events.index("train_fit") < events.index("test_evaluation")
    assert report.get("test_loading_after_all_training") is not True
    assert report["test_evaluation_after_all_training"] is True
