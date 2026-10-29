import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.forecasting.baselines import predict_baseline
from src.forecasting.data import PoseSequence, build_samples, load_manifest, target_indices, validate_horizons, validate_transforms
from src.forecasting.evaluate import aggregate_reports, evaluate_sequence
from src.forecasting.online import OnlineForecaster, replay_sequence
from src.forecasting.render import compose_frame


def make_sequence(count=95, rate=30):
    timestamps = np.arange(count, dtype=np.float64) / rate
    hands = np.zeros((count, 2, 21, 3), dtype=np.float32)
    hands[:, 0] = np.nan
    hands[:, 1, :, 0] = 0.2 + timestamps[:, None] * 0.1
    hands[:, 1, :, 1] = 0.4
    objects = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1, 1))
    objects[:, 0, 2, 3] = 1
    objects[:, 0, 0, 3] = timestamps * 0.05
    boxes = np.tile([10, 10, 30, 30], (count, 1, 1))
    return PoseSequence("fixture", "fixture_group", hands, objects, boxes, timestamps, {},
                        [{"stem": str(index)} for index in range(count)], Path("test_only"))


def test_persistence_copies_latest_without_aliasing():
    sequence = make_sequence()
    output = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32])
    np.testing.assert_equal(output["hand_future"][0], sequence.hands[31])
    np.testing.assert_allclose(output["object_future"][1], sequence.objects[31])
    output["object_future"][0, 0, 0, 3] = 99
    assert sequence.objects[31, 0, 0, 3] != 99


def test_constant_velocity_predicts_linear_motion():
    sequence = make_sequence()
    output = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32], method="constant_velocity")
    np.testing.assert_allclose(output["hand_future"][1, 1], sequence.hands[61, 1], atol=1e-6)
    np.testing.assert_allclose(output["object_future"][1], sequence.objects[61], atol=1e-6)
    assert np.isnan(output["hand_future"][:, 0]).all()


def test_rotation_extrapolation_is_so3():
    sequence = make_sequence()
    for index in range(len(sequence.timestamps)):
        sequence.objects[index, 0, :3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, sequence.timestamps[index] * 0.2]))[0]
    output = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32], method="constant_velocity")
    validate_transforms(output["object_future"])
    np.testing.assert_allclose(output["object_future"][1, 0, :3, :3], sequence.objects[61, 0, :3, :3], atol=1e-5)


def test_missing_current_entity_not_fabricated():
    sequence = make_sequence()
    sequence.hands[31] = np.nan
    sequence.objects[31] = np.nan
    output = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32], method="constant_velocity")
    assert np.isnan(output["hand_future"]).all()
    assert np.isnan(output["object_future"]).all()


def test_recovery_does_not_use_velocity_across_missing_frame():
    sequence = make_sequence()
    sequence.hands[30] = np.nan
    sequence.objects[30] = np.nan
    output = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32], method="constant_velocity")
    np.testing.assert_equal(output["hand_future"][0], sequence.hands[31])
    np.testing.assert_equal(output["object_future"][0], sequence.objects[31])


@pytest.mark.parametrize("horizons", [[], [0], [-1], [1, 0.5], [0.5, 0.5], [float("nan")]])
def test_invalid_horizons(horizons):
    with pytest.raises(ValueError):
        validate_horizons(horizons)


@pytest.mark.parametrize("damage", ["last_row", "reflection", "partial_nan"])
def test_invalid_transforms_rejected(damage):
    poses = np.eye(4)[None]
    if damage == "last_row":
        poses[0, 3, 0] = 1
    elif damage == "reflection":
        poses[0, 0, 0] = -1
    else:
        poses[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_transforms(poses)


def test_sample_horizons_and_input_prefix():
    sequence = make_sequence()
    samples, rejected = build_samples(sequence)
    first = samples[0]
    assert first["observation_index"] == 31
    np.testing.assert_equal(first["target_indices"], [46, 61])
    np.testing.assert_allclose(first["actual_horizons"], [0.5, 1.0])
    np.testing.assert_equal(first["hand_seq"], sequence.hands[:32])
    sequence.hands[32:, 1] += 100
    changed, _ = build_samples(sequence)
    np.testing.assert_equal(first["hand_seq"], changed[0]["hand_seq"])
    np.testing.assert_equal(first["obj_seq"], changed[0]["obj_seq"])
    assert not np.array_equal(first["hand_future"], changed[0]["hand_future"], equal_nan=True)
    assert rejected["no_future_or_gap"] == 30


def test_short_44_frame_sequence_cannot_train_default_window():
    samples, rejected = build_samples(make_sequence(44))
    assert not samples
    assert rejected["no_future_or_gap"] == 13


def test_gap_targets_are_not_synthesized():
    times = np.arange(100) / 30
    times[45:] += 1
    assert target_indices(times, 31, [0.5, 1.0]) is None


def test_partial_future_supervision_preserves_nan():
    sequence = make_sequence()
    sequence.hands[46] = np.nan
    samples, _ = build_samples(sequence)
    assert np.isnan(samples[0]["hand_future"][0]).all()
    assert np.isfinite(samples[0]["hand_future"][1, 1]).all()


def test_split_leakage_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("src.forecasting.data.load_sequence", lambda *args: make_sequence())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"contract_version": "pose_forecast_v1", "sequences": [
        {"sequence_id": "camera1", "group_id": "same_capture", "split": "train", "observation_dir": "first"},
        {"sequence_id": "camera2", "group_id": "same_capture", "split": "test", "observation_dir": "second"},
    ]}))
    with pytest.raises(ValueError, match="leaks"):
        load_manifest(path)


def test_online_warmup_and_causality():
    sequence = make_sequence()
    original = replay_sequence(sequence, OnlineForecaster("constant_velocity"))
    assert not original["ready"][:31].any()
    assert original["ready"][31:].all()
    sequence.hands[40:, 1] += 10
    sequence.objects[40:, 0, 0, 3] += 4
    changed = replay_sequence(sequence, OnlineForecaster("constant_velocity"))
    np.testing.assert_equal(original["hand_future"][:40], changed["hand_future"][:40])
    np.testing.assert_equal(original["object_future"][:40], changed["object_future"][:40])


def test_duplicate_stream_timestamp_rejected():
    sequence = make_sequence()
    forecaster = OnlineForecaster()
    forecaster.update(sequence.hands[0], sequence.objects[0], sequence.bboxes[0], 0)
    with pytest.raises(ValueError, match="increasing"):
        forecaster.update(sequence.hands[0], sequence.objects[0], sequence.bboxes[0], 0)


def test_learned_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        OnlineForecaster("learned")


def test_forecast_metrics_zero_for_true_linear_model():
    sequence = make_sequence()
    predictions = replay_sequence(sequence, OnlineForecaster("constant_velocity"))
    report = evaluate_sequence(sequence, predictions)
    for horizon in report["horizons"]:
        assert horizon["joint_coverage"] == 1
        assert horizon["hand_xy_pixels"]["mean"] < 1e-3
        assert horizon["object_translation_mm"]["mean"] < 1e-3
        assert horizon["object_rotation_degrees"]["mean"] == 0
    assert report["independent_accuracy"] is None


def test_persistence_metrics_have_explicit_units():
    sequence = make_sequence()
    predictions = replay_sequence(sequence, OnlineForecaster("persistence"))
    report = evaluate_sequence(sequence, predictions)
    assert report["horizons"][1]["hand_xy_pixels"]["mean"] == pytest.approx(64, abs=1e-3)
    assert report["horizons"][1]["object_translation_mm"]["mean"] == pytest.approx(50, abs=1e-3)


def test_missing_predictions_reduce_coverage_not_error_to_zero():
    sequence = make_sequence()
    predictions = replay_sequence(sequence, OnlineForecaster())
    predictions["hand_future"][:] = np.nan
    report = evaluate_sequence(sequence, predictions)
    assert report["horizons"][0]["joint_coverage"] == 0
    assert report["horizons"][0]["hand_xy_pixels"]["mean"] is None
    json.dumps(report, allow_nan=False)


def test_aggregation_rejects_multiview_duplicate_groups():
    sequence = make_sequence()
    report = evaluate_sequence(sequence, replay_sequence(sequence, OnlineForecaster()))
    with pytest.raises(ValueError, match="acquisition"):
        aggregate_reports([report, report])


def test_video_left_panel_never_reads_future_image():
    sequence = make_sequence()
    current = np.zeros((120, 160, 3), np.uint8)
    future = np.full_like(current, 255)
    forecast = predict_baseline(sequence.hands[:32], sequence.objects[:32], sequence.timestamps[:32])
    forecast["ready"] = True
    intrinsics = np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1]])
    corners = np.array([[left, top, depth] for left in (-0.1, 0.1) for top in (-0.1, 0.1) for depth in (-0.1, 0.1)])
    first = compose_frame(current, sequence.hands[31], sequence.objects[31], forecast, intrinsics,
                          corners, [0.5, 1.0], 1.0, "test_only", future, sequence.hands[61], sequence.objects[61])
    second = compose_frame(current, sequence.hands[31], sequence.objects[31], forecast, intrinsics,
                           corners, [0.5, 1.0], 1.0, "test_only", current, sequence.hands[61], sequence.objects[61])
    np.testing.assert_equal(first[:, :160], second[:, :160])
    assert not np.array_equal(first[:, 160:], second[:, 160:])
