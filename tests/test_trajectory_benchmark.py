"""CPU-only protocol regressions; fixtures are synthetic, not benchmark results."""
import json
from pathlib import Path
import subprocess
import sys
from uuid import UUID

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mamba import trajectory_benchmark as benchmark


def make_sequence(count=90, interval=1 / 30, offset=0.0):
    timestamps = np.arange(count, dtype=np.float64) * interval
    velocity = np.array([[0.2, -0.1, 0.3], [-0.1, 0.4, 0.2]])
    wrists = timestamps[:, None, None] * velocity + np.asarray(offset)
    return {
        "timestamps": timestamps,
        "frame_index": np.arange(count, dtype=np.int64),
        "left_hand_world": np.repeat(wrists[:, 0, None], 21, axis=1),
        "right_hand_world": np.repeat(wrists[:, 1, None], 21, axis=1),
        "hand_valid": np.ones((count, 2), dtype=bool),
    }


def make_staging(root, count=70):
    root.mkdir()
    for number in reversed(range(1, 11)):
        source_uuid = str(UUID(int=number))
        directory = root / source_uuid
        directory.mkdir()
        sequence = make_sequence(count=count, offset=number * 10.0)
        for name in ("left_hand_world", "right_hand_world"):
            sequence[name] *= number / 5.0
        np.savez_compressed(directory / "sequence.npz", **sequence)
        benchmark.write_json(directory / "meta.json", {
            "source_episode_uuid": source_uuid, "coordinate_system": "world",
            "frame_count": count, "task_name": f"task {number}",
            "world_to_camera_transform_applied": False,
        })
    return root


def make_pack(sequence=None, split="train", source_uuid="episode"):
    return benchmark.build_episode_samples(
        make_sequence() if sequence is None else sequence, source_uuid, split,
    )


def test_fixed_uuid_split_has_no_leakage(tmp_path):
    staging = make_staging(tmp_path / "staging")
    episodes, splits = benchmark.discover_splits(staging)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [6, 2, 2]
    assert splits["train"] + splits["val"] + splits["test"] == sorted(episodes)
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["val"]) & set(splits["test"])
    for split, uuids in splits.items():
        samples = benchmark.load_samples(episodes, uuids, split)
        assert set(samples.uuids) == set(uuids)
        assert {row["split"] for row in samples.provenance} == {split}


def test_requires_exactly_ten_episodes(tmp_path):
    with pytest.raises(ValueError, match="exactly 10"):
        benchmark.discover_splits(tmp_path)


def test_timestamp_target_and_observation_offsets():
    sequence = make_sequence(count=80, interval=0.07, offset=123.0)
    samples = make_pack(sequence)
    first = samples.provenance[0]
    assert first["observation_end_index"] == 15
    assert first["target_index"] == 30
    assert first["target_timestamp"] >= first["requested_target_timestamp"]
    assert sequence["timestamps"][first["target_index"] - 1] < first["requested_target_timestamp"]
    assert first["actual_horizon_seconds"] == pytest.approx(1.05)
    assert first["target_index"] > max(first["observation_frame_indices"])
    assert samples.history.shape[1:] == (16, 6)
    np.testing.assert_array_equal(samples.history[:, -1], 0)
    np.testing.assert_allclose(samples.target[0], [0.21, -0.105, 0.315, -0.105, 0.42, 0.21])
    assert samples.provenance[1]["start_index"] == 5
    translated = make_sequence(count=80, interval=0.07, offset=-35.0)
    np.testing.assert_allclose(samples.history, make_pack(translated).history, atol=1e-12)


@pytest.mark.parametrize("hand", [0, 1])
@pytest.mark.parametrize("location", ["observation", "target"])
def test_bad_hand_filters_entire_sample(hand, location):
    sequence = make_sequence()
    frame = 3 if location == "observation" else 45
    sequence["hand_valid"][frame, hand] = False
    samples = make_pack(sequence)
    assert not any(row["start_index"] == 0 for row in samples.provenance)
    assert samples.audits[0]["rejected"][f"invalid_{location}"] >= 1


def test_nonfinite_wrist_is_filtered_even_with_valid_flag():
    sequence = make_sequence()
    sequence["left_hand_world"][3, 0, 0] = np.nan
    samples = make_pack(sequence)
    assert samples.audits[0]["rejected"]["invalid_observation"] >= 1
    assert np.isfinite(samples.history).all()


@pytest.mark.parametrize("gap_start,reason", [(5, "observation_time_gap"), (45, "target_time_gap")])
def test_gaps_rejected(gap_start, reason):
    sequence = make_sequence()
    sequence["timestamps"][gap_start:] += 0.25
    samples = make_pack(sequence)
    assert not any(row["start_index"] == 0 for row in samples.provenance)
    assert samples.audits[0]["rejected"][reason] >= 1
    audit = samples.audits[0]
    assert audit["accepted"] + sum(audit["rejected"].values()) == audit["candidates"]


def test_nonmonotonic_timestamps_fail():
    sequence = make_sequence()
    sequence["timestamps"][1] = sequence["timestamps"][0]
    with pytest.raises(ValueError, match="increase strictly"):
        make_pack(sequence)


def test_normalization_train_history_only_and_degenerate_fallback():
    train = make_pack()
    expected_scale = float(np.std(train.history))
    train.target[:] = 1e9
    normalization = benchmark.fit_normalization(train)
    assert normalization["scale"] == expected_scale
    assert normalization["fit_split"] == "train"
    assert normalization["fit_uuids"] == ["episode"]
    assert normalization["targets_used_for_fit"] is False
    for split in ("val", "test"):
        with pytest.raises(ValueError, match="only be fitted on train"):
            benchmark.fit_normalization(make_pack(split=split))
    train.history[:] = 0
    assert benchmark.fit_normalization(train)["scale"] == 1.0


def test_constant_velocity_baseline_uses_timestamp_interval():
    sequence = make_sequence(interval=0.07)
    samples = make_pack(sequence)
    np.testing.assert_allclose(samples.velocity_prediction, samples.target, atol=1e-13)
    velocity = benchmark.endpoint_metrics(samples.velocity_prediction, samples)
    persistence = benchmark.endpoint_metrics(np.zeros_like(samples.target), samples)
    assert velocity["episode_equal"]["combined"] < 1e-13
    assert persistence["episode_equal"]["combined"] > 0.1
    assert benchmark.persistence_improvement(velocity, persistence)["episode_equal"]["combined"] == pytest.approx(100)


def test_episode_and_sample_weighting_differ():
    samples = make_pack()
    samples.target = np.zeros((3, 6))
    samples.uuids = np.array(["first", "first", "second"])
    prediction = np.array([[1, 0, 0, 3, 0, 0], [1, 0, 0, 3, 0, 0], [7, 0, 0, 9, 0, 0]])
    metrics = benchmark.endpoint_metrics(prediction, samples)
    assert metrics["sample_weighted"]["combined"] == 4
    assert metrics["episode_equal"]["combined"] == 5
    assert metrics["episode_equal"]["left"] == 4
    assert metrics["episode_equal"]["right"] == 6
    assert metrics["per_uuid"]["first"]["samples"] == 2


def test_zero_persistence_improvement_is_null():
    samples = make_pack()
    metrics = benchmark.endpoint_metrics(samples.target, samples)
    improvement = benchmark.persistence_improvement(metrics, metrics)
    assert improvement["episode_equal"]["combined"] is None


def test_cpu_training_checkpoint_and_test_quarantine(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    staging = make_staging(tmp_path / "staging")
    output = tmp_path / "result"
    events = []
    original_load = benchmark.load_samples
    original_train = benchmark.train_model
    original_predict = benchmark.predict

    def observed_load(episodes, uuids, split):
        events.append(("load", split))
        return original_load(episodes, uuids, split)

    def observed_train(*args):
        result = original_train(*args)
        events.append(("trained", result["seed"]))
        return result

    def observed_predict(model, samples, *args):
        events.append(("predict", samples.split))
        return original_predict(model, samples, *args)

    monkeypatch.setattr(benchmark, "load_samples", observed_load)
    monkeypatch.setattr(benchmark, "train_model", observed_train)
    monkeypatch.setattr(benchmark, "predict", observed_predict)
    report = benchmark.run_benchmark(
        staging, output, models=["lstm"], seeds=[17, 29], epochs=2, batch_size=128, device="cpu",
    )
    assert len(report["runs"]) == 2
    assert events.index(("load", "test")) > max(events.index(("trained", seed)) for seed in [17, 29])
    assert events.count(("load", "test")) == 1
    assert events.count(("predict", "test")) == 2
    assert (output / "sample_provenance.jsonl").is_file()
    assert (output / "splits.json").is_file()
    assert (output / "normalization.json").is_file()
    assert (output / "metrics.json").is_file()
    for result in report["runs"]:
        run_dir = Path(result["checkpoint"]).parent
        entries = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
        best = min(entries, key=lambda entry: entry["val"]["episode_equal"]["combined"])
        assert result["best_epoch"] == best["epoch"]
        assert result["reload_check"]["passed"]
        assert result["reload_check"]["prediction_max_abs_difference"] == 0
        assert result["test_evaluations"] == 1
        assert result["peak_cuda_allocated_bytes"] == 0
        assert result["total_seconds"] > 0
        assert all(entry["loss_grad_parameters_finite"] for entry in entries)
        model, checkpoint = benchmark.reload_checkpoint(result["checkpoint"], torch.device("cpu"))
        assert checkpoint["epoch"] == best["epoch"]
        assert checkpoint["normalization"]["fit_split"] == "train"
        with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as prediction_file:
            assert set(prediction_file["source_episode_uuid"]) == set(report["splits"]["test"])
            assert np.isfinite(prediction_file["prediction"]).all()
        assert isinstance(model.backbone, benchmark.LSTMBackbone)
    with pytest.raises(FileExistsError, match="new directory"):
        benchmark.run_benchmark(staging, output, models=["lstm"], device="cpu")


def test_nonfinite_training_loss_fails(tmp_path):
    torch.set_num_threads(1)
    training = make_pack()
    training.target[:] = 1e30
    config = {"device": "cpu", "d_model": 32, "learning_rate": 1e-3, "batch_size": 128, "epochs": 1}
    with pytest.raises(FloatingPointError, match="training loss"):
        benchmark.train_model(training, make_pack(split="val"), benchmark.fit_normalization(training),
                              tmp_path / "failed", config, "lstm", 17)


def test_direct_cli_cpu_smoke(tmp_path):
    staging = make_staging(tmp_path / "staging", count=55)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "src/mamba/trajectory_benchmark.py"),
         "--staging-root", str(staging), "--output-dir", str(tmp_path / "cli_result"),
         "--models", "lstm", "--seeds", "43", "--epochs", "1", "--device", "cpu"],
        cwd=tmp_path, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "cli_result/metrics.json").read_text())
    assert len(report["runs"]) == 1
    assert "NOT HOI" in report["limitation"]


@pytest.mark.parametrize("kwargs", [
    {"seeds": [17, 17]}, {"models": ["unknown"]}, {"epochs": 0},
    {"learning_rate": float("nan")}, {"d_model": 16}, {"seeds": [-1]},
])
def test_invalid_run_options_rejected_before_creating_output(tmp_path, kwargs):
    options = {"models": ["lstm"], "device": "cpu", **kwargs}
    with pytest.raises(ValueError):
        benchmark.run_benchmark(tmp_path / "absent", tmp_path / "new", **options)
    assert not (tmp_path / "new").exists()
