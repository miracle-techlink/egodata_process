import copy
import json

import numpy as np
import pytest
import torch

from src.encoding.encoder import HOIEncoder
from src.forecasting.model import PoseForecastModel, masked_pose_loss, rotation_6d_to_matrix
from src.forecasting.train import ForecastSamples, _run_epoch, _training_config, fit_forecaster, load_forecaster
from src.mamba.train import LSTMBackbone


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config(**overrides):
    config = {"d_model": 8, "hidden_dim": 16, "n_heads": 2, "dim_ff": 16,
              "window_size": 4, "n_max": 2, "epochs": 3, "batch_size": 2,
              "learning_rate": 0.005, "device": "cpu", "seed": 19}
    config.update(overrides)
    return config


def example_batch(batch_size=2):
    generator = torch.Generator().manual_seed(45)
    hands = torch.rand(batch_size, 4, 2, 21, 3, generator=generator)
    objects = torch.zeros(batch_size, 4, 2, 13)
    objects[..., :3] = torch.tensor([0.1, 0.2, 1.0])
    objects[..., 3:9] = torch.tensor([0., 1., 0., -1., 0., 0.])
    objects[..., 9] = 30720
    return {"hand_seq": hands, "obj_seq": objects, "obj_mask": torch.ones(batch_size, 2, dtype=torch.bool)}


def expected_targets(batch):
    hands = batch["hand_seq"][:, -1:, ...].repeat(1, 2, 1, 1, 1).clone()
    objects = torch.eye(4).expand(len(hands), 2, 2, 4, 4).clone()
    objects[..., :3, :3] = rotation_6d_to_matrix(batch["obj_seq"][:, -1:, :, 3:9])
    objects[..., :3, 3] = batch["obj_seq"][:, -1:, :, :3]
    return {"hand_future": hands, "object_future": objects}


def assert_finite_gradients(model):
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def samples(split, count=4):
    batch = example_batch(count)
    targets = expected_targets(batch)
    targets["hand_future"] += 0.04
    targets["object_future"][..., :3, 3] += 0.03
    values = dict(batch, **targets)
    return [dict({name: value[index].numpy().copy() for name, value in values.items()},
                 sequence_id=f"{split}-sequence", group_id=f"{split}-acquisition") for index in range(count)]


def test_shapes_reuse_and_zero_initialized_hold_pose():
    model = PoseForecastModel(small_config()).eval()
    batch = example_batch()
    forecast = model(**batch)
    assert isinstance(model.encoder, HOIEncoder)
    assert isinstance(model.backbone, LSTMBackbone)
    assert forecast["hand_future"].shape == (2, 2, 2, 21, 3)
    assert forecast["object_future"].shape == (2, 2, 2, 4, 4)
    expected = expected_targets(batch)
    for name in expected:
        torch.testing.assert_close(forecast[name], expected[name])
    assert forecast["hand_valid"].shape == (2, 2, 2)
    assert forecast["object_valid"].all()


def test_latest_missing_is_not_forward_filled_and_padding_is_nan():
    model = PoseForecastModel(small_config())
    batch = example_batch()
    batch["hand_seq"][0, -1, 0, 5, 1] = torch.nan
    batch["obj_seq"][0, -1, 0] = torch.nan
    batch["obj_mask"][1, 1] = False
    result = model(**batch)
    assert result["hand_future"][0, :, 0].isnan().all()
    assert result["object_future"][0, :, 0].isnan().all()
    assert result["object_future"][1, :, 1].isnan().all()
    assert result["object_future"][1, :, 0].isfinite().all()
    assert result["hand_future"][0, :, 1].isfinite().all()


@pytest.mark.parametrize("rotation", [[0., 0., 0., 0., 0., 0.], [1., 0., 0., 2., 0., 0.],
                                     [0., 1., 0., 0., 1., 0.], [2., 3., -4., 1., -1., 2.],
                                     [1., 1., 1., 1e6, 1e6, 1e6],
                                     [1e30, 1e30, 1e30, 1e30, -1e30, 1e30]])
def test_rotation_degeneracies_produce_so3_and_finite_gradients(rotation):
    values = torch.tensor(rotation, requires_grad=True)
    matrix = rotation_6d_to_matrix(values)
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.tensor(1.), atol=1e-5, rtol=1e-5)
    matrix.sum().backward()
    assert values.grad.isfinite().all()


def test_nonzero_rotation_residual_and_homogeneous_rows():
    model = PoseForecastModel(small_config()).eval()
    with torch.no_grad():
        model.decoder.object_head[-1].bias.copy_(torch.tensor([0.1, 0.2, 0.3, -1., 1., 0., -1., -1., 0.] * 2))
    result = model(**example_batch())["object_future"]
    rotations = result[..., :3, :3]
    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations,
                               torch.eye(3).expand_as(rotations), atol=1e-5, rtol=1e-5)
    assert (torch.linalg.det(rotations) > 0.9999).all()
    torch.testing.assert_close(result[..., 3, :], torch.tensor([0., 0., 0., 1.]).expand(2, 2, 2, 4))
    expected_delta = rotation_6d_to_matrix(torch.tensor([0., 1., 0., -1., 0., 0.]))
    current_rotation = expected_targets(example_batch())["object_future"][0, 0, 0, :3, :3]
    torch.testing.assert_close(result[0, 0, 0, :3, :3], expected_delta @ current_rotation)


def test_degenerate_observed_rotation_is_missing_not_identity_prediction():
    batch = example_batch()
    batch["obj_seq"][:, -1, 0, 3:9] = 0
    result = PoseForecastModel(small_config())(**batch)
    assert result["object_future"][:, :, 0].isnan().all()
    assert result["object_future"][:, :, 1].isfinite().all()


@pytest.mark.parametrize("depth", [0., -0.1])
def test_nonpositive_current_depth_is_missing(depth):
    batch = example_batch()
    batch["obj_seq"][:, -1, 0, 2] = depth
    result = PoseForecastModel(small_config())(**batch)
    assert result["object_future"][:, :, 0].isnan().all()
    assert not result["object_valid"][:, :, 0].any()
    assert result["object_valid"][:, :, 1].all()


def test_negative_future_depth_is_explicitly_invalid():
    model = PoseForecastModel(small_config()).eval()
    with torch.no_grad():
        model.decoder.object_head[-1].bias[2] = -20
    result = model(**example_batch())
    assert result["object_future"][:, 0].isnan().all()
    assert not result["object_valid"][:, 0].any()
    assert result["object_valid"][:, 1].all()


@pytest.mark.parametrize("training", [False, True])
def test_negative_prediction_cannot_evade_supervision_through_nan_mask(training):
    config = _training_config(small_config())
    model = PoseForecastModel(config)
    with torch.no_grad():
        model.decoder.object_head[-1].bias[2] = -20
    loader = torch.utils.data.DataLoader(ForecastSamples(samples("train"), config, "train"), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters()) if training else None
    with pytest.raises(FloatingPointError, match="loss masking must not hide failure"):
        _run_epoch(model, loader, config, optimizer)


def test_partial_nan_invalidates_entire_hand_or_object_without_imputation():
    batch = example_batch()
    batch["hand_seq"][0, -1, 0, 0, 1] = torch.nan
    batch["obj_seq"][0, -1, 1, 10] = torch.nan
    result = PoseForecastModel(small_config())(**batch)
    assert result["hand_future"][0, :, 0].isnan().all()
    assert result["object_future"][0, :, 1].isnan().all()
    assert not result["hand_valid"][0, :, 0].any()
    assert not result["object_valid"][0, :, 1].any()
    assert result["hand_future"][0, :, 1].isfinite().all()


def test_mixed_space_and_explicit_pixel_area_scaling():
    model = PoseForecastModel(small_config()).eval()
    observed = []
    handle = model.encoder.obj_mlp.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0].detach().clone()))
    model(**example_batch())
    handle.remove()
    assert model.encoder.coordinate_mode == "mixed"
    torch.testing.assert_close(observed[0][..., 9], torch.full((2, 4, 2), 0.1))
    torch.testing.assert_close(observed[0][..., :3], example_batch()["obj_seq"][..., :3])


def test_batch_future_labels_are_never_consumed_by_forward():
    model = PoseForecastModel(small_config()).eval()
    batch = example_batch()
    first = model(batch)
    batch.update(hand_future=torch.full((2, 2, 2, 21, 3), 1e10), object_future="must not be read")
    second = model(batch)
    for name in first:
        torch.testing.assert_close(first[name], second[name])


def test_masked_loss_partial_missing_backward_is_finite():
    model = PoseForecastModel(small_config())
    batch = example_batch()
    targets = expected_targets(batch)
    batch["hand_seq"][0, :, 0] = torch.nan
    batch["obj_seq"][0, :, 0] = torch.inf
    targets["hand_future"][1, 0, 1] = torch.nan
    targets["object_future"][1, 1, 1] = torch.nan
    targets["hand_future"] += 0.1
    targets["object_future"][..., :3, 3] += 0.1
    losses = masked_pose_loss(model(**batch), targets)
    assert losses["hand_count"] == 5
    assert losses["object_count"] == 5
    assert losses["loss"] > 0
    losses["loss"].backward()
    assert_finite_gradients(model)


@pytest.mark.parametrize("missing", ["observations", "targets", "explicit_masks"])
def test_all_missing_loss_is_differentiable_zero(missing):
    model = PoseForecastModel(small_config())
    batch = example_batch()
    targets = expected_targets(batch)
    if missing == "observations":
        batch["hand_seq"][:] = torch.nan
        batch["obj_seq"][:] = torch.nan
    elif missing == "targets":
        for target in targets.values():
            target[:] = torch.nan
    else:
        targets.update(hand_valid=torch.zeros(2, 2, 2, dtype=torch.bool),
                       object_valid=torch.zeros(2, 2, 2, dtype=torch.bool))
    loss = masked_pose_loss(model(**batch), targets)
    assert loss["hand_count"] == loss["object_count"] == 0
    assert loss["loss"].item() == 0
    loss["loss"].backward()
    assert_finite_gradients(model)
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters())


@pytest.mark.parametrize("config", [{"coordinate_mode": "camera"}, {"horizons_seconds": [1., 0.5]},
                                    {"horizons_seconds": [0.5, 0.5]}, {"horizons_seconds": [float("nan")]},
                                    {"d_model": 7}, {"object_area_scale": 0}, {"window_size": 0}])
def test_invalid_model_configs_rejected(config):
    with pytest.raises(ValueError):
        PoseForecastModel(small_config(**config))


@pytest.mark.parametrize("key", ["hand_seq", "obj_seq", "obj_mask"])
def test_invalid_input_shapes_rejected(key):
    batch = example_batch()
    batch[key] = batch[key][:, :-1]
    with pytest.raises(ValueError):
        PoseForecastModel(small_config())(**batch)


def test_mamba_factory_is_selected_without_cpu_gpu_execution(monkeypatch):
    calls = []

    def fake_mamba(*arguments):
        calls.append(arguments)
        return LSTMBackbone(arguments[0], arguments[1], arguments[-1])

    monkeypatch.setattr("src.forecasting.model.MambaBackbone", fake_mamba)
    model = PoseForecastModel(small_config(backbone="mamba"))
    assert calls == [(16, 1, 16, 4, 2, 0.)]
    assert model(**example_batch())["hand_future"].shape == (2, 2, 2, 21, 3)


def test_cpu_fit_reload_val_selection_and_fixed_scaling(tmp_path):
    config = small_config(epochs=4)
    train_samples, val_samples = samples("train"), samples("val")
    before = copy.deepcopy(train_samples)
    report = fit_forecaster(train_samples, val_samples, tmp_path / "fit", config)
    assert report["status"] == "completed"
    assert report["provenance"]["test_accessed"] is False
    assert report["provenance"]["evaluation_scope"] == "pseudo_label_consistency_not_independent_ground_truth"
    assert report["provenance"]["normalization"]["object_bbox_area_divisor_pixels_squared"] == 307200
    assert report["best_val_loss"] == min(entry["val"]["loss"] for entry in report["history"])
    assert report["history"][-1]["train"]["loss"] < report["history"][1]["train"]["loss"]
    assert report["best_val_loss"] < report["history"][0]["val"]["loss"]
    assert json.loads((tmp_path / "fit/training_report.json").read_text()) == report
    model, metadata = load_forecaster(report["checkpoint_path"])
    assert model.training is False
    assert next(model.parameters()).device.type == "cpu"
    assert metadata["epoch"] == report["best_epoch"]
    assert metadata["config"]["horizons_seconds"] == [0.5, 1.0]
    batch = example_batch()
    reloaded, _ = load_forecaster(report["checkpoint_path"])
    for key, prediction in model(**batch).items():
        torch.testing.assert_close(prediction, reloaded(**batch)[key], rtol=0, atol=0)
    for sample, original in zip(train_samples, before):
        np.testing.assert_array_equal(sample["obj_seq"], original["obj_seq"])


@pytest.mark.parametrize("field", ["sequence_id", "group_id"])
def test_training_rejects_sequence_or_acquisition_leakage(tmp_path, field):
    train_samples, val_samples = samples("train"), samples("val")
    val_samples[0][field] = train_samples[0][field]
    with pytest.raises(ValueError, match="overlap"):
        fit_forecaster(train_samples, val_samples, tmp_path / "fit", small_config())
    assert not (tmp_path / "fit").exists()


def test_training_rejects_test_samples_and_malformed_rotations(tmp_path):
    train_samples, val_samples = samples("train"), samples("val")
    val_samples[0]["split"] = "test"
    with pytest.raises(ValueError, match="test inputs"):
        fit_forecaster(train_samples, val_samples, tmp_path / "fit", small_config())
    del val_samples[0]["split"]
    val_samples[0]["object_future"][0, 0, 0, 0] = 10
    with pytest.raises(ValueError, match="rigid homogeneous"):
        fit_forecaster(train_samples, val_samples, tmp_path / "fit", small_config())


def test_training_rejects_all_missing_validation_and_existing_output(tmp_path):
    train_samples, val_samples = samples("train"), samples("val")
    for sample in val_samples:
        sample["hand_future"][:] = np.nan
        sample["object_future"][:] = np.nan
    with pytest.raises(ValueError, match="val has no valid"):
        fit_forecaster(train_samples, val_samples, tmp_path / "fit", small_config())
    (tmp_path / "existing").mkdir()
    (tmp_path / "existing/important.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        fit_forecaster(train_samples, samples("val"), tmp_path / "existing", small_config())
    assert (tmp_path / "existing/important.txt").read_text() == "preserve"


def test_cpu_fit_is_reproducible_without_global_rng_changes(tmp_path):
    torch.manual_seed(737)
    original_rng = torch.random.get_rng_state().clone()
    first = fit_forecaster(samples("train"), samples("val"), tmp_path / "one", small_config(epochs=1))
    torch.testing.assert_close(torch.random.get_rng_state(), original_rng)
    second = fit_forecaster(samples("train"), samples("val"), tmp_path / "two", small_config(epochs=1))
    assert first["history"] == second["history"]
    assert first["provenance"]["splits"] == second["provenance"]["splits"]


def test_random_initialization_is_not_a_trained_checkpoint(tmp_path):
    model = PoseForecastModel(small_config())
    checkpoint = tmp_path / "random.pt"
    torch.save({"contract_version": "pose_forecast_v1", "config": model.config,
                "model_state_dict": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="training provenance"):
        load_forecaster(checkpoint)


@pytest.fixture
def trained_checkpoint(tmp_path):
    fit_forecaster(samples("train"), samples("val"), tmp_path / "source", small_config(epochs=1))
    return torch.load(tmp_path / "source/last.pt", map_location="cpu", weights_only=True)


@pytest.mark.parametrize("key_path,value", [
    (("epoch",), -1), (("epoch",), 100), (("optimizer_steps",), 0),
    (("selection",), "chosen_using_test"), (("checkpoint_role",), "unknown"),
    (("model_status",), "validation_selected_untrained_persistence_initialization"),
    (("validation", "loss"), float("nan")), (("validation", "loss"), 100.),
    (("validation", "hand_count"), 10000), (("validation", "optimizer_steps"), 1),
    (("provenance", "test_accessed"), True), (("provenance", "source_sha256"), {}),
    (("provenance", "source_sha256", "src/forecasting/model.py"), "g" * 64),
    (("provenance", "normalization", "object_bbox_area_divisor_pixels_squared"), 1.),
    (("provenance", "splits", "val", "group_ids"), ["train-acquisition"]),
    (("provenance", "splits", "train", "sha256"), "missing"),
    (("provenance", "evaluation_scope"), "independent_ground_truth"),
    (("config",), {"d_model": 8}),
])
def test_checkpoint_metadata_tampering_is_rejected(tmp_path, trained_checkpoint, key_path, value):
    target = trained_checkpoint
    for key in key_path[:-1]:
        target = target[key]
    target[key_path[-1]] = value
    path = tmp_path / "invalid.pt"
    torch.save(trained_checkpoint, path)
    with pytest.raises(ValueError):
        load_forecaster(path)


def test_checkpoint_nonfinite_state_is_rejected(tmp_path, trained_checkpoint):
    trained_checkpoint["model_state_dict"]["decoder.hand_head.2.bias"][0] = torch.nan
    path = tmp_path / "invalid.pt"
    torch.save(trained_checkpoint, path)
    with pytest.raises(ValueError, match="finite tensors"):
        load_forecaster(path)


@pytest.mark.parametrize("buffer_name", ["object_scale", "decoder.identity_6d"])
def test_checkpoint_cannot_override_fixed_scaling_or_rotation_identity(tmp_path, trained_checkpoint, buffer_name):
    trained_checkpoint["model_state_dict"][buffer_name][0] = 0
    path = tmp_path / "invalid.pt"
    torch.save(trained_checkpoint, path)
    with pytest.raises(ValueError, match="geometry or normalization buffers"):
        load_forecaster(path)


def test_epoch_zero_selected_initialization_has_explicit_untrained_status(tmp_path):
    train_samples, val_samples = samples("train"), samples("val")
    for sample in train_samples + val_samples:
        batch = {name: torch.from_numpy(sample[name][None]) for name in ("hand_seq", "obj_seq", "obj_mask")}
        for name, target in expected_targets(batch).items():
            sample[name] = target[0].numpy()
    report = fit_forecaster(train_samples, val_samples, tmp_path / "fit", small_config(epochs=1))
    expected_status = "validation_selected_untrained_persistence_initialization"
    assert report["best_epoch"] == 0
    assert report["model_status"] == expected_status
    assert report["best_optimizer_steps"] == 0
    _, metadata = load_forecaster(report["checkpoint_path"])
    assert metadata["model_status"] == expected_status
    assert metadata["checkpoint_role"] == "best"
    _, last_metadata = load_forecaster(tmp_path / "fit/last.pt")
    assert last_metadata["model_status"] == "trained"
    assert last_metadata["optimizer_steps"] > 0
    assert last_metadata["selection"] == "last_training_epoch_not_validation_selected"
    checkpoint = torch.load(report["checkpoint_path"], weights_only=True)
    checkpoint["model_state_dict"]["decoder.hand_head.2.bias"][0] = 0.1
    torch.save(checkpoint, tmp_path / "not_persistence.pt")
    with pytest.raises(ValueError, match="zero-residual persistence"):
        load_forecaster(tmp_path / "not_persistence.pt")
