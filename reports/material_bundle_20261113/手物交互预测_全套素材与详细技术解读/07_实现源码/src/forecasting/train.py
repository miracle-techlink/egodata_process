"""Train/validation-only fitting and reloadable pose forecast checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.forecasting.model import DEFAULT_CONFIG, PoseForecastModel, masked_pose_loss, observed_masks, resolve_config


TRAIN_DEFAULTS = {
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "seed": 42,
    "device": "cpu",
    "num_workers": 0,
    "target_provenance": "future_estimator_pseudo_labels",
    "camera_assumption": "fixed_camera_uncompensated",
    "source_metadata": {},
}

ARRAY_KEYS = ("hand_seq", "obj_seq", "obj_mask", "hand_future", "object_future")
MASK_KEYS = ("hand_valid", "object_valid")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_config(config):
    resolved = dict(TRAIN_DEFAULTS)
    resolved.update(copy.deepcopy(config or {}))
    resolved = resolve_config(resolved)
    for name in ("epochs", "batch_size", "num_workers", "seed"):
        minimum = 1 if name in {"epochs", "batch_size"} else 0
        if isinstance(resolved[name], bool) or int(resolved[name]) != resolved[name] or resolved[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        resolved[name] = int(resolved[name])
    if resolved["num_workers"] != 0:
        raise ValueError("this in-memory trainer uses num_workers=0")
    for name in ("learning_rate", "weight_decay", "grad_clip"):
        resolved[name] = float(resolved[name])
        if not math.isfinite(resolved[name]) or resolved[name] < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if resolved["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    json.dumps(resolved, allow_nan=False)
    return resolved


class ForecastSamples(Dataset):
    """Eager validation, no augmentation, no learned normalization or test reads."""

    def __init__(self, samples, config, split):
        if not samples:
            raise ValueError(f"{split} samples must not be empty")
        self.samples = []
        self.sequence_ids = set()
        self.group_ids = set()
        self.target_counts = {"hand": 0, "object": 0}
        digest = hashlib.sha256()
        window_size, object_count = config["window_size"], config["n_max"]
        horizon_count = len(config["horizons_seconds"])
        shapes = {"hand_seq": (window_size, 2, 21, 3), "obj_seq": (window_size, object_count, 13),
                  "obj_mask": (object_count,), "hand_future": (horizon_count, 2, 21, 3),
                  "object_future": (horizon_count, object_count, 4, 4)}
        for sample in samples:
            sequence_id = sample.get("sequence_id")
            group_id = sample.get("group_id", sequence_id)
            if not isinstance(sequence_id, str) or not sequence_id or not isinstance(group_id, str) or not group_id:
                raise ValueError("each sample needs nonempty string sequence_id/group_id")
            if sample.get("split", split) != split:
                raise ValueError(f"sample split must be {split}; test inputs are forbidden")
            self.sequence_ids.add(sequence_id)
            self.group_ids.add(group_id)
            digest.update(json.dumps([sequence_id, group_id]).encode())
            tensors = {}
            for name, shape in shapes.items():
                array = np.asarray(sample[name])
                if array.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
                if name == "obj_mask":
                    if not np.isin(array, [0, 1]).all():
                        raise ValueError("obj_mask must contain boolean or 0/1 values")
                    array = np.array(array, dtype=np.bool_, copy=True)
                else:
                    array = np.array(array, dtype=np.float32, copy=True)
                    if np.isinf(array).any():
                        raise ValueError(f"{name} must use NaN, not infinity, for missing data")
                tensors[name] = torch.from_numpy(array)
                digest.update(name.encode())
                digest.update(array.tobytes())
            for name, shape in (("hand_valid", (horizon_count, 2)),
                                ("object_valid", (horizon_count, object_count))):
                mask = np.asarray(sample.get(name, np.ones(shape, dtype=bool)))
                if mask.shape != shape or not np.isin(mask, [0, 1]).all():
                    raise ValueError(f"{name} must be a boolean mask with shape {shape}")
                tensors[name] = torch.from_numpy(np.array(mask, dtype=np.bool_, copy=True))
                digest.update(tensors[name].numpy().tobytes())
            self._validate_target_transforms(tensors["object_future"])
            hand_current, object_current = observed_masks(
                tensors["hand_seq"][None], tensors["obj_seq"][None], tensors["obj_mask"][None]
            )
            for name, current, target in (("hand", hand_current[0, -1], tensors["hand_future"]),
                                          ("object", object_current[0, -1], tensors["object_future"])):
                valid = torch.isfinite(target).all(dim=(-1, -2)) & current[None] & tensors[name + "_valid"]
                self.target_counts[name] += int(valid.sum())
            self.samples.append(tensors)
        self.fingerprint = digest.hexdigest()

    @staticmethod
    def _validate_target_transforms(transforms):
        finite = torch.isfinite(transforms).all(dim=(-1, -2))
        rotations = transforms[finite][..., :3, :3]
        if not len(rotations):
            return
        identity = torch.eye(3, dtype=rotations.dtype).expand_as(rotations)
        bottom = torch.tensor([0., 0., 0., 1.]).expand(len(rotations), -1)
        if (not torch.allclose(rotations.transpose(-1, -2) @ rotations, identity, atol=2e-3)
                or not torch.allclose(torch.linalg.det(rotations), torch.ones(len(rotations)), atol=2e-3)
                or not torch.allclose(transforms[finite][..., 3, :], bottom, atol=1e-5)):
            raise ValueError("object_future finite targets must be rigid homogeneous SO(3) transforms")
        if (transforms[finite][..., 2, 3] <= 0).any():
            raise ValueError("object_future finite targets require positive camera Z")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def metadata(self):
        return {"sample_count": len(self), "sequence_ids": sorted(self.sequence_ids),
                "group_ids": sorted(self.group_ids), "sha256": self.fingerprint,
                "valid_target_counts": dict(self.target_counts)}


def _run_epoch(model, loader, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"hand_loss": 0., "translation_loss": 0., "rotation_loss": 0.}
    counts = {"hand_count": 0, "object_count": 0}
    skipped_batches = 0
    optimizer_steps = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {name: value.to(config["device"]) for name, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(batch["hand_seq"], batch["obj_seq"], batch["obj_mask"])
            hand_current, object_current = observed_masks(batch["hand_seq"], batch["obj_seq"], batch["obj_mask"])
            for name, observed in (("hand", hand_current), ("object", object_current)):
                finite = torch.isfinite(predictions[name + "_future"]).all(dim=(-1, -2))
                if (predictions[name + "_valid"] & ~finite).any():
                    raise FloatingPointError(f"nonfinite {name} forecast for a currently valid entity")
                supervised = observed[:, -1:, :] & batch[name + "_valid"]
                supervised = supervised & torch.isfinite(batch[name + "_future"]).all(dim=(-1, -2))
                if (supervised & ~predictions[name + "_valid"]).any():
                    raise FloatingPointError(f"invalid {name} forecast for a supervised entity; loss masking must not hide failure")
            losses = masked_pose_loss(predictions, batch, config)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("nonfinite forecast loss")
            has_targets = (losses["hand_count"] > 0 and config["hand_loss_weight"] > 0) or (
                losses["object_count"] > 0 and (config["translation_loss_weight"] > 0 or config["rotation_loss_weight"] > 0)
            )
            if training and has_targets:
                losses["loss"].backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                    raise FloatingPointError("nonfinite or missing forecast gradients")
                if config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
            elif not has_targets:
                skipped_batches += 1
            for name in totals:
                count_key = "hand_count" if name == "hand_loss" else "object_count"
                totals[name] += float(losses[name].detach()) * losses[count_key]
            for name in counts:
                counts[name] += losses[name]
    metrics = {name: total / max(1, counts["hand_count" if name == "hand_loss" else "object_count"])
               for name, total in totals.items()}
    metrics["loss"] = sum(metrics[name + "_loss"] * config[name + "_loss_weight"]
                          for name in ("hand", "translation", "rotation"))
    return dict(metrics, **counts, skipped_batches=skipped_batches, optimizer_steps=optimizer_steps)


def _checkpoint(model, config, provenance, epoch, validation, optimizer_steps=0, role="best"):
    model_status = "trained" if epoch else "validation_selected_untrained_persistence_initialization"
    selection = "minimum_validation_masked_loss_only" if role == "best" else "last_training_epoch_not_validation_selected"
    return {"contract_version": "pose_forecast_v1", "model_state_dict": {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }, "config": copy.deepcopy(config), "provenance": provenance, "epoch": epoch,
        "validation": validation, "selection": selection, "checkpoint_role": role,
        "model_status": model_status, "optimizer_steps": optimizer_steps}


def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"checkpoint {name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("contract_version") != "pose_forecast_v1":
        raise ValueError("checkpoint is not pose_forecast_v1")
    required = {"config", "provenance", "epoch", "validation", "selection", "checkpoint_role",
                "model_status", "optimizer_steps", "model_state_dict"}
    if required - checkpoint.keys():
        raise ValueError(f"checkpoint missing required training provenance fields: {sorted(required - checkpoint.keys())}")
    config = checkpoint["config"]
    if not isinstance(config, dict) or (set(DEFAULT_CONFIG) | set(TRAIN_DEFAULTS)) - config.keys():
        raise ValueError("checkpoint requires complete model and training config")
    resolved = _training_config(config)
    if resolved != config:
        raise ValueError("checkpoint config is not in canonical resolved form")
    epoch, optimizer_steps = checkpoint["epoch"], checkpoint["optimizer_steps"]
    _positive_integer(epoch, "epoch", allow_zero=True)
    _positive_integer(optimizer_steps, "optimizer_steps", allow_zero=True)
    if epoch > config["epochs"] or (epoch == 0) != (optimizer_steps == 0):
        raise ValueError("checkpoint epoch and actual optimizer steps are inconsistent")
    expected_status = "trained" if epoch else "validation_selected_untrained_persistence_initialization"
    if checkpoint["model_status"] != expected_status:
        raise ValueError("checkpoint model_status does not match training epoch")
    role = checkpoint["checkpoint_role"]
    expected_selection = {"best": "minimum_validation_masked_loss_only",
                          "last": "last_training_epoch_not_validation_selected"}
    if role not in expected_selection or checkpoint["selection"] != expected_selection[role]:
        raise ValueError("checkpoint selection policy is invalid")
    if role == "last" and epoch != config["epochs"]:
        raise ValueError("last checkpoint must correspond to the final training epoch")
    provenance = checkpoint["provenance"]
    if not isinstance(provenance, dict) or provenance.get("test_accessed") is not False:
        raise ValueError("checkpoint needs explicit train/val-only provenance")
    if provenance.get("evaluation_scope") != "pseudo_label_consistency_not_independent_ground_truth":
        raise ValueError("checkpoint must retain the pseudo-label consistency scope")
    source_hashes = provenance.get("source_sha256", {})
    required_sources = {"src/forecasting/model.py", "src/forecasting/train.py",
                        "src/encoding/encoder.py", "src/mamba/train.py"}
    if (not isinstance(source_hashes, dict) or required_sources - source_hashes.keys()
            or any(not _valid_digest(source_hashes[name]) for name in required_sources)):
        raise ValueError("checkpoint is missing valid source code SHA256 provenance")
    normalization = provenance.get("normalization", {})
    normalization_keys = {"hand_divisor": "hand_unit_scale", "object_position_divisor_m": "object_position_scale_m",
                          "object_velocity_divisor_mps": "object_velocity_scale_mps",
                          "object_bbox_area_divisor_pixels_squared": "object_area_scale"}
    if (not isinstance(normalization, dict)
            or normalization.get("strategy") != "explicit_fixed_unit_scaling_no_data_fit"
            or any(normalization.get(name) != config[config_name] for name, config_name in normalization_keys.items())):
        raise ValueError("checkpoint fixed-unit normalization provenance does not match config")
    losses = provenance.get("loss_definition", {})
    if not isinstance(losses, dict) or losses.get("cross_coordinate_geometry") is not False:
        raise ValueError("checkpoint must exclude cross-coordinate geometry")
    for name in ("hand", "translation", "rotation", "aggregation"):
        if not isinstance(losses.get(name), str) or not losses[name]:
            raise ValueError("checkpoint loss definition is incomplete")
    split_metadata = provenance.get("splits", {})
    if not isinstance(split_metadata, dict) or set(split_metadata) != {"train", "val"}:
        raise ValueError("checkpoint requires exactly train and val split provenance")
    for name, split in split_metadata.items():
        if not isinstance(split, dict):
            raise ValueError("checkpoint split metadata must be dictionaries")
        _positive_integer(split.get("sample_count"), name + " sample_count")
        if not _valid_digest(split.get("sha256")):
            raise ValueError("checkpoint sample SHA256 is missing")
        for identifiers in ("sequence_ids", "group_ids"):
            values = split.get(identifiers)
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or not value for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError("checkpoint split identifiers are invalid")
        counts = split.get("valid_target_counts", {})
        if not isinstance(counts, dict) or set(counts) != {"hand", "object"}:
            raise ValueError("checkpoint valid-target counts are missing")
        for kind, count in counts.items():
            _positive_integer(count, kind + " valid_target_count", allow_zero=True)
        if not sum(counts.values()):
            raise ValueError("checkpoint split must contain supervised targets")
    for identifiers in ("sequence_ids", "group_ids"):
        if set(split_metadata["train"][identifiers]) & set(split_metadata["val"][identifiers]):
            raise ValueError("checkpoint train/val provenance contains split leakage")
    validation = checkpoint["validation"]
    if not isinstance(validation, dict):
        raise ValueError("checkpoint validation metrics are missing")
    for name in ("loss", "hand_loss", "translation_loss", "rotation_loss"):
        value = validation.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("checkpoint validation losses must be nonnegative and finite")
    for name in ("hand_count", "object_count", "skipped_batches", "optimizer_steps"):
        _positive_integer(validation.get(name), "validation " + name, allow_zero=True)
    if validation["optimizer_steps"] != 0:
        raise ValueError("validation must not update model parameters")
    for name in ("hand", "object"):
        if validation[name + "_count"] != split_metadata["val"]["valid_target_counts"][name]:
            raise ValueError("checkpoint validation coverage does not match recorded targets")
    expected_loss = sum(validation[name + "_loss"] * config[name + "_loss_weight"]
                        for name in ("hand", "translation", "rotation"))
    if not math.isclose(validation["loss"], expected_loss, abs_tol=1e-8, rel_tol=1e-6):
        raise ValueError("checkpoint validation total does not match configured weights")
    state = checkpoint["model_state_dict"]
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values())):
        raise ValueError("checkpoint model state must contain only finite tensors")
    object_scale = [config["object_position_scale_m"]] * 3 + [1.] * 6
    object_scale += [config["object_area_scale"]] + [config["object_velocity_scale_mps"]] * 3
    expected_buffers = {"object_scale": torch.tensor(object_scale),
                        "decoder.identity_6d": torch.tensor([1., 0., 0., 0., 1., 0.])}
    for name, expected in expected_buffers.items():
        if name not in state or not torch.equal(state[name], expected):
            raise ValueError("checkpoint geometry or normalization buffers do not match the config")
    if epoch == 0:
        for head in ("hand_head", "object_head"):
            for kind in ("weight", "bias"):
                name = f"decoder.{head}.2.{kind}"
                if name not in state or torch.count_nonzero(state[name]):
                    raise ValueError("epoch-zero checkpoint must be the zero-residual persistence initialization")


def load_forecaster(checkpoint_path, device="cpu"):
    """Return (eval-mode model, checkpoint metadata); never consume future labels."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint)
    model = PoseForecastModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    metadata = {name: value for name, value in checkpoint.items() if name != "model_state_dict"}
    return model, metadata


def fit_forecaster(train_samples, val_samples, out_dir, config=None):
    """Fit on train, select on val; inputs from overlapping acquisitions are rejected."""
    resolved = _training_config(config)
    train_data = ForecastSamples(train_samples, resolved, "train")
    val_data = ForecastSamples(val_samples, resolved, "val")
    if train_data.sequence_ids & val_data.sequence_ids or train_data.group_ids & val_data.group_ids:
        raise ValueError("train/val sequence or acquisition group overlap; split leakage forbidden")
    for split, dataset in (("train", train_data), ("val", val_data)):
        useful_hands = dataset.target_counts["hand"] and resolved["hand_loss_weight"] > 0
        useful_objects = dataset.target_counts["object"] and (
            resolved["translation_loss_weight"] > 0 or resolved["rotation_loss_weight"] > 0
        )
        if not useful_hands and not useful_objects:
            raise ValueError(f"{split} has no valid forecast targets with positive loss weight")
    destination = Path(out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("out_dir must be new or empty; previous checkpoints are not overwritten")
    device = torch.device(resolved["device"])
    if resolved["backbone"] == "mamba" and device.type != "cuda":
        raise ValueError("Mamba training requires explicit CUDA device; use lstm for CPU validation")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    destination.mkdir(parents=True, exist_ok=True)
    module_root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), Path(__file__).resolve().with_name("model.py"),
                    module_root / "encoding/encoder.py", module_root / "mamba/train.py"]
    provenance = {
        "evaluation_scope": "pseudo_label_consistency_not_independent_ground_truth",
        "normalization": {"strategy": "explicit_fixed_unit_scaling_no_data_fit", "hand_divisor": resolved["hand_unit_scale"],
                          "object_position_divisor_m": resolved["object_position_scale_m"],
                          "object_velocity_divisor_mps": resolved["object_velocity_scale_mps"],
                          "object_bbox_area_divisor_pixels_squared": resolved["object_area_scale"]},
        "loss_definition": {"hand": "mean SmoothL1 of scaled 21x3 joints per valid hand",
                            "translation": "mean SmoothL1 of scaled camera-frame translation",
                            "rotation": "mean squared Frobenius rotation difference / 8; no symmetry adjustment",
                            "aggregation": "valid-entity/horizon weighted within each component",
                            "cross_coordinate_geometry": False},
        "missing_policy": {"partial_nonfinite": "invalidate_entire_entity; never impute from old observations",
                           "object_camera_depth": "current_or_predicted_z_le_zero_is_missing",
                           "supervised_invalid_prediction": "raise; never reduce loss by dropping valid targets"},
        "splits": {"train": train_data.metadata(), "val": val_data.metadata()},
        "test_accessed": False,
        "source_sha256": {str(path.relative_to(module_root.parent)): _file_hash(path) for path in source_files},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": str(np.__version__), "device": str(device)},
    }
    (destination / "config.json").write_text(json.dumps(resolved, indent=2, allow_nan=False) + "\n")
    generator = torch.Generator().manual_seed(resolved["seed"])
    train_loader = DataLoader(train_data, batch_size=resolved["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=resolved["batch_size"], shuffle=False)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(resolved["seed"])
        model = PoseForecastModel(resolved).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=resolved["learning_rate"], weight_decay=resolved["weight_decay"])
        validation = _run_epoch(model, val_loader, resolved)
        best_loss, best_epoch = validation["loss"], 0
        history = [{"epoch": 0, "train": None, "val": validation}]
        torch.save(_checkpoint(model, resolved, provenance, 0, validation), destination / "best.pt")
        optimizer_steps = 0
        for epoch in range(1, resolved["epochs"] + 1):
            training = _run_epoch(model, train_loader, resolved, optimizer)
            optimizer_steps += training["optimizer_steps"]
            validation = _run_epoch(model, val_loader, resolved)
            history.append({"epoch": epoch, "train": training, "val": validation})
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                torch.save(_checkpoint(model, resolved, provenance, epoch, validation, optimizer_steps), destination / "best.pt")
        torch.save(_checkpoint(model, resolved, provenance, resolved["epochs"], validation, optimizer_steps, "last"), destination / "last.pt")
        reloaded, metadata = load_forecaster(destination / "best.pt", device)
        reloaded_validation = _run_epoch(reloaded, val_loader, resolved)
        if not math.isclose(reloaded_validation["loss"], best_loss, rel_tol=1e-5, abs_tol=1e-8):
            raise RuntimeError("reloaded best checkpoint does not reproduce validation loss")
    report = {"status": "completed", "contract_version": "pose_forecast_v1", "config": resolved,
              "provenance": provenance, "history": history, "best_epoch": best_epoch,
              "best_val_loss": best_loss, "reloaded_best_val": reloaded_validation,
              "model_status": metadata["model_status"], "best_optimizer_steps": metadata["optimizer_steps"],
              "selection": metadata["selection"], "checkpoint_path": str((destination / "best.pt").resolve()),
              "checkpoint_sha256": _file_hash(destination / "best.pt"),
              "last_checkpoint_sha256": _file_hash(destination / "last.pt"),
              "report_path": str((destination / "training_report.json").resolve())}
    (destination / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
