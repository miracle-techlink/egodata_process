"""Causal hand/object residual forecasting under the pose_forecast_v1 contract."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional

from src.encoding.encoder import HOIEncoder
from src.mamba.train import LSTMBackbone, MambaBackbone


DEFAULT_CONFIG = {
    "contract_version": "pose_forecast_v1",
    "d_model": 32,
    "hidden_dim": 64,
    "n_max": 1,
    "window_size": 32,
    "horizons_seconds": [0.5, 1.0],
    "coordinate_mode": "mixed",
    "backbone": "lstm",
    "encoder_layers": 1,
    "backbone_layers": 1,
    "n_heads": 4,
    "dim_ff": 64,
    "dropout": 0.0,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "contact_threshold": 0.15,
    "hand_unit_scale": 1.0,
    "object_position_scale_m": 1.0,
    "object_velocity_scale_mps": 1.0,
    "object_area_scale": 307200.0,
    "hand_residual_scale": 0.1,
    "translation_residual_scale_m": 0.1,
    "rotation_residual_scale": 1.0,
    "hand_loss_weight": 1.0,
    "translation_loss_weight": 1.0,
    "rotation_loss_weight": 1.0,
}


def resolve_config(config=None):
    resolved = copy.deepcopy(DEFAULT_CONFIG)
    resolved.update(copy.deepcopy(config or {}))
    if resolved["contract_version"] != "pose_forecast_v1":
        raise ValueError("unsupported contract_version")
    if resolved["coordinate_mode"] != "mixed":
        raise ValueError("pose_forecast_v1 only supports mixed coordinates; no cross-space geometry")
    if resolved["backbone"] not in {"lstm", "mamba"}:
        raise ValueError("backbone must be lstm or mamba")
    for name in ("d_model", "hidden_dim", "n_max", "window_size", "encoder_layers",
                 "backbone_layers", "n_heads", "dim_ff", "d_state", "d_conv", "expand"):
        if isinstance(resolved[name], bool) or int(resolved[name]) != resolved[name] or resolved[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
        resolved[name] = int(resolved[name])
    if resolved["d_model"] % resolved["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    horizons = [float(value) for value in resolved["horizons_seconds"]]
    if not horizons or any(not math.isfinite(value) or value <= 0 for value in horizons):
        raise ValueError("horizons_seconds must be positive and finite")
    if horizons != sorted(set(horizons)):
        raise ValueError("horizons_seconds must be strictly increasing")
    resolved["horizons_seconds"] = horizons
    for name in ("hand_unit_scale", "object_position_scale_m", "object_velocity_scale_mps",
                 "object_area_scale", "hand_residual_scale", "translation_residual_scale_m",
                 "rotation_residual_scale", "contact_threshold"):
        resolved[name] = float(resolved[name])
        if not math.isfinite(resolved[name]) or resolved[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("hand_loss_weight", "translation_loss_weight", "rotation_loss_weight"):
        resolved[name] = float(resolved[name])
        if not math.isfinite(resolved[name]) or resolved[name] < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if not any(resolved[name] for name in ("hand_loss_weight", "translation_loss_weight", "rotation_loss_weight")):
        raise ValueError("at least one loss weight must be positive")
    if not 0 <= resolved["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    return resolved


def rotation_6d_to_matrix(rotation, epsilon=1e-6):
    """First-two-column convention, with finite SO(3) fallbacks at degeneracies."""
    if rotation.shape[-1] != 6 or not rotation.is_floating_point():
        raise ValueError("rotation must be floating point with final dimension 6")
    clean = torch.where(torch.isfinite(rotation), rotation, torch.zeros_like(rotation))
    first, second = clean[..., :3], clean[..., 3:]
    first = first / first.abs().amax(dim=-1, keepdim=True).clamp_min(1)
    second = second / second.abs().amax(dim=-1, keepdim=True).clamp_min(1)
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    identity_first = torch.zeros_like(first)
    identity_first[..., 0] = 1
    first = torch.where(first_norm > epsilon, first / first_norm.clamp_min(epsilon), identity_first)
    third = torch.linalg.cross(first, second, dim=-1)
    third_norm = torch.linalg.vector_norm(third, dim=-1, keepdim=True)
    fallback = functional.one_hot(first.abs().argmin(dim=-1), num_classes=3).to(first.dtype)
    fallback = torch.linalg.cross(first, fallback, dim=-1)
    fallback = fallback / torch.linalg.vector_norm(fallback, dim=-1, keepdim=True).clamp_min(epsilon)
    third = torch.where(third_norm > epsilon, third / third_norm.clamp_min(epsilon), fallback)
    second = torch.linalg.cross(third, first, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def observed_masks(hand_seq, obj_seq, obj_mask):
    """Any nonfinite value invalidates an entity; object poses also require positive camera Z."""
    hand_valid = torch.isfinite(hand_seq).all(dim=(-1, -2))
    clean_objects = torch.where(torch.isfinite(obj_seq), obj_seq, torch.zeros_like(obj_seq))
    first, second = clean_objects[..., 3:6], clean_objects[..., 6:9]
    rotation_valid = (torch.linalg.vector_norm(first, dim=-1) > 1e-6) & (
        torch.linalg.vector_norm(torch.linalg.cross(first, second, dim=-1), dim=-1) > 1e-6
    )
    object_valid = torch.isfinite(obj_seq).all(dim=-1) & rotation_valid & obj_mask[:, None, :].bool()
    object_valid = object_valid & (obj_seq[..., 2] > 0)
    return hand_valid, object_valid


class PoseDecoder(nn.Module):
    """Zero-initialized hold-pose baseline plus learned future residuals."""

    def __init__(self, config=None):
        super().__init__()
        self.config = resolve_config(config)
        self.horizon_count = len(self.config["horizons_seconds"])
        hidden_dim = self.config["hidden_dim"]
        self.hand_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, self.horizon_count * 63))
        self.object_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, self.horizon_count * 9))
        for head in (self.hand_head, self.object_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer("identity_6d", torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, features, current_hands, current_objects, hand_valid, object_valid):
        batch_size = features.shape[0]
        hand_delta = self.hand_head(features[:, 1:3]).reshape(batch_size, 2, self.horizon_count, 21, 3)
        hand_delta = hand_delta.permute(0, 2, 1, 3, 4) * self.config["hand_residual_scale"]
        safe_hands = current_hands.masked_fill(~hand_valid[..., None, None], 0)
        hand_future = safe_hands[:, None] + hand_delta
        object_delta = self.object_head(features[:, 3:]).reshape(
            batch_size, self.config["n_max"], self.horizon_count, 9
        ).permute(0, 2, 1, 3)
        safe_objects = current_objects.masked_fill(~object_valid[..., None], 0)
        current_rotation = rotation_6d_to_matrix(safe_objects[..., 3:9])
        rotation_delta = rotation_6d_to_matrix(
            self.identity_6d + object_delta[..., 3:] * self.config["rotation_residual_scale"]
        )
        future_rotation = rotation_delta @ current_rotation[:, None]
        future_translation = safe_objects[:, None, :, :3] + (
            object_delta[..., :3] * self.config["translation_residual_scale_m"]
        )
        upper = torch.cat((future_rotation, future_translation[..., None]), dim=-1)
        bottom = torch.zeros_like(upper[..., :1, :])
        bottom[..., 0, 3] = 1
        object_future = torch.cat((upper, bottom), dim=-2)
        hand_valid = hand_valid[:, None].expand(-1, self.horizon_count, -1)
        object_valid = object_valid[:, None].expand(-1, self.horizon_count, -1)
        object_valid = object_valid & (future_translation[..., 2] > 0)
        return {
            "hand_future": hand_future.masked_fill(~hand_valid[..., None, None], torch.nan),
            "object_future": object_future.masked_fill(~object_valid[..., None, None], torch.nan),
            "hand_valid": hand_valid,
            "object_valid": object_valid,
        }


class PoseForecastModel(nn.Module):
    """HOIEncoder -> per-entity LSTM/Mamba -> independent geometric PoseDecoder."""

    def __init__(self, config=None):
        super().__init__()
        self.config = resolve_config(config)
        encoder_config = dict(self.config, n_layers=self.config["encoder_layers"],
                              max_frames=self.config["window_size"], use_gaze=False)
        self.encoder = HOIEncoder(encoder_config)
        self.temporal_projection = nn.Linear(self.config["d_model"], self.config["hidden_dim"])
        if self.config["backbone"] == "lstm":
            self.backbone = LSTMBackbone(self.config["hidden_dim"], self.config["backbone_layers"],
                                         self.config["dropout"])
        else:
            self.backbone = MambaBackbone(self.config["hidden_dim"], self.config["backbone_layers"],
                                          self.config["d_state"], self.config["d_conv"],
                                          self.config["expand"], self.config["dropout"])
        self.decoder = PoseDecoder(self.config)
        object_scale = [self.config["object_position_scale_m"]] * 3 + [1.] * 6
        object_scale += [self.config["object_area_scale"]] + [self.config["object_velocity_scale_mps"]] * 3
        self.register_buffer("object_scale", torch.tensor(object_scale))

    def forward(self, hand_seq, obj_seq=None, obj_mask=None):
        if isinstance(hand_seq, dict):
            batch = hand_seq
            hand_seq, obj_seq, obj_mask = batch["hand_seq"], batch["obj_seq"], batch["obj_mask"]
        if hand_seq.ndim != 5 or hand_seq.shape[1:] != (self.config["window_size"], 2, 21, 3):
            raise ValueError("hand_seq must have configured shape (B,W,2,21,3)")
        batch_size, window_size = hand_seq.shape[:2]
        if obj_seq is None or obj_seq.shape != (batch_size, window_size, self.config["n_max"], 13):
            raise ValueError("obj_seq must have configured shape (B,W,N,13)")
        if obj_mask is None or obj_mask.shape != (batch_size, self.config["n_max"]):
            raise ValueError("obj_mask must have shape (B,N)")
        if not hand_seq.is_floating_point() or not obj_seq.is_floating_point():
            raise ValueError("pose inputs must be floating point")
        if not torch.all((obj_mask == 0) | (obj_mask == 1)):
            raise ValueError("obj_mask must contain only boolean or 0/1 values")
        hand_observed, object_observed = observed_masks(hand_seq, obj_seq, obj_mask)
        clean_hands = hand_seq.masked_fill(~hand_observed[..., None, None], torch.nan)
        clean_objects = obj_seq.masked_fill(~object_observed[..., None], torch.nan)
        tokens = self.encoder(clean_hands / self.config["hand_unit_scale"], clean_objects / self.object_scale,
                              obj_mask=obj_mask[:, None].expand(-1, window_size, -1))
        tokens = self.temporal_projection(tokens)
        entity_count, hidden_dim = tokens.shape[2:]
        trajectories = tokens.permute(0, 2, 1, 3).reshape(batch_size * entity_count, window_size, hidden_dim)
        features = self.backbone(trajectories)[:, -1].reshape(batch_size, entity_count, hidden_dim)
        return self.decoder(features, hand_seq[:, -1], obj_seq[:, -1],
                            hand_observed[:, -1], object_observed[:, -1])


def masked_pose_loss(predictions, targets, config=None):
    """Mask before arithmetic; rotation loss is squared chordal distance / 8, not degrees."""
    resolved = resolve_config(config)
    predicted_hands, predicted_objects = predictions["hand_future"], predictions["object_future"]
    target_hands, target_objects = targets["hand_future"], targets["object_future"]
    if predicted_hands.shape != target_hands.shape or predicted_objects.shape != target_objects.shape:
        raise ValueError("forecast target shapes must match predictions")
    hand_valid = torch.isfinite(predicted_hands).all(dim=(-1, -2)) & torch.isfinite(target_hands).all(dim=(-1, -2))
    object_valid = torch.isfinite(predicted_objects).all(dim=(-1, -2)) & torch.isfinite(target_objects).all(dim=(-1, -2))
    for payload in (predictions, targets):
        for key, valid in (("hand_valid", hand_valid), ("object_valid", object_valid)):
            if key in payload:
                if payload[key].shape != valid.shape:
                    raise ValueError(f"{key} must match entity/horizon dimensions")
                if not torch.all((payload[key] == 0) | (payload[key] == 1)):
                    raise ValueError(f"{key} must contain boolean or 0/1 values")
                valid &= payload[key].bool()
    valid_hands, valid_hand_targets = predicted_hands[hand_valid], target_hands[hand_valid]
    valid_objects, valid_object_targets = predicted_objects[object_valid], target_objects[object_valid]
    zero = valid_hands.sum() * 0 + valid_objects.sum() * 0
    hand_count, object_count = int(hand_valid.sum()), int(object_valid.sum())
    hand_loss = functional.smooth_l1_loss(valid_hands / resolved["hand_unit_scale"],
                                         valid_hand_targets / resolved["hand_unit_scale"]) if hand_count else zero
    translation_loss, rotation_loss = zero, zero
    if object_count:
        translation_loss = functional.smooth_l1_loss(
            valid_objects[..., :3, 3] / resolved["object_position_scale_m"],
            valid_object_targets[..., :3, 3] / resolved["object_position_scale_m"]
        )
        rotation_loss = (valid_objects[..., :3, :3] - valid_object_targets[..., :3, :3]).square().sum(dim=(-1, -2)).mean() / 8
    total = hand_loss * resolved["hand_loss_weight"] + translation_loss * resolved["translation_loss_weight"]
    total = total + rotation_loss * resolved["rotation_loss_weight"]
    return {"loss": total, "hand_loss": hand_loss, "translation_loss": translation_loss,
            "rotation_loss": rotation_loss, "hand_count": hand_count, "object_count": object_count}
