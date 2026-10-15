"""Interaction encoder (task 5, arc).

Per-frame entity tokenization for ego HOI samples (contract: .agent/memory/03_data_pipeline.md sec.4/5).

Entity order per frame: [person, left_hand, right_hand, obj_0, ..., obj_{n_max-1}].
Token = self-state MLP + aggregated pairwise-relation MLP + entity-type embedding + time embedding,
then n_layers of masked self-attention across entities within each frame (no temporal mixing;
the time dim is preserved for the downstream Mamba in task 6).

Inputs:
    hand_seq : (B, W, 2, 21, 3) float   left/right hand keypoints; NaN = not detected that frame
    obj_seq  : (B, W, n_max, 13) float  [pos3, rot6d, bbox_area, vel3]; NaN = not detected that frame
    obj_mask : (B, W, n_max) bool/float, 1 = slot holds a real object, 0 = padding.
               Optional; when None, a slot is inferred valid iff observed in >= 1 frame of the window.
    gaze_seq : (B, W, gaze_dim) optional, required iff cfg.use_gaze is True.

Output: tokens (B, W, 3 + n_max, d_model); padding-object tokens are exactly zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn

HAND_KEYPOINTS = 21
HAND_DIM = HAND_KEYPOINTS * 3
OBJ_DIM = 13
REL_DIM = 8  # dist(1) + rel_pos(3) + rel_vel(3) + contact(1)
N_FIXED_ENTITIES = 3  # person, left_hand, right_hand


def cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
    )


class HOIEncoder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.d_model = int(cfg_get(cfg, "d_model", 128))
        self.n_max = int(cfg_get(cfg, "n_max", 8))
        self.n_heads = int(cfg_get(cfg, "n_heads", 4))
        self.n_layers = int(cfg_get(cfg, "n_layers", 2))
        self.dim_ff = int(cfg_get(cfg, "dim_ff", 256))
        self.dropout_p = float(cfg_get(cfg, "dropout", 0.1))
        self.max_frames = int(cfg_get(cfg, "max_frames", 64))
        self.contact_threshold = float(cfg_get(cfg, "contact_threshold", 0.15))
        self.use_gaze = bool(cfg_get(cfg, "use_gaze", False))
        self.gaze_dim = int(cfg_get(cfg, "gaze_dim", 3))
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        self.n_entities = N_FIXED_ENTITIES + self.n_max

        person_in = 2 * 3 + 2 + (self.gaze_dim if self.use_gaze else 0)  # both wrists + obs flags [+ gaze]
        hand_in = HAND_DIM + 1  # flattened keypoints + obs flag
        obj_in = OBJ_DIM + 2  # features + obs flag + slot-valid flag
        self.person_mlp = _mlp(person_in, self.d_model, self.d_model, self.dropout_p)
        self.hand_mlp = _mlp(hand_in, self.d_model, self.d_model, self.dropout_p)
        self.obj_mlp = _mlp(obj_in, self.d_model, self.d_model, self.dropout_p)
        self.rel_mlp = _mlp(REL_DIM, max(self.d_model // 2, 16), self.d_model, self.dropout_p)

        self.type_emb = nn.Parameter(torch.zeros(4, self.d_model))  # person / left / right / object
        self.time_emb = nn.Embedding(self.max_frames, self.d_model)
        self.in_norm = nn.LayerNorm(self.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_ff,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation = nn.TransformerEncoder(layer, num_layers=self.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(self.d_model)

        nn.init.trunc_normal_(self.type_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb.weight, std=0.02)

    def forward(self, hand_seq, obj_seq, obj_mask=None, gaze_seq=None):
        if hand_seq.dim() != 5 or hand_seq.shape[2:] != (2, HAND_KEYPOINTS, 3):
            raise ValueError(f"hand_seq must be (B, W, 2, {HAND_KEYPOINTS}, 3), got {tuple(hand_seq.shape)}")
        if obj_seq.dim() != 4 or obj_seq.shape[-1] != OBJ_DIM:
            raise ValueError(f"obj_seq must be (B, W, n_max, {OBJ_DIM}), got {tuple(obj_seq.shape)}")
        B, W = hand_seq.shape[0], hand_seq.shape[1]
        n_obj = obj_seq.shape[2]
        if n_obj != self.n_max:
            raise ValueError(f"obj_seq has n_max={n_obj}, encoder was built with n_max={self.n_max}")
        if W > self.max_frames:
            raise ValueError(f"W={W} exceeds max_frames={self.max_frames}; increase cfg.max_frames")

        # NaN -> 0, observed flag appended (contract: undetected frames are NaN)
        hand_obs = ~torch.isnan(hand_seq).any(dim=(-1, -2))  # (B, W, 2)
        hands = torch.nan_to_num(hand_seq, nan=0.0)
        obj_obs = ~torch.isnan(obj_seq).any(dim=-1)  # (B, W, n_max)
        objs = torch.nan_to_num(obj_seq, nan=0.0)

        if obj_mask is None:
            slot_valid = obj_obs.any(dim=1, keepdim=True).expand(B, W, n_obj)
        else:
            if obj_mask.shape != (B, W, n_obj):
                raise ValueError(f"obj_mask must be (B, W, n_max), got {tuple(obj_mask.shape)}")
            slot_valid = obj_mask.bool()
        obj_valid = slot_valid & obj_obs

        # --- self-state tokens ---
        hand_feat = torch.cat([hands.flatten(3), hand_obs[..., None].float()], dim=-1)
        hand_tok = self.hand_mlp(hand_feat)  # (B, W, 2, D)

        wrists = hands[..., 0, :]  # (B, W, 2, 3) keypoint 0 = wrist
        person_parts = [wrists.flatten(2), hand_obs.float()]
        if self.use_gaze:
            if gaze_seq is None:
                raise ValueError("cfg.use_gaze is True but gaze_seq was not provided")
            person_parts.append(torch.nan_to_num(gaze_seq, nan=0.0))
        person_tok = self.person_mlp(torch.cat(person_parts, dim=-1))  # (B, W, D)

        obj_feat = torch.cat([objs, obj_obs[..., None].float(), slot_valid[..., None].float()], dim=-1)
        obj_tok = self.obj_mlp(obj_feat)  # (B, W, n_max, D)

        self_tok = torch.cat([person_tok.unsqueeze(2), hand_tok, obj_tok], dim=2)  # (B, W, E, D)
        E = self.n_entities

        # --- pairwise relation features (per frame) ---
        wflag = hand_obs[..., None].float()  # (B, W, 2, 1)
        person_pos = (wrists * wflag).sum(dim=2) / wflag.sum(dim=2).clamp(min=1.0)  # (B, W, 3)
        person_valid = hand_obs.any(dim=-1)  # (B, W)
        pos = torch.cat([person_pos.unsqueeze(2), wrists, objs[..., :3]], dim=2)  # (B, W, E, 3)
        usable = torch.cat([person_valid.unsqueeze(-1), hand_obs, obj_valid], dim=-1)  # (B, W, E)

        vel = torch.diff(pos, dim=1, prepend=pos[:, :1])  # (B, W, E, 3), vel[:, 0] = 0

        rel_pos = pos[:, :, None, :, :] - pos[:, :, :, None, :]  # (B, W, E_i, E_j, 3) = pos_j - pos_i
        rel_vel = vel[:, :, None, :, :] - vel[:, :, :, None, :]
        dist = rel_pos.norm(dim=-1, keepdim=True)  # (B, W, E, E, 1)
        pair_ok = usable[:, :, :, None] & usable[:, :, None, :]  # (B, W, E, E)
        eye = torch.eye(E, dtype=torch.bool, device=pos.device)
        pair_ok = pair_ok & ~eye
        contact = ((dist.squeeze(-1) < self.contact_threshold) & pair_ok)[..., None].float()
        pair_feat = torch.cat([dist, rel_pos, rel_vel, contact], dim=-1) * pair_ok[..., None].float()
        rel_agg = pair_feat.sum(dim=3) / pair_ok.sum(dim=-1, keepdim=True).clamp(min=1).float()
        rel_tok = self.rel_mlp(rel_agg)  # (B, W, E, D)

        # --- assemble tokens ---
        type_ids = torch.tensor([0, 1, 2] + [3] * self.n_max, device=pos.device)
        h = self_tok + rel_tok + self.type_emb[type_ids]
        h = h + self.time_emb.weight[:W][None, :, None, :]
        h = self.in_norm(h)

        # --- relation layers: masked self-attention across entities within each frame ---
        slot = torch.cat(
            [torch.ones(B, W, N_FIXED_ENTITIES, dtype=torch.bool, device=pos.device), slot_valid], dim=-1
        )  # (B, W, E)
        h = h.reshape(B * W, E, self.d_model)
        h = self.relation(h, src_key_padding_mask=~slot.reshape(B * W, E))
        h = self.out_norm(h.reshape(B, W, E, self.d_model))
        return h * slot[..., None].float()


def build_encoder(cfg=None) -> nn.Module:
    return HOIEncoder(cfg)
