"""Interaction decoder (task 5, arc).

Three-way prediction head for the next-moment interaction (contract: 03_data_pipeline.md sec.5).

Input:
    tokens   : (B, W, E, d_model) from the encoder (or a temporally-modeled variant from task 6;
               (B, E, d_model) and (B, d_model) are also accepted). When W is present the
               last time step is used.
    obj_mask : (B, W, n_max) or (B, n_max) bool/float, same convention as the encoder.
               Optional; padding-object logits are set to -inf when provided.

Output dict:
    subject          : (B, 3)          index 0=left, 1=right, 2=none
    object           : (B, n_max + 1)  index 0..n_max-1 = object slots, index n_max = none
    interaction_type : (B, 7)          index order: none, reach, grasp, hold, move, place, release
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import N_FIXED_ENTITIES, cfg_get

SUBJECT_CLASSES = ("left", "right", "none")
INTERACTION_TYPES = ("none", "reach", "grasp", "hold", "move", "place", "release")
OBJECT_NONE_INDEX_OFFSET = 0  # "none" is the last object logit: index n_max


class HOIDecoder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.d_model = int(cfg_get(cfg, "d_model", 128))
        self.n_max = int(cfg_get(cfg, "n_max", 8))
        self.n_subject = int(cfg_get(cfg, "n_subject", len(SUBJECT_CLASSES)))
        self.n_types = int(cfg_get(cfg, "n_interaction_types", len(INTERACTION_TYPES)))
        dropout = float(cfg_get(cfg, "dropout", 0.1))

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.subject_head = nn.Linear(self.d_model, self.n_subject)
        self.object_head = nn.Linear(self.d_model, self.n_max + 1)
        self.type_head = nn.Linear(self.d_model, self.n_types)

    def _pool_entities(self, h: torch.Tensor, obj_mask: torch.Tensor | None) -> torch.Tensor:
        # h: (B, E, D); masked mean over valid entities (fixed entities always valid)
        B, E, _ = h.shape
        device, dtype = h.device, h.dtype
        fixed = torch.ones(B, N_FIXED_ENTITIES, dtype=dtype, device=device)
        if obj_mask is not None:
            n_obj = E - N_FIXED_ENTITIES
            if obj_mask.shape != (B, n_obj):
                raise ValueError(f"obj_mask must be (B, {n_obj}), got {tuple(obj_mask.shape)}")
            em = torch.cat([fixed, obj_mask.to(dtype)], dim=-1)
        else:
            em = torch.ones(B, E, dtype=dtype, device=device)
        return (h * em[..., None]).sum(dim=1) / em.sum(dim=1, keepdim=True).clamp(min=1.0)

    def forward(self, tokens: torch.Tensor, obj_mask: torch.Tensor | None = None):
        if tokens.dim() == 4:
            h = tokens[:, -1]  # last time step
            if obj_mask is not None and obj_mask.dim() == 3:
                obj_mask = obj_mask[:, -1]
        else:
            h = tokens
            if obj_mask is not None and obj_mask.dim() == 3:
                obj_mask = obj_mask[:, -1]
        if h.dim() == 3:
            h = self._pool_entities(h, obj_mask)
        if h.dim() != 2 or h.shape[-1] != self.d_model:
            raise ValueError(f"expected (B, d_model={self.d_model}) after pooling, got {tuple(h.shape)}")

        h = self.dropout(self.norm(h))
        object_logits = self.object_head(h)  # (B, n_max + 1), last index = none
        if obj_mask is not None:
            pad = ~obj_mask.bool()
            object_logits = torch.cat(
                [object_logits[:, : self.n_max].masked_fill(pad, float("-inf")),
                 object_logits[:, self.n_max :]],
                dim=-1,
            )
        return {
            "subject": self.subject_head(h),
            "object": object_logits,
            "interaction_type": self.type_head(h),
        }


def build_decoder(cfg=None) -> nn.Module:
    return HOIDecoder(cfg)
