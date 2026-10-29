"""Causal, entity-permutation-equivariant Mamba with directed graph targets.

Only historical entity_features/entity_valid/entity_kind enter the model.
Metadata, labels, and future values are never inspected by the encoder.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphV5Encoder(nn.Module):
    """Shared per-entity projection and same-frame masked set attention."""

    def __init__(self, input_dim, d_model=128, n_heads=4, dropout=0.0):
        super().__init__()
        if input_dim < 1 or d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError("Invalid feature/attention dimensions")
        self.input_dim, self.d_model = input_dim, d_model
        self.features = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # Object kind is shared by ALL object slots. No object-index embedding.
        self.kind = nn.Embedding(4, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, features, entity_valid, entity_kind):
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError("entity_features must have shape [B,W,E,input_dim]")
        B, W, E, _ = features.shape
        if B < 1 or W < 1 or E < 1 or not features.is_floating_point():
            raise ValueError("Historical features must be nonempty floating point tensors")
        if entity_valid.shape != (B, W, E) or entity_valid.dtype != torch.bool:
            raise ValueError("entity_valid must be bool [B,W,E]")
        if entity_kind.shape != (B, E) or entity_kind.dtype != torch.long:
            raise ValueError("entity_kind must be long [B,E]")
        if torch.any((entity_kind < 0) | (entity_kind > 3)):
            raise ValueError("Entity kind must be left/right/both/object (0..3)")
        if not torch.isfinite(features[entity_valid]).all():
            raise ValueError("Observed entity features contain NaN/Inf")
        # Sanitize BEFORE any learnable operation to keep backward gradients
        # finite even when unobserved slots contain NaN or arbitrary values.
        safe = torch.where(entity_valid[..., None], features, torch.zeros_like(features))
        tokens = self.features(safe) + self.kind(entity_kind)[:, None]
        tokens = tokens.masked_fill(~entity_valid[..., None], 0)
        flat = self.pre_norm(tokens).reshape(B * W, E, self.d_model)
        padding = ~entity_valid.reshape(B * W, E)
        # An all-missing frame must not softmax over an all -inf row. The dummy
        # key contains zeros; all outputs from such a frame are masked away.
        all_missing = padding.all(-1)
        safe_padding = padding.clone()
        safe_padding[all_missing, 0] = False
        attended, _ = self.attention(flat, flat, flat, key_padding_mask=safe_padding, need_weights=False)
        tokens = self.post_norm(tokens + attended.reshape(B, W, E, self.d_model))
        return tokens.masked_fill(~entity_valid[..., None], 0)


class DirectedGraphV5Head(nn.Module):
    """Shared ordered-pair head; source and destination stay distinguishable."""

    def __init__(self, d_model=128, event_type_count=66):
        super().__init__()
        if d_model < 1 or event_type_count < 1:
            raise ValueError("Invalid graph head dimensions")
        self.event_type_count = event_type_count
        self.pair = nn.Sequential(nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, event_type_count))

    def forward(self, states, entity_seen):
        if states.ndim != 3 or entity_seen.shape != states.shape[:2] or entity_seen.dtype != torch.bool:
            raise ValueError("Graph states/mask must be [B,E,D] / bool [B,E]")
        if not torch.isfinite(states[entity_seen]).all():
            raise ValueError("Observed graph states contain NaN/Inf")
        states = states.masked_fill(~entity_seen[..., None], 0)
        count = states.shape[1]
        source = states[:, :, None].expand(-1, -1, count, -1)
        target = states[:, None, :].expand(-1, count, -1, -1)
        pair = torch.cat((source, target, source - target, source * target), -1)
        prediction_mask = entity_seen[:, :, None] & entity_seen[:, None, :]
        logits = self.pair(pair).masked_fill(~prediction_mask[..., None], 0)
        return logits, prediction_mask


class GraphV5Mamba(nn.Module):
    """Real shared Mamba along each entity's native observation-time axis."""

    def __init__(self, input_dim, event_type_count=66, *, d_model=128, n_layer=2,
                 d_state=16, d_conv=4, expand=2, n_heads=4, dropout=0.0):
        super().__init__()
        if n_layer < 1:
            raise ValueError("At least one real Mamba layer is required")
        from src.mamba.train import MambaBackbone
        self.encoder = GraphV5Encoder(input_dim, d_model, n_heads, dropout)
        self.backbone = MambaBackbone(d_model, n_layer, d_state, d_conv, expand, dropout)
        self.graph_head = DirectedGraphV5Head(d_model, event_type_count)
        self.input_dim, self.event_type_count = input_dim, event_type_count

    def encode_history(self, batch):
        # These are intentionally the ONLY batch fields read by the model.
        valid = batch["entity_valid"]
        tokens = self.encoder(batch["entity_features"], valid, batch["entity_kind"])
        B, W, E, D = tokens.shape
        temporal = tokens.permute(0, 2, 1, 3).reshape(B * E, W, D)
        temporal = self.backbone(temporal)
        history = temporal.reshape(B, E, W, D).permute(0, 2, 1, 3)
        # Missing frames after a real observation may retain historical state.
        # Never-seen entities cannot acquire identity from biases or padding.
        seen = valid.to(torch.int32).cumsum(1) > 0
        return history.masked_fill(~seen[..., None], 0)

    def forward(self, batch):
        history = self.encode_history(batch)
        seen = batch["entity_valid"].any(1)
        states = history[:, -1]
        logits, prediction_mask = self.graph_head(states, seen)
        return {"graph_logits": logits, "graph_prediction_mask": prediction_mask,
                "entity_states": states, "entity_prediction_mask": seen}


def _prediction_mask(logits, prediction_mask):
    if prediction_mask is None:
        return None
    if prediction_mask.dtype != torch.bool:
        raise ValueError("Prediction mask must be boolean")
    if prediction_mask.shape == logits.shape[:-1]:
        return prediction_mask[..., None].expand_as(logits)
    if prediction_mask.shape == logits.shape:
        return prediction_mask
    raise ValueError("Prediction mask must be [B,E,E] or [B,E,E,K]")


def sparse_graph_bce(logits, indices, values, prediction_mask=None, *, require_supervision=True):
    """BCE over unique COO [batch,source,target,type] cells only.

    Missing cells are unknown. Duplicate cells are rejected to avoid silently
    changing their weights. An empty batch cannot masquerade as training.
    """
    if logits.ndim != 4 or logits.shape[1] != logits.shape[2] or not logits.is_floating_point():
        raise ValueError("graph_logits must be floating [B,E,E,K]")
    if indices.ndim != 2 or indices.shape[1] != 4 or indices.dtype != torch.long:
        raise ValueError("Sparse graph indices must be long [L,4]")
    if values.ndim != 1 or len(values) != len(indices):
        raise ValueError("Sparse graph values must have shape [L]")
    if indices.device != logits.device or values.device != logits.device:
        raise ValueError("Graph predictions and supervision must share a device")
    predicted = _prediction_mask(logits, prediction_mask)
    count = len(indices)
    if not count:
        if require_supervision:
            raise ValueError("No explicit graph supervision; skip this batch")
        return {"loss": logits.reshape(-1)[:0].sum(), "valid_count": 0,
                "positive_count": 0, "negative_count": 0, "skipped": True}
    bounds = torch.tensor(logits.shape, device=indices.device)
    if torch.any(indices < 0) or torch.any(indices >= bounds):
        raise ValueError("Graph supervision index out of range")
    if len(torch.unique(indices, dim=0)) != count:
        raise ValueError("Duplicate sparse graph supervision cell")
    at = tuple(indices.unbind(-1))
    if predicted is not None and not predicted[at].all():
        raise ValueError("Graph supervision references an entity never observed in history")
    selected = logits[at]
    if not torch.isfinite(selected).all() or not torch.isfinite(values).all():
        raise ValueError("Supervised graph values/logits must be finite")
    if not ((values == 0) | (values == 1)).all():
        raise ValueError("Graph labels must be explicit zero or one")
    # FP32 accumulation is deliberate even when the Mamba forward uses BF16.
    loss = F.binary_cross_entropy_with_logits(selected.float(), values.float())
    positive = int((values == 1).sum())
    return {"loss": loss, "valid_count": count, "positive_count": positive,
            "negative_count": count - positive, "skipped": False}


def masked_graph_bce(logits, values, label_valid, prediction_mask=None, *, require_supervision=True):
    """Dense interface with sparse supervision semantics; unknowns are unread."""
    if values.shape != logits.shape or label_valid.shape != logits.shape or label_valid.dtype != torch.bool:
        raise ValueError("Graph values and bool label mask must match graph logits")
    if values.device != logits.device or label_valid.device != logits.device:
        raise ValueError("Graph values and mask must share the prediction device")
    return sparse_graph_bce(logits, label_valid.nonzero(), values[label_valid], prediction_mask,
                            require_supervision=require_supervision)


def decode_graph(logits, event_types, metadata, *, prediction_mask=None, threshold=0.5, top_k=20):
    """Decode directed scores to Chinese IDs/names without altering model input."""
    if logits.ndim != 4 or logits.shape[1] != logits.shape[2] or len(event_types) != logits.shape[-1]:
        raise ValueError("Graph logits/event vocabulary shape mismatch")
    if len(metadata) != logits.shape[0] or not 0 <= threshold <= 1 or top_k < 1:
        raise ValueError("Invalid graph decoding metadata or threshold")
    predicted = _prediction_mask(logits, prediction_mask)
    if predicted is None:
        raise ValueError("Decoding requires an explicit observed-entity prediction mask")
    scores = logits.detach().float().sigmoid().cpu()
    keep = predicted.detach().cpu() & torch.isfinite(scores) & (scores >= threshold)
    decoded = []
    for b, meta in enumerate(metadata):
        identities = meta["entity_ids"]
        names = meta.get("entity_names_zh", identities)
        if len(identities) != logits.shape[1] or len(names) != len(identities):
            raise ValueError("Decoding entity IDs must preserve every graph slot")
        candidates = []
        for source, target, kind in keep[b].nonzero().tolist():
            if not identities[source] or not identities[target]:
                continue
            event = event_types[kind]
            candidates.append({"source_slot": source, "target_slot": target,
                "source_id": identities[source], "target_id": identities[target],
                "source_name_zh": names[source], "target_name_zh": names[target],
                "layer": event["layer"], "type": event.get("code", event.get("type")),
                "type_name_zh": event.get("name_zh", event.get("code", event.get("type"))),
                "probability": float(scores[b, source, target, kind]),
                "annotation_source": "model_prediction", "human_label_valid": False})
        decoded.append(sorted(candidates, key=lambda x: -x["probability"])[:top_k])
    return decoded
