"""Encoding module for ego HOI intent prediction (task 5, arc).

Usage (consumed by task 6, mamba-training):
    from src.encoding import build_encoder, build_decoder
    enc = build_encoder(cfg)  # hand_seq, obj_seq, obj_mask -> (B, W, 3 + n_max, d_model)
    dec = build_decoder(cfg)  # tokens -> dict(subject=, object=, interaction_type=) logits

cfg may be a dict or any object with attributes (dataclass, SimpleNamespace, ...).
Recognized keys (defaults shown):
    d_model=128, n_max=8, n_heads=4, n_layers=2, dim_ff=256, dropout=0.1,
    max_frames=64, contact_threshold=0.15, use_gaze=False, gaze_dim=3,
    n_subject=3, n_interaction_types=7  (decoder)
"""
from .decoder import (
    INTERACTION_TYPES,
    SUBJECT_CLASSES,
    HOIDecoder,
    build_decoder,
)
from .encoder import HOIEncoder, build_encoder

__all__ = [
    "build_encoder",
    "build_decoder",
    "HOIEncoder",
    "HOIDecoder",
    "SUBJECT_CLASSES",
    "INTERACTION_TYPES",
]
