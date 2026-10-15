"""Forward/unit tests for src/encoding (task 5).

Runs with pytest or standalone: python tests/test_encoding.py
Uses synthetic tensors per the contract shapes (03_data_pipeline.md sec.4):
hand_seq (B, W, 2, 21, 3), obj_seq (B, W, n_max, 13), obj_mask (B, W, n_max).
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.encoding import INTERACTION_TYPES, SUBJECT_CLASSES, build_decoder, build_encoder

CFG = dict(d_model=64, n_max=5, n_heads=4, n_layers=2, dim_ff=128, dropout=0.0, max_frames=64)
B, W = 2, 16


def make_batch(n_valid_list, nan_hand_frames=None, nan_obj_frames=None, seed=0):
    """n_valid_list: per-sample number of valid object slots (rest are padding)."""
    g = torch.Generator().manual_seed(seed)
    n_max = CFG["n_max"]
    hand_seq = torch.randn(B, W, 2, 21, 3, generator=g)
    obj_seq = torch.randn(B, W, n_max, 13, generator=g)
    obj_mask = torch.zeros(B, W, n_max, dtype=torch.bool)
    for b, n in enumerate(n_valid_list):
        obj_mask[b, :, :n] = True
    if nan_hand_frames:
        for b, w, h in nan_hand_frames:
            hand_seq[b, w, h] = float("nan")
    if nan_obj_frames:
        for b, w, o in nan_obj_frames:
            obj_seq[b, w, o] = float("nan")
    return hand_seq, obj_seq, obj_mask


def test_encoder_shapes():
    enc = build_encoder(CFG)
    hand_seq, obj_seq, obj_mask = make_batch([3, 5])
    tokens = enc(hand_seq, obj_seq, obj_mask)
    assert tokens.shape == (B, W, 3 + CFG["n_max"], CFG["d_model"]), tokens.shape
    assert torch.isfinite(tokens).all()


def test_nan_inputs_produce_finite_output():
    enc = build_encoder(CFG)
    hand_seq, obj_seq, obj_mask = make_batch(
        [3, 5],
        nan_hand_frames=[(0, 3, 0), (1, 7, 1)],  # left hand of sample 0 at frame 3, etc.
        nan_obj_frames=[(0, 2, 1), (1, 9, 4)],
    )
    tokens = enc(hand_seq, obj_seq, obj_mask)
    assert torch.isfinite(tokens).all()


def test_variable_object_count_and_zero_padding():
    enc = build_encoder(CFG)
    hand_seq, obj_seq, obj_mask = make_batch([2, 5])
    tokens = enc(hand_seq, obj_seq, obj_mask)
    pad0 = tokens[0, :, 3 + 2 :]  # sample 0: slots 2..4 are padding
    assert (pad0 == 0).all(), "padding-object tokens must be zero"
    assert tokens[0, :, : 3 + 2].abs().sum() > 0
    assert torch.isfinite(tokens[1]).all()


def test_obj_mask_inferred_when_none():
    enc = build_encoder(CFG)
    hand_seq, obj_seq, _ = make_batch([5, 5])
    obj_seq[0, :, 3:] = float("nan")  # slots 3..4 never observed -> inferred padding
    tokens = enc(hand_seq, obj_seq, obj_mask=None)
    assert (tokens[0, :, 3 + 3 :] == 0).all()
    assert torch.isfinite(tokens).all()


def test_decoder_shapes_and_padding_logits():
    enc, dec = build_encoder(CFG), build_decoder(CFG)
    hand_seq, obj_seq, obj_mask = make_batch([2, 5])
    tokens = enc(hand_seq, obj_seq, obj_mask)
    out = dec(tokens, obj_mask)
    assert out["subject"].shape == (B, len(SUBJECT_CLASSES))
    assert out["object"].shape == (B, CFG["n_max"] + 1)
    assert out["interaction_type"].shape == (B, len(INTERACTION_TYPES))
    assert torch.isinf(out["object"][0, 2:-1]).all() and (out["object"][0, 2:-1] < 0).all()
    assert torch.isfinite(out["object"][0, [0, 1, -1]]).all()
    assert torch.isfinite(out["object"][1]).all()
    assert torch.isfinite(out["subject"]).all() and torch.isfinite(out["interaction_type"]).all()


def test_decoder_accepts_pooled_inputs():
    dec = build_decoder(CFG)
    enc = build_encoder(CFG)
    hand_seq, obj_seq, obj_mask = make_batch([3, 4])
    tokens = enc(hand_seq, obj_seq, obj_mask)
    out3 = dec(tokens[:, -1], obj_mask[:, -1])  # (B, E, D) + (B, n_max)
    assert out3["object"].shape == (B, CFG["n_max"] + 1)
    assert torch.isinf(out3["object"][0, 3:-1]).all()
    out2 = dec(tokens[:, -1].mean(dim=1))  # (B, D), no mask
    assert out2["subject"].shape == (B, len(SUBJECT_CLASSES))
    assert torch.isfinite(out2["object"]).all()


def test_backward_gradients_flow():
    enc, dec = build_encoder(CFG), build_decoder(CFG)
    hand_seq, obj_seq, obj_mask = make_batch([3, 5])
    out = dec(enc(hand_seq, obj_seq, obj_mask), obj_mask)
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads), "encoder must receive gradients"
    assert all(torch.isfinite(g).all() for g in grads)


def test_cfg_as_dataclass_like_object():
    cfg = SimpleNamespace(d_model=32, n_max=3, n_heads=2, n_layers=1, dim_ff=64, dropout=0.0, max_frames=32)
    enc, dec = build_encoder(cfg), build_decoder(cfg)
    hand_seq = torch.randn(1, 8, 2, 21, 3)
    obj_seq = torch.randn(1, 8, 3, 13)
    obj_mask = torch.ones(1, 8, 3, dtype=torch.bool)
    out = dec(enc(hand_seq, obj_seq, obj_mask), obj_mask)
    assert out["object"].shape == (1, 4)


def test_gaze_channel_optional():
    cfg = dict(CFG, use_gaze=True, gaze_dim=3)
    enc = build_encoder(cfg)
    hand_seq, obj_seq, obj_mask = make_batch([3, 3])
    gaze = torch.randn(B, W, 3)
    tokens = enc(hand_seq, obj_seq, obj_mask, gaze_seq=gaze)
    assert tokens.shape == (B, W, 3 + CFG["n_max"], CFG["d_model"])
    try:
        enc(hand_seq, obj_seq, obj_mask)
        raise AssertionError("expected ValueError when use_gaze=True but gaze_seq is None")
    except ValueError:
        pass


def test_bad_shapes_raise():
    enc = build_encoder(CFG)
    try:
        enc(torch.randn(B, W, 2, 21, 2), torch.randn(B, W, CFG["n_max"], 13))
        raise AssertionError("expected ValueError for bad hand_seq")
    except ValueError:
        pass
    try:
        enc(torch.randn(B, W, 2, 21, 3), torch.randn(B, W, CFG["n_max"] + 1, 13))
        raise AssertionError("expected ValueError for mismatched n_max")
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
