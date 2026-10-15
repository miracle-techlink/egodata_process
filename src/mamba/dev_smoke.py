"""Dev-only smoke harness for task 6 (D1: synthetic-data smoke).

Generates synthetic samples in the real task-4 layout (keys/NaN/mask rules
follow src/dataset/build_samples.py make_sample), then runs train.main.
Uses the real src.encoding when importable; otherwise installs a stand-in
with the same interface so the data path can still be exercised.

Usage:
    python -m src.mamba.dev_smoke [--out /tmp/ego_smoke] [--epochs 2] [--baseline lstm]
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

W, N_MAX = 32, 4


def make_synthetic_samples(root: Path, n_train=16, n_val=8, seed=0):
    rng = np.random.default_rng(seed)
    index = {}
    for split, n in (("train", n_train), ("val", n_val)):
        (root / split).mkdir(parents=True, exist_ok=True)
        ids = []
        for i in range(n):
            sid = f"synth_{split}_{i:03d}"
            hand = rng.normal(size=(W, 2, 21, 3)).astype(np.float32)
            miss = rng.random(W) < 0.1  # undetected frames: whole frame NaN (contract)
            hand[miss] = np.nan
            n_obj = int(rng.integers(1, N_MAX + 1))
            obj = np.zeros((W, N_MAX, 13), np.float32)  # padded slots are zeros (task-4 layout)
            obj[:, :n_obj] = rng.normal(size=(W, n_obj, 13)).astype(np.float32)
            gone = rng.random((W, n_obj)) < 0.1  # object undetected in some frames
            obj[:, :n_obj][gone] = np.nan
            obj_mask = np.zeros(N_MAX, np.float32)
            obj_mask[:n_obj] = 1.0
            object_ids = np.array([f"obj{j}" for j in range(n_obj)] + [""] * (N_MAX - n_obj))
            interaction = int(rng.integers(0, 7))
            if interaction == 0:  # none => no subject / no object (label_rules convention)
                subject, slot = 2, -1
            else:
                subject = int(rng.integers(0, 2))
                slot = int(rng.integers(0, n_obj))  # label must point at a valid slot
            np.savez(
                root / split / f"{sid}.npz",
                hand_seq=hand,
                obj_seq=obj,
                obj_mask=obj_mask,
                object_ids=object_ids,
                label_subject=np.int64(subject),
                label_object_slot=np.int64(slot),
                label_interaction=np.int64(interaction),
            )
            ids.append(sid)
        index[split] = ids
    with open(root / "dataset_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def install_stub_encoding():
    """Stand-in for task 5's src.encoding, same interface (pre-integration only)."""
    import torch
    import torch.nn as nn

    mod = types.ModuleType("src.encoding")

    def cfg_get(cfg, key, default):
        if cfg is None:
            return default
        return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

    class StubEncoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            d = int(cfg_get(cfg, "d_model", 128))
            self.n_max = int(cfg_get(cfg, "n_max", 8))
            self.hand_proj = nn.Linear(2 * 21 * 3, d)
            self.obj_proj = nn.Linear(13, d)
            self.person_proj = nn.Linear(8, d)

        def forward(self, hand_seq, obj_seq, obj_mask=None, gaze_seq=None):
            b, w = hand_seq.shape[:2]
            d = self.hand_proj.out_features
            h = self.hand_proj(torch.nan_to_num(hand_seq, nan=0.0).reshape(b, w, -1))
            person = self.person_proj(torch.zeros(b, w, 8, device=h.device, dtype=h.dtype))
            o = self.obj_proj(torch.nan_to_num(obj_seq, nan=0.0))
            toks = torch.cat([person.unsqueeze(2), h.unsqueeze(2), h.unsqueeze(2), o], dim=2)
            if obj_mask is not None:
                slot = torch.cat([
                    torch.ones(b, w, 3, dtype=torch.bool, device=h.device),
                    obj_mask.bool(),
                ], dim=-1)
                toks = toks * slot[..., None].float()
            return toks  # (B, W, 3 + n_max, d)

    class StubDecoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            d = int(cfg_get(cfg, "d_model", 128))
            self.n_max = int(cfg_get(cfg, "n_max", 8))
            self.subject_head = nn.Linear(d, 3)
            self.object_head = nn.Linear(d, self.n_max + 1)
            self.type_head = nn.Linear(d, int(cfg_get(cfg, "n_interaction_types", 7)))

        def forward(self, tokens, obj_mask=None):
            h = tokens[:, -1].mean(dim=1)  # last frame, mean over entities
            obj_logits = self.object_head(h)
            if obj_mask is not None:
                pad = torch.cat(
                    [obj_logits[:, : self.n_max].masked_fill(~obj_mask.bool(), float("-inf")),
                     obj_logits[:, self.n_max:]], dim=-1)
                obj_logits = pad
            return {
                "subject": self.subject_head(h),
                "object": obj_logits,
                "interaction_type": self.type_head(h),
            }

    mod.build_encoder = lambda cfg=None: StubEncoder(cfg)
    mod.build_decoder = lambda cfg=None: StubDecoder(cfg)
    sys.modules["src.encoding"] = mod
    return mod


def ensure_encoding():
    try:
        import src.encoding  # noqa: F401
        print("[dev_smoke] 使用真实 src.encoding（⑤ 已落地）")
    except ImportError:
        install_stub_encoding()
        print("[dev_smoke] 未找到 src.encoding，注入同接口 stub（仅冒烟用）")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.mamba.dev_smoke")
    p.add_argument("--out", default="/tmp/ego_smoke", help="smoke 工作目录（合成数据+runs）")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--baseline", choices=["lstm"], default="lstm",
                   help="本机冒烟固定 lstm；mamba 需 CUDA 环境")
    args = p.parse_args(argv)

    out = Path(args.out)
    data_root = out / "samples"
    make_synthetic_samples(data_root)
    ensure_encoding()

    cfg_path = out / "smoke.yaml"
    cfg_path.write_text(
        "\n".join([
            f"run_name: dev_smoke_{args.baseline}",
            "device: cpu",
            f"runs_root: {out}/runs",
            "data:",
            f"  root: {data_root}",
            f"  index: {data_root}/dataset_index.json",
            "model:",
            f"  backbone: {args.baseline}",
            "  d_model: 64",
            "  n_layer: 2",
            "  encoder:",
            "    n_layers: 1",
            "    n_heads: 4",
            "  decoder:",
            f"    n_max: {N_MAX}",
            "train:",
            f"  epochs: {args.epochs}",
            "  batch_size: 8",
            "  amp_bf16: true   # CPU 上应自动降级 fp32 并告警",
            "  log_interval: 1",
        ]) + "\n",
        encoding="utf-8",
    )

    from src.mamba import train
    return train.main(["--config", str(cfg_path)])


if __name__ == "__main__":
    sys.exit(main())
