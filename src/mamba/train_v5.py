"""Config-driven trainer for the admitted V5 directed-graph interaction task.

Reuses the frozen V5 dataset (admission enforced by ``EgoDemoGraphV5Dataset``),
the shared graph model/loss from ``src.mamba.graph_v5``, and the LSTM baseline
backbone from ``src.mamba.train``.  This trainer never passes ``diagnostic`` —
only datasets that passed release admission can be trained on.

Usage:
    python -m src.mamba.train_v5 --config configs/graph_v5_base.yaml
    python -m src.mamba.train_v5 --config configs/graph_v5_base.yaml --backbone lstm --device cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as e:  # pragma: no cover - environment guard
    raise SystemExit(f"[train_v5] 需要 PyTorch：{e}") from None

from src.mamba.dataset_v5 import EgoDemoGraphV5Dataset, collate_graph_v5, sha
from src.mamba.eval_v5 import GraphMetricAccumulator
from src.mamba.graph_v5 import GraphV5Mamba, masked_graph_bce
from src.mamba.train import LSTMBackbone

DEFAULT_CONFIG = {
    "run_name": "graph_v5_base",
    "seed": 20261024,
    "contract_version": "v5-2026-10-24",
    "device": "auto",  # cuda | cpu | auto; mamba requires cuda
    "runs_root": "runs",
    "data": {
        "root": "runs/egodemo_training_v5_300_20261024",
        "cache_all_sequences": True,  # full in-memory cache; the base loader keeps 1
        "num_workers": 0,
    },
    "model": {
        "backbone": "mamba",  # mamba | lstm
        "input_dim": 132,  # v6: 120 causal + 12 pose block
        "event_type_count": 66,
        "d_model": 128,
        "n_layer": 2,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "n_heads": 4,
        "dropout": 0.0,
    },
    "train": {
        "epochs": 50,
        "batch_size": 16,
        "lr": 3.0e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "amp_bf16": True,
        "log_interval": 10,
        "selection_metric": "val_loss",  # val_loss | val_micro_f1 | val_macro_f1_seen
    },
    "eval": {
        "threshold": 0.5,
        "top_k": 20,
        "decode_samples": 5,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_v5(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    if cfg["model"]["backbone"] not in ("mamba", "lstm", "transformer"):
        raise SystemExit(f"[train_v5] model.backbone 只支持 mamba|lstm|transformer，收到 {cfg['model']['backbone']!r}")
    if cfg["train"]["selection_metric"] not in ("val_loss", "val_micro_f1", "val_macro_f1_seen"):
        raise SystemExit(f"[train_v5] 未知 selection_metric {cfg['train']['selection_metric']!r}")
    return cfg


class TransformerBackbone(nn.Module):
    """Causal Transformer baseline over each entity's temporal trajectory."""

    def __init__(self, d_model, n_layer, n_heads, dropout=0.0, max_frames=32):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.position = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        if x.shape[1] > self.position.shape[1]:
            raise ValueError("Transformer sequence exceeds configured max_frames")
        length = x.shape[1]
        causal = torch.full((length, length), float("-inf"), device=x.device, dtype=x.dtype)
        causal = torch.triu(causal, diagonal=1)
        return self.norm(self.layers(x + self.position[:, :length], mask=causal))


class GraphV5LSTM(nn.Module):
    """CPU baseline: identical encoder/head, LSTM instead of the Mamba stack."""

    def __init__(self, input_dim, event_type_count=66, *, d_model=128, n_layer=2,
                 n_heads=4, dropout=0.0):
        super().__init__()
        from src.mamba.graph_v5 import DirectedGraphV5Head, GraphV5Encoder
        self.encoder = GraphV5Encoder(input_dim, d_model, n_heads, dropout)
        self.backbone = LSTMBackbone(d_model, n_layer, dropout)
        self.graph_head = DirectedGraphV5Head(d_model, event_type_count)
        self.input_dim, self.event_type_count = input_dim, event_type_count

    def forward(self, batch):
        # Identical wiring to GraphV5Mamba.forward/encode_history.
        return GraphV5Mamba.forward(self, batch)

    def encode_history(self, batch):
        return GraphV5Mamba.encode_history(self, batch)


class GraphV5Transformer(nn.Module):
    """CPU-capable Transformer baseline with the same encoder/head wiring."""

    def __init__(self, input_dim, event_type_count=66, *, d_model=128, n_layer=2,
                 n_heads=4, dropout=0.0):
        super().__init__()
        from src.mamba.graph_v5 import DirectedGraphV5Head, GraphV5Encoder
        self.encoder = GraphV5Encoder(input_dim, d_model, n_heads, dropout)
        self.backbone = TransformerBackbone(d_model, n_layer, n_heads, dropout)
        self.graph_head = DirectedGraphV5Head(d_model, event_type_count)
        self.input_dim, self.event_type_count = input_dim, event_type_count

    def forward(self, batch):
        return GraphV5Mamba.forward(self, batch)

    def encode_history(self, batch):
        return GraphV5Mamba.encode_history(self, batch)


class MemoryCachedV5Dataset(EgoDemoGraphV5Dataset):
    """Cache every sequence in memory; admission/SHA checks still run once per sequence."""

    def __init__(self, root, split="train"):
        super().__init__(root, split, diagnostic=False)
        self._full_cache = {}

    def _load_sequence(self, uid):
        if uid not in self._full_cache:
            self._full_cache[uid] = super()._load_sequence(uid)
        return self._full_cache[uid]


def build_v5_dataset(cfg: dict, split: str) -> EgoDemoGraphV5Dataset:
    root = cfg["data"]["root"]
    if cfg["data"].get("cache_all_sequences", True):
        return MemoryCachedV5Dataset(root, split)
    return EgoDemoGraphV5Dataset(root, split, diagnostic=False)


def build_v5_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["backbone"] == "mamba":
        return GraphV5Mamba(m["input_dim"], m["event_type_count"], d_model=m["d_model"],
                            n_layer=m["n_layer"], d_state=m["d_state"], d_conv=m["d_conv"],
                            expand=m["expand"], n_heads=m["n_heads"], dropout=m["dropout"])
    if m["backbone"] == "transformer":
        return GraphV5Transformer(m["input_dim"], m["event_type_count"], d_model=m["d_model"],
                                  n_layer=m["n_layer"], n_heads=m["n_heads"], dropout=m["dropout"])
    return GraphV5LSTM(m["input_dim"], m["event_type_count"], d_model=m["d_model"],
                       n_layer=m["n_layer"], n_heads=m["n_heads"], dropout=m["dropout"])


def scan_seen_types(dataset) -> dict:
    """Positive label counts per (layer, code) across the train split."""
    from src.mamba.eval_v5 import scan_seen_types as scan
    return scan(dataset)


def cross_check_seen_types(seen: dict, release: dict):
    absent = set(release.get("positive_types_absent_from_train") or [])
    seen_names = set(seen)
    problem = absent & seen_names
    if problem:
        raise SystemExit(f"[train_v5] release 声明缺失但 train 实际出现的类型：{sorted(problem)}")
    return {"seen_types": sorted(seen_names), "release_absent_types": sorted(absent)}


def run_epoch_v5(model, loader, device, cfg, amp_ok, optimizer=None, seen_types=None):
    """One train or eval epoch; unknown cells never contribute loss or metrics."""
    training = optimizer is not None
    model.train() if training else model.eval()
    event_types = loader.dataset.event_types
    threshold = cfg["eval"]["threshold"]
    accumulator = GraphMetricAccumulator(event_types, threshold)
    loss_total, loss_batches = 0.0, 0
    t0 = time.perf_counter()
    for step, batch in enumerate(loader):
        moved = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if amp_ok else nullcontext())
        with torch.set_grad_enabled(training), ctx:
            output = model(moved)
            stats = masked_graph_bce(output["graph_logits"], moved["graph_values"],
                                     moved["graph_label_valid"], output["graph_prediction_mask"])
            loss = stats["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"[train_v5] 非有限 loss 于 step {step + 1}，中止训练")
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = cfg["train"]["grad_clip"]
            if clip and clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        loss_total += loss.item()
        loss_batches += 1
        accumulator.update(output["graph_logits"], moved["graph_values"],
                           moved["graph_label_valid"], output["graph_prediction_mask"])
        if training and cfg["train"]["log_interval"] and (step + 1) % cfg["train"]["log_interval"] == 0:
            print(f"    step {step + 1}/{len(loader)} loss={loss.item():.4f}")
    metrics = accumulator.summarize(set(seen_types or ()))
    metrics["loss"] = loss_total / max(loss_batches, 1)
    metrics["seconds"] = time.perf_counter() - t0
    metrics["windows_per_second"] = metrics["windows"] / metrics["seconds"] if metrics["seconds"] else 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        metrics["peak_cuda_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return metrics


def save_checkpoint_v5(path: Path, model, optimizer, epoch, cfg, best_metric, data_binding):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "best_metric": best_metric,
        "data_binding": data_binding,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, path)


def load_checkpoint_v5(path: Path, model, optimizer, data_binding=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if data_binding is not None:
        saved = checkpoint.get("data_binding") or {}
        for key, expected in data_binding.items():
            if saved.get(key) != expected:
                raise SystemExit(f"[train_v5] resume 拒绝：checkpoint 数据绑定 {key} 不一致")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    rng = checkpoint.get("rng")
    if rng:
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return checkpoint


def _metric_value(metrics: dict, name: str) -> float:
    if name == "val_loss":
        return metrics["loss"]
    if name == "val_micro_f1":
        return metrics["micro"]["f1"]
    if name == "val_macro_f1_seen":
        return metrics["macro_f1_seen"]
    raise ValueError(name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.mamba.train_v5",
                                description="V5 有向图交互训练（配置驱动，admission 强制）")
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", choices=["mamba", "lstm", "transformer"], default=None)
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--runs-root", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="checkpoint 路径，恢复训练")
    return p.parse_args(argv)


def main(argv=None):
    args = argv if isinstance(argv, argparse.Namespace) else parse_args(argv)
    cfg = load_config_v5(args.config)
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.runs_root:
        cfg["runs_root"] = args.runs_root
    if args.run_name:
        cfg["run_name"] = args.run_name

    device_cfg = cfg["device"]
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available()
                          else device_cfg if device_cfg != "auto" else "cpu")
    if cfg["model"]["backbone"] == "mamba" and device.type != "cuda":
        raise SystemExit("[train_v5] mamba_ssm 需要 CUDA；本机跑基线请加 --backbone lstm --device cpu")
    amp_ok = bool(cfg["train"]["amp_bf16"]) and device.type == "cuda"

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    data_root = Path(cfg["data"]["root"])
    run_dir = Path(cfg["runs_root"]) / cfg["run_name"]
    try:
        run_dir.resolve().relative_to(data_root.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("[train_v5] run 目录不得位于冻结数据包内部")

    train_set = build_v5_dataset(cfg, "train")
    val_set = build_v5_dataset(cfg, "val")
    event_types = train_set.event_types
    m = cfg["model"]
    if m["input_dim"] not in (120, 132) or len(event_types) != m["event_type_count"]:
        raise SystemExit(f"[train_v5] 契约不符：input_dim={m['input_dim']}，event_types={len(event_types)}/{m['event_type_count']}")

    index_sha = sha(data_root / "training_index.json")
    release = json.loads((data_root / "training_index.json").read_text())
    admission_path = data_root / release["admission_path"]
    release_sha = sha(admission_path)
    release_doc = json.loads(admission_path.read_text())
    data_binding = {"training_index_sha256": index_sha, "training_release_sha256": release_sha}

    seen = scan_seen_types(train_set)
    seen_check = cross_check_seen_types(seen, release_doc)

    loader_kw = dict(batch_size=cfg["train"]["batch_size"], num_workers=cfg["data"]["num_workers"],
                     collate_fn=collate_graph_v5, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    model = build_v5_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps({
        "contract_version": cfg["contract_version"],
        "device": str(device),
        "backbone": m["backbone"],
        "params": n_params,
        "train_windows": len(train_set), "val_windows": len(val_set),
        **data_binding,
        **seen_check,
    }, ensure_ascii=False, indent=2))
    (run_dir / "seen_types.json").write_text(json.dumps(seen, ensure_ascii=False, indent=2))
    print(f"[train_v5] device={device} backbone={m['backbone']} params={n_params / 1e6:.2f}M "
          f"train={len(train_set)} val={len(val_set)} amp_bf16={amp_ok}")

    start_epoch = 1
    metric_name = cfg["train"]["selection_metric"]
    higher_better = metric_name != "val_loss"
    best = -float("inf") if higher_better else float("inf")
    if args.resume:
        checkpoint = load_checkpoint_v5(Path(args.resume), model, optimizer, data_binding)
        start_epoch = int(checkpoint["epoch"]) + 1
        best = checkpoint["best_metric"]
        print(f"[train_v5] resume 自 epoch {checkpoint['epoch']}，best={best:.4f}")

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        tr = run_epoch_v5(model, train_loader, device, cfg, amp_ok, optimizer=optimizer, seen_types=seen)
        va = run_epoch_v5(model, val_loader, device, cfg, amp_ok, seen_types=seen)
        print(f"[epoch {epoch:03d}] train loss={tr['loss']:.4f} micro_f1={tr['micro']['f1']:.3f} "
              f"| val loss={va['loss']:.4f} micro_f1={va['micro']['f1']:.3f} "
              f"macro_seen={va['macro_f1_seen']:.3f} ({va['seconds']:.1f}s)")
        current = _metric_value(va, metric_name)
        save_checkpoint_v5(run_dir / "last.pt", model, optimizer, epoch, cfg, best, data_binding)
        improved = current > best if higher_better else current < best
        if improved:
            best = current
            save_checkpoint_v5(run_dir / "best.pt", model, optimizer, epoch, cfg, best, data_binding)
            print(f"  <- best ({metric_name}={best:.4f})")

    print(f"[train_v5] 完成。run_dir={run_dir} best_{metric_name}={best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
