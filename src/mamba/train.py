"""Task 6 training entry: config-driven intent-prediction training.

Usage:
    python -m src.mamba.train --config configs/mamba_base.yaml
    python -m src.mamba.train --config configs/mamba_base.yaml --baseline lstm

Pipeline: samples/{train,val}/*.npz (task 4) -> build_encoder (task 5)
-> Mamba / LSTM backbone -> build_decoder (task 5) -> 3-way CE loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise SystemExit(
        "[train] 需要 PyTorch：conda activate caron_mamba\n"
        "        安装细节见 .agent/skills/mamba-training/SKILL.md\n"
        f"        原始错误: {e}"
    ) from None

# Label vocabularies follow .agent/memory/03_data_pipeline.md section 4.
# Index conventions match src/dataset/label_rules.py and src/encoding/decoder.py:
# subject 0=left, 1=right, 2=none; object 0..n_max-1=slot, n_max=none.
INTERACTION_TYPES = ["none", "reach", "grasp", "hold", "move", "place", "release"]
SUBJECT_MAP = {
    "left": 0, "左": 0, "左手": 0,
    "right": 1, "右": 1, "右手": 1,
    "none": 2, "无": 2,
}
TYPE_MAP = {name: i for i, name in enumerate(INTERACTION_TYPES)}

DEFAULT_CONFIG = {
    "run_name": "run",
    "seed": 42,
    "contract_version": "2026-10-22",
    "device": "auto",
    "runs_root": "runs",
    "data": {
        "root": "samples",
        "index": None,  # default: <data.root>/dataset_index.json
        "cache_in_memory": False,
        # False: ⑤ 编码器原生吃 NaN（提取 obs 标志）；仅调试/自研编码器时开 true
        "nan_to_num": False,
        "use_gaze": False,
        # 仅 legacy/synthetic smoke 可显式开启；不代表 accepted 人工审核或正式数据许可
        "allow_legacy_missing_label_valid": False,
    },
    "model": {
        "backbone": "mamba",  # mamba | lstm
        "d_model": 128,
        "n_layer": 4,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dropout": 0.1,
        "encoder": {},
        "decoder": {"n_max": 8, "n_interaction_types": len(INTERACTION_TYPES)},
    },
    "train": {
        "epochs": 50,
        "batch_size": 64,
        "lr": 3.0e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "num_workers": 0,
        "amp_bf16": True,
        "log_interval": 20,
        "selection_metric": "val_loss",  # val_loss | val_acc_macro
        "loss_weights": {"subject": 1.0, "object": 1.0, "interaction_type": 1.0},
    },
}


# ---------------------------------------------------------------- config


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("[train] 需要 pyyaml：pip install pyyaml") from None
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    if cfg["model"]["backbone"] not in ("mamba", "lstm"):
        raise SystemExit(f"[train] model.backbone 只支持 mamba|lstm，收到 {cfg['model']['backbone']!r}")
    n_types = cfg["model"]["decoder"]["n_interaction_types"]
    if n_types != len(INTERACTION_TYPES):
        raise SystemExit(
            f"[train] decoder.n_interaction_types={n_types} 与契约标签体系 "
            f"{INTERACTION_TYPES} 不一致；改标签需先改契约"
        )
    return cfg


# ---------------------------------------------------------------- dataset


def _scalar(v):
    if isinstance(v, np.ndarray):
        v = v.item() if v.ndim == 0 else v.tolist()
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return v


def _first_key(keys, *candidates):
    for c in candidates:
        if c in keys:
            return c
    return None


def parse_labels(z, keys, n_max: int) -> dict:
    """Accept either a pickled `label` dict or separate label_* keys."""
    if "label" in keys:
        raw = _scalar(z["label"])
        if isinstance(raw, str):  # json-encoded variant
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError(f"label 字段应为 dict 或 json 字符串，收到 {type(raw)}")
        subject = raw.get("subject")
        obj = raw.get("object_slot", raw.get("object_id", raw.get("object")))
        itype = raw.get("interaction_type", raw.get("type"))
    else:
        ks = _first_key(keys, "label_subject", "subject")
        ko = _first_key(keys, "label_object_slot", "label_object", "label_object_id",
                        "object_id", "object")
        kt = _first_key(keys, "label_interaction_type", "label_interaction",
                        "label_type", "interaction_type", "type")
        if not (ks and ko and kt):
            raise ValueError(f"npz 缺少 label 字段（keys={sorted(keys)}）")
        subject, obj, itype = _scalar(z[ks]), _scalar(z[ko]), _scalar(z[kt])

    subject = _scalar(subject)
    if isinstance(subject, str):
        if subject.lower() not in SUBJECT_MAP:
            raise ValueError(f"未知 subject 标签 {subject!r}")
        subject = SUBJECT_MAP[subject.lower()]
    subject = int(subject)

    itype = _scalar(itype)
    if isinstance(itype, str):
        if itype.lower() not in TYPE_MAP:
            raise ValueError(f"未知 interaction_type 标签 {itype!r}")
        itype = TYPE_MAP[itype.lower()]
    itype = int(itype)

    obj = _scalar(obj)
    if isinstance(obj, str):
        if obj.strip().lower() in ("none", "", "-1"):
            obj = n_max
        elif "object_ids" in keys:
            ids = [str(_scalar(x)) for x in z["object_ids"].tolist()]
            if obj not in ids:
                raise ValueError(f"object 标签 {obj!r} 不在样本 object_ids {ids} 中")
            obj = ids.index(obj)
        else:
            try:
                obj = int(obj)
            except ValueError:
                raise ValueError(
                    f"object 标签为字符串 {obj!r} 但样本内无 object_ids 列表，无法映射到槽位；"
                    "应由 ④ 在样本中写入 object_ids，或使用整数 label_object_slot"
                ) from None
    obj = int(obj)
    if obj < 0:
        obj = n_max
    if not 0 <= obj <= n_max:
        raise ValueError(f"object 标签 {obj} 超出 [0, {n_max}]（n_max=none 类）")
    return {"subject": subject, "object": obj, "interaction_type": itype}


def resolve_split_ids(root: Path, split: str, index_path: Path | None) -> list[str]:
    if index_path is None:
        index_path = root / "dataset_index.json"
    if index_path.is_file():
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        if isinstance(idx, dict):
            entries = idx.get(split, [])
        elif isinstance(idx, list):  # flat list of {"sample_id", "split"}
            entries = [e for e in idx if isinstance(e, dict) and e.get("split") == split]
        else:
            raise ValueError(f"无法解析 {index_path}")
        ids = []
        for e in entries:
            sid = e if isinstance(e, str) else e.get("sample_id") or e.get("id") or e.get("path")
            if sid:
                ids.append(Path(str(sid)).stem)
        return ids
    split_dir = root / split
    print(f"[train] 警告：未找到 {index_path}，回退为 glob {split_dir}/*.npz（无法校验 split 隔离）")
    if not split_dir.is_dir():
        return []
    return sorted(p.stem for p in split_dir.glob("*.npz"))


def _warn_legacy_missing_label_valid(context):
    print(
        "[train] 警告：LEGACY/SYNTHETIC SMOKE ONLY — "
        f"{context}: 兼容模式允许缺失 label_valid 并按 True 处理；"
        "这不代表 accepted 人工审核，不是正式数据许可，输出不得作为正式结果"
    )


class SamplesDataset(Dataset):
    """Lazy npz loader for task-4 samples; optional in-memory cache."""

    def __init__(self, root, split, n_max, index_path=None, cache=False,
                 nan_to_num=False, use_gaze=False, allow_legacy_missing_label_valid=False):
        self.root = Path(root)
        self.split = split
        self.n_max = n_max
        self.nan_to_num = nan_to_num
        self.use_gaze = use_gaze
        self.allow_legacy_missing_label_valid = allow_legacy_missing_label_valid
        self._warned_legacy_missing_label_valid = False
        self.ids = resolve_split_ids(self.root, split, Path(index_path) if index_path else None)
        self._cache: dict[str, dict] = {} if cache else None

    def __len__(self):
        return len(self.ids)

    def _load_npz(self, sid: str) -> dict:
        path = self.root / self.split / f"{sid}.npz"
        with np.load(path, allow_pickle=True) as z:
            keys = set(z.files)
            hand = np.asarray(z["hand_seq"], dtype=np.float32)      # (W, 2, 21, 3)
            obj = np.asarray(z["obj_seq"], dtype=np.float32)        # (W, N, 13)
            if hand.ndim != 4 or hand.shape[1:] != (2, 21, 3):
                raise ValueError(f"{path}: hand_seq 形状 {hand.shape}，契约要求 (W,2,21,3)")
            if obj.ndim != 3 or obj.shape[2] != 13:
                raise ValueError(f"{path}: obj_seq 形状 {obj.shape}，契约要求 (W,N_max,13)")
            if "obj_mask" in keys:
                mask = np.asarray(z["obj_mask"]).astype(bool)      # (N,) 槽位级（④ 真实格式）
            elif "valid_mask" in keys:
                mask = np.asarray(z["valid_mask"]).astype(bool)    # (W, N) 逐帧（⑦ fixture 变体）
            else:
                mask = ~np.isnan(obj).all(axis=-1)                 # (W, N) 逐帧观测
            if mask.ndim == 2:
                # 槽位级 = 窗口内至少一帧观测到（与 ⑤ 编码器无 mask 时的推断语义一致）
                mask = mask.any(axis=0)
            if "label_valid" not in keys:
                if not self.allow_legacy_missing_label_valid:
                    raise ValueError(
                        f"{path}: 缺少 label_valid，正式训练默认拒绝；仅 legacy/synthetic smoke "
                        "可显式设置 data.allow_legacy_missing_label_valid=true "
                        "（SamplesDataset 同名参数），且不构成正式数据许可"
                    )
                label_valid = True
                if not self._warned_legacy_missing_label_valid:
                    _warn_legacy_missing_label_valid(path)
                    self._warned_legacy_missing_label_valid = True
            else:
                raw_valid = np.asarray(z["label_valid"])
                if raw_valid.size != 1 or raw_valid.dtype.kind not in "buif":
                    raise ValueError(f"{path}: label_valid 必须是 0/1 标量")
                value = float(raw_valid.reshape(-1)[0])
                if not np.isfinite(value) or value not in (0.0, 1.0):
                    raise ValueError(f"{path}: label_valid 必须是 0/1 标量")
                label_valid = bool(value)
            labels = parse_labels(z, keys, self.n_max)
            gaze = None
            if self.use_gaze and "gaze_seq" in keys:
                gaze = np.asarray(z["gaze_seq"], dtype=np.float32)

        n, n_max = obj.shape[1], self.n_max
        if n > n_max:
            raise ValueError(f"{path}: N_max={n} 超过配置 decoder.n_max={n_max}，请调大配置")
        if n < n_max:  # right-pad object slots with NaN, mask=False
            pad = np.full((obj.shape[0], n_max - n, obj.shape[2]), np.nan, dtype=np.float32)
            obj = np.concatenate([obj, pad], axis=1)
            mask = np.concatenate([mask, np.zeros(n_max - n, dtype=bool)])
        if self.nan_to_num:
            hand = np.nan_to_num(hand)
            obj = np.nan_to_num(obj)
        return {"hand_seq": hand, "obj_seq": obj, "obj_mask": mask,
                "gaze_seq": gaze, "labels": labels, "label_valid": label_valid}

    def __getitem__(self, i: int) -> dict:
        sid = self.ids[i]
        if self._cache is not None:
            if sid not in self._cache:
                self._cache[sid] = self._load_npz(sid)
            item = self._cache[sid]
        else:
            item = self._load_npz(sid)
        out = {
            "hand_seq": torch.from_numpy(item["hand_seq"]),
            "obj_seq": torch.from_numpy(item["obj_seq"]),
            "obj_mask": torch.from_numpy(item["obj_mask"]),
            "subject": torch.tensor(item["labels"]["subject"], dtype=torch.long),
            "object": torch.tensor(item["labels"]["object"], dtype=torch.long),
            "interaction_type": torch.tensor(item["labels"]["interaction_type"], dtype=torch.long),
            "label_valid": torch.tensor(item["label_valid"], dtype=torch.bool),
            "gaze_seq": torch.from_numpy(item["gaze_seq"]) if item["gaze_seq"] is not None else None,
        }
        return out


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "hand_seq": torch.stack([b["hand_seq"] for b in batch]),
        "obj_seq": torch.stack([b["obj_seq"] for b in batch]),
        "obj_mask": torch.stack([b["obj_mask"] for b in batch]),
        "subject": torch.stack([b["subject"] for b in batch]),
        "object": torch.stack([b["object"] for b in batch]),
        "interaction_type": torch.stack([b["interaction_type"] for b in batch]),
        "label_valid": torch.stack([b["label_valid"] for b in batch]),
    }
    gazes = [b["gaze_seq"] for b in batch]
    out["gaze_seq"] = None if all(g is None for g in gazes) else torch.stack(gazes)
    return out


# ---------------------------------------------------------------- model


class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except (ImportError, RuntimeError, OSError) as e:
            raise SystemExit(
                "[train] 无法导入 mamba_ssm（Mamba 骨干依赖）。\n"
                "  最常见原因：GPU 未挂载进容器（triton 初始化报 '0 active drivers'，属预期，非安装错误）。\n"
                "  排查：1) nvidia-smi 确认 GPU 已挂载；2) conda activate caron_mamba；\n"
                "        3) 安装判据见 .agent/skills/mamba-training/SKILL.md\n"
                "  只想验证数据链路/基线：加 --baseline lstm（不依赖 mamba_ssm）。\n"
                f"  原始错误: {type(e).__name__}: {e}"
            ) from None
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mamba(self.norm(x)))


class MambaBackbone(nn.Module):
    def __init__(self, d_model, n_layer, d_state, d_conv, expand, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            ResidualMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layer)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # (B, W, d_model) -> (B, W, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LSTMBackbone(nn.Module):
    """Same-shape baseline: d_model/n_layer/dropout match the Mamba stack."""

    def __init__(self, d_model, n_layer, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=n_layer,
            batch_first=True, dropout=dropout if n_layer > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.norm(out)


def _accepts_kw(fn, name) -> bool:
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


class IntentModel(nn.Module):
    """encoder (task 5) -> temporal backbone -> decoder (task 5).

    The encoder emits per-frame entity tokens (B, W, E, d_model); the backbone
    models each entity's temporal trajectory (reshaped to (B*E, W, d_model))
    and the result is restored to (B, W, E, d_model) for the decoder.
    """

    def __init__(self, encoder, backbone, decoder):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.decoder = decoder
        self._enc_obj_mask = _accepts_kw(getattr(encoder, "forward", None), "obj_mask")
        self._enc_gaze = _accepts_kw(getattr(encoder, "forward", None), "gaze_seq")
        self._dec_obj_mask = _accepts_kw(getattr(decoder, "forward", None), "obj_mask")

    def forward(self, batch) -> dict:
        hand, obj = batch["hand_seq"], batch["obj_seq"]
        batch_size, window_size = hand.shape[:2]
        slot_mask = batch["obj_mask"]  # (B, n_max) bool, slot validity
        enc_kw = {}
        if self._enc_obj_mask:
            enc_kw["obj_mask"] = slot_mask[:, None, :].expand(batch_size, window_size, -1)
        if self._enc_gaze:
            enc_kw["gaze_seq"] = batch.get("gaze_seq")
        tokens = self.encoder(hand, obj, **enc_kw)
        if tokens.dim() == 4:  # (B, W, E, D) per-entity tokens
            entity_count, feature_dim = tokens.shape[2:]
            trajectories = tokens.permute(0, 2, 1, 3).reshape(
                batch_size * entity_count, window_size, feature_dim
            )
            seq = self.backbone(trajectories).reshape(
                batch_size, entity_count, window_size, feature_dim
            ).permute(0, 2, 1, 3)
        else:  # (B, W, D) already pooled per frame
            seq = self.backbone(tokens)
        dec_kw = {"obj_mask": slot_mask} if self._dec_obj_mask else {}
        return self.decoder(seq, **dec_kw)


def build_model(cfg: dict) -> IntentModel:
    try:
        from src.encoding import build_decoder, build_encoder
    except ImportError as e:
        raise SystemExit(
            "[train] 未找到 src.encoding.build_encoder/build_decoder（任务⑤ 的交付物）。\n"
            "  该模块尚未合入仓库时，可先用 python -m src.mamba.dev_smoke 验证数据链路。\n"
            f"  原始错误: {e}"
        ) from None
    m = cfg["model"]
    # ⑤ 的 cfg 为扁平键（cfg_get 兼容 dict/对象）：model.encoder 节内容上提合并
    encdec_cfg = {k: v for k, v in m.items() if k not in ("encoder", "decoder")}
    encdec_cfg.update(m.get("encoder") or {})
    encdec_cfg.update(m.get("decoder") or {})
    encdec_cfg.update({
        "use_gaze": bool(cfg["data"]["use_gaze"]),
        "interaction_types": INTERACTION_TYPES,
    })
    encoder = build_encoder(encdec_cfg)
    decoder = build_decoder(encdec_cfg)
    if m["backbone"] == "mamba":
        backbone = MambaBackbone(m["d_model"], m["n_layer"], m["d_state"],
                                 m["d_conv"], m["expand"], m["dropout"])
    else:
        backbone = LSTMBackbone(m["d_model"], m["n_layer"], m["dropout"])
    return IntentModel(encoder, backbone, decoder)


# ---------------------------------------------------------------- logging


class RunLogger:
    """TensorBoard scalars in <run_dir>; JSONL fallback if tb is missing."""

    def __init__(self, run_dir: Path):
        self.writer = None
        self._jsonl = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(run_dir))
        except ImportError:
            print("[train] 警告：未安装 tensorboard，指标回退写入 "
                  f"{run_dir/'metrics.jsonl'}（pip install tensorboard 可恢复 tb 日志）")
            self._jsonl = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def add_scalars(self, metrics: dict, step: int):
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        else:
            self._jsonl.write(json.dumps({"step": step, **metrics}) + "\n")
            self._jsonl.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self._jsonl is not None:
            self._jsonl.close()


# ---------------------------------------------------------------- train


def mask_object_logits(logits: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
    """logits (B, n_max+1); class n_max = none, never masked."""
    masked = logits.clone()
    masked[:, :-1] = masked[:, :-1].masked_fill(~obj_mask.bool(), float("-inf"))
    return masked


def _label_valid_mask(batch, allow_legacy_missing_label_valid, context):
    if "label_valid" in batch:
        return batch["label_valid"].bool(), False
    if not allow_legacy_missing_label_valid:
        raise ValueError(
            f"{context}: 缺少 label_valid，正式训练默认拒绝；仅 legacy/synthetic smoke "
            "可显式启用 allow_legacy_missing_label_valid，且不构成正式数据许可"
        )
    return torch.ones_like(batch["subject"], dtype=torch.bool), True


def compute_losses(logits: dict, batch: dict, weights: dict, *,
                   allow_legacy_missing_label_valid=False) -> tuple:
    """Average each head over valid targets; an unsupervised head contributes zero."""
    label_valid, legacy_missing = _label_valid_mask(
        batch, allow_legacy_missing_label_valid, "compute_losses"
    )
    if legacy_missing:
        _warn_legacy_missing_label_valid("compute_losses")
    losses = {}
    for head in ("subject", "object", "interaction_type"):
        head_logits = (mask_object_logits(logits[head], batch["obj_mask"])
                       if head == "object" else logits[head])
        valid_targets = (batch[head] != -1) & label_valid
        valid_count = valid_targets.sum().clamp_min(1)
        losses[head] = F.cross_entropy(
            head_logits[valid_targets], batch[head][valid_targets], reduction="sum"
        ) / valid_count
    total = sum(weights[head] * loss for head, loss in losses.items())
    return total, {head: loss.item() for head, loss in losses.items()}


def run_epoch(model, loader, device, cfg, amp_ok, optimizer=None):
    """Return metrics excluding -1 targets and label_valid=False samples.

    f1_macro_<head> averages class F1 over the head's full output vocabulary;
    classes absent from both targets and predictions contribute zero. acc_joint
    requires all three targets to be valid and all three predictions correct.
    acc_macro remains the mean of the three head accuracies for compatibility,
    not a class-level macro-F1. Empty denominators return zero, including losses.
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    weights = cfg["train"]["loss_weights"]
    log_interval = cfg["train"]["log_interval"]
    allow_legacy = cfg.get("data", {}).get("allow_legacy_missing_label_valid", False)
    legacy_missing_label_valid = False
    totals = {"subject": 0.0, "object": 0.0, "interaction_type": 0.0}
    correct = {"subject": 0, "object": 0, "interaction_type": 0}
    valid_counts = {head: 0 for head in totals}
    confusion_matrices = {}
    joint_correct = 0
    joint_count = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        label_valid, legacy_missing = _label_valid_mask(batch, allow_legacy, f"epoch batch {step}")
        if legacy_missing:
            batch["label_valid"] = label_valid
            if not legacy_missing_label_valid:
                _warn_legacy_missing_label_valid(f"epoch batch {step}")
            legacy_missing_label_valid = True
        valid_targets = {head: (batch[head] != -1) & label_valid for head in totals}
        batch_counts = {head: valid.sum().item() for head, valid in valid_targets.items()}
        if not any(batch_counts.values()):
            continue
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if amp_ok else nullcontext())
        with torch.set_grad_enabled(training), ctx:
            logits = model(batch)
            loss, parts = compute_losses(logits, batch, weights)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = cfg["train"]["grad_clip"]
            if clip and clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        for head in parts:
            totals[head] += parts[head] * batch_counts[head]
            valid_counts[head] += batch_counts[head]
        with torch.no_grad():
            predictions = {
                head: (mask_object_logits(logits[head].float(), batch["obj_mask"])
                       if head == "object" else logits[head]).argmax(-1)
                for head in totals
            }
            joint_valid = torch.stack(list(valid_targets.values())).all(dim=0)
            joint_matches = joint_valid.clone()
            for head, predicted in predictions.items():
                valid = valid_targets[head]
                matches = predicted == batch[head]
                correct[head] += (matches & valid).sum().item()
                joint_matches &= matches
                class_count = logits[head].shape[-1]
                counts = torch.bincount(
                    batch[head][valid] * class_count + predicted[valid],
                    minlength=class_count * class_count,
                ).reshape(class_count, class_count).cpu()
                if head not in confusion_matrices:
                    confusion_matrices[head] = counts
                else:
                    confusion_matrices[head] += counts
            joint_correct += joint_matches.sum().item()
            joint_count += joint_valid.sum().item()
        if training and log_interval and (step + 1) % log_interval == 0:
            print(f"    step {step + 1}/{len(loader)} loss={loss.item():.4f}")
    metrics = {}
    for head in totals:
        metrics[f"loss_{head}"] = totals[head] / max(valid_counts[head], 1)
        metrics[f"acc_{head}"] = correct[head] / max(valid_counts[head], 1)
        confusion = confusion_matrices.get(head)
        if confusion is None:
            metrics[f"f1_macro_{head}"] = 0.0
        else:
            denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
            class_f1 = 2.0 * confusion.diag().double() / denominator.clamp_min(1)
            metrics[f"f1_macro_{head}"] = class_f1.mean().item()
    metrics["loss"] = sum(weights[head] * metrics[f"loss_{head}"] for head in totals)
    metrics["acc_macro"] = sum(metrics[f"acc_{head}"] for head in totals) / len(totals)
    metrics["acc_joint"] = joint_correct / max(joint_count, 1)
    metrics["warning_legacy_missing_label_valid"] = float(
        allow_legacy or legacy_missing_label_valid
    )
    metrics["seconds"] = time.time() - t0
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch, cfg, best_metric):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "contract_version": cfg["contract_version"],
        "best_metric": best_metric,
    }, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.mamba.train",
                                description="任务⑥ Mamba/LSTM 意图预测训练（配置驱动）")
    p.add_argument("--config", required=True, help="configs/<run>.yaml 路径")
    p.add_argument("--baseline", choices=["mamba", "lstm"], default=None,
                   help="覆盖 config 的 model.backbone（lstm 不依赖 mamba_ssm，可用于无 GPU 冒烟）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.baseline and args.baseline != cfg["model"]["backbone"]:
        cfg["model"]["backbone"] = args.baseline
        cfg["run_name"] = f"{cfg['run_name']}_{args.baseline}"
        print(f"[train] --baseline {args.baseline} 覆盖骨干，run_name -> {cfg['run_name']}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    dev_cfg = cfg["device"]
    if dev_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_cfg)
    if cfg["model"]["backbone"] == "mamba" and device.type != "cuda":
        print("[train] 警告：mamba_ssm 需要 CUDA；当前 device 非 cuda，大概率会在 import/forward 失败。"
              "无 GPU 环境请用 --baseline lstm。")
    amp_ok = bool(cfg["train"]["amp_bf16"]) and device.type == "cuda"
    if cfg["train"]["amp_bf16"] and not amp_ok:
        print(f"[train] bf16 autocast 仅 CUDA 生效，当前 device={device.type}，本 run 使用 fp32")
    print(f"[train] device={device} backbone={cfg['model']['backbone']} amp_bf16={amp_ok}")

    data_root = Path(cfg["data"]["root"])
    n_max = int(cfg["model"]["decoder"]["n_max"])
    dcfg = cfg["data"]
    if dcfg["allow_legacy_missing_label_valid"]:
        _warn_legacy_missing_label_valid("配置已显式启用兼容模式")
    train_set = SamplesDataset(
        data_root, "train", n_max, dcfg["index"], dcfg["cache_in_memory"],
        dcfg["nan_to_num"], dcfg["use_gaze"], dcfg["allow_legacy_missing_label_valid"],
    )
    val_set = SamplesDataset(
        data_root, "val", n_max, dcfg["index"], dcfg["cache_in_memory"],
        dcfg["nan_to_num"], dcfg["use_gaze"], dcfg["allow_legacy_missing_label_valid"],
    )
    if len(train_set) == 0:
        raise SystemExit(f"[train] 训练集为空：{data_root}/train/*.npz（检查 data.root/data.index 配置）")
    if len(val_set) == 0:
        print("[train] 警告：val split 为空，跳过验证，best.pt 等同 last.pt")
    print(f"[train] 样本数 train={len(train_set)} val={len(val_set)} "
          f"cache={dcfg['cache_in_memory']} use_gaze={dcfg['use_gaze']}")

    tcfg = cfg["train"]
    loader_kw = dict(batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
                     collate_fn=collate_fn, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw) if len(val_set) else None

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] 参数量 {n_params / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])

    run_dir = Path(cfg["runs_root"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    logger = RunLogger(run_dir)

    metric_name = tcfg["selection_metric"]
    higher_better = metric_name == "val_acc_macro"
    key = "loss" if metric_name == "val_loss" else "acc_macro"
    best = -float("inf") if higher_better else float("inf")

    global_step = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        tr = run_epoch(model, train_loader, device, cfg, amp_ok, optimizer=optimizer)
        global_step += len(train_loader)
        logger.add_scalars({f"train/{k}": v for k, v in tr.items() if k != "seconds"}, epoch)
        line = (f"[epoch {epoch:03d}] train loss={tr['loss']:.4f} "
                f"acc(s/o/t)={tr['acc_subject']:.3f}/{tr['acc_object']:.3f}/{tr['acc_interaction_type']:.3f} "
                f"({tr['seconds']:.1f}s)")
        if val_loader is not None:
            va = run_epoch(model, val_loader, device, cfg, amp_ok)
            logger.add_scalars({f"val/{k}": v for k, v in va.items() if k != "seconds"}, epoch)
            line += (f" | val loss={va['loss']:.4f} "
                     f"acc(s/o/t)={va['acc_subject']:.3f}/{va['acc_object']:.3f}/{va['acc_interaction_type']:.3f}")
            current = va[key]
        else:
            current = tr[key]
        improved = current > best if higher_better else current < best
        if improved:
            best = current
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, cfg, best)
            line += f"  <- best ({metric_name}={best:.4f})"
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, cfg, best)
        print(line)

    logger.close()
    print(f"[train] 完成。run_dir={run_dir} best_{metric_name}={best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
