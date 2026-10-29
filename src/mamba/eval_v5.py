"""Evaluation and benchmark utilities for the admitted V5 graph task.

Unknown graph cells are never treated as negatives.  The evaluator reports
cell-level and window-level metrics, including the train-seen/unseen split.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.mamba.graph_v5 import decode_graph, masked_graph_bce


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def _type_name(event_types, index):
    event = event_types[index]
    return f"{event['layer']}:{event.get('code', event.get('type'))}"


def _metric_from_counts(counts):
    return prf(counts["tp"], counts["fp"], counts["fn"])


class GraphMetricAccumulator:
    """Accumulate masked graph metrics without reading unknown cells."""

    def __init__(self, event_types, threshold=0.5):
        self.event_types = list(event_types)
        self.threshold = float(threshold)
        self.counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "known": 0, "positive": 0})
        self.window_known_positive = 0
        self.window_positive_hit = 0
        self.windows = 0
        self.loss_sum = 0.0
        self.loss_cells = 0

    def update(self, logits, values, label_valid, prediction_mask=None, metadata=None):
        if logits.shape != values.shape or logits.shape != label_valid.shape:
            raise ValueError("logits, values, and label_valid must have the same shape")
        if logits.ndim != 4 or label_valid.dtype != torch.bool:
            raise ValueError("graph metrics require [B,E,E,K] logits and bool mask")
        known = label_valid
        if prediction_mask is not None:
            if prediction_mask.shape != known.shape[:3]:
                raise ValueError("prediction_mask must be [B,E,E]")
            known = known & prediction_mask[..., None]
        if known.any():
            selected_logits = logits[known].float()
            selected_values = values[known].float()
            self.loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(
                selected_logits, selected_values, reduction="sum").item()
            self.loss_cells += int(known.sum())
        predicted = logits.float().sigmoid() >= self.threshold
        for b in range(logits.shape[0]):
            sample_known = known[b]
            true = values[b] >= 0.5
            hit = predicted[b] & true & sample_known
            true_positive = (true & sample_known).any()
            self.windows += 1
            if true_positive:
                self.window_known_positive += 1
                if hit.any():
                    self.window_positive_hit += 1
            for source, target, kind in sample_known.nonzero().tolist():
                key = _type_name(self.event_types, kind)
                cell = self.counts[key]
                actual = bool(true[source, target, kind])
                guess = bool(predicted[b, source, target, kind])
                cell["known"] += 1
                cell["positive"] += int(actual)
                if actual and guess:
                    cell["tp"] += 1
                elif not actual and guess:
                    cell["fp"] += 1
                elif actual:
                    cell["fn"] += 1
                else:
                    cell["tn"] += 1

    def summarize(self, seen_types=None):
        seen_types = set(seen_types or ())
        total = {k: sum(v[k] for v in self.counts.values()) for k in ("tp", "fp", "fn", "tn", "known", "positive")}
        all_metric = _metric_from_counts(total)
        per_type = {}
        for key, counts in sorted(self.counts.items()):
            metric = _metric_from_counts(counts)
            per_type[key] = {**counts, **metric, "seen": key in seen_types}
        layers = {}
        for key, counts in self.counts.items():
            layer = key.split(":", 1)[0]
            dest = layers.setdefault(layer, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "known": 0, "positive": 0})
            for field in dest:
                dest[field] += counts[field]
        layer_metrics = {layer: {**counts, **_metric_from_counts(counts)} for layer, counts in sorted(layers.items())}
        present = [key for key, value in per_type.items() if value["known"]]
        positive_types = [key for key, value in per_type.items() if value["positive"]]
        def macro(keys):
            return sum(per_type[k]["f1"] for k in keys) / len(keys) if keys else 0.0
        seen_present = [k for k in present if k in seen_types]
        unseen_present = [k for k in present if k not in seen_types]
        positive_rate = total["positive"] / total["known"] if total["known"] else 0.0
        return {
            "loss": self.loss_sum / self.loss_cells if self.loss_cells else 0.0,
            "counts": total,
            "micro": all_metric,
            "macro_f1": macro(present),
            "macro_f1_positive_types": macro(positive_types),
            "macro_f1_seen": macro(seen_present),
            "macro_f1_unseen": macro(unseen_present),
            "seen_types": seen_present,
            "unseen_types": unseen_present,
            "per_layer": layer_metrics,
            "per_type": per_type,
            "windows": self.windows,
            "windows_with_known_positive": self.window_known_positive,
            "windows_with_positive_hit": self.window_positive_hit,
            "window_positive_hit_rate": self.window_positive_hit / self.window_known_positive if self.window_known_positive else 0.0,
            "positive_rate": positive_rate,
        }


def seen_unseen_split(per_type, seen_types):
    seen = set(seen_types)
    return {
        "seen": {k: v for k, v in per_type.items() if k in seen},
        "unseen": {k: v for k, v in per_type.items() if k not in seen},
    }


def all_positive_baseline(counts):
    positive = counts.get("positive", 0)
    known = counts.get("known", 0)
    return {"precision": positive / known if known else 0.0, "recall": 1.0 if positive else 0.0,
            "f1": 2 * positive / (known + positive) if known + positive else 0.0}


def all_negative_baseline(counts):
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def scan_seen_types(dataset):
    """Scan explicit positive labels in train without using future model inputs."""
    seen = defaultdict(int)
    for i in range(len(dataset)):
        sample = dataset[i]
        for kind in sample["graph_label_valid"].nonzero()[:, -1].unique().tolist():
            values = sample["graph_values"][..., kind]
            valid = sample["graph_label_valid"][..., kind]
            seen[_type_name(dataset.event_types, kind)] += int((values.bool() & valid).sum())
    return dict(sorted(seen.items()))


def evaluate_model(model, loader, device, event_types, seen_types, threshold=0.5):
    accumulator = GraphMetricAccumulator(event_types, threshold)
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output = model(moved)
            accumulator.update(output["graph_logits"], moved["graph_values"], moved["graph_label_valid"],
                               output["graph_prediction_mask"], moved.get("metadata"))
    elapsed = time.perf_counter() - start
    result = accumulator.summarize(seen_types)
    result["seconds"] = elapsed
    result["windows_per_second"] = result["windows"] / elapsed if elapsed else 0.0
    result["all_positive_baseline"] = all_positive_baseline(result["counts"])
    result["all_negative_baseline"] = all_negative_baseline(result["counts"])
    return result


def _cli():
    parser = argparse.ArgumentParser(description="Evaluate an admitted V5 graph checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main(argv=None):
    args = _cli() if argv is None else argv
    if not isinstance(args, argparse.Namespace):
        raise TypeError("main expects parsed CLI args or no args")
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config_resolved.json").read_text())
    import yaml
    from torch.utils.data import DataLoader
    from src.mamba.dataset_v5 import EgoDemoGraphV5Dataset, collate_graph_v5
    from src.mamba.train_v5 import build_v5_model
    device_cfg = config["device"]
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    model = build_v5_model(config).to(device)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_set = EgoDemoGraphV5Dataset(config["data"]["root"], "train")
    seen = scan_seen_types(train_set)
    seen_names = set(seen)
    output_dir = run_dir / "eval"
    output_dir.mkdir(exist_ok=True)
    for split in args.splits:
        dataset = EgoDemoGraphV5Dataset(config["data"]["root"], split)
        loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False,
                            num_workers=0, collate_fn=collate_graph_v5)
        report = evaluate_model(model, loader, device, dataset.event_types, seen_names, args.threshold)
        report["split"] = split
        report["threshold"] = args.threshold
        report["caveats_zh"] = json.loads((Path(config["data"]["root"]) / "training_release.json").read_text()).get("evaluation_limit_zh", "")
        (output_dir / f"{split}_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    main()
