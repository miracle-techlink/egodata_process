"""RSI loop for the admitted V5 graph benchmark (research-self-improve cycle).

Each round: read ``genome.yaml`` -> mutate within the whitelist -> train one
candidate -> evaluate on val only -> replace the genome if the score improves
beyond the noise floor.  Test split is never touched by the loop.

Discipline (borrowed from piper-dagger-rsi / gpu-rsi-plugin):
- only numeric/sweep hyperparameters may mutate; paths, admission fields,
  ``input_dim``, ``event_type_count``, window semantics are immutable;
- a crash, a non-zero exit, or a missing benchmark is ``ok=false`` and can
  never win;
- improvements below the noise floor (default 2% relative) do not count;
- every candidate keeps its own run directory, so the loop is fully
  resumable: existing ``eval/val_benchmark.json`` marks a finished job.

Usage:
    python -m src.mamba.rsi_v5 --loop-root runs/rsi_v5 [--rounds 8] [--concurrency 2]
    python -m src.mamba.rsi_v5 --loop-root runs/rsi_v5 --report   # aggregate only
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

MUTABLE_NUMERIC = {
    "model": {"d_model": [64, 128, 256], "n_layer": [1, 2, 3, 4], "dropout": [0.0, 0.1, 0.2]},
    "train": {
        "batch_size": [8, 16, 32],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 0.01, 0.1],
        "selection_metric": ["val_loss", "val_micro_f1", "val_macro_f1_seen"],
    },
}
BACKBONES = ["mamba", "lstm", "transformer"]
NOISE_FLOOR = 0.02  # <2% relative improvement is noise, not progress
IMMUTABLE_KEYS = {"input_dim", "event_type_count", "root", "runs_root", "run_name", "seed"}


def deep_get(cfg, path):
    for key in path:
        cfg = cfg[key]
    return cfg


def deep_set(cfg, path, value):
    for key in path[:-1]:
        cfg = cfg[key]
    cfg[path[-1]] = value


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def mutate(genome, rng):
    """Return one candidate derived from the genome; never touches immutable keys."""
    candidate = copy.deepcopy(genome)
    candidate["model"]["backbone"] = rng.choice(BACKBONES)
    section, options = rng.choice(list(MUTABLE_NUMERIC.items()))
    key, values = rng.choice(list(options.items()))
    deep_set(candidate, [section, key], rng.choice(values))
    return candidate


def fingerprint(candidate):
    parts = [candidate["model"]["backbone"]]
    for section, options in MUTABLE_NUMERIC.items():
        for key in options:
            parts.append(f"{key}={deep_get(candidate, [section, key])}")
    return "g_" + "_".join(parts).replace("/", "-").replace(" ", "")


def run_one(python_bin, config_path, run_dir, log_path, device="auto"):
    command = [
        python_bin, "-m", "src.mamba.train_v5",
        "--config", str(config_path), "--runs-root", str(run_dir.parent),
        "--run-name", run_dir.name,
    ]
    if device != "auto":
        command += ["--device", device]
    with open(log_path, "a", encoding="utf-8") as log:
        started = time.time()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=".")
    return result.returncode, time.time() - started


def evaluate_one(python_bin, run_dir, log_path):
    command = [
        python_bin, "-m", "src.mamba.eval_v5",
        "--run-dir", str(run_dir), "--checkpoint", "best.pt", "--splits", "val",
    ]
    with open(log_path, "a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return None
    report = run_dir / "eval" / "val_benchmark.json"
    if not report.is_file():
        return None
    return json.loads(report.read_text())


def score_of(benchmark):
    """Primary score: val micro-F1; val loss kept for the report."""
    return float(benchmark["micro"]["f1"])


def better(new_score, old_score):
    if old_score is None:
        return True
    return new_score > old_score * (1.0 + NOISE_FLOOR)


def one_round(loop_root, python_bin, round_id, device, epochs=None):
    rng = random.Random(round_id * 9973 + 17)
    genome = load_yaml(loop_root / "genome.yaml")
    candidate = mutate(genome, rng)
    if epochs:
        candidate["train"]["epochs"] = int(epochs)
    name = fingerprint(candidate)
    run_dir = loop_root / "candidates" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "candidate.yaml"
    write_yaml(config_path, candidate)
    log_path = run_dir / "round.log"
    score_path = run_dir / "score.json"

    if score_path.is_file():  # already finished in an earlier interrupted round
        record = json.loads(score_path.read_text())
        print(f"[rsi] {name} already scored {record.get('score')}")
        return record

    print(f"[rsi] round {round_id}: training {name} (backbone={candidate['model']['backbone']})")
    returncode, seconds = run_one(python_bin, config_path, run_dir, log_path, device)
    if returncode != 0:
        record = {"ok": False, "name": name, "round": round_id,
                  "error": f"train exit {returncode}"}
        score_path.write_text(json.dumps(record, ensure_ascii=False))
        return record
    benchmark = evaluate_one(python_bin, run_dir, log_path)
    if benchmark is None:
        record = {"ok": False, "name": name, "round": round_id,
                  "error": "eval failed or benchmark missing"}
        score_path.write_text(json.dumps(record, ensure_ascii=False))
        return record
    score = score_of(benchmark)
    record = {
        "ok": True, "name": name, "round": round_id, "score": score,
        "val_loss": benchmark["loss"], "macro_f1_seen": benchmark["macro_f1_seen"],
        "macro_f1_unseen": benchmark["macro_f1_unseen"], "seconds": seconds,
        "config": candidate,
    }
    score_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def select(loop_root):
    """Promote the best scoring candidate into genome.yaml (noise floor applies)."""
    genome = load_yaml(loop_root / "genome.yaml")
    history_path = loop_root / "history.jsonl"
    best_score = None
    best_record = None
    if history_path.is_file():
        for line in history_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("ok") and (best_score is None or record["score"] > best_score):
                best_score, best_record = record["score"], record
    if best_record is not None and better(best_record["score"], score_genome(loop_root)):
        promoted = {k: v for k, v in best_record["config"].items()}
        write_yaml(loop_root / "genome.yaml", promoted)
        print(f"[rsi] genome promoted: {best_record['name']} score={best_record['score']:.4f}")
    else:
        print("[rsi] no promotion this round")


def score_genome(loop_root):
    """Score of the current genome if it has already been evaluated."""
    genome = load_yaml(loop_root / "genome.yaml")
    name = fingerprint(genome)
    score_path = loop_root / "candidates" / name / "score.json"
    if not score_path.is_file():
        return None
    record = json.loads(score_path.read_text())
    return record.get("score") if record.get("ok") else None


def report(loop_root):
    rows = []
    for score_path in sorted((loop_root / "candidates").glob("*/score.json")):
        record = json.loads(score_path.read_text())
        rows.append(record)
    rows.sort(key=lambda r: -(r.get("score") or -1))
    lines = ["# RSI loop report", "", "| rank | name | ok | score | val_loss | macro_seen | rounds |", "|---|---|---|---|---|---|---|"]
    for rank, row in enumerate(rows, 1):
        lines.append(f"| {rank} | {row.get('name')} | {row.get('ok')} | {row.get('score', '-')} | "
                     f"{row.get('val_loss', '-')} | {row.get('macro_f1_seen', '-')} | {row.get('round')} |")
    (loop_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return rows


def ensure_layout(loop_root, seed_config):
    loop_root.mkdir(parents=True, exist_ok=True)
    (loop_root / "candidates").mkdir(exist_ok=True)
    if not (loop_root / "genome.yaml").is_file():
        write_yaml(loop_root / "genome.yaml", seed_config)
    history = loop_root / "history.jsonl"
    if not history.exists():
        history.touch()
    return load_yaml(loop_root / "genome.yaml")


def append_history(loop_root, record):
    with open(loop_root / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.mamba.rsi_v5",
                                description="V5 benchmark RSI loop (val-only selection, test untouched)")
    p.add_argument("--loop-root", required=True)
    p.add_argument("--seed-config", default="configs/graph_v5_base.yaml")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--epochs", type=int, default=None, help="override train epochs per candidate")
    p.add_argument("--device", default="auto")
    p.add_argument("--report", action="store_true", help="aggregate existing candidates only")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    loop_root = Path(args.loop_root)
    if args.report:
        report(loop_root)
        return 0
    seed = load_yaml(args.seed_config)
    ensure_layout(loop_root, seed)
    for round_id in range(1, args.rounds + 1):
        record = one_round(loop_root, args.python_bin, round_id, args.device, args.epochs)
        append_history(loop_root, record)
        if record.get("ok"):
            select(loop_root)
    report(loop_root)
    print("[rsi] done. Test split untouched; run eval_v5 --splits val test manually on the winner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
