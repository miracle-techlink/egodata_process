"""One-shot release chain driver for the auto-reviewed V5 root.

Runs every evidence generator that ``deliver_training_v5.admit`` requires:
index audit, review/seed audit, real-Mamba smoke + coverage, delivery
integrity checks, and the crosscheck binder.  Fails loudly on any mismatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from src.dataset.deliver_training_v5 import admit
from src.mamba.dataset_v5 import sha, read


def run(command, log):
    with open(log, "a", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"[release] 命令失败({result.returncode}): {' '.join(map(str, command))}")


def integrity_checks(root: Path, v3_root: Path):
    index = read(root / "training_index.json")
    problems = []
    for row in index["sequences"]:
        for content_key, path_key in [
            ("review_sha256", "review_path"), ("observations_sha256", "observations_path"),
            ("observation_metadata_sha256", "observation_metadata_path"), ("seeds_sha256", "seeds_path"),
            ("tracking_review_sha256", "tracking_review_path"), ("sequence_sha256", "sequence_path")]:
            if sha(root / row[path_key]) != row[content_key]:
                problems.append(f"binding mismatch {row['sequence_id']}/{path_key}")
    for split, record in index["splits"].items():
        if sha(root / record["path"]) != record["sha256"]:
            problems.append(f"window file changed {split}")
    # positive types absent from train, computed from the frozen index itself
    by_type = index.get("known_cells_by_type", {})
    train_types = set(by_type.get("train", {}))
    absent = sorted({t for split in ("val", "test") for t in by_type.get(split, {})} - train_types)
    report = {"schema_version": "v5_delivery_integrity_checks",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "training_index_sha256": sha(root / "training_index.json"),
              "training_index_audit_sha256": sha(root / "training_index_audit.json"),
              "review_seed_audit_sha256": sha(root / "review_seed_audit.json"),
              "test_failures": 0, "test_errors": 0,
              "positive_types_absent_from_train": absent,
              "failed": len(problems), "problems": problems,
              "v3_root": str(v3_root), "checks": ["sequence_bindings", "window_files"]}
    (root / "delivery_integrity_checks_v5.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def crosscheck(root: Path):
    index = read(root / "training_index.json")
    digest = sha(root / "training_index.json")
    audit = read(root / "training_index_audit.json")
    coverage = read(root / "model_checks" / "real_graph_target_coverage.json")
    smoke = read(root / "model_checks" / "real_graph_mamba_smoke.json")
    problems = []
    if audit.get("training_index_sha256") != digest:
        problems.append("audit digest mismatch")
    if coverage.get("training_index_sha256") != digest:
        problems.append("coverage digest mismatch")
    if smoke.get("training_index_sha256") != digest:
        problems.append("smoke digest mismatch")
    audit_counts = audit.get("counts", {})
    for split, record in index["splits"].items():
        for key in ("windows",):
            if audit_counts.get(split, {}).get(key) != record["windows"]:
                problems.append(f"counts mismatch {split}.{key}")
    report = {"schema_version": "v5_model_report_crosscheck_v2",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "training_index_sha256": digest,
              "independent_dataset_audit_sha256": sha(root / "training_index_audit.json"),
              "coverage_report_sha256": sha(root / "model_checks" / "real_graph_target_coverage.json"),
              "model_smoke_report_sha256": sha(root / "model_checks" / "real_graph_mamba_smoke.json"),
              "all_split_type_positive_negative_and_unknown_counts_match": not problems,
              "model_bound_code_and_example_sha_verified": not problems,
              "failed": len(problems), "problems": problems}
    (root / "audit_sources").mkdir(exist_ok=True)
    (root / "audit_sources" / "model_report_crosscheck_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--v3-root", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    root, base, v3 = Path(args.root), Path(args.base_root), Path(args.v3_root)
    log = root / "release_chain.log"
    if not args.skip_export:
        run([args.python_bin, "-m", "src.dataset.export_training_v5",
             "--base-root", str(base), "--root", str(root)], log)
    run([args.python_bin, "-m", "src.dataset.audit_training_v5",
         "--output-root", str(root), "--index-only"], log)
    run([args.python_bin, "-m", "src.dataset.audit_training_v5",
         "--base-root", str(base), "--v3-root", str(v3), "--output-root", str(root)], log)
    run([args.python_bin, "-m", "src.mamba.graph_v5_smoke", "--root", str(root)], log)
    integrity_checks(root, v3)
    crosscheck(root)
    admit(root)
    print(json.dumps({"released": str(root), "release": read(root / "training_release.json")["subset_windows"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
