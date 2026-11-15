"""Immutable acquisition/environment splits for the private EgoDemo v2 export.

Source environment IDs are grouping evidence, not verified participant identities.
No window, geometry quality score, or future label participates in this split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile


SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _identifier(value: object) -> str | None:
    if value is None or str(value).strip().lower() in {"", "unknown", "none", "null", "pending"}:
        return None
    return str(value).strip()


def build_groups(records: list[dict], subject_identity_verified: bool = False) -> list[dict]:
    """Union shared environment, scene, acquisition and content aliases transitively."""
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        keys = ["environment_id", "scene_id", "source_episode_uuid"]
        if subject_identity_verified or record.get("subject_identity_verified") is True:
            keys.append("subject_id")
        tokens = [(key, _identifier(record.get(key))) for key in keys]
        # Data/video duplicates must not cross partitions even if their UUID differs.
        tokens.extend(("content_" + item["role"], item.get("expected_sha256"))
                      for item in record.get("planned_assets", [])
                      if item["role"] in {"data", "head_left_video"})
        for kind, value in tokens:
            if value is None:
                continue
            token = (kind, value)
            if token in seen:
                parents[find(index)] = find(seen[token])
            else:
                seen[token] = index
    grouped: dict[int, list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)
    result = []
    for members in grouped.values():
        ids = sorted(r["source_episode_uuid"] for r in members)
        result.append({"group_id": canonical_hash(ids), "sequence_ids": ids,
                       "weight": len(ids),
                       "scene_ids": sorted({_identifier(r.get("scene_id")) for r in members} - {None}),
                       "environment_ids": sorted({_identifier(r.get("environment_id")) for r in members} - {None})})
    return sorted(result, key=lambda row: row["group_id"])


def freeze_split(selection_path: str | Path, output_dir: str | Path, *,
                 prior_splits_paths: tuple | list = (), seed: int = 20261023) -> dict:
    selection_path, output_dir = Path(selection_path), Path(output_dir)
    source = json.loads(selection_path.read_text())
    records = source["records"]
    ids = [row["source_episode_uuid"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("source selection has duplicate acquisition UUIDs")
    groups = build_groups(records, source.get("subject_identity_verified") is True)
    seen_ids = {row["source_episode_uuid"] for row in records if row.get("previously_downloaded")}
    prior, prior_memberships = [], defaultdict(list)
    for value in prior_splits_paths:
        path = Path(value)
        split = json.loads(path.read_text())
        prior.append({"filename": path.name, "sha256": sha256(path)})
        for name in SPLITS:
            for uid in split.get(name, []):
                if not isinstance(uid, str):
                    raise ValueError("prior split must contain UUID strings")
                seen_ids.add(uid)
                prior_memberships[uid].append(name)
        # The older human_dataset manifest used records; candidate is not a holdout.
        for row in split.get("records", []):
            if row.get("split") in SPLITS and row.get("sequence_id", "").startswith("egodemo_"):
                uid = row["sequence_id"].rsplit("_", 1)[-1]
                seen_ids.add(uid)
                prior_memberships[uid].append(row["split"])
    counts = dict.fromkeys(SPLITS, 0)
    target = dict(zip(SPLITS, (len(ids) * .7, len(ids) * .15, len(ids) * .15)))
    for group in groups:
        group["previously_seen_sequence_ids"] = sorted(set(group["sequence_ids"]) & seen_ids)
        if group["previously_seen_sequence_ids"]:
            group["split"] = "train"
            group["assignment_reason"] = "previous_development_exposure_or_linked_environment"
            counts["train"] += group["weight"]
    pending = sorted((group for group in groups if "split" not in group),
                     key=lambda group: (-group["weight"], canonical_hash([seed, group["group_id"]])))
    for group in pending:
        # Minimize acquisition-count error without breaking a connected component.
        def cost(name: str) -> tuple:
            trial = dict(counts)
            trial[name] += group["weight"]
            return (sum((trial[s] - target[s]) ** 2 / max(target[s], 1) for s in SPLITS),
                    canonical_hash([seed, group["group_id"], name]))
        name = min(SPLITS, key=cost)
        group.update(split=name, assignment_reason="deterministic_group_count_balance")
        counts[name] += group["weight"]
    assignments = {uid: group["split"] for group in groups for uid in group["sequence_ids"]}
    sequence_group = {uid: group["group_id"] for group in groups for uid in group["sequence_ids"]}
    bindings = []
    for record in sorted(records, key=lambda row: row["source_episode_uuid"]):
        uid = record["source_episode_uuid"]
        staging = selection_path.parent / "staging" / uid
        bindings.append({"sequence_id": uid, "split": assignments[uid], "group_id": sequence_group[uid],
                         "scene_id": record.get("scene_id"), "environment_id": record.get("environment_id"),
                         "subject_id": record.get("subject_id"), "task_name": record["task_name"],
                         "native_frame_count": record["manifest_frame_count"],
                         "staging_sequence_sha256": sha256(staging / "sequence.npz"),
                         "staging_metadata_sha256": sha256(staging / "meta.json"),
                         "planned_assets_sha256": canonical_hash(record["planned_assets"]),
                         "previous_split_membership": sorted(set(prior_memberships[uid]))})
    manifest = {"schema_version": "egodemo_mamba_v2_split", "seed": seed,
                "source_selection_sha256": sha256(selection_path),
                "policy": "union of source environment/scene/acquisition/content aliases before windowing",
                "assignments": assignments, "sequence_group": sequence_group, "groups": groups,
                "records": bindings, "counts": counts, "target_ratios": dict(zip(SPLITS, (.7, .15, .15))),
                "group_counts": dict(Counter(group["split"] for group in groups)),
                "prior_split_bindings": prior, "normalization_fit_split": "train",
                "subject_disjoint_verified": False, "training_allowed": False,
                "residual_risks": ["No verified participant identity; cross-subject generalization is not established.",
                    "Source environment/scene identifiers are not independently verified physical-environment identities.",
                    "All sequences underwent automated offline preprocessing; purposefully selected visual QA is not a blind benchmark.",
                    "Source task names and object categories can overlap splits; this is not an unseen-task/object evaluation.",
                    "Previously downloaded or previously used pilot acquisitions and linked environments are development-only train groups."]}
    destination = output_dir / "splits" / "split_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError("immutable split differs; use a new output version")
        return manifest
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd, temp = tempfile.mkstemp(prefix=".split-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
        os.link(temp, destination)  # Exclusive publication; never clobber a frozen split.
    finally:
        os.unlink(temp)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-splits", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    result = freeze_split(args.source_selection, args.output_dir,
                          prior_splits_paths=args.prior_splits, seed=args.seed)
    print(json.dumps({"counts": result["counts"], "group_counts": result["group_counts"],
                      "source_selection_sha256": result["source_selection_sha256"]}))


if __name__ == "__main__":
    main()
