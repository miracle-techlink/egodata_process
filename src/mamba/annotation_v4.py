"""Explicit continuous-review positives/negatives over unchanged Mamba inputs."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.dataset.export_annotation_v4_windows import validate_review, reviewed_record
from src.mamba.annotation_v3 import EgoDemoAnnotationV3Dataset, SUBJECTS, sha256


def dense_targets(review, metadata, event_types, anchor, target):
    """Preserve whole visual identities; component mappings are not aliases."""
    objects = metadata["object_ids"]
    object_map = {name: i for i, name in enumerate(objects) if name}
    event_map = {(e["layer"], e["code"]): i for i, e in enumerate(event_types)}
    values = torch.zeros((len(objects) + 1, 5, len(event_types)), dtype=torch.float32)
    valid = torch.zeros_like(values, dtype=torch.bool)
    graph, unmapped = [], []
    records = []
    for record in review.get("state_segments", []):
        if (reviewed_record(record) and record["start_frame"] <= target < record["end_frame"]
                and target in record["evidence_frames"]):
            records.append((record, True, record["value"], "reviewed_state_at_target"))
    for record in review.get("transition_events", []):
        if not reviewed_record(record):
            continue
        start, end = record["boundary_frame_range"]
        if not (end > anchor + 1 and start <= target):
            continue
        exact = (end - start == 1 and record.get("frame_resolved") is True
                 and record.get("censored") is False and anchor < start <= target
                 and start in record["evidence_frames"])
        records.append((record, exact, 1 if exact else None,
                        "frame_resolved_transition_in_horizon" if exact else "uncertain_boundary_overlaps_horizon"))
    for record, temporal_valid, value, temporal_scope in records:
        identity = record.get("object_id")
        second = record.get("secondary_object_id")
        edge = {**record, "object_slot": object_map.get(identity), "secondary_object_slot": object_map.get(second),
                "participant_object_slots": [object_map.get(p) for p in record.get("participant_object_ids", [])],
                "temporal_supervision_valid": temporal_valid, "temporal_scope": temporal_scope,
                "supervision_value": value, "annotation_version": "assistant_annotation_v4_dense"}
        graph.append(edge)
        reason = None
        if not temporal_valid:
            reason = "transition_boundary_not_frame_resolved"
        elif second is not None or len(record.get("participant_object_ids", [])) > 1:
            reason = "multi_object_edge_preserved_in_graph"
        elif identity is None:
            reason = "unknown_object_not_an_explicit_dense_target"
        elif identity not in object_map:
            reason = "whole_entity_not_in_observed_slots_no_component_substitution"
        elif (record["layer"], record["type"]) not in event_map:
            reason = "event_outside_frozen_v2_vocab"
        elif record.get("subject") not in SUBJECTS or record.get("subject") in (None, "unknown"):
            reason = "subject_not_explicitly_identified"
        if reason:
            unmapped.append({"record_id": record.get("id"), "reason": reason, "object_id": identity})
            continue
        at = (object_map[identity], SUBJECTS[record["subject"]], event_map[(record["layer"], record["type"])])
        if valid[at] and values[at].item() != value:
            raise ValueError("Conflicting dense labels for the same prediction cell")
        values[at], valid[at] = value, True
    return values, valid, graph, unmapped


class EgoDemoAnnotationV4Dataset(EgoDemoAnnotationV3Dataset):
    """V2 observations, V3 exclusions, separately masked V4 dense supervision."""

    def __init__(self, root, annotation_root, dense_root, split="train", **kwargs):
        super().__init__(root, annotation_root, split=split, **kwargs)
        self.dense_root = Path(dense_root)
        self.dense_index_path = self.dense_root / "mamba_dense_index.json"
        self.dense_index = json.loads(self.dense_index_path.read_text())
        if (self.dense_index["base_dataset_index_sha256"] != sha256(self.root / "dataset_index.json")
                or self.dense_index["v3_overlay_index_sha256"] != sha256(self.annotation_root / "mamba_overlay_index.json")
                or self.dense_index["split_manifest_sha256"] != self.index["split_manifest_sha256"]):
            raise ValueError("Dense index belongs to different frozen V2/V3 sources")
        self.dense_entries = {e["sequence_id"]: e for e in self.dense_index["sequences"]}
        for row in self.windows:
            if "dense_review_sha256" in row:
                entry = self.dense_entries.get(row["sequence_id"])
                if (entry is None or entry["split"] != split or row["dense_review_sha256"] != entry["review_sha256"]
                        or row["dense_review_path"] != entry["review_path"]):
                    raise ValueError("Window dense review binding is stale or crosses split")
        for name, info in self.dense_index["splits"].items():
            expected_path = (self.dense_root / info["window_path"]).resolve()
            if Path(self.windows_path).resolve() == expected_path:
                if name != split or sha256(expected_path) != info["sha256"]:
                    raise ValueError("Published dense windows changed or cross split")
        self._dense_cache = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        uid = sample["metadata"]["sequence_id"]
        entry = self.dense_entries.get(uid)
        review = {"state_segments": [], "transition_events": []}
        if entry:
            if self._dense_cache is None or self._dense_cache[0] != uid:
                path = self.dense_root / entry["review_path"]
                if sha256(path) != entry["review_sha256"]:
                    raise ValueError("Dense review changed after index freeze")
                review = json.loads(path.read_text())
                validate_review(review, review["frame_count"])
                self._dense_cache = uid, review
            review = self._dense_cache[1]
        values, valid, graph, unmapped = dense_targets(review, sample["metadata"], self.event_types,
            sample["metadata"]["anchor_frame"], sample["metadata"]["target_frame"])
        sample["dense_event_values"] = values
        sample["dense_event_label_valid"] = valid
        sample["v3_graph_targets"] = sample["graph_targets"]
        sample["graph_targets"] = graph
        sample["metadata"].update(annotation_version="assistant_annotation_v4_dense",
            dense_review_available=entry is not None, unmapped_dense_targets=unmapped,
            dense_supervision_is_independent_of_v3_silver=True,
            dense_positive_cells=int(((values == 1) & valid).sum()),
            dense_negative_cells=int(((values == 0) & valid).sum()),
            partial_geometry_never_substituted_for_whole_object=True)
        return sample


def collate_annotation_v4(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    lists = {"metadata", "graph_targets", "v3_graph_targets"}
    return {key: ([s[key] for s in samples] if key in lists else torch.stack([s[key] for s in samples]))
            for key in samples[0]}
