"""Evidence-scoped assistant annotations over immutable EgoDemo v2 geometry.

Silver labels remain separate from human labels. A sampled positive is never
expanded over an interval; unobserved entries are unknown, not negative.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.mamba.egodemo_v2 import EgoDemoV2Dataset


SUBJECTS = {"left_hand": 0, "right_hand": 1, "both_hands": 2, "object": 3,
            "unknown": 4, None: 4}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_window_bindings(windows, base_index, annotation_records, split):
    """An explicit window path must not silently select another frozen split."""
    sources = {r["sequence_id"]: r for r in base_index["sequences"]}
    for row in windows:
        uid = row["sequence_id"]
        if uid not in sources or uid not in annotation_records:
            raise ValueError("Window references an unknown acquisition")
        source = sources[uid]
        if not row["split"] == source["split"] == annotation_records[uid]["split"] == split:
            raise ValueError("Window path crosses the requested frozen split")
        for key in ("sequence_path", "metadata_path", "events_path"):
            if row[key] != source[key]:
                raise ValueError("Window source path differs from frozen acquisition")
        if (row["sequence_sha256"] != source["sha256"]["sequence"]
                or row["source_selection_sha256"] != base_index["source_selection_sha256"]
                or row["split_manifest_sha256"] != base_index["split_manifest_sha256"]):
            raise ValueError("Window provenance differs from frozen source")
        if "annotation_sha256" in row and row["annotation_sha256"] != annotation_records[uid]["sha256"]:
            raise ValueError("Window annotation binding is stale")


def temporal_support(record, anchor, target):
    """Return only positives justified at target or entirely within horizon."""
    if (record.get("annotation_source") != "assistant_visual_review"
            or record.get("review_status") != "assistant_reviewed"
            or record.get("silver_label_valid") is not True):
        return False
    frames = record.get("evidence_frames", [])
    if not frames or any(type(f) is not int for f in frames):
        return False
    if record.get("layer") == "transition":
        boundary = record.get("boundary_frame_range", record.get("boundary_range"))
        if boundary is not None:
            if len(boundary) != 2 or not all(type(f) is int for f in boundary):
                return False
            start, end = boundary
            # Partial overlap of uncertain onset range is not a known positive.
            return anchor < start < end and end - 1 <= target
        return not record.get("censored", True) and any(anchor < f <= target for f in frames)
    return record.get("temporal_coverage") == "sampled_points" and target in frames


def silver_targets(records, metadata, event_types, anchor, target):
    """Keep graph edges with two objects outside the single-object event head."""
    objects = metadata["object_ids"]
    object_map = {identity: slot for slot, identity in enumerate(objects) if identity}
    event_map = {(r["layer"], r["code"]): i for i, r in enumerate(event_types)}
    candidate = torch.zeros((len(objects) + 1, 5, len(event_types)), dtype=torch.float32)
    valid = torch.zeros_like(candidate, dtype=torch.bool)
    graph, unmapped = [], []
    for record in records:
        if not temporal_support(record, anchor, target):
            continue
        edge = dict(record)
        identity = record.get("object_id", record.get("target_entity_id"))
        second = record.get("secondary_object_id")
        edge["object_slot"] = object_map.get(identity)
        edge["secondary_object_slot"] = object_map.get(second)
        edge["participant_object_slots"] = [object_map.get(p) for p in record.get("participant_object_ids", [])]
        graph.append(edge)
        if second is not None or len(record.get("participant_object_ids", [])) > 1:
            unmapped.append({"reason": "two_object_edge_preserved_in_graph", "record_id": record.get("id")})
            continue
        if identity is not None and identity not in object_map:
            unmapped.append({"reason": "known_entity_not_observed_in_window", "object_id": identity})
            continue
        kind = (record.get("layer"), record.get("type"))
        if kind not in event_map:
            unmapped.append({"reason": "event_outside_frozen_v2_vocab", "type": kind})
            continue
        subject = record.get("subject")
        if subject not in SUBJECTS:
            unmapped.append({"reason": "unresolved_subject_format", "subject": subject})
            continue
        at = (object_map[identity] if identity is not None else len(objects),
              SUBJECTS[subject], event_map[kind])
        candidate[at] = 1
        valid[at] = True
    return candidate, valid, graph, unmapped


def exclude_rejected_geometry(sample, exclusions):
    """Apply explicit review rejection without overwriting frozen v2 files."""
    excluded = {r["object_id"] for r in exclusions if r["scope"] == "entity_until_reidentified"}
    applied = []
    for slot, identity in enumerate(sample["metadata"]["object_ids"]):
        if not identity or identity not in excluded:
            continue
        applied.append(identity)
        sample["obj_pose_seq"][:, slot] = float("nan")
        sample["surface_seq"][:, slot] = float("nan")
        sample["target_obj_pose"][slot] = float("nan")
        sample["target_surface"][slot] = float("nan")
        sample["obj_valid"][:, slot] = False
        sample["surface_valid"][:, slot] = False
        for name in ("object_mask", "point_mask", "target_obj_valid", "target_surface_valid"):
            sample[name][slot] = False
        for name in ("event_candidates", "event_weak_mask", "event_label_valid"):
            sample[name][slot] = 0
        sample["object_kind"][slot] = 0
        sample["metadata"]["object_ids"][slot] = ""
        sample["metadata"]["object_names_zh"][slot] = ""
        sample["metadata"]["object_indices"][slot] = -1
        sample["metadata"]["point_keys"][slot] = [""] * len(sample["metadata"]["point_keys"][slot])
        sample["metadata"]["point_indices"][slot] = [-1] * len(sample["metadata"]["point_indices"][slot])
    sample["metadata"]["geometry_exclusions_applied"] = applied


class EgoDemoAnnotationV3Dataset(EgoDemoV2Dataset):
    """Same numeric inputs, additional independently masked silver targets."""

    def __init__(self, root, annotation_root, split="train", **kwargs):
        super().__init__(root, split, **kwargs)
        self.annotation_root = Path(annotation_root)
        overlay = json.loads((self.annotation_root / "mamba_overlay_index.json").read_text())
        if overlay["base_dataset_index_sha256"] != sha256(self.root / "dataset_index.json"):
            raise ValueError("Annotation overlay belongs to a different v2 dataset")
        self.annotation_records = {r["sequence_id"]: r for r in overlay["sequences"]}
        validate_window_bindings(self.windows, self.index, self.annotation_records, split)
        self._annotation_cache = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        uid = sample["metadata"]["sequence_id"]
        entry = self.annotation_records[uid]
        path = self.annotation_root / entry["annotation_path"]
        if self._annotation_cache is None or self._annotation_cache[0] != uid:
            if sha256(path) != entry["sha256"]:
                raise ValueError("Annotation content changed after delivery")
            annotation = json.loads(path.read_text())
            self._annotation_cache = uid, annotation
        annotation = self._annotation_cache[1]
        # V2 regex candidates include known noun/adjective false matches. Keep
        # their values for provenance, but do not supervise the v3 overlay with
        # stale weak masks; reviewed silver positives have a separate mask.
        sample["metadata"]["base_weak_candidate_count"] = int(sample["event_weak_mask"].sum())
        sample["metadata"]["base_weak_supervision_disabled"] = True
        sample["event_weak_mask"].zero_()
        exclude_rejected_geometry(sample, annotation.get("geometry_exclusions", []))
        candidate, valid, graph, unmapped = silver_targets(
            annotation["silver_event_records"], sample["metadata"], self.event_types,
            sample["metadata"]["anchor_frame"], sample["metadata"]["target_frame"])
        sample["silver_event_candidates"] = candidate
        sample["silver_event_label_valid"] = valid
        sample["graph_targets"] = graph
        sample["metadata"]["annotation_version"] = "assistant_annotation_v3"
        sample["metadata"]["unmapped_silver_targets"] = unmapped
        sample["metadata"]["annotation_human_verified"] = False
        return sample


def collate_annotation_v3(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {key: ([sample[key] for sample in samples] if key in ("metadata", "graph_targets")
                  else torch.stack([sample[key] for sample in samples])) for key in samples[0]}
