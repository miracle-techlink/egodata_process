"""EgoDemo v2: native-time, masked geometry and separate interaction vocabularies.

This module deliberately does not reinterpret the legacy seven-class samples.
Current exports are pending/offline estimates: loading requires diagnostic=True.
NaNs remain in Dataset outputs and are sanitized only inside the encoder.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


LAYERS = ("operation", "relation", "transition", "cooperation")
SUBJECT_NAMES_ZH = ("左手", "右手", "双手", "物体", "未知主体")
SCHEMA_VERSION = "egodemo_mamba_v2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def vocabulary_entries(vocab):
    """Flatten only for tensor storage; each entry retains its annotation layer."""
    entries = []
    for layer in LAYERS:
        spec = vocab[layer]
        codes, names = spec["codes"], spec["names_zh"]
        if len(codes) != len(names) or len(codes) != len(set(codes)):
            raise ValueError(f"Invalid vocabulary: {layer}")
        entries.extend({"layer": layer, "code": c, "name_zh": n}
                       for c, n in zip(codes, names))
    return entries


def anchor_points(points, world_to_anchor):
    return (np.asarray(points) @ world_to_anchor[:3, :3].T
            + world_to_anchor[:3, 3]).astype(np.float32)


def _mask_nan(values, valid):
    out = np.asarray(values, dtype=np.float32).copy()
    out[~valid] = np.nan
    return out


class EgoDemoV2Dataset(Dataset):
    """Load independent acquisition shards and construct observation-only slots.

    N/P truncation is deterministic: first observed frame then stable key. Neither
    event targets nor future observations can influence entity/point selection.
    sequence_cache_size bounds the number of decompressed acquisition shards.
    """

    def __init__(self, root, split="train", *, diagnostic=False, n_max=8,
                 points_per_object=16, sequence_cache_size=1, windows_path=None):
        self.root = Path(root)
        self.index = read_json(self.root / "dataset_index.json")
        if self.index.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Expected egodemo_mamba_v2 dataset")
        self.diagnostic = bool(diagnostic)
        self._require_access(self.index)
        self.n_max, self.points_per_object = int(n_max), int(points_per_object)
        if self.n_max < 1 or self.points_per_object < 1:
            raise ValueError("n_max and points_per_object must be positive")
        self.sequence_cache_size = max(0, int(sequence_cache_size))
        self._cache = OrderedDict()
        self.vocab = read_json(self.root / self.index.get("vocab_path", "vocab.json"))
        self.event_types = vocabulary_entries(self.vocab)
        self._event_index = {(e["layer"], e["code"]): i
                             for i, e in enumerate(self.event_types)}
        path = windows_path or self.index["splits"][split]["window_path"]
        self.windows_path = self._path(path)
        self.windows = [json.loads(line) for line in self.windows_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def _path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _require_access(self, metadata):
        if not self.diagnostic and (metadata.get("training_allowed") is not True
                                    or metadata.get("allowed_as_model_input") is not True):
            raise PermissionError("正式联合训练被拦截：标注未审核或几何非因果。"
                                  "仅结构诊断可显式使用 diagnostic=True；不得据此训练。")

    def __len__(self):
        return len(self.windows)

    def _load_sequence(self, row):
        key = row["sequence_path"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with np.load(self._path(key), allow_pickle=False) as z:
            sequence = {name: z[name] for name in z.files}
        metadata = read_json(self._path(row["metadata_path"]))
        self._require_access(metadata)
        events = read_json(self._path(row["events_path"]))["records"]
        value = sequence, metadata, events
        if self.sequence_cache_size:
            self._cache[key] = value
            while len(self._cache) > self.sequence_cache_size:
                self._cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        row = self.windows[index]
        z, metadata, records = self._load_sequence(row)
        start, end, target = (int(row[k]) for k in
                              ("start_frame", "end_frame", "target_frame"))
        anchor = int(row["anchor_frame"])
        if not (0 <= start < end <= target < len(z["timestamps"])) or anchor != end - 1:
            raise ValueError("Invalid observation/target boundary")
        camera = z["camera_to_world"][anchor].astype(np.float64)
        if (not np.isfinite(camera).all()
                or not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(camera[:3, :3]), 1, atol=1e-4)):
            raise ValueError("Anchor camera pose must be finite SE(3)")
        world_to_anchor = np.linalg.inv(camera)
        W, N, P = end - start, self.n_max, self.points_per_object
        observed = z["object_observed"][start:end].astype(bool)
        candidates = np.flatnonzero(observed.any(axis=0)).tolist()
        candidates.sort(key=lambda j: (int(np.flatnonzero(observed[:, j])[0]),
                                       str(z["object_ids"][j])))
        selected = candidates[:N]
        object_map = {obj: slot for slot, obj in enumerate(selected)}
        object_mask = np.arange(N) < len(selected)
        object_ids = [str(z["object_ids"][j]) for j in selected] + [""] * (N - len(selected))
        kind = np.zeros(N, np.int64)
        kind[:len(selected)] = z["object_kind"][selected]
        if np.any((kind < 0) | (kind > 3)):
            raise ValueError("Invalid object_kind")
        pose = np.full((W, N, 4, 4), np.nan, np.float32)
        pose_valid = np.zeros((W, N), bool)
        future_pose = np.full((N, 4, 4), np.nan, np.float32)
        future_pose_valid = np.zeros(N, bool)
        for slot, obj in enumerate(selected):
            valid = z["object_pose_valid"][start:end, obj].astype(bool)
            valid &= np.isfinite(z["object_pose_world"][start:end, obj]).all(axis=(-1, -2))
            pose_valid[:, slot] = valid
            pose[valid, slot] = (world_to_anchor @ z["object_pose_world"][start:end, obj][valid])
            future_pose_valid[slot] = (bool(z["object_pose_valid"][target, obj])
                                      and np.isfinite(z["object_pose_world"][target, obj]).all())
            if future_pose_valid[slot]:
                future_pose[slot] = world_to_anchor @ z["object_pose_world"][target, obj]

        surface = np.full((W, N, P, 3), np.nan, np.float32)
        surface_valid = np.zeros((W, N, P), bool)
        future_surface = np.full((N, P, 3), np.nan, np.float32)
        future_surface_valid = np.zeros((N, P), bool)
        point_mask = np.zeros((N, P), bool)
        point_indices = np.full((N, P), -1, np.int64)
        point_ids = [[""] * P for _ in range(N)]
        point_map = {}
        lo, hi = (int(z["surface_frame_ptr"][f]) for f in (start, end))
        obs_indices = range(lo, hi)
        for slot, obj in enumerate(selected):
            first = {}
            for s in obs_indices:
                if int(z["surface_object_index"][s]) == obj and np.isfinite(z["surface_world"][s]).all():
                    p = int(z["surface_point_index"][s])
                    if int(z["point_object_index"][p]) != obj:
                        raise ValueError("Surface point identity does not match object")
                    first.setdefault(p, int(z["surface_frame_index"][s]))
            points = sorted(first, key=lambda p: (first[p], str(z["point_keys"][p])))[:P]
            for pslot, p in enumerate(points):
                point_map[p] = (slot, pslot)
                point_mask[slot, pslot] = True
                point_indices[slot, pslot] = p
                point_ids[slot][pslot] = str(z["point_keys"][p])
        target_lo, target_hi = (int(z["surface_frame_ptr"][f]) for f in (target, target + 1))
        for begin, stop, is_target in ((lo, hi, False), (target_lo, target_hi, True)):
            for s in range(begin, stop):
                p = int(z["surface_point_index"][s])
                if p not in point_map or not np.isfinite(z["surface_world"][s]).all():
                    continue
                slot, pslot = point_map[p]
                if int(z["surface_object_index"][s]) != selected[slot]:
                    raise ValueError("Surface target changed object identity")
                xyz = anchor_points(z["surface_world"][s], world_to_anchor)
                if is_target:
                    future_surface[slot, pslot] = xyz
                    future_surface_valid[slot, pslot] = True
                else:
                    f = int(z["surface_frame_index"][s]) - start
                    if not 0 <= f < W:
                        raise ValueError("Invalid sparse surface frame pointer")
                    surface[f, slot, pslot] = xyz
                    surface_valid[f, slot, pslot] = True

        hands = anchor_points(z["hands_world"][start:end], world_to_anchor)
        hand_valid = z["hand_valid"][start:end].astype(bool) & np.isfinite(hands).all(axis=(-1, -2))
        future_hands = anchor_points(z["hands_world"][target], world_to_anchor)
        future_hand_valid = z["hand_valid"][target].astype(bool) & np.isfinite(future_hands).all(axis=(-1, -2))
        event_candidates = np.zeros((N + 1, 5, len(self.event_types)), np.float32)
        event_weak = np.zeros_like(event_candidates, dtype=bool)
        event_valid = np.zeros_like(event_candidates, dtype=bool)
        dropped_events = 0
        for record in records:
            event_key = record["layer"], record["type"]
            if event_key not in self._event_index:
                raise ValueError(f"Event not in vocabulary: {event_key}")
            a, b = int(record["start_frame"]), int(record["end_frame"])
            # Transition intervals are uncertain event-location ranges. A weak
            # candidate intersects the horizon; never invent an exact onset.
            active = ((b > anchor + 1 and a <= target) if record["layer"] == "transition"
                      else (a <= target < b))
            if not active:
                continue
            obj = int(record.get("object_index", -1))
            if obj >= 0 and obj not in object_map:
                dropped_events += 1
                continue  # excluded known object is not the unknown-object class
            subject = int(record.get("subject", -1))
            if subject not in (-1, 0, 1, 2, 3):
                raise ValueError("Invalid event subject")
            loc = (object_map[obj] if obj >= 0 else N,
                   subject if subject >= 0 else 4, self._event_index[event_key])
            event_candidates[loc] = 1.0
            event_weak[loc] |= bool(record.get("weak_positive", False))
            # Pending exports cannot be promoted by a stray per-record flag.
            event_valid[loc] |= (bool(record.get("label_valid", False))
                                 and metadata.get("training_allowed") is True
                                 and self.index.get("training_allowed") is True)

        arrays = {
            "hand_seq": _mask_nan(hands, hand_valid), "hand_valid": hand_valid,
            "obj_pose_seq": pose, "obj_valid": pose_valid,
            "surface_seq": surface, "surface_valid": surface_valid,
            "object_kind": kind, "object_mask": object_mask, "point_mask": point_mask,
            "target_hand": _mask_nan(future_hands, future_hand_valid),
            "target_hand_valid": future_hand_valid,
            "target_obj_pose": future_pose, "target_obj_valid": future_pose_valid,
            "target_surface": future_surface, "target_surface_valid": future_surface_valid,
            "event_candidates": event_candidates, "event_weak_mask": event_weak,
            "event_label_valid": event_valid,
            "timestamps": z["timestamps"][start:end].astype(np.float64),
            "relative_time": (z["timestamps"][start:end] - z["timestamps"][anchor]).astype(np.float32),
            "actual_horizon_s": np.asarray(z["timestamps"][target] - z["timestamps"][anchor], np.float32),
            "anchor_camera_to_world": camera,
        }
        result = {k: torch.from_numpy(v) for k, v in arrays.items()}
        entity_metadata = {entity["id"]: entity for entity in metadata.get("entities", [])}
        result["metadata"] = {"sequence_id": row["sequence_id"], "start_frame": start,
                              "end_frame": end, "target_frame": target,
                              "anchor_frame": anchor,
                              "object_ids": object_ids, "point_keys": point_ids,
                              "object_names_zh": [entity_metadata.get(obj, {}).get("name_zh", obj)
                                                  if obj else "填充槽位" for obj in object_ids],
                              "object_pose_scope": [entity_metadata.get(obj, {}).get("pose_scope", "unknown")
                                                    if obj else "padding" for obj in object_ids],
                              "object_indices": selected + [-1] * (N - len(selected)),
                              "point_indices": point_indices.tolist(),
                              "subject_names_zh": list(SUBJECT_NAMES_ZH),
                              "unknown_object_slot": N, "unknown_subject_slot": 4,
                              "event_absence_means": "unknown_not_negative",
                              "dropped_observed_objects": max(0, len(candidates) - N),
                              "dropped_target_events_unselected_objects": dropped_events,
                              "training_allowed": self.index.get("training_allowed") is True,
                              "allowed_as_model_input": self.index.get("allowed_as_model_input") is True,
                              "diagnostic": self.diagnostic}
        return result


def collate_egodemo_v2(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {key: ([s[key] for s in samples] if key == "metadata"
                  else torch.stack([s[key] for s in samples])) for key in samples[0]}


class EgoDemoV2Encoder(nn.Module):
    """Per-frame entity encoding; no target tensors or temporal lookahead used.

    Surface points have their own branch. Missing pose is never interpreted as
    a rigid object at the origin. Explicit validity and padding flags survive.
    """

    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.hand = nn.Sequential(nn.Linear(64, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pose = nn.Sequential(nn.Linear(17, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.point = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.kind = nn.Embedding(4, d_model)
        self.entity_type = nn.Embedding(4, d_model)
        self.time = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, batch):
        hands = batch["hand_seq"]
        B, W = hands.shape[:2]
        N = batch["obj_pose_seq"].shape[2]
        hvalid = batch["hand_valid"] & torch.isfinite(hands).all(dim=-1).all(dim=-1)
        hands = torch.nan_to_num(hands).masked_fill(~hvalid[..., None, None], 0)
        h = self.hand(torch.cat((hands.flatten(-2), hvalid[..., None].to(hands.dtype)), -1))
        pose = batch["obj_pose_seq"]
        pvalid = batch["obj_valid"] & torch.isfinite(pose).all(dim=-1).all(dim=-1)
        pose = torch.nan_to_num(pose).masked_fill(~pvalid[..., None, None], 0)
        xyz = batch["surface_seq"]
        svalid = batch["surface_valid"] & torch.isfinite(xyz).all(dim=-1)
        xyz = torch.nan_to_num(xyz).masked_fill(~svalid[..., None], 0)
        count = svalid.sum(-1, keepdim=True).clamp_min(1)
        centroid = xyz.sum(-2) / count
        point_features = self.point(xyz) * svalid[..., None]
        surface_features = point_features.sum(-2) / count
        center = torch.where(pvalid[..., None], pose[..., :3, 3], centroid)
        center_valid = pvalid | svalid.any(-1)
        delta = center[..., None, :] - hands[:, :, None, :, 0, :]
        pair_valid = center_valid[..., None] & hvalid[:, :, None, :]
        delta = delta.masked_fill(~pair_valid[..., None], 0).flatten(-2)
        # Translation + first two rotation columns + pose valid + surface ratio
        # + left/right wrist relative positions = 17 features.
        pfeatures = torch.cat((pose[..., :3, 3], pose[..., :3, :2].transpose(-1, -2).flatten(-2),
                               pvalid[..., None].to(hands.dtype),
                               svalid.float().mean(-1, keepdim=True), delta), -1)
        objects = self.pose(pfeatures) + surface_features + self.kind(batch["object_kind"])[:, None]
        object_mask = batch["object_mask"]
        entities = torch.cat((h, objects), dim=2)
        nonpadding = torch.cat((torch.ones((B, 2), dtype=torch.bool, device=hands.device), object_mask), -1)
        person = (entities * nonpadding[:, None, :, None]).sum(2, keepdim=True) / nonpadding.sum(-1)[:, None, None, None]
        tokens = torch.cat((person, entities), 2)
        type_ids = torch.tensor([0, 1, 2] + [3] * N, device=hands.device)
        tokens = tokens + self.entity_type(type_ids)[None, None]
        tokens = tokens + self.time(batch["relative_time"][..., None])[:, :, None, :]
        token_mask = torch.cat((torch.ones((B, 3), dtype=torch.bool, device=hands.device), object_mask), -1)
        flat = self.norm(tokens).reshape(B * W, N + 3, self.d_model)
        padding = (~token_mask)[:, None].expand(B, W, N + 3).reshape(B * W, N + 3)
        attended, _ = self.attention(flat, flat, flat, key_padding_mask=padding, need_weights=False)
        tokens = self.out_norm(tokens + attended.reshape(B, W, N + 3, self.d_model))
        return tokens.masked_fill(~token_mask[:, None, :, None], 0)


def rotation_6d_to_matrix(rotation):
    """Two predicted columns -> right-handed rotation, including zero fallback."""
    a, b = rotation[..., :3], rotation[..., 3:]
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    x = torch.where(a.norm(dim=-1, keepdim=True) > 1e-8, a, fallback)
    x = torch.nn.functional.normalize(x, dim=-1)
    y = b - (b * x).sum(-1, keepdim=True) * x
    axis = torch.nn.functional.one_hot(x.abs().argmin(-1), 3).to(x.dtype)
    y = torch.where(y.norm(dim=-1, keepdim=True) > 1e-8, y,
                    axis - (axis * x).sum(-1, keepdim=True) * x)
    y = torch.nn.functional.normalize(y, dim=-1)
    return torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), dim=-1)


class EgoDemoV2Mamba(nn.Module):
    """Real Mamba per entity over time, with explicit v2 future geometry heads."""

    def __init__(self, event_type_count, *, d_model=128, n_layer=4, d_state=16,
                 d_conv=4, expand=2, dropout=0.1, n_heads=4):
        super().__init__()
        from src.mamba.train import MambaBackbone
        self.encoder = EgoDemoV2Encoder(d_model, n_heads, dropout)
        self.backbone = MambaBackbone(d_model, n_layer, d_state, d_conv, expand, dropout)
        self.hand_head = nn.Linear(d_model, 63)
        self.pose_head = nn.Linear(d_model, 9)
        self.surface_head = nn.Sequential(nn.Linear(d_model + 3, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.event_head = nn.Linear(d_model, 5 * event_type_count)
        self.event_type_count = event_type_count

    def forward(self, batch):
        tokens = self.encoder(batch)
        B, W, E, D = tokens.shape
        # Never flatten (W,E) directly: Mamba must see one entity across time.
        temporal = tokens.permute(0, 2, 1, 3).reshape(B * E, W, D)
        last = self.backbone(temporal)[:, -1].reshape(B, E, D)
        hand = self.hand_head(last[:, 1:3]).reshape(B, 2, 21, 3)
        raw_pose = self.pose_head(last[:, 3:])
        pose = torch.eye(4, device=last.device, dtype=last.dtype).expand(B, E - 3, 4, 4).clone()
        pose[..., :3, :3] = rotation_6d_to_matrix(raw_pose[..., 3:])
        pose[..., :3, 3] = raw_pose[..., :3]
        # Each point prediction keeps its observation-selected key. The last
        # available position is used even when that point is hidden at anchor.
        observations = batch["surface_seq"]
        valid = batch["surface_valid"] & torch.isfinite(observations).all(-1)
        time = torch.arange(W, device=last.device)[None, :, None, None]
        last_index = torch.where(valid, time, -1).amax(1)
        point_seen = last_index >= 0
        gather = last_index.clamp_min(0)[:, None, ..., None].expand(-1, 1, -1, -1, 3)
        position = torch.gather(observations, 1, gather).squeeze(1)
        position = torch.nan_to_num(position).masked_fill(~point_seen[..., None], 0)
        context = last[:, 3:, None, :].expand(-1, -1, position.shape[2], -1)
        surface = position + self.surface_head(torch.cat((context, position), -1))
        # Unknown object uses the global token, never a padded object slot.
        event_context = torch.cat((last[:, 3:], last[:, :1]), dim=1)
        logits = self.event_head(event_context).reshape(B, E - 2, 5, self.event_type_count)
        return {"hand": hand, "obj_pose": pose, "obj_rotation_6d": raw_pose[..., 3:],
                "surface": surface, "event_logits": logits,
                "hand_prediction_mask": batch["hand_valid"].any(1),
                "obj_prediction_mask": batch["obj_valid"].any(1) & batch["object_mask"],
                "surface_prediction_mask": point_seen & batch["point_mask"],
                "event_object_mask": torch.cat((batch["object_mask"], torch.ones(
                    (B, 1), dtype=torch.bool, device=last.device)), dim=-1)}


def decode_events(logits, event_types, metadata, *, threshold=0.5, top_k=20):
    """Decode multilabel probabilities to Chinese, retaining unknown assignments.

    An untrained smoke model's decoded scores are not valid annotations.
    Padding is excluded; threshold never changes the human review status.
    """
    scores = logits.detach().float().sigmoid().cpu()
    decoded = []
    for b, meta in enumerate(metadata):
        N = len(meta["object_ids"])
        candidates = []
        for obj, subject, event in (scores[b] >= threshold).nonzero().tolist():
            if obj < N and not meta["object_ids"][obj]:
                continue
            candidates.append({**event_types[event], "subject": subject,
                               "subject_name_zh": SUBJECT_NAMES_ZH[subject],
                               "object_id": meta["object_ids"][obj] if obj < N else None,
                               "object_name_zh": (meta.get("object_names_zh", meta["object_ids"])[obj]
                                                  if obj < N else "未知物体"),
                               "object_slot": obj if obj < N else None,
                               "probability": float(scores[b, obj, subject, event]),
                               "review_status": "pending"})
        decoded.append(sorted(candidates, key=lambda x: -x["probability"])[:top_k])
    return decoded
