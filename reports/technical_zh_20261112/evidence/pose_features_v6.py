"""Attach v10 world poses to V5 object slots by Chinese-name matching.

Maps visual_* seed identities to joint_annotations obj_XXX entities via
name_zh equality (normalized), producing per-slot world poses for a new
feature block. Poses remain offline_noncausal candidates — the new features
carry a dedicated flag so the causal inputs stay distinguishable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _normalize(name: str) -> str:
    import re
    return re.sub(r"[\s\-_（）()/、0-9]+", "", str(name or "")).lower()


def _name_similarity(a: str, b: str) -> float:
    """Token-overlap score in [0,1]: exact > substring > char-bigram overlap."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.9
    # char bigram Jaccard, so 胶带/黄色胶带卷 and 绿色袋子/绿色快递袋 score high
    def bigrams(x):
        return {x[i:i+2] for i in range(len(x)-1)} or {x}
    ga, gb = bigrams(na), bigrams(nb)
    inter = len(ga & gb)
    return inter / len(ga | gb)


class PoseTable:
    """Per-sequence (T, N_slots, 4, 4) world pose lookup for visual identities."""

    def __init__(self, refine_root: str, uid: str, visual_identities: list[str],
                 seed_names: dict[str, str], frame_count: int):
        folder = Path(refine_root) / "joint_annotations" / uid
        self.ok = False
        self.poses = None
        if not folder.exists():
            return
        try:
            z = np.load(folder / "objects.npz", allow_pickle=False)
            proposal = json.loads((folder / "proposal.json").read_text())
        except Exception:
            return
        ids = [str(v) for v in z["object_ids"]]
        raw_names = {e["id"]: str(e.get("name_zh", "")) for e in proposal.get("entities", [])}
        all_poses = z["poses"]            # (N, T, 4, 4) camera frame
        valid = z["pose_valid"]           # (N, T)
        # Camera→world from the base sequence for the shared timeline.
        base = json.loads((Path(refine_root) / ".." / "egodemo_mamba_v2_300_20261023" /
                           "dataset_index.json").read_text())
        row = {r["sequence_id"]: r for r in base["sequences"]}[uid]
        seq = np.load(Path(refine_root).parent / "egodemo_mamba_v2_300_20261023" / row["sequence_path"])
        c2w = seq["camera_to_world"]      # (T,4,4)
        if all_poses.shape[1] != frame_count:
            return
        # world pose = c2w @ pose: (T,4,4) batched over N entities
        world = np.einsum("tij,ntjk->ntik", c2w, all_poses)
        # Candidate pool: all v10 names split on the alias separator " / "
        candidates = []
        for n, oid in enumerate(ids):
            for alias in raw_names.get(oid, "").split("/"):
                alias = alias.strip()
                if alias:
                    candidates.append((n, alias))
        slots = []
        for identity in visual_identities:
            target = seed_names.get(identity, identity)
            best, best_score = None, 0.0
            for n, alias in candidates:
                score = _name_similarity(target, alias)
                if score > best_score:
                    best, best_score = n, score
            if best_score < 0.34:  # below bigram-chance floor → unknown
                slots.append((None, None))
            else:
                slots.append((world[best], valid[best]))
        if all(p is None for p, _ in slots):
            return
        self.poses = slots
        self.ok = True

    def world_series(self, frame_count: int):
        """Return (world (T,N,4,4), valid (T,N)) aligned to the visual slots."""
        if not self.ok:
            return None
        worlds, valids = [], []
        for pose, valid in self.poses:
            if pose is None:
                worlds.append(np.full((frame_count, 4, 4), np.nan, np.float32))
                valids.append(np.zeros(frame_count, bool))
            else:
                worlds.append(pose.astype(np.float32))
                valids.append(valid.astype(bool))
        return np.stack(worlds, 1), np.stack(valids, 1)

    def features(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (pos  (N,3), rot6 (N,6), valid (N,)) at the last frame."""
        if not self.ok:
            return None
        out_pos, out_rot, out_valid = [], [], []
        for pose, valid in self.poses:
            if pose is None or not valid.any():
                out_pos.append(np.full(3, np.nan)); out_rot.append(np.full(6, np.nan)); out_valid.append(False)
                continue
            f = int(np.flatnonzero(valid)[-1])
            m = pose[f]
            r = m[:3, :3]
            # rotation-6d: first two columns
            rot6 = np.concatenate([r[:, 0], r[:, 1]])
            out_pos.append(m[:3, 3]); out_rot.append(rot6); out_valid.append(True)
        return np.asarray(out_pos, np.float32), np.asarray(out_rot, np.float32), np.asarray(out_valid, bool)
