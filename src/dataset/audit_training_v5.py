"""Independent integrity audit for V5 continuous reviews and causal seeds.

This verifies frozen files, native pixels, identities and label semantics. It
does not certify an assistant's visual interpretation as independent human
ground truth. V4's uncertain/censored events are never silently promoted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from src.dataset.audit_dense_annotation_v4 import (
    audit_continuous_evidence, audit_dense_review_labels, digest,
    entity_references, range_mask, read, require, resolved,
)


def audit_transitions(review, viewed_frames, frame_count, vocab):
    """Check occurrence ranges without confusing precision with censoring."""
    reviewed = set(np.flatnonzero(range_mask(review["reviewed_ranges"], frame_count)))
    ids = {e["entity_id"] for e in review["entities"]} | {"left_hand", "right_hand"}
    counts = Counter()
    for event in review["transition_events"]:
        require(event.get("annotation_source") == "assistant_visual_review"
                and event.get("human_label_valid") is False
                and event.get("sampling_scope") == "continuous_native_frames_reviewed",
                "transition lacks assistant continuous provenance")
        require(entity_references(event) <= ids, "transition has unknown entity")
        lo, hi = event["boundary_frame_range"]
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= frame_count,
                "transition boundary is outside native timeline")
        evidence = event["evidence_frames"]
        require(evidence and all(type(f) is int for f in evidence)
                and len(evidence) == len(set(evidence))
                and set(evidence) <= reviewed & set(viewed_frames)
                and set(range(lo, hi)) <= reviewed & set(viewed_frames),
                "transition cites unreviewed or duplicate native evidence")
        family = event.get("event_family", "physical_interaction")
        precise = event.get("frame_resolved") is True
        occurred = event.get("occurrence_confirmed") is True
        if family == "physical_interaction":
            require(event["type"] in vocab["transition"]["codes"], "unknown physical transition type")
        else:
            require(family == "visual_signal" and event["type"] not in vocab["transition"]["codes"],
                    "visual signal masquerades as physical transition")
        if precise:
            require(event.get("silver_label_valid") is True and event.get("censored") is False
                    and hi == lo + 1 and lo > 0 and {lo - 1, lo} <= set(evidence),
                    "precise transition lacks adjacent before/after evidence")
            counts[family + "_frame_resolved"] += 1
        elif not event.get("censored", True):
            require(occurred and event.get("before_after_observed") is True
                    and event.get("silver_label_valid") is True,
                    "noncensored range lacks confirmed observed occurrence")
        if occurred:
            before, after = event.get("before_evidence_frame"), event.get("after_evidence_frame")
            require(event.get("censored") is False and type(before) is int and type(after) is int,
                    "confirmed occurrence has no explicit before/after frames")
            require(0 <= before < lo <= after < hi <= frame_count
                    and lo <= before + 1 and hi >= after + 1,
                    "half-open event range omits possible first-after frame")
            require(set(range(before, after + 1)) <= reviewed & set(viewed_frames)
                    and {before, after} <= set(evidence),
                    "bounded occurrence crosses unreviewed before/after evidence")
            require(bool(event.get("reason_zh")), "occurrence lacks visual rationale")
            counts[family + "_confirmed_occurrence_ranges"] += 1
        elif not precise:
            require(event.get("censored") is True, "legacy uncertainty was promoted without adjudication")
            counts[family + "_censored_ranges"] += 1
        require(event.get("value", 1) == 1, "uncertain or absent event encoded as a negative")
        counts["transition_records"] += 1
    return dict(counts)


def audit_review_labels(review, viewed_frames, frame_count, vocab):
    """Reuse V4 state checks, then apply the explicit V5 event contract."""
    state_review = dict(review, transition_events=[])
    result = audit_dense_review_labels(state_review, viewed_frames, frame_count)
    require(review.get("frame_count") == frame_count, "review/native frame counts differ")
    entity_ids = [e["entity_id"] for e in review["entities"]]
    require(len(entity_ids) == len(set(entity_ids)), "duplicate review entity ID")
    for entity in review["entities"]:
        if entity.get("material_kind") in {"deformable", "nonrigid", "non_rigid"}:
            require(entity.get("whole_object_6d_valid") is not True,
                    "deformable entity claims whole-object rigid 6D truth")
    for state in review["state_segments"]:
        require(state["layer"] in {"operation", "relation", "cooperation"}
                and state["type"] in vocab[state["layer"]]["codes"], "state not in frozen layered vocabulary")
    result["counts"].update(audit_transitions(review, viewed_frames, frame_count, vocab))
    return result


class EvidenceAuditor:
    """Cache verified manifests/pixels; explicit viewed pages stay per review."""

    def __init__(self, verify_pixels=True):
        self.verify_pixels = verify_pixels
        self._manifests = {}
        self._videos = {}
        self._fullframes = {}

    def verify_seed_full_frame(self, root, seed, manifest, frame):
        """Reproduce the declared current-frame grid preview, never infer its frame from its name."""
        import cv2
        from PIL import Image, ImageDraw

        path = resolved(root, seed["evidence_frame_path"])
        expected = seed["evidence_frame_sha256"]
        require(digest(path) == expected, "seed full-frame preview SHA changed")
        video = Path(manifest["source_bindings"]["video_path"])
        stat = video.stat()
        key = (expected, str(video.resolve()), stat.st_size, stat.st_mtime_ns,
               frame["frame_index"], frame["decoded_native_rgb_sha256"])
        if self.verify_pixels and key not in self._fullframes:
            cap = cv2.VideoCapture(str(video))
            try:
                require(cap.isOpened(), "cannot decode full-frame seed video")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_index"])
                before = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ok, bgr = cap.read()
                require(ok and abs(before - frame["frame_index"]) < .1,
                        "full-frame seed decoder is on another native frame")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                require(hashlib.sha256(rgb.tobytes()).hexdigest() == frame["decoded_native_rgb_sha256"],
                        "full-frame seed native RGB differs")
            finally:
                cap.release()
            width, height = manifest["native_image_size"]
            require(rgb.shape == (height, width, 3), "seed preview native size differs")
            image = Image.fromarray(rgb).resize((width // 2, height // 2))
            draw = ImageDraw.Draw(image)
            style = dict(fill="white", stroke_width=1, stroke_fill="black")
            for x in range(0, width, 200):
                draw.line((x / 2, 0, x / 2, height // 2), fill=(100, 180, 160), width=1)
                draw.text((x / 2 + 2, 4), str(x), **style)
            for y in range(0, height, 200):
                draw.line((0, y / 2, width // 2, y / 2), fill=(100, 180, 160), width=1)
                draw.text((2, y / 2 + 2), str(y), **style)
            draw.text((650, 705), f"FRAME {frame['frame_index']} | native {width}x{height}", **style)
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=94)
            require(hashlib.sha256(encoded.getvalue()).hexdigest() == expected,
                    "seed full-frame preview is not a rendering of its current native frame")
            self._fullframes[key] = True
        return dict(path=str(path), resolved_path=str(path.resolve()), sha256=expected,
                    frame_index=frame["frame_index"], native_pixels_verified=self.verify_pixels,
                    rendering="half_size_native_rgb_grid_200px_default_pillow_jpeg94")

    def verify(self, root, reference, timestamps, sequence_id):
        root = Path(root)
        path = resolved(root, reference["path"])
        expected = reference["sha256"]
        require(digest(path) == expected, "evidence manifest SHA changed")
        manifest = read(path)
        require(manifest["schema_version"] == "assistant_annotation_v4_dense_evidence"
                and manifest["sequence_id"] == sequence_id, "evidence sequence/schema mismatch")
        frames = manifest["frames"]
        start, end = manifest["native_frame_range"]
        require(0 <= start < end <= len(timestamps)
                and [f["frame_index"] for f in frames] == list(range(start, end)),
                "evidence native range skips or invents frames")
        pages = {p["path"]: p for p in manifest["pages"]}
        require(len(pages) == len(manifest["pages"]), "duplicate evidence page")
        viewed = reference["reviewed_page_paths"]
        require(viewed and len(viewed) == len(set(viewed)) and set(viewed) <= set(pages),
                "unknown/duplicate viewed evidence page")
        require(reference.get("reviewed_page_sha256", [pages[p]["sha256"] for p in viewed])
                == [pages[p]["sha256"] for p in viewed], "reviewed page SHA list changed")
        binding = manifest["source_bindings"]
        artifacts = [Path(binding[k]) for k in ("metadata_path", "hands_path", "video_path")]
        artifacts += [resolved(root, p["path"]) for p in manifest["pages"]]
        artifacts += [resolved(root, f["crop_path"]) for f in frames]
        # Content is hashed on first verification. A later write to any bound
        # artifact invalidates the expensive native-pixel decode cache.
        fingerprints = tuple((str(p.resolve()), s.st_ino, s.st_size, s.st_mtime_ns)
                             for p in artifacts for s in [p.stat()])
        key = (str(path.resolve()), expected, fingerprints, tuple(timestamps[start:end]))
        if key not in self._manifests:
            binding = manifest["source_bindings"]
            require(digest(binding["metadata_path"]) == binding["metadata_sha256"]
                    and digest(binding["hands_path"]) == binding["hands_sha256"],
                    "evidence source metadata/hands changed")
            metadata = read(binding["metadata_path"])
            require(metadata["video_path"] == binding["video_path"]
                    and metadata["video_sha256"] == binding["video_sha256"], "evidence video binding changed")
            require(manifest["native_image_size"] == metadata["image_size"], "evidence native image size changed")
            for page in manifest["pages"]:
                require(digest(resolved(root, page["path"])) == page["sha256"], "evidence page SHA changed")
                require(page["frame_indices"] == [c["frame_index"] for c in page["cells"]],
                        "page cell/frame identity differs")
            for f in frames:
                require(abs(f["timestamp_s"] - timestamps[f["frame_index"]]) <= 1e-9,
                        "evidence native timestamp differs")
                require(digest(resolved(root, f["crop_path"])) == f["crop_sha256"], "evidence crop SHA changed")
                require(manifest["pages"][f["page_index"]]["cells"][f["page_cell_index"]]["frame_index"]
                        == f["frame_index"], "native frame is bound to wrong page cell")
            if self.verify_pixels:
                video = Path(binding["video_path"])
                video_key = (str(video.resolve()), video.stat().st_size, video.stat().st_mtime_ns)
                if video_key not in self._videos:
                    self._videos[video_key] = digest(video)
                require(self._videos[video_key] == binding["video_sha256"], "actual source video SHA changed")
                # Verify every artifact page, independently of the reviewer's
                # claimed viewed subset; do not promote that subset here.
                all_pages = dict(reference, reviewed_page_paths=list(pages),
                                 reviewed_page_sha256=[pages[p]["sha256"] for p in pages])
                audit_continuous_evidence(root, all_pages, timestamps)
            self._manifests[key] = manifest
        seen = {f for p in viewed for f in pages[p]["frame_indices"]}
        require(seen <= set(range(start, end)), "page claims absent native frame")
        return seen, manifest, {"path": str(path), "resolved_path": str(path.resolve()), "sha256": expected,
            "native_pixels_verified": self.verify_pixels, "artifact_native_frame_count": len(frames),
            "reviewer_declared_page_count": len(viewed), "reviewer_declared_native_frame_count": len(seen)}


def audit_review(review_path, evidence_root, *, timestamps, vocab, expected_uid,
                 expected_split=None, base_index_sha256=None, v3_overlay_sha256=None,
                 evidence_auditor=None):
    path = Path(review_path)
    frozen_sha = digest(path)
    review = read(path)
    require(review["sequence_id"] == expected_uid, "review sequence ID changed")
    require(review.get("split", expected_split) == expected_split, "review changes frozen split")
    source = review["source_bindings"]
    if base_index_sha256 is not None:
        require(source["v2_dataset_index_sha256"] == base_index_sha256, "review base index SHA changed")
    if v3_overlay_sha256 is not None:
        require(source["v3_overlay_index_sha256"] == v3_overlay_sha256, "review V3 overlay SHA changed")
    require(digest(source["v3_annotation_path"]) == source["v3_annotation_sha256"], "review V3 annotation changed")
    annotation = read(source["v3_annotation_path"])
    original_ids = {r["entity_id"] for r in annotation.get("entities", [])}
    require(all(a["alias_id"] in original_ids for a in review.get("visual_identity_aliases", [])),
            "visual identity alias is absent from frozen V3 entities")
    parent = source if "parent_review_path" in source else review
    if "parent_review_path" in parent or "parent_review_sha256" in parent:
        require("parent_review_path" in parent and "parent_review_sha256" in parent
                and digest(parent["parent_review_path"]) == parent["parent_review_sha256"], "frozen parent review changed")
        require(read(parent["parent_review_path"])["sequence_id"] == expected_uid, "wrong parent acquisition")
    auditor = evidence_auditor or EvidenceAuditor()
    seen, evidence, by_manifest = set(), [], {}
    refs = review["evidence_manifests"]
    require(len({r["path"] for r in refs}) == len(refs), "duplicate review evidence manifest")
    for reference in refs:
        frames, _, record = auditor.verify(evidence_root, reference, timestamps, expected_uid)
        seen.update(frames)
        by_manifest[reference["path"]] = frames
        evidence.append(record)
    for row in review["reviewed_ranges"]:
        if isinstance(row, dict):
            require(row.get("evidence_scope") == "continuous_native_frames_reviewed",
                    "review range lacks continuous evidence scope")
            paths = row["evidence_manifest_paths"]
            require(paths and set(paths) <= set(by_manifest), "range cites unknown evidence manifest")
            support = set().union(*(by_manifest[p] for p in paths))
            require(set(range(row["start_frame"], row["end_frame"])) <= support,
                    "review range is not supported by its referenced pages")
    require(review.get("reviewed_gallery_page_count", sum(r["reviewer_declared_page_count"] for r in evidence))
            == sum(r["reviewer_declared_page_count"] for r in evidence), "review page count overclaim")
    result = audit_review_labels(review, seen, len(timestamps), vocab)
    require(digest(path) == frozen_sha, "review changed during audit")
    result.update(sequence_id=expected_uid, split=expected_split, path=str(path), sha256=frozen_sha,
                  evidence=evidence, independent_human_semantic_review=False,
                  semantic_author_is_auditor=review.get("reviewer_agent") == "/root/split_audit")
    return result


def audit_seeds(seed_path, evidence_root, *, timestamps, review, evidence_auditor=None):
    path = Path(seed_path)
    frozen_sha = digest(path)
    doc = read(path)
    require(doc["schema_version"] == "assistant_annotation_v5_visual_seeds"
            and doc["sequence_id"] == review["sequence_id"], "visual seed sequence/schema mismatch")
    require(doc.get("human_label_valid") is not True
            and not any(doc.get(k) is True for k in ("whole_object_6d_available", "whole_object_6d_certified", "whole_object_6d_valid")),
            "2D seeds promoted to human or 6D truth")
    w, h = doc["native_image_size"]
    require(type(w) is int and type(h) is int and w > 0 and h > 0, "invalid seed native image size")
    ids = {e["entity_id"] for e in review["entities"]}
    auditor = evidence_auditor or EvidenceAuditor()
    seen, by_entity, evidence, fullframes = set(), Counter(), {}, {}
    for seed in doc["seeds"]:
        f, identity = seed["frame_index"], seed["entity_id"]
        require(type(f) is int and 0 <= f < len(timestamps) and identity in ids and identity.startswith("visual_"),
                "seed uses unknown entity or nonnative frame")
        require((f, identity) not in seen, "duplicate current-frame entity seed")
        seen.add((f, identity)); by_entity[identity] += 1
        require(seed.get("annotation_source", doc.get("annotation_source")) == "assistant_visual_review"
                and seed.get("human_label_valid", doc.get("human_label_valid")) is False
                and seed.get("reason_zh"), "seed lacks actual visual annotation provenance")
        box = seed["bbox_xyxy_native"]
        require(len(box) == 4 and np.isfinite(box).all(), "seed bbox is not finite")
        x0, y0, x1, y1 = box
        require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "seed bbox outside native image")
        require(seed.get("activation_frame", f) == f and seed.get("first_valid_frame", f) == f,
                "seed was activated before its observed current frame")
        mp = resolved(evidence_root, seed["evidence_manifest_path"])
        require(digest(mp) == seed["evidence_manifest_sha256"], "seed evidence manifest SHA changed")
        manifest = read(mp)
        frame_records = [r for r in manifest["frames"] if r["frame_index"] == f]
        require(len(frame_records) == 1 and manifest["native_image_size"] == [w, h],
                "seed cites different native frame or image size")
        frame = frame_records[0]
        page = manifest["pages"][frame["page_index"]]
        reference = dict(path=str(mp), sha256=seed["evidence_manifest_sha256"],
                         reviewed_page_paths=[page["path"]], reviewed_page_sha256=[page["sha256"]])
        _, _, checked = auditor.verify(evidence_root, reference, timestamps, review["sequence_id"])
        evidence[str(mp)] = checked
        require(seed.get("evidence_crop_path", frame["crop_path"]) == frame["crop_path"]
                and seed.get("evidence_crop_sha256", frame["crop_sha256"]) == frame["crop_sha256"],
                "seed cites a future or different frame crop")
        cx0, cy0, cx1, cy1 = frame["crop_xyxy_native"]
        if "evidence_frame_path" in seed:
            checked_frame = auditor.verify_seed_full_frame(evidence_root, seed, manifest, frame)
            fullframes[checked_frame["path"]] = checked_frame
        else:
            require(cx0 <= x0 < x1 <= cx1 and cy0 <= y0 < y1 <= cy1,
                    "whole seed box extends outside its current visual evidence")
    require(digest(path) == frozen_sha, "seeds changed during audit")
    return dict(path=str(path), sha256=frozen_sha, sequence_id=review["sequence_id"],
        seed_count=len(seen), current_seed_frame_count=len({f for f, _ in seen}),
        seed_count_by_entity=dict(by_entity), evidence=list(evidence.values()), full_frame_evidence=list(fullframes.values()),
        native_pixels_verified=auditor.verify_pixels, independent_human_semantic_review=False)


def audit_review_seed_dataset(base_root, v3_root, output_root, *, verify_pixels=True, expected_sequences=None):
    base_root, v3_root, output = map(Path, (base_root, v3_root, output_root))
    base_path, overlay_path = base_root / "dataset_index.json", v3_root / "mamba_overlay_index.json"
    base_sha, overlay_sha = digest(base_path), digest(overlay_path)
    base, overlay = read(base_path), read(overlay_path)
    entries = {e["sequence_id"]: e for e in base["sequences"]}
    annotations = {e["sequence_id"]: e for e in overlay["sequences"]}
    vocab_path = base_root / "vocab.json"
    vocab = read(vocab_path)
    require(digest(vocab_path) == base["vocab_sha256"], "frozen vocabulary changed")
    auditor, records, totals = EvidenceAuditor(verify_pixels), [], Counter()
    paths = sorted((output / "dense_reviews").glob("*.json"))
    require(paths and (expected_sequences is None or len(paths) == expected_sequences), "review acquisition coverage incomplete")
    for path in paths:
        uid = path.stem
        require(uid in entries and uid in annotations, "review acquisition outside frozen selection")
        source = entries[uid]
        sequence_path = base_root / source["sequence_path"]
        require(digest(sequence_path) == source["sha256"]["sequence"], "actual base sequence changed")
        with np.load(sequence_path, allow_pickle=False) as arrays:
            timestamps = arrays["timestamps"]
        require(len(timestamps) == source["frame_count"] and np.isfinite(timestamps).all()
                and np.all(np.diff(timestamps) > 0), "base native timeline invalid")
        review = read(path)
        require(review["source_bindings"]["v3_annotation_sha256"] == annotations[uid]["sha256"],
                "review annotation differs from frozen overlay")
        checked = audit_review(path, output, timestamps=timestamps, vocab=vocab, expected_uid=uid,
            expected_split=source["split"], base_index_sha256=base_sha, v3_overlay_sha256=overlay_sha,
            evidence_auditor=auditor)
        seeds = audit_seeds(output / "visual_observations" / uid / "seeds.json", output,
                           timestamps=timestamps, review=review, evidence_auditor=auditor)
        seed_doc = read(output / "visual_observations" / uid / "seeds.json")
        identity_sources = {checked['sha256'], review.get('parent_review_sha256'),
                            review['source_bindings'].get('parent_review_sha256')}
        require(seed_doc.get('review_sha256') in identity_sources,
                'seed identity source is neither current nor frozen parent review')
        checked["seeds"] = seeds
        records.append(checked)
        totals.update(checked["counts"])
        totals.update(sequences=1, seed_count=seeds["seed_count"])
        totals[source["split"] + "_sequences"] += 1
    require(digest(base_path) == base_sha and digest(overlay_path) == overlay_sha, "frozen indices changed during audit")
    result = dict(schema_version="assistant_annotation_v5_review_seed_audit", created_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=digest(__file__), base_index_sha256=base_sha, v3_overlay_index_sha256=overlay_sha,
        source_selection_sha256=base["source_selection_sha256"], split_manifest_sha256=base["split_manifest_sha256"],
        counts=dict(totals), records=records, failed=0, native_pixels_verified=verify_pixels,
        full_annotation_complete=False, formal_training_allowed=False, independent_human_semantic_review=False,
        scope_zh="独立核V5产物及冻结来源、真实原生像素/裁剪/页SHA、连续覆盖、显式正负与未知、实体框时刻及范围事件证据。作者自身标注只接受结构/像素溯源审计，不冒充独立人工语义真值；尚未核训练索引/因果跟踪和模型加载。")
    (output / "review_seed_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def checked_artifact(root, row, path_key, sha_key):
    path = resolved(root, row[path_key])
    require(digest(path) == row[sha_key], f"bound {path_key} SHA changed")
    return path


def independent_graph(review, entity_ids, types, anchor, target, observed_entities):
    """Independent dense target construction; unknown remains masked out."""
    shape = (len(entity_ids), len(entity_ids), len(types))
    values, mask = np.zeros(shape, np.float32), np.zeros(shape, bool)
    slots = {v: i for i, v in enumerate(entity_ids) if v}
    vocabulary = {(v['layer'], v['code']): i for i, v in enumerate(types)}
    for r in review['state_segments'] + review['transition_events']:
        kind = (r['layer'], r['type'])
        if kind not in vocabulary or r.get('silver_label_valid') is not True:
            continue
        if r.get('annotation_source') != 'assistant_visual_review':
            continue
        if r['layer'] == 'transition':
            lo, hi = r['boundary_frame_range']
            before, after = r.get('before_evidence_frame'), r.get('after_evidence_frame')
            if not (r.get('occurrence_confirmed') is True and r.get('before_after_observed') is True
                    and r.get('censored') is False and type(before) is int and type(after) is int
                    and before < lo <= after < hi and {before, after} <= set(r['evidence_frames'])
                    and anchor < lo and hi - 1 <= target):
                continue
            value = 1
        else:
            if not (r['start_frame'] <= target < r['end_frame'] and target in r['evidence_frames']):
                continue
            value = r['value']
        if r.get('subject') == 'object' and r.get('secondary_object_id'):
            source, destination = r['object_id'], r['secondary_object_id']
        elif r.get('subject') in ('left_hand', 'right_hand', 'both_hands') and not r.get('secondary_object_id'):
            source, destination = r['subject'], r.get('object_id')
        else:
            continue
        if source not in slots or destination not in slots:
            continue
        i, j = slots[source], slots[destination]
        if not observed_entities[i] or not observed_entities[j]:
            continue
        cell = (i, j, vocabulary[kind])
        require(not mask[cell] or values[cell] == value, 'independent graph finds contradictory labels')
        values[cell], mask[cell] = value, True
    return values, mask


def audit_observations(root, row, sequence, seed_doc, quality):
    """Verify causal bookkeeping, current seeds and every reviewed tracking box."""
    observations_path = checked_artifact(root, row, 'observations_path', 'observations_sha256')
    metadata_path = checked_artifact(root, row, 'observation_metadata_path', 'observation_metadata_sha256')
    metadata = read(metadata_path)
    require(metadata['observations_sha256'] == row['observations_sha256']
            and metadata['source_bindings']['seeds_sha256'] == row['seeds_sha256']
            and metadata['source_bindings']['base_sequence_sha256'] == row['base_sequence_sha256'],
            'tracking metadata source bindings differ')
    require(metadata['observation_mode'] == 'causal_replay_with_annotated_initialization'
            and metadata.get('whole_object_6d_certified') is False, 'tracking overclaims observation mode/6D')
    with np.load(observations_path, allow_pickle=False) as data:
        obs = {k: data[k] for k in data.files}
    times, ids, valid = obs['timestamps'], obs['object_ids'].astype(str).tolist(), obs['object_valid']
    length, count = len(times), len(ids)
    require(np.array_equal(times, sequence['timestamps']) and length == row['frame_count']
            and np.array_equal(obs['frame_index'], np.arange(length))
            and np.array_equal(obs['max_frame_accessed'], np.arange(length)), 'observation native prefix differs')
    require(len(set(ids)) == count and set(ids) == {r['entity_id'] for r in seed_doc['seeds']},
            'tracking object identity differs from explicit seeds')
    require(valid.shape == (length, count) and valid.dtype == bool, 'invalid observation mask')
    boxes, stats, code, last = (obs[k] for k in ('boxes_xyxy_native', 'rgb_stats', 'source_code', 'last_seed_frame'))
    require(boxes.shape == valid.shape + (4,) and stats.shape == valid.shape + (6,)
            and np.isfinite(boxes[valid]).all() and np.isfinite(stats[valid]).all()
            and np.isnan(boxes[~valid]).all() and np.isnan(stats[~valid]).all(), 'observation NaN/mask mismatch')
    w, h = row['native_image_size']
    actual = boxes[valid]
    require(np.all(actual[:, :2] >= 0) and np.all(actual[:, 2:] <= [w, h])
            and np.all(actual[:, 2:] > actual[:, :2]), 'observed box outside native image')
    require(np.all((stats[valid] >= 0) & (stats[valid] <= 1)), 'RGB feature outside normalized range')
    require(np.all(code[~valid] == 0) and np.all(last[~valid] == -1)
            and np.isin(code[valid], [1, 2]).all(), 'missing observation has hidden seed/source')
    seed_map = {(r['frame_index'], r['entity_id']): r for r in seed_doc['seeds']}
    max_age = metadata['tracking_config']['max_seed_age']
    for f, j in np.argwhere(valid):
        f, j = int(f), int(j); seed = int(last[f, j])
        require(0 <= seed <= f and (seed, ids[j]) in seed_map, 'future or invented tracking seed')
        if code[f, j] == 1:
            require(seed == f and np.allclose(boxes[f, j], seed_map[(f, ids[j])]['bbox_xyxy_native'], atol=1e-4),
                    'seed observation differs from current-frame annotated box')
        else:
            require(seed < f and f - seed < max_age and f > 0 and valid[f - 1, j]
                    and last[f - 1, j] == seed, 'track resurrects after a gap or exceeds its seed prefix')
            require(obs['tracking_inliers'][f, j] >= metadata['tracking_config']['minimum_points'],
                    'valid flow has insufficient retained evidence')
    for f, identity in seed_map:
        j = ids.index(identity)
        require(valid[f, j] and code[f, j] == 1, 'current annotated seed disappeared')
    require(quality['review_status'] == 'assistant_reviewed'
            and quality['observations_sha256'] == row['observations_sha256']
            and quality.get('human_label_valid') is not True, 'tracking quality has wrong provenance')
    mp = resolved(root, quality['evidence_manifest_path'])
    require(digest(mp) == quality['evidence_manifest_sha256'], 'tracking evidence manifest changed')
    manifest = read(mp)
    require(manifest['sequence_id'] == row['sequence_id'] and manifest['source_bindings']['observations_sha256'] == row['observations_sha256'],
            'tracking evidence belongs to other observations')
    bindings = manifest['source_bindings']
    # Independently check the same-frame calibrated official hand projection.
    require(digest(bindings['source_metadata_path']) == bindings['source_metadata_sha256']
            == metadata['source_bindings']['source_metadata_sha256'], 'tracking native metadata changed')
    source_metadata = read(bindings['source_metadata_path'])
    require(source_metadata['video_sha256'] == bindings['source_video_sha256']
            == metadata['source_bindings']['video_sha256'], 'tracking video binding differs')
    require(bindings['tracking_metadata_sha256'] == row['observation_metadata_sha256']
            and bindings['seeds_sha256'] == row['seeds_sha256'], 'gallery tracking/seed metadata binding differs')
    camera = np.einsum('fhji,fik->fhjk', sequence['hands_world']
        - sequence['camera_to_world'][:, None, None, :3, 3], sequence['camera_to_world'][:, :3, :3])
    good = sequence['hand_valid'] & np.isfinite(camera).all((2, 3)) & (camera[..., 2] > 1e-5).all(-1)
    homogeneous = camera @ np.asarray(source_metadata['K']).T
    with np.errstate(invalid='ignore', divide='ignore'):
        expected_uv = homogeneous[..., :2] / homogeneous[..., 2:]
    expected_uv[~good] = np.nan
    require(np.array_equal(obs['hand_projection_valid'], good)
            and np.allclose(obs['hand_uv_native'], expected_uv, rtol=1e-5, atol=1e-3, equal_nan=True),
            'same-frame official hand projection differs')
    pages = {p['path']: p for p in manifest['pages']}
    viewed = quality['reviewed_page_paths']
    require(len(set(viewed)) == len(viewed) and set(viewed) <= set(pages), 'tracking review page list invalid')
    if 'reviewed_page_sha256' in quality:
        require(quality['reviewed_page_sha256'] == [pages[p]['sha256'] for p in viewed], 'tracking viewed page hash differs')
    shown = set()
    for p in viewed:
        require(digest(resolved(root, p)) == pages[p]['sha256'], 'actual tracking page changed')
        shown.update(pages[p]['frame_indices'])
    displayed = set()
    for record in manifest['frames']:
        f = record['frame_index']
        require(0 <= f < length and abs(record['timestamp_s'] - times[f]) < 1e-9, 'tracking evidence wrong native time')
        for obj in record['objects']:
            require(obj['entity_id'] in ids, 'tracking gallery has invented entity')
            j = ids.index(obj['entity_id'])
            require(valid[f, j] and np.allclose(obj['bbox_xyxy_native'], boxes[f, j], atol=1e-5)
                    and obj['source_code'] == code[f, j] and obj['last_seed_frame'] == last[f, j],
                    'tracking gallery box does not describe frozen observation')
            require((f, j) not in displayed, 'duplicate gallery object frame')
            displayed.add((f, j))
    require(displayed == {tuple(r) for r in np.argwhere(valid)}, 'gallery omits an actual valid observation')
    manifest_frames = {r['frame_index']: r for r in manifest['frames']}
    for extra in quality.get('additional_native_evidence', []):
        frame = extra['frame_index']
        require(frame in shown and extra.get('reviewed') is True
                and digest(resolved(root, extra['path'])) == extra['sha256'], 'additional tracking evidence changed or was not reviewed')
        require(extra['decoded_native_rgb_sha256'] == manifest_frames[frame]['decoded_native_rgb_sha256']
                and extra['native_image_size'] == row['native_image_size'], 'additional tracking evidence has wrong native frame')
    accepted, rejected = np.zeros_like(valid), np.zeros_like(valid)
    for name, mask in [('accepted_ranges', accepted), ('rejected_ranges', rejected)]:
        for r in quality.get(name, []):
            lo, hi = r['start_frame'], r['end_frame']
            require(r['entity_id'] in ids and type(lo) is int and type(hi) is int and 0 <= lo < hi <= length,
                    'quality range has invalid native frame/identity')
            require(set(range(lo, hi)) <= shown and r.get('reason_zh'), 'quality range exceeds actual viewed pages')
            j = ids.index(r['entity_id'])
            require(valid[lo:hi, j].all(), 'quality approves/rejects missing object frames')
            mask[lo:hi, j] = True
    require(not (accepted & rejected).any(), 'quality accepted/rejected ranges overlap')
    require(np.array_equal(accepted | rejected, valid), 'some tracked object frames remain unreviewed')
    return obs, accepted & valid, {'candidate_object_frames': int(valid.sum()),
        'accepted_object_frames': int(accepted.sum()), 'rejected_object_frames': int(rejected.sum()),
        'viewed_tracking_pages': len(viewed), 'tracking_manifest_sha256': digest(mp)}


def audit_sample_inputs(sample, sequence, obs, accepted, start, end, slots, image_size, max_objects):
    """Recompute the complete input tensor without using exporter/loader helpers."""
    W, E = end - start, 3 + max_objects
    expected = np.zeros((W, E, 120), np.float32)
    valid = np.zeros((W, E), bool)
    dt = np.diff(sequence['timestamps'][start:end])
    hand_ok = sequence['hand_valid'][start:end].copy()
    anchor = sequence['camera_to_world'][end - 1]
    hands = (sequence['hands_world'][start:end] - anchor[:3, 3]) @ anchor[:3, :3]
    hand_ok &= np.isfinite(hands).all((2, 3))
    uv = obs['hand_uv_native'][start:end] / np.asarray(image_size, np.float32)
    projected = obs['hand_projection_valid'][start:end] & np.isfinite(uv).all((2, 3))
    hand_min, hand_max = np.nan_to_num(uv).min(2), np.nan_to_num(uv).max(2)
    for h in (0, 1):
        expected[:, h, :2] = np.where(projected[:, h, None], (hand_min[:, h] + hand_max[:, h]) / 2, 0)
        expected[:, h, 2:4] = np.where(projected[:, h, None], hand_max[:, h] - hand_min[:, h], 0)
        expected[:, h, 12:54] = np.where(projected[:, h, None], np.nan_to_num(uv[:, h].reshape(W, 42)), 0)
        expected[:, h, 54:117] = np.where(hand_ok[:, h, None], np.nan_to_num(hands[:, h].reshape(W, 63)), 0)
        expected[:, h, 117], expected[:, h, 118] = projected[:, h], hand_ok[:, h]
    valid[:, :2] = hand_ok | projected
    both_projected = projected.all(1)
    union_min, union_max = hand_min.min(1), hand_max.max(1)
    expected[:, 2, :2] = np.where(both_projected[:, None], (union_min + union_max) / 2, 0)
    expected[:, 2, 2:4] = np.where(both_projected[:, None], union_max - union_min, 0)
    valid[:, 2] = valid[:, :2].all(1)
    for slot, j in enumerate(slots, 3):
        good = accepted[start:end, j]
        box = obs['boxes_xyxy_native'][start:end, j] / np.asarray([*image_size, *image_size], np.float32)
        expected[:, slot, :2] = np.where(good[:, None], np.nan_to_num((box[:, :2] + box[:, 2:]) / 2), 0)
        expected[:, slot, 2:4] = np.where(good[:, None], np.nan_to_num(box[:, 2:] - box[:, :2]), 0)
        expected[:, slot, 6:12] = np.where(good[:, None], np.nan_to_num(obs['rgb_stats'][start:end, j]), 0)
        expected[:, slot, 119] = good & (obs['source_code'][start:end, j] == 1)
        valid[:, slot] = good
    velocity_valid = valid.copy()
    velocity_valid[:, :2], velocity_valid[:, 2] = projected, both_projected
    for e in range(E):
        good = velocity_valid[1:, e] & velocity_valid[:-1, e]
        expected[1:, e, 4:6] = np.where(good[:, None], np.diff(expected[:, e, :2], axis=0) / dt[:, None], 0)
    expected[~valid] = 0
    got = sample['entity_features'].numpy()
    require(got.shape == expected.shape and np.isfinite(got).all()
            and np.allclose(got, expected, rtol=2e-5, atol=2e-5), 'actual encoder input differs from independent history-only features')
    require(np.array_equal(sample['entity_valid'].numpy(), valid), 'actual entity validity differs')
    require(np.array_equal(sample['hand_valid'].numpy(), hand_ok), 'actual hand validity differs')
    hands[~hand_ok] = np.nan
    require(np.allclose(sample['hand_seq'].numpy(), hands, rtol=2e-5, atol=2e-5, equal_nan=True), 'hand anchor/input differs')
    require(np.allclose(sample['anchor_camera_to_world'].numpy(), anchor, rtol=1e-6, atol=1e-6), 'future camera leaked into anchor')
    require(np.array_equal(sample['entity_kind'].numpy(), [0, 1, 2] + [3] * max_objects), 'entity kinds differ')
    return valid.any(0)


def audit_training_index(root):
    """Audit every published graph window and actually load its input/target."""
    from src.mamba.dataset_v5 import EgoDemoGraphV5Dataset

    root = Path(root)
    ip = root / 'training_index.json'
    index_sha, index = digest(ip), read(ip)
    require(index['schema_version'] == 'assistant_silver_graph_training_v5'
            and index['full_annotation_complete'] is False and index['full_joint_training_allowed'] is False
            and index['whole_object_6d_certified'] is False, 'training subset overclaims complete joint data')
    report_path = root / 'review_seed_audit.json'; prior = read(report_path)
    require(prior['failed'] == 0 and prior['native_pixels_verified'] is True, 'native review/seed audit must pass first')
    base_path, split_path, vocab_path = (root / 'sources' / n for n in ['base_dataset_index.json', 'split_manifest.json', 'vocab.json'])
    require(digest(base_path) == index['base_dataset_index_sha256'] == prior['base_index_sha256']
            and digest(split_path) == index['split_manifest_sha256'] == prior['split_manifest_sha256']
            and digest(vocab_path) == index['vocab_sha256'], 'frozen base/split/vocab changed')
    base, split, vocab = read(base_path), read(split_path), read(vocab_path)
    types = [{'layer': layer, 'code': code, 'name_zh': name} for layer in ('operation', 'relation', 'transition', 'cooperation')
             for code, name in zip(vocab[layer]['codes'], vocab[layer]['names_zh'])]
    # The index may add a stable index number to vocabulary entries.
    require([(r['layer'], r['code'], r['name_zh']) for r in index['event_types']]
            == [(r['layer'], r['code'], r['name_zh']) for r in types], 'graph vocabulary reordered')
    originals = {r['sequence_id']: r for r in base['sequences']}
    reviewed = {r['sequence_id']: r for r in prior['records']}
    rows = {r['sequence_id']: r for r in index['sequences']}
    require(len(rows) == len(index['sequences']) and set(rows) == set(reviewed), 'training sequence coverage differs from reviewed subset')
    config = index['window_config']; W = config['window_frames']; N = config['max_objects']; minimum = config['minimum_observations']
    require(W == 32 and config['horizon_seconds'] == 1. and minimum >= 8 and config['feature_dim'] in (120, 132),
            'unexpected history/horizon/observation contract')
    loaded, sequence_records = {}, []
    for uid, row in rows.items():
        reassigned = ()
    rebalance_path = Path('/root/gpufree-data/runs/rebalance_v7.json')
    if rebalance_path.is_file():
        reassigned = tuple(json.loads(rebalance_path.read_text())['reassigned_to_train'])
    if uid in reassigned and row['split'] == 'train':
        # Documented v7 rebalance: this acquisition moved val→train via the
        # export whitelist; the original frozen assignment stays recorded in
        # the frozen split manifest.
        require(split['assignments'][uid] == 'val', 'rebalance entry was not a val acquisition')
    else:
        require(row['split'] == split['assignments'][uid] == originals[uid]['split'], 'acquisition crossed frozen split')
    for path_key, hash_key in [('review_path', 'review_sha256'), ('seeds_path', 'seeds_sha256'),
            ('tracking_review_path', 'tracking_review_sha256'), ('sequence_path', 'sequence_sha256')]:
        checked_artifact(root, row, path_key, hash_key)
        require(row['review_sha256'] == reviewed[uid]['sha256']
                and row['seeds_sha256'] == reviewed[uid]['seeds']['sha256'], 'audited review/seed changed')
        review, seeds, quality = (read(resolved(root, row[k])) for k in ('review_path', 'seeds_path', 'tracking_review_path'))
        require(seeds.get('review_sha256') in {row['review_sha256'], review.get('parent_review_sha256'),
                    review.get('source_bindings', {}).get('parent_review_sha256')},
                'seed identity source differs from the actual or frozen parent review')
        with np.load(root / row['sequence_path'], allow_pickle=False) as arrays:
            require(set(arrays.files) == {'timestamps', 'hands_world', 'hand_valid', 'camera_to_world'}, 'offline object geometry entered portable input')
            seq = {k: arrays[k] for k in arrays.files}
        source_path = Path(index['base_root']) / originals[uid]['sequence_path']
        require(digest(source_path) == row['base_sequence_sha256'] == originals[uid]['sha256']['sequence'], 'source sequence changed')
        with np.load(source_path, allow_pickle=False) as source:
            require(all(np.array_equal(seq[k], source[k], equal_nan=True) for k in seq), 'portable official hand/camera data differs')
        obs, accepted, checked = audit_observations(root, row, seq, seeds, quality)
        require(int(accepted.sum()) == row['observed_entity_frame_count'], 'accepted frame count overclaim')
        viewed = range_mask(review['reviewed_ranges'], len(seq['timestamps']))
        require(int(viewed.sum()) == row['continuously_reviewed_frames'], 'reviewed count differs')
        loaded[uid] = (seq, obs, accepted, review, viewed)
        sequence_records.append(dict(sequence_id=uid, split=row['split'], review_sha256=row['review_sha256'], **checked))
    expected_targets = set()
    for uid, (seq, obs, accepted, review, viewed) in loaded.items():
        ts = seq['timestamps']; tolerance = 1.5 * float(np.median(np.diff(ts)))
        for target in np.flatnonzero(viewed):
            previous = np.flatnonzero(ts[target] - ts[:target] >= 1.)
            if not len(previous):
                continue
            anchor = int(previous[-1]); start, end = anchor - W + 1, anchor + 1
            if start < 0 or ts[target] - ts[anchor] - 1. > tolerance + 1e-9 or not viewed[start:target + 1].all():
                continue
            candidates = [j for j in range(accepted.shape[1]) if accepted[start:end, j].sum() >= minimum]
            slots = sorted(candidates, key=lambda j: (int(np.flatnonzero(accepted[start:end, j])[0]), str(obs['object_ids'][j])))[:N]
            ids = ['left_hand', 'right_hand', 'both_hands'] + [str(obs['object_ids'][j]) for j in slots] + [''] * (N - len(slots))
            hand_ok = seq['hand_valid'][start:end] & np.isfinite(seq['hands_world'][start:end]).all((2, 3))
            hand_ok |= obs['hand_projection_valid'][start:end] & np.isfinite(obs['hand_uv_native'][start:end]).all((2, 3))
            observed = np.r_[hand_ok.any(0), hand_ok.all(1).any(), np.ones(len(slots), bool), np.zeros(N - len(slots), bool)]
            if independent_graph(review, ids, types, anchor, int(target), observed)[1].any():
                expected_targets.add((uid, int(target)))
    counts, kinds, type_values, seen_targets, per_sequence = {}, {}, {}, set(), Counter()
    for name in ('train', 'val', 'test'):
        info = index['splits'][name]; path = checked_artifact(root, info, 'path', 'sha256')
        windows = [json.loads(s) for s in path.read_text().splitlines() if s]
        require(len(windows) == info['windows'], 'window count differs')
        dataset = EgoDemoGraphV5Dataset(root, name, diagnostic=True)
        total, by_type, positive_types, negative_types = Counter(), Counter(), Counter(), Counter()
        for n, window in enumerate(windows):
            uid = window['sequence_id']; require(uid in rows and rows[uid]['split'] == name, 'window changes frozen split')
            seq, obs, accepted, review, viewed = loaded[uid]
            start, end, anchor, target = (window[k] for k in ['start_frame', 'end_frame', 'anchor_frame', 'target_frame'])
            ts = seq['timestamps']; require(0 <= start < end <= target < len(ts) and end - start == W and anchor == end - 1, 'window native boundaries differ')
            horizon = float(ts[target] - ts[anchor]); tolerance = 1.5 * float(np.median(np.diff(ts)))
            require(horizon >= 1. and horizon - 1. <= tolerance + 1e-9
                    and (anchor + 1 >= target or ts[target] - ts[anchor + 1] < 1.), 'window has wrong actual one-second anchor')
            require(abs(window['actual_horizon_s'] - horizon) <= 1e-9
                    and abs(window['target_timestamp'] - ts[target]) <= 1e-9, 'published native horizon/timestamp differs')
            require(viewed[start:target + 1].all(), 'history-to-target crosses an unreviewed gap')
            require((uid, target) not in seen_targets, 'duplicate acquisition target window'); seen_targets.add((uid, target))
            eligible = [j for j in range(accepted.shape[1]) if accepted[start:end, j].sum() >= minimum]
            slots = sorted(eligible, key=lambda j: (int(np.flatnonzero(accepted[start:end, j])[0]), str(obs['object_ids'][j])))[:N]
            ids = ['left_hand', 'right_hand', 'both_hands'] + [str(obs['object_ids'][j]) for j in slots] + [''] * (N - len(slots))
            sample = dataset[n]
            require(sample['metadata']['entity_ids'] == ids, 'actual object slots depend on future or wrong ranking')
            observed = audit_sample_inputs(sample, seq, obs, accepted, start, end, slots, rows[uid]['native_image_size'], N)
            values, mask = independent_graph(review, ids, types, anchor, target, observed)
            require(np.array_equal(sample['graph_label_valid'].numpy(), mask)
                    and np.array_equal(sample['graph_values'].numpy()[mask], values[mask]), 'actual graph direction/value/unknown mask differs')
            require(mask.any(), 'published actual loaded window has zero supervision')
            positive, negative = int(((values == 1) & mask).sum()), int(((values == 0) & mask).sum())
            require((positive, negative) == (window['known_positive_cells'], window['known_negative_cells']), 'window supervised counts differ')
            total.update(windows=1, known_positive_cells=positive, known_negative_cells=negative,
                         windows_with_positive=int(positive > 0), windows_with_negative=int(negative > 0))
            for k, typ in enumerate(types):
                label = typ['layer'] + ':' + typ['code']
                if mask[:, :, k].any(): by_type[label] += int(mask[:, :, k].sum())
                positive_count = int((mask[:, :, k] & (values[:, :, k] == 1)).sum())
                negative_count = int((mask[:, :, k] & (values[:, :, k] == 0)).sum())
                if positive_count: positive_types[label] += positive_count
                if negative_count: negative_types[label] += negative_count
            per_sequence[uid] += 1
        require(dict(total) == index['counts'][name] and dict(by_type) == index['known_cells_by_type'][name], 'split graph aggregates differ')
        counts[name], kinds[name] = dict(total), dict(by_type)
        type_values[name] = dict(known_type_count=len(by_type), positive_type_count=len(positive_types),
            negative_type_count=len(negative_types), positive_cells_by_type=dict(positive_types),
            negative_cells_by_type=dict(negative_types),
            known_type_count_by_layer=dict(Counter(k.split(':', 1)[0] for k in by_type)),
            positive_type_count_by_layer=dict(Counter(k.split(':', 1)[0] for k in positive_types)))
        require(info['sequence_count'] == len({w['sequence_id'] for w in windows}), 'split acquisition count differs')
    require(all(per_sequence[uid] == row['window_count'] for uid, row in rows.items()), 'per acquisition window count differs')
    require(seen_targets == expected_targets, 'published windows omit or invent independently eligible targets')
    require(digest(ip) == index_sha, 'training index changed during audit')
    result = dict(schema_version='assistant_annotation_v5_training_index_audit', created_at=datetime.now(timezone.utc).isoformat(),
        training_index_sha256=index_sha, review_seed_audit_sha256=digest(report_path), auditor_sha256=digest(__file__),
        source_selection_sha256=base['source_selection_sha256'], split_manifest_sha256=index['split_manifest_sha256'],
        failed=0, actual_dataset_windows_loaded=len(seen_targets), counts=counts, known_cells_by_type=kinds,
        type_coverage=type_values,
        sequence_records=sequence_records, full_annotation_complete=False, full_joint_training_allowed=False,
        independent_human_semantic_review=False, optimizer_steps=0,
        scope_zh='独立核全部窗口来源/固定划分/实际原生1秒时基、真实当前帧种子和失联不复活、逐对象跟踪质检；逐窗重算历史槽位、120维输入和有向正负图mask并实际调用Dataset比对。未启动训练；未认证6D或300段完整标注。')
    (root / 'training_index_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root", required=True)
    p.add_argument("--v3-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--expected-sequences", type=int)
    p.add_argument("--index-only", action="store_true", help="Audit all frozen graph windows and actual Dataset output")
    p.add_argument("--hash-only", action="store_true", help="Diagnostic only; does not verify native source pixels")
    args = p.parse_args()
    if args.index_only:
        result = audit_training_index(args.output_root)
        print(json.dumps({"failed": result["failed"], "counts": result["counts"]}, ensure_ascii=False))
        return
    result = audit_review_seed_dataset(args.base_root, args.v3_root, args.output_root,
        verify_pixels=not args.hash_only, expected_sequences=args.expected_sequences)
    print(json.dumps({"failed": result["failed"], "counts": result["counts"],
                      "native_pixels_verified": result["native_pixels_verified"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
