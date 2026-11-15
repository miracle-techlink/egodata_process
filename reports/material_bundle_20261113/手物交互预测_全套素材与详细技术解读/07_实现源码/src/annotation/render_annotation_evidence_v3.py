"""Render immutable, native-frame evidence sheets for assistant silver review.

Rendering creates evidence only. It never creates a reviewed/accepted annotation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sample_frames(start, end):
    if not 0 <= start < end:
        raise ValueError("invalid half-open interval")
    return sorted({int(round(start + (end - 1 - start) * fraction)) for fraction in (0, .25, .5, .75, 1)})


def review_shard(index):
    if index < 100:
        return "entity_shard_a"
    if index < 200:
        return "event_shard_b"
    if index < 280:
        return "audit_shard_c"
    return "root_shard_d"


def _text(draw, xy, text, font, color=(240, 242, 245), width=110):
    text = str(text)
    for offset in range(0, len(text), width):
        draw.text((xy[0], xy[1] + 28 * (offset // width)), text[offset:offset + width], font=font, fill=color)


def render_sequence(source, output, uid, font_path, sorted_index):
    cv2.setNumThreads(1)
    source, output = Path(source), Path(output)
    proposal_path = source / "joint_annotations" / uid / "proposal.json"
    proposal = json.loads(proposal_path.read_text())
    source_meta = json.loads((source / "sequences" / uid / "metadata.json").read_text())
    destination = output / "evidence" / uid
    manifest_path = destination / "manifest.json"
    proposal_sha = digest(proposal_path)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["source_bindings"]["proposal_sha256"] != proposal_sha:
            raise ValueError("Source changed after evidence freeze")
        for page in existing["pages"]:
            if digest(output / page["path"]) != page["sha256"]:
                raise ValueError("Evidence page changed after freeze")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    intervals = proposal["source_semantic_intervals"]
    selected = {r["source_interval_id"]: sample_frames(r["start_frame"], r["end_frame"]) for r in intervals}
    needed = sorted(set(frame for frames in selected.values() for frame in frames))
    font = ImageFont.truetype(str(font_path), 22)
    label_font = ImageFont.truetype(str(font_path), 29)
    with np.load(source / "sequences" / uid / "hands.npz", allow_pickle=False) as z:
        hands = z["hand_uv"]
        hand_valid = z["hand_valid"]
        times = z["timestamps"]
    frame_objects = {frame: [] for frame in needed}
    for entity in proposal["entities"]:
        for track in entity["tracklets"]:
            with np.load(source / "native_tracks" / uid / (track + ".npz"), allow_pickle=False) as z:
                frame_ids, valid, uv = z["frame_index"], z["point_valid"], z["points_uv"]
                lookup = {int(frame): index for index, frame in enumerate(frame_ids)}
                for frame in needed:
                    if frame not in lookup:
                        continue
                    index = lookup[frame]
                    points = uv[index, 0]
                    keep = valid[index] & np.isfinite(points).all(axis=-1)
                    points = points[keep]
                    if len(points) < 2:
                        continue
                    lo, hi = np.quantile(points, [.05, .95], axis=0)
                    frame_objects[frame].append({"object_id": entity["id"], "name_zh": entity["name_zh"],
                                                  "tracklet_id": track, "bbox": [*lo.tolist(), *hi.tolist()]})
    cap = cv2.VideoCapture(source_meta["video_path"])
    if not cap.isOpened():
        raise ValueError("Cannot open native source video")
    frame_records, thumbnails = {}, {}
    chains = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    try:
        for frame in needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - frame - 1) > .1:
                raise ValueError("Native video frame seek/read failed: " + str(frame))
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            for side, color in enumerate([(40, 195, 255), (255, 115, 40)]):
                if not hand_valid[frame, side]:
                    continue
                joints = hands[frame, side]
                for chain in chains:
                    for a, b in zip(chain[:-1], chain[1:]):
                        if np.isfinite(joints[[a, b]]).all():
                            draw.line([tuple(joints[a]), tuple(joints[b])], fill=color, width=4)
                if np.isfinite(joints[0]).all():
                    draw.text(tuple(joints[0]), ["左手 L", "右手 R"][side], font=label_font,
                              fill=color, stroke_width=1, stroke_fill="black")
            ids = []
            for item in frame_objects[frame]:
                identity = item["object_id"]
                if identity in ids:
                    continue
                ids.append(identity)
                color = tuple(90 + (int(hashlib.sha256(identity.encode()).hexdigest()[i:i + 2], 16) % 155)
                              for i in (0, 2, 4))
                x0, y0, x1, y1 = item["bbox"]
                draw.rectangle((x0, y0, x1, y1), outline=color, width=5)
                label = identity + " " + item["name_zh"][:14] + " [候选]"
                draw.text((max(0, x0), max(0, y0 - 38)), label, font=label_font,
                          fill=color, stroke_width=2, stroke_fill="black")
            draw.rectangle((0, 0, image.width, 42), fill=(15, 18, 24))
            draw.text((12, 3), f"{uid[:8]} | 原生帧 {frame} | {float(times[frame]):.3f}s | 候选覆盖不等于已确认", font=label_font, fill="white")
            single = destination / "frames" / f"{frame:06d}.jpg"
            single.parent.mkdir(exist_ok=True)
            image.save(single, quality=89)
            thumb = image.resize((512, round(image.height * 512 / image.width)), Image.Resampling.LANCZOS)
            thumbnails[frame] = thumb
            frame_records[frame] = {"frame_index": frame, "timestamp": float(times[frame]),
                                   "path": str(single.relative_to(output)), "sha256": digest(single),
                                   "visible_candidate_objects": ids}
    finally:
        cap.release()
    page_records, interval_records = [], []
    for page_index in range(0, len(intervals), 2):
        rows = intervals[page_index:page_index + 2]
        image_height = max(thumb.height for thumb in thumbnails.values())
        row_height = image_height + 128
        canvas = Image.new("RGB", (2560, 82 + row_height * len(rows)), (19, 24, 33))
        draw = ImageDraw.Draw(canvas)
        _text(draw, (16, 7), f"采集 {uid}  |  {source_meta['task_name']}  |  原生区间采样证据", font)
        _text(draw, (16, 38), "仅抽取可见时刻；物体框/身份均为待核对候选。未看见接触或转换时必须保留不确定，不能按标题通过。", font)
        interval_ids = []
        for row_index, interval in enumerate(rows):
            y = 82 + row_index * row_height
            iid = interval["source_interval_id"]
            interval_ids.append(iid)
            header = f"{iid}  [{interval['start_frame']}, {interval['end_frame']})  中文：{interval.get('description_zh', '')}"
            _text(draw, (12, y), header, font, width=110)
            _text(draw, (12, y + 58), "原文：" + interval["description_original"], font, (190, 200, 215), width=150)
            for column, frame in enumerate(selected[iid]):
                canvas.paste(thumbnails[frame], (column * 512, y + 120))
            interval_records.append({"source_interval_id": iid, "source_event_id": interval.get("source_event_id"),
                                     "start_frame": interval["start_frame"], "end_frame": interval["end_frame"],
                                     "description_zh": interval.get("description_zh"), "description_original": interval["description_original"],
                                     "operation_candidates": interval["operations"], "frames": selected[iid],
                                     "page_index": len(page_records)})
        page_path = destination / f"page_{len(page_records):03d}.jpg"
        canvas.save(page_path, quality=92)
        page_records.append({"path": str(page_path.relative_to(output)), "sha256": digest(page_path),
                             "interval_ids": interval_ids, "width": canvas.width, "height": canvas.height})
    manifest = {"schema_version": "assistant_annotation_v3_evidence", "sequence_id": uid,
                "sorted_index": sorted_index, "review_shard": review_shard(sorted_index),
                "source_bindings": {"proposal_sha256": proposal_sha, "video_sha256": source_meta["video_sha256"],
                                    "hands_sha256": digest(source / "sequences" / uid / "hands.npz"),
                                    "track_manifest_sha256": digest(source / "native_tracks" / uid / "tracklets.json"),
                                    "source_selection_sha256": proposal["source_selection_sha256"]},
                "source_root": str(source), "renderer_sha256": digest(__file__),
                "sample_policy": "five native frame samples at start/25%/50%/75%/end-1; no interpolation",
                "pages": page_records, "intervals": interval_records, "frames": list(frame_records.values()),
                "annotation_created": False, "human_label_valid": False, "training_allowed": False}
    save(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    ids = sorted(path.name for path in (args.source / "joint_annotations").iterdir() if (path / "proposal.json").exists())
    # Deliver early examples from every reviewer shard before the rest of the pool.
    priority = [i for begin in (0, 100, 200, 280) for i in range(begin, min(begin + 3, len(ids)))]
    order = priority + [i for i in range(len(ids)) if i not in priority]
    if args.limit:
        order = order[:args.limit]
    results = []
    with ProcessPoolExecutor(max_workers=min(6, max(1, args.workers))) as pool:
        futures = {pool.submit(render_sequence, args.source, args.output, ids[i], args.font, i): ids[i] for i in order}
        for future in as_completed(futures):
            value = future.result()
            results.append({"sequence_id": value["sequence_id"], "review_shard": value["review_shard"],
                            "manifest_path": f"evidence/{value['sequence_id']}/manifest.json",
                            "manifest_sha256": digest(args.output / "evidence" / value["sequence_id"] / "manifest.json"),
                            "intervals": len(value["intervals"]), "pages": len(value["pages"])})
            print(json.dumps({"done": len(results), "sequence_id": value["sequence_id"],
                              "review_shard": value["review_shard"]}), flush=True)
    save(args.output / "evidence" / "index.json", {"schema_version": "assistant_annotation_v3_evidence_index",
         "records": sorted(results, key=lambda row: row["sequence_id"]), "expected_sequences": len(ids),
         "complete": len(results) == len(ids), "annotation_created": False})


if __name__ == "__main__":
    main()
