"""Read-only full-frame evidence for every valid native tracking observation.

Rendering never marks a frame visually reviewed or accepts its object identity.
"""
from __future__ import annotations

import argparse
import colorsys
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.annotation.track_visual_entities_v5 import read, sha


def render_tracking(root, refine_root, uid, *, font_path=None, tile_width=640, columns=3, frames_per_page=12):
    root, refine_root = Path(root).resolve(), Path(refine_root).resolve()
    if tile_width < 320 or columns < 1 or frames_per_page < 1:
        raise ValueError("Invalid evidence page dimensions")
    source = root / "visual_observations" / uid
    observations_path, seeds_path = source / "observations.npz", source / "seeds.json"
    metadata_path = source / "metadata.json"
    observations_sha, seeds_sha = sha(observations_path), sha(seeds_path)
    metadata, seeds = read(metadata_path), read(seeds_path)
    if (metadata["observations_sha256"] != observations_sha
            or metadata["source_bindings"]["seeds_sha256"] != seeds_sha):
        raise ValueError("Tracking observations/seeds changed after production")
    if metadata["sequence_id"] != uid or seeds["sequence_id"] != uid:
        raise ValueError("Tracking acquisition differs")
    native_metadata_path = refine_root / "sequences" / uid / "metadata.json"
    native = read(native_metadata_path)
    video_path = Path(native["video_path"])
    video_sha = sha(video_path)
    if video_sha != native["video_sha256"] or metadata["source_bindings"]["video_sha256"] != video_sha:
        raise ValueError("Source native video changed")
    if sha(native_metadata_path) != metadata["source_bindings"]["source_metadata_sha256"]:
        raise ValueError("Source native metadata changed")
    bindings = {"observations_path": str(observations_path.relative_to(root)), "observations_sha256": observations_sha,
        "seeds_path": str(seeds_path.relative_to(root)), "seeds_sha256": seeds_sha,
        "tracking_metadata_path": str(metadata_path.relative_to(root)), "tracking_metadata_sha256": sha(metadata_path),
        "source_video_path": str(video_path), "source_video_sha256": video_sha,
        "source_metadata_path": str(native_metadata_path), "source_metadata_sha256": sha(native_metadata_path)}
    destination = root / "tracking_evidence" / uid
    manifest_path = destination / "manifest.json"
    config = {"tile_width": tile_width, "columns": columns, "frames_per_page": frames_per_page,
              "image_scope": "complete_native_frame", "sampling": "every_frame_with_any_valid_object_observation"}
    if manifest_path.exists():
        prior = read(manifest_path)
        if prior["source_bindings"] != bindings or prior["render_config"] != config:
            raise ValueError("Published tracking evidence has different sources/config")
        for page in prior["pages"]:
            if sha(root / page["path"]) != page["sha256"]:
                raise ValueError("Published tracking evidence page changed")
        return prior
    with np.load(observations_path, allow_pickle=False) as z:
        ids = z["object_ids"].astype(str).tolist()
        valid, boxes = z["object_valid"], z["boxes_xyxy_native"]
        times, frame_index = z["timestamps"], z["frame_index"]
        codes, seed_frames = z["source_code"], z["last_seed_frame"]
        inliers = z["tracking_inliers"]
    count = len(times)
    if not np.array_equal(frame_index, np.arange(count)) or valid.shape != (count, len(ids)):
        raise ValueError("Tracking frame index or shape mismatch")
    if boxes.shape != valid.shape + (4,) or not np.isfinite(boxes[valid]).all():
        raise ValueError("Valid tracking observation has invalid bbox")
    width, height = native["image_size"]
    if metadata["native_image_size"] != [width, height]:
        raise ValueError("Tracking image size mismatch")
    active = boxes[valid]
    if np.any(active[:, :2] < 0) or np.any(active[:, 2:] > [width, height]) or np.any(active[:, 2:] <= active[:, :2]):
        raise ValueError("Valid tracking box is outside native pixels")
    needed = np.flatnonzero(valid.any(1)).tolist()
    font_path = Path(font_path) if font_path else refine_root.parent / "egodemo_expansion_300_20261023/tools/NotoSansCJK-Regular.ttc"
    font, label_font = ImageFont.truetype(str(font_path), 22), ImageFont.truetype(str(font_path), 16)
    names = {r["entity_id"]: r.get("name_zh", r["entity_id"]) for r in seeds["seeds"]}
    colors = {identity: tuple(int(v * 255) for v in colorsys.hsv_to_rgb(
        int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) / 2**32, .75, 1.)) for identity in ids}
    legend = [{"entity_id": identity, "short_id": f"O{j+1:02d}", "name_zh": names.get(identity, identity),
               "color_rgb": list(colors[identity])} for j, identity in enumerate(ids)]
    tile_height = round(height * tile_width / width)
    header, caption = 80, 34
    scale = np.asarray([tile_width / width, tile_height / height] * 2)
    destination.mkdir(parents=True, exist_ok=True)
    pages, frame_records, pending = [], [], []

    def finish_page():
        if not pending:
            return
        page_index = len(pages)
        rows = (len(pending) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_width, header + rows * (tile_height + caption)), (22, 28, 38))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 5), f"{uid}  跟踪质检页 {page_index + 1}", font=font, fill="white")
        draw.text((10, 37), "所有有效观测原生帧；完整画面；种子/跟踪分开；尚未视觉审核", font=font, fill=(235, 218, 150))
        for cell, (frame, image, record) in enumerate(pending):
            x, y = (cell % columns) * tile_width, header + (cell // columns) * (tile_height + caption)
            sheet.paste(image, (x, y + caption))
            draw.text((x + 5, y + 3), f"原生帧 {frame:06d} | {times[frame]:.6f}s", font=font, fill="white")
            record.update(page_index=page_index, page_cell_index=cell,
                          full_frame_region_on_page_xyxy=[x, y + caption, x + tile_width, y + caption + tile_height])
            frame_records.append(record)
        path = destination / f"page_{page_index:04d}.jpg"
        sheet.save(path, quality=94, subsampling=0)
        pages.append({"path": str(path.relative_to(root)), "sha256": sha(path),
                      "frame_indices": [frame for frame, _, _ in pending], "width": sheet.width, "height": sheet.height})
        pending.clear()

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    needed_set = set(needed)
    try:
        if not cap.isOpened():
            raise ValueError("Native source video cannot be opened")
        for frame in range(needed[-1] + 1 if needed else 0):
            ok, bgr = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - frame - 1) > .1:
                raise ValueError("Native decoder skipped or substituted a frame")
            if frame not in needed_set:
                continue
            if bgr.shape[:2] != (height, width):
                raise ValueError("Native decoded image dimensions differ")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decoded_sha = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            image = Image.fromarray(rgb).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            draw, objects = ImageDraw.Draw(image), []
            for j in np.flatnonzero(valid[frame]):
                identity = ids[j]
                code = int(codes[frame, j])
                if code not in (1, 2) or not 0 <= seed_frames[frame, j] <= frame:
                    raise ValueError("Valid observation has impossible seed/source code")
                box = boxes[frame, j].astype(float)
                scaled = box * scale
                color = colors[identity]
                draw.rectangle(tuple(scaled.tolist()), outline=color, width=3)
                label = f"O{j+1:02d} {'种子' if code == 1 else '跟踪'}"
                x = min(max(0, float(scaled[0])), tile_width - 115)
                y = min(max(0, float(scaled[1]) - 24), tile_height - 24)
                draw.rectangle((x, y, x + 112, y + 23), fill=(0, 0, 0))
                draw.text((x + 2, y + 1), label, font=label_font, fill=color)
                objects.append({"entity_id": identity, "short_id": f"O{j+1:02d}",
                    "bbox_xyxy_native": box.tolist(), "source_code": code,
                    "last_seed_frame": int(seed_frames[frame, j]), "tracking_inliers": int(inliers[frame, j])})
            record = {"frame_index": frame, "timestamp_s": float(times[frame]),
                      "decoded_native_rgb_sha256": decoded_sha, "objects": objects,
                      "native_to_tile_scale_xy": scale[:2].tolist(), "crop_xyxy_native": [0, 0, width, height]}
            pending.append((frame, image, record))
            if len(pending) == frames_per_page:
                finish_page()
        finish_page()
    finally:
        cap.release()
    if [r["frame_index"] for r in frame_records] != needed:
        raise ValueError("An observed native frame is missing from rendered evidence")
    if sha(observations_path) != observations_sha or sha(seeds_path) != seeds_sha:
        raise ValueError("Tracking sources changed during evidence rendering")
    manifest = {"schema_version": "assistant_annotation_v5_tracking_evidence", "sequence_id": uid,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_bindings": bindings,
        "renderer_sha256": sha(__file__), "render_config": config, "native_image_size": [width, height],
        "entity_legend": legend, "frame_count": count, "rendered_native_frames": len(needed),
        "valid_object_observations": int(valid.sum()), "frames": frame_records, "pages": pages,
        "every_valid_observation_frame_rendered": True, "visual_reviewed": False,
        "annotation_created": False, "human_label_valid": False, "training_allowed": False}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--refine-root", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--font")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--frames-per-page", type=int, default=12)
    args = parser.parse_args()
    manifest = render_tracking(args.root, args.refine_root, args.sequence_id, font_path=args.font,
        tile_width=args.tile_width, columns=args.columns, frames_per_page=args.frames_per_page)
    print(json.dumps({"sequence_id": manifest["sequence_id"], "rendered_native_frames": manifest["rendered_native_frames"],
                      "pages": len(manifest["pages"]), "valid_object_observations": manifest["valid_object_observations"]}))


if __name__ == "__main__":
    main()
