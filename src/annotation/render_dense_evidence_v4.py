"""Render bounded, consecutive native-frame crops for actual dense review.

Rendering evidence does not create a reviewed label. The source video and
timestamps remain authoritative; contact sheets are browsing derivatives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_request(start, end, frame_count, crop, image_size, max_frames=512):
    if any(type(v) is not int for v in (start, end, frame_count, max_frames)):
        raise ValueError("Native frame boundaries must be integers")
    if not 0 <= start < end <= frame_count or end - start > max_frames:
        raise ValueError("Invalid native range or request exceeds bounded batch size")
    if len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError("Crop must use integer native pixel coordinates")
    x0, y0, x1, y1 = crop
    width, height = image_size
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("Crop falls outside native image")


def frame_batches(start, end, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 24:
        raise ValueError("Each contact sheet must contain between 1 and 24 frames")
    return [list(range(i, min(i + batch_size, end))) for i in range(start, end, batch_size)]


def render_dense(source_root, output_root, uid, start, end, crop, *, font_path=None,
                 tile_width=512, columns=3, frames_per_page=12, max_frames=512):
    """Save every requested frame, retaining native crop pixels and provenance."""
    started = time.monotonic()
    cv2.setNumThreads(1)
    source_root, output_root = Path(source_root), Path(output_root)
    metadata_path = source_root / "sequences" / uid / "metadata.json"
    hands_path = source_root / "sequences" / uid / "hands.npz"
    metadata = json.loads(metadata_path.read_text())
    if metadata["sequence_id"] != uid:
        raise ValueError("Wrong acquisition metadata")
    image_size = metadata["image_size"]
    validate_request(start, end, metadata["frame_count"], crop, image_size, max_frames)
    batches = frame_batches(start, end, frames_per_page)
    if type(columns) is not int or not 1 <= columns <= 4 or not 128 <= tile_width <= 1024:
        raise ValueError("Invalid contact-sheet layout")
    with np.load(hands_path, allow_pickle=False) as arrays:
        timestamps = arrays["timestamps"].copy()
    if (len(timestamps) != metadata["frame_count"] or not np.isfinite(timestamps).all()
            or not np.all(np.diff(timestamps) > 0)):
        raise ValueError("Native timestamps must be finite, ordered and complete")
    video = Path(metadata["video_path"])
    actual_video_sha = digest(video)
    if actual_video_sha != metadata["video_sha256"]:
        raise ValueError("Source video changed after source metadata was recorded")
    configuration = {"native_frame_range": [start, end], "crop_xyxy_native": list(crop),
                     "tile_width": tile_width, "columns": columns, "frames_per_page": frames_per_page}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:12]
    font = ImageFont.truetype(str(font_path), 19) if font_path else ImageFont.load_default(size=19)
    caption_font = ImageFont.truetype(str(font_path), 21) if font_path else ImageFont.load_default(size=21)
    destination = output_root / "dense_evidence" / uid / f"{start:06d}_{end:06d}_{config_hash}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "crops").mkdir()
    cap = cv2.VideoCapture(str(video))
    records, pages = [], []
    x0, y0, x1, y1 = crop
    crop_width, crop_height = x1 - x0, y1 - y0
    tile_height = round(crop_height * tile_width / crop_width)
    header_height, caption_height = 64, 32
    try:
        if not cap.isOpened() or int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) != metadata["frame_count"]:
            raise ValueError("Native video cannot be opened or frame count differs")
        # A single native seek followed by consecutive decode, never seeking
        # independently at each requested frame or dropping intermediate frames.
        if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError("Native start seek failed")
        for page_index, frame_ids in enumerate(batches):
            rows = (len(frame_ids) + columns - 1) // columns
            canvas = Image.new("RGB", (columns * tile_width, header_height + rows * (tile_height + caption_height)), (18, 22, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 4), f"{uid}  原生帧 [{frame_ids[0]}, {frame_ids[-1] + 1})", font=font, fill="white")
            draw.text((10, 31), "连续每一帧；裁剪坐标保留；渲染完成不等于已复核", font=font, fill=(220, 224, 232))
            cells = []
            for position, frame_id in enumerate(frame_ids):
                before = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ok, bgr = cap.read()
                after = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if not ok or abs(before - frame_id) > .1 or abs(after - frame_id - 1) > .1:
                    raise ValueError(f"Native consecutive decode failed at frame {frame_id}")
                if bgr.shape != (image_size[1], image_size[0], 3):
                    raise ValueError("Decoded native image shape changed")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                raw_sha = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
                image = Image.fromarray(rgb[y0:y1, x0:x1])
                path = destination / "crops" / f"{frame_id:06d}.jpg"
                image.save(path, format="JPEG", quality=90, subsampling=0)
                row, column = divmod(position, columns)
                tx, ty = column * tile_width, header_height + row * (tile_height + caption_height)
                draw.text((tx + 5, ty + 3), f"原生帧 {frame_id} | {float(timestamps[frame_id]):.6f}s", font=caption_font, fill="white")
                canvas.paste(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS), (tx, ty + caption_height))
                record = {"frame_index": frame_id, "timestamp_s": float(timestamps[frame_id]),
                          "decoded_native_rgb_sha256": raw_sha, "crop_path": str(path.relative_to(output_root)),
                          "crop_sha256": digest(path), "crop_xyxy_native": list(crop),
                          "page_index": page_index, "page_cell_index": position,
                          "video_decoder_position_before": before, "video_decoder_position_after": after}
                records.append(record)
                cells.append({"frame_index": frame_id, "crop_region_on_page_xyxy": [tx, ty + caption_height, tx + tile_width, ty + caption_height + tile_height],
                              "native_to_page_scale_xy": [tile_width / crop_width, tile_height / crop_height]})
            path = destination / f"page_{page_index:04d}.jpg"
            canvas.save(path, quality=85, subsampling=1)
            pages.append({"path": str(path.relative_to(output_root)), "sha256": digest(path),
                          "frame_indices": frame_ids, "width": canvas.width, "height": canvas.height, "cells": cells})
    finally:
        cap.release()
    if [r["frame_index"] for r in records] != list(range(start, end)):
        raise AssertionError("Dense evidence contains missing or repeated native frames")
    manifest = {"schema_version": "assistant_annotation_v4_dense_evidence", "sequence_id": uid,
                "created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(source_root),
                "source_bindings": {"video_path": str(video), "video_sha256": actual_video_sha,
                                    "metadata_path": str(metadata_path), "metadata_sha256": digest(metadata_path),
                                    "hands_path": str(hands_path), "hands_sha256": digest(hands_path)},
                "renderer_path": str(Path(__file__).resolve()), "renderer_sha256": digest(__file__),
                "opencv_version": cv2.__version__, "native_image_size": image_size,
                "decoded_native_rgb_hash_definition": "SHA256 of contiguous uint8 RGB HxWx3 bytes, full native frame before crop",
                "native_to_crop_transform": [[1, 0, -x0], [0, 1, -y0], [0, 0, 1]],
                **configuration, "frames": records, "pages": pages, "rendered_frame_count": len(records),
                "every_native_frame_rendered_in_requested_range": True, "evidence_rendered": True,
                "visual_reviewed": False, "annotation_created": False, "human_label_valid": False,
                "training_allowed": False, "elapsed_seconds": time.monotonic() - started}
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"manifest_path": str(path), "manifest_sha256": digest(path),
            "rendered_frame_count": len(records), "page_count": len(pages), "elapsed_seconds": manifest["elapsed_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--crop", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--font")
    parser.add_argument("--tile-width", type=int, default=512)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--frames-per-page", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(render_dense(args.source_root, args.output_root, args.sequence_id, args.start, args.end,
                                  args.crop, font_path=args.font, tile_width=args.tile_width, columns=args.columns,
                                  frames_per_page=args.frames_per_page, max_frames=args.max_frames), ensure_ascii=False))


if __name__ == "__main__":
    main()
