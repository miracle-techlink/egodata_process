"""Freeze, acquire and audit the EgoDemo long auxiliary pool without training."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote


REPOSITORY = "LightwheelAI/EgoDemo"
UUID_RE = re.compile(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")
REQUIRED = {"info", "annotation", "bad_frame_ratio", "tasks", "episodes", "data", "head_left_video"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def identify_role(relative):
    if relative == "meta/info.json":
        return "info"
    if relative == "annotation.json":
        return "annotation"
    if relative == "bad_frame_ratio.json":
        return "bad_frame_ratio"
    if relative == "meta/tasks.parquet":
        return "tasks"
    if relative == "meta/stats.json":
        return "stats"
    if relative.startswith("meta/episodes/") and relative.endswith(".parquet"):
        return "episodes"
    if relative.startswith("data/") and relative.endswith(".parquet"):
        return "data"
    if relative.startswith("videos/observation.images.head_left/") and relative.endswith(".mp4"):
        return "head_left_video"
    return None


def asset_entry(path, repository_root, role):
    with path.open("rb") as stream:
        prefix = stream.read(256)
    pointer = re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n", prefix)
    return {
        "role": role, "source_path": path.relative_to(repository_root).as_posix(),
        "expected_sha256": pointer[1].decode() if pointer else sha256(path),
        "expected_size": int(pointer[2]) if pointer else path.stat().st_size,
        "source_storage": "git_lfs" if pointer else "git_blob",
    }


def choose_candidates(records, count, byte_limit):
    pending = list(records)
    selected, task_counts, scene_counts = [], Counter(), Counter()
    total = 0
    while pending and len(selected) < count:
        pending.sort(key=lambda record: (
            not 20 <= record["manifest_duration_seconds"] <= 60,
            task_counts[record["task_name"]], scene_counts[record["scene_id"]],
            record["planned_bytes"], record["acquisition_id"],
        ))
        record = pending.pop(0)
        if total + record["planned_bytes"] > byte_limit:
            continue
        selected.append(record)
        total += record["planned_bytes"]
        task_counts[record["task_name"]] += 1
        scene_counts[record["scene_id"]] += 1
    return selected


def catalog(root, raw, pilot, output, count, byte_limit):
    if output.exists():
        raise ValueError("Frozen catalog exists; do not overwrite")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    raw_ids = {path.name for path in raw.rglob("*") if path.is_dir() and UUID_RE.fullmatch(path.name)}
    old_ids = {path.parent.parent.name for path in pilot.rglob("info.json")}
    manifest = root / "EgoProStandard/manifests/train.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    eligible, excluded = [], []
    for row in rows:
        identifier = row["source_episode_uuid"]
        if identifier not in raw_ids:
            continue
        reason = None
        if identifier in old_ids:
            reason = "old_pilot"
        duration = float(row["duration_s"])
        if not 10 <= duration <= 60:
            reason = reason or "outside_10_60_seconds"
        base = root / row["lerobot_path"]
        info = json.loads((base / "meta/info.json").read_text())
        annotation = json.loads((base / "annotation.json").read_text())
        description = annotation.get("episode_description", "")
        if info.get("robot_type") or re.search(r"\brobot(?:ic)?\b|机械手|机器人", description, re.I):
            reason = "explicit_robot_metadata"
        if reason:
            excluded.append({"acquisition_id": identifier, "reason": reason})
            continue
        assets = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                role = identify_role(path.relative_to(base).as_posix())
                if role:
                    assets.append(asset_entry(path, root, role))
        if not REQUIRED <= {asset["role"] for asset in assets}:
            excluded.append({"acquisition_id": identifier, "reason": "incomplete_pairing"})
            continue
        eligible.append({
            "sequence_id": "egodemo_standard_" + identifier, "dataset": "EgoDemo",
            "acquisition_id": identifier, "subject_id": None, "scene_id": row.get("scene_id") or None,
            "environment_id": row.get("environment_id") or None, "object_ids": [],
            "action_labels": [], "task_name": row["task_name"], "source_task_label": row["task_name"],
            "task_namespace": "LightwheelAI/EgoDemo/official_task_name",
            "source_episode_description": description,
            "action_label_note": "Official task and subtask text retained, not mapped to seven-class HOI",
            "source_ids": {"demo_episode_id": row["demo_episode_id"], "source_episode_uuid": identifier,
                           "source_repo": row.get("source_repo"), "repository": REPOSITORY,
                           "revision": revision, "lerobot_path": row["lerobot_path"]},
            "camera_id": "head_left", "view": "egocentric", "variant": "EgoProStandard",
            "manifest_frame_count": int(row["frame_count"]), "manifest_fps": float(row["fps"]),
            "manifest_duration_seconds": duration, "frame_count": None, "duration_seconds": None,
            "fps": None, "timestamp_source": "pending_source_parquet_and_standard_video_PTS",
            "human_verified": False, "human_review": "pending_user_contactsheet_review",
            "machine_exclusion": "explicit robot metadata excluded; visual check still required",
            "split": None, "pool": "auxiliary", "joint_training_ready": False,
            "training_allowed": False, "download_status": "not_started",
            "capabilities": {"rgb": False, "depth": False, "intrinsics": False, "mesh": False,
                             "hand_metric_gt": False, "object_pose_gt": False,
                             "hand_world_reference": False,
                             "hand_observations": False, "object_observations": False,
                             "camera_motion_compensated": False},
            "missing_capabilities": ["depth", "intrinsics", "mesh", "object_pose_gt", "camera_motion_compensated"],
            "quality": {**{key: "pending" for key in ["decode", "timestamps", "calibration", "coordinates",
                                                         "annotation_alignment"]}, "visual_review": "needs_full_review"},
            "planned_assets": assets, "assets": [], "planned_bytes": sum(asset["expected_size"] for asset in assets),
        })
    selected = choose_candidates(eligible, count, byte_limit)
    document = {
        "schema_version": "human_pose_dataset_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY, "revision": revision, "catalog_frozen_before_download": True,
        "official_endpoint": "https://huggingface.co", "catalog_manifest_sha256": sha256(manifest),
        "old_pilot_excluded": sorted(old_ids), "raw_uuid_count": len(raw_ids),
        "eligible_count": len(eligible), "excluded": excluded,
        "unselected_eligible_ids": [record["acquisition_id"] for record in eligible if record not in selected],
        "byte_limit": byte_limit, "planned_bytes": sum(record["planned_bytes"] for record in selected),
        "training_allowed": False, "records": selected,
    }
    save_json(output, document, exclusive=True)
    print(json.dumps({"frozen": len(selected), "bytes": document["planned_bytes"], "eligible": len(eligible),
                      "tasks": len({record["task_name"] for record in selected}),
                      "scenes": len({record["scene_id"] for record in selected})}), flush=True)


def download_asset(asset, root, revision, token, deadline):
    import requests

    destination = root / "assets" / asset["source_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == asset["expected_size"] and sha256(destination) == asset["expected_sha256"]:
            return
        raise ValueError("Existing payload fails expected SHA; refusing overwrite")
    if time.monotonic() >= deadline:
        raise TimeoutError("download_deadline")
    url = "https://huggingface.co/datasets/" + REPOSITORY + "/resolve/" + revision + "/" + quote(asset["source_path"], safe="/")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest, size = hashlib.sha256(), 0
    try:
        with requests.get(url, headers={"Authorization": "Bearer " + token}, stream=True, timeout=(12, 30)) as response:
            response.raise_for_status()
            with temporary.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("download_deadline")
                    size += len(chunk)
                    if size > asset["expected_size"]:
                        raise ValueError("Response exceeds frozen asset size")
                    stream.write(chunk)
                    digest.update(chunk)
        if size != asset["expected_size"] or digest.hexdigest() != asset["expected_sha256"]:
            raise ValueError("Payload size or source SHA mismatch")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def fetch(catalog_path, root, token_path, workers, seconds):
    document = json.loads(catalog_path.read_text())
    if document["planned_bytes"] > document["byte_limit"]:
        raise ValueError("Catalog exceeds byte limit")
    token = token_path.read_text().strip()
    if not token:
        raise ValueError("Empty credential")
    deadline = time.monotonic() + seconds
    receipt = {"catalog_sha256": sha256(catalog_path), "files": [], "training_allowed": False}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_asset, asset, root, document["revision"], token, deadline): asset
                   for record in document["records"] for asset in record["planned_assets"]}
        for future in as_completed(futures):
            asset = futures[future]
            result = {"source_path": asset["source_path"]}
            try:
                future.result()
                result.update(status="downloaded", bytes=asset["expected_size"], sha256=asset["expected_sha256"])
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__)
            receipt["files"].append(result)
            save_json(root / "download_receipt.json", receipt)
            print(json.dumps({"done": len(receipt["files"]), "total": len(futures), "status": result["status"],
                              "role": asset["role"]}), flush=True)


def compare_timestamps(timestamps, video_pts, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid official fps")
    finite = all(math.isfinite(value) for value in timestamps + video_pts)
    ordered = all(right > left for values in [timestamps, video_pts] for left, right in zip(values, values[1:]))
    if not finite or not ordered or not timestamps or len(timestamps) != len(video_pts):
        return {"pass": False, "reason": "nonfinite_nonmonotonic_or_count_mismatch"}
    differences = [abs(source - video) for source, video in zip(timestamps, video_pts)]
    tolerance = max(0.002, 0.1 / fps)
    return {"pass": max(differences) <= tolerance, "max_absolute_error_seconds": max(differences),
            "tolerance_seconds": tolerance, "source_first": timestamps[0], "source_last": timestamps[-1],
            "video_first": video_pts[0], "video_last": video_pts[-1], "compared_frames": len(timestamps)}


def command_json(arguments):
    return json.loads(subprocess.check_output(arguments, text=True, stderr=subprocess.DEVNULL, timeout=90))


def write_timestamp_asset(record, root, timestamps, video_pts=None):
    path = root / "timestamps" / record["acquisition_id"] / "timestamps.json"
    payload = {"timestamps": timestamps, "unit": "seconds", "source": "official_frame_parquet.timestamp",
               "frame_count": len(timestamps), "source_episode_uuid": record["acquisition_id"],
               "source_parquet_sha256": [asset["sha256"] for asset in record["assets"] if asset["role"] == "data"],
               "paired_standard_video_alignment": record["timestamp_alignment"]}
    if video_pts is not None:
        payload["video_pts"] = video_pts
    save_json(path, payload)
    record["timestamp_source_detail"] = "source_parquet.timestamp + paired_standard_video_PTS"
    record["timestamp_source"] = "source_timestamps"
    record["duration_definition"] = "frame_count_over_fps"
    record.setdefault("evidence", {})["timestamp_source"] = {
        "description": "Actual official frame-parquet seconds, compared against every decoded paired standard-video PTS",
        "asset_role": "timestamps",
    }
    record["assets"].append({"role": "timestamps", "path": str(path.resolve()), "sha256": sha256(path),
                             "size_bytes": path.stat().st_size, "generated": True})


def audit_record(record, root):
    import pyarrow.parquet as parquet

    spec = importlib.util.spec_from_file_location("egodemo_native_import", Path(__file__).with_name("import_egodemo.py"))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    record["quality"]["visual_review"] = "needs_full_review"

    paths = {}
    for asset in record["planned_assets"]:
        path = root / "assets" / asset["source_path"]
        if not path.is_file():
            raise FileNotFoundError(asset["source_path"])
        digest = sha256(path)
        if digest != asset["expected_sha256"] or path.stat().st_size != asset["expected_size"]:
            raise ValueError("SHA mismatch")
        record["assets"].append({"role": "rgb" if asset["role"] == "head_left_video" else asset["role"],
                                 "source_role": asset["role"], "path": str(path.resolve()),
                                 "sha256": digest, "size_bytes": path.stat().st_size})
        paths.setdefault(asset["role"], []).append(path)
    record["download_status"] = "downloaded"
    info = json.loads(paths["info"][0].read_text())
    rows = [row for path in paths["data"] for row in parquet.read_table(path).to_pylist()]
    bad_frames = json.loads(paths["bad_frame_ratio"][0].read_text())
    native_sequence = native.normalize_sequence(rows, info, bad_frames)
    record["hand_valid_frame_counts"] = native_sequence["hand_valid"].sum(axis=0).tolist()
    record["hand_bad_frame_count"] = len(bad_frames["hand_bad_frame_details"])
    times = [float(row["timestamp"]) for row in rows]
    count, fps = len(rows), float(info["fps"])
    record.update(frame_count=count, fps=fps, timestamp_source="source_parquet.timestamp + paired_standard_video_PTS")
    record["duration_seconds"] = times[-1] - times[0] + 1 / fps
    video = paths["head_left_video"][0]
    probe = command_json(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
                          "-show_frames", "-show_streams", "-show_entries",
                          "frame=best_effort_timestamp_time:stream=width,height,nb_frames,duration,avg_frame_rate,r_frame_rate",
                          "-of", "json", str(video)])
    points = [float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
    alignment = compare_timestamps(times, points, fps)
    frame_indices = [int(row["frame_index"]) for row in rows]
    structural = (count == int(info["total_frames"]) == record["manifest_frame_count"]
                  and frame_indices == list(range(count)) and len({row["episode_index"] for row in rows}) == 1)
    alignment["structural_alignment"] = structural
    record["timestamp_alignment"] = alignment
    write_timestamp_asset(record, root, times, points)
    record["video_probe"] = probe["streams"][0]
    record["quality"]["timestamps"] = "pass" if alignment["pass"] else "fail"
    annotation = json.loads(paths["annotation"][0].read_text())
    events = native.validate_events(annotation)
    events_valid = all(0 <= float(event["start_time"]) <= float(event["end_time"]) <= record["duration_seconds"] + 1 / fps for event in events)
    record["source_action_events"] = events
    record["quality"]["annotation_alignment"] = "pass" if alignment["pass"] and structural and events_valid else "fail"
    for role in ["tasks", "episodes"]:
        for path in paths[role]:
            parquet.read_table(path)
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(video),
                             "-map", "0:v:0", "-f", "null", "-"], capture_output=True, timeout=120)
    record["quality"]["decode"] = "pass" if result.returncode == 0 and not result.stderr.strip() else "fail"
    record["decode_returncode"] = result.returncode
    record["decode_error_bytes"] = len(result.stderr)
    hand_fields = ["observation.state.hand_left_world", "observation.state.hand_right_world"]
    record["hand_world_annotation_present"] = all(key in rows[0] for key in hand_fields)
    record["hand_coordinate_note"] = "Official world-coordinate hand annotation; not object GT; metric/camera calibration not independently certified"
    record["capabilities"].update(rgb=True, hand_observations=False,
                                  hand_world_reference=record["hand_world_annotation_present"])
    record["quality"]["calibration"] = "fail"
    contact = root / "contacts" / (record["acquisition_id"] + ".jpg")
    contact.parent.mkdir(parents=True, exist_ok=True)
    selected = [0, count // 2, count - 1]
    selection = "+".join("eq(n\\,%d)" % index for index in selected)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(video),
                    "-vf", "select=" + selection + ",scale=384:-2,tile=3x1", "-frames:v", "1", "-y", str(contact)],
                   check=True, capture_output=True, timeout=90)
    record["contactsheet"] = {"path": str(contact.resolve()), "sha256": sha256(contact),
                              "frame_indices": selected, "timestamps": [times[index] for index in selected]}
    record["technical_long_pass"] = record["duration_seconds"] >= 10 and all(record["quality"][key] == "pass" for key in ["decode", "timestamps", "annotation_alignment"])
    record["qualified_long_human"] = False


def contact_pages(document, root):
    from PIL import Image, ImageDraw, ImageFont

    records = [record for record in document["records"] if record.get("contactsheet")]
    pages = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(records), 6):
        group = records[start:start + 6]
        images = [Image.open(record["contactsheet"]["path"]).convert("RGB") for record in group]
        height = max(image.height for image in images) + 58
        page = Image.new("RGB", (1152, height * len(group)), (245, 245, 245))
        draw = ImageDraw.Draw(page)
        for offset, (record, image) in enumerate(zip(group, images)):
            position = offset * height
            draw.text((8, position + 2), record["acquisition_id"] + " | " + record["task_name"], fill="black", font=font)
            draw.text((8, position + 26), "%s | %.2fs | frames %s | PREVIEW ONLY; needs full review" % (
                record["scene_id"], record["duration_seconds"], record["contactsheet"]["frame_indices"]), fill="black", font=font)
            page.paste(image, (0, position + 58))
        path = root / "contacts" / ("contactsheet_%02d.jpg" % (start // 6 + 1))
        page.save(path, quality=88)
        pages.append({"path": str(path.resolve()), "sha256": sha256(path), "acquisition_ids": [record["acquisition_id"] for record in group]})
    document["contactsheet_pages"] = pages


def attach_server_receipt(document, root):
    receipt_path = root / "server_sha_receipt.json"
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text())
    matches = {item["source_path"]: item for item in receipt["files"]}
    for record in document["records"]:
        for asset in record["assets"]:
            if asset.get("generated"):
                continue
            source_path = Path(asset["path"]).relative_to((root / "assets").resolve()).as_posix()
            match = matches.get(source_path)
            if match and match["sha256"] == asset["sha256"] and match["size_bytes"] == asset["size_bytes"]:
                asset["server_path"] = match["server_path"]
                asset["server_sha_verified"] = True
    document["server_replica"] = {"root": receipt["server_root"], "receipt_path": str(receipt_path.resolve()),
                                  "receipt_sha256": sha256(receipt_path), "all_match": receipt["all_match"]}


def audit(catalog_path, root, workers=4):
    document = json.loads(catalog_path.read_text())
    document["catalog_sha256"] = sha256(catalog_path)
    document["asset_root"] = str(root.resolve())
    def check_record(record):
        record["quality"]["visual_review"] = "needs_full_review"
        try:
            audit_record(record, root)
        except Exception as error:
            record["audit_error_type"] = type(error).__name__
            if record["download_status"] != "downloaded":
                record["download_status"] = "incomplete_or_failed"
            record["technical_long_pass"] = False
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(check_record, record) for record in document["records"]]
        for future in as_completed(futures):
            record = future.result()
            print(json.dumps({"uuid": record["acquisition_id"], "status": record["download_status"],
                              "technical_long_pass": record["technical_long_pass"], "error": record.get("audit_error_type")}), flush=True)
    document["summary"] = {
        "candidates": len(document["records"]),
        "downloaded": sum(record["download_status"] == "downloaded" for record in document["records"]),
        "technical_long_pass": sum(record["technical_long_pass"] for record in document["records"]),
        "human_verified": 0, "qualified_long_human": 0, "joint_training_ready": 0,
        "downloaded_asset_bytes": sum(asset["size_bytes"] for record in document["records"] for asset in record["assets"] if not asset.get("generated")),
        "tasks": len({record["task_name"] for record in document["records"]}),
        "scenes": len({record["scene_id"] for record in document["records"]}),
    }
    contact_pages(document, root)
    attach_server_receipt(document, root)
    save_json(root / "candidates.json", document)
    print(json.dumps(document["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["catalog", "fetch", "audit"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--byte-limit", type=int, default=5_000_000_000)
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".cache/huggingface/token")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=540)
    args = parser.parse_args()
    if args.mode == "catalog":
        catalog(args.root, args.raw, args.pilot, args.catalog, args.count, args.byte_limit)
    elif args.mode == "fetch":
        fetch(args.catalog, args.root, args.token_file, args.workers, args.seconds)
    else:
        audit(args.catalog, args.root, min(args.workers, 4))


if __name__ == "__main__":
    main()
