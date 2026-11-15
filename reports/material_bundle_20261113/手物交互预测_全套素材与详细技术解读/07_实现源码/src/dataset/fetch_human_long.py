"""Bounded, data-only HO-Cap acquisition; never trains or chooses a model.

The HF automatic ``train`` partition is NOT the official benchmark split.
Use the publisher's subject/sequence indices to exclude held-out captures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request


REPOSITORY = "JWRoboticsVision/HO-Cap-Dataset"
REVISION = "2b24836d5e51ad39e56ed4db3fc0c166e755332e"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
MAX_BYTES = 15_000_000_000
MAX_SECONDS = 25 * 60
MEMBER_PATTERN = re.compile(r"(\d{12}|hololens_[a-z0-9]+)-(\d{6})\.(color\.jpg|depth\.png|label\.npy)")
HEADER_PATTERN = re.compile(rb"(?:[0-9]{12}|hololens_[a-z0-9]+)-[0-9]{6}\.(?:color\.jpg|depth\.png|label\.npy)\x00")
CAMERA_SELECTION_BASIS = "Primary assistant visual inspection of RGB-only contact sheet before training; no GT/model error selection"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return str(path)


def capture_split(subject: str, sequence: str, info: dict, splits: dict) -> str:
    identity = (info["subject_ids"].index(subject), info["sequence_ids"].index(sequence))
    memberships = {
        split for split, rows in splits.items()
        if any(tuple(row[:2]) == identity for row in rows)
    }
    if memberships == {"train"}:
        return "train"
    if "test" in memberships:
        return "test"
    if "valid" in memberships:
        return "valid"
    return "unknown"


def select_archives(tree: list, info: dict, splits: dict, subjects: list[str]) -> list:
    selected = []
    for subject in subjects:
        candidates = []
        for item in tree:
            match = re.fullmatch(r"(subject_\d+)-(\d{8}_\d{6})\.tar", item["path"])
            if not match or match[1] != subject:
                continue
            if capture_split(subject, match[2], info, splits) != "train":
                continue
            candidates.append({
                "name": item["path"], "size": item["size"],
                "sha256": item["lfs"]["oid"], "subject_id": subject,
                "acquisition_id": match[2], "official_split": "train",
                "url": BASE_URL + item["path"],
            })
        if not candidates:
            raise ValueError(f"No exclusively training capture for {subject}")
        selected.append(min(candidates, key=lambda item: (item["size"], item["name"])))
    return selected


def plan(root: Path, subjects: list[str]) -> dict:
    import yaml

    official = root / "official"
    tree = json.loads((official / "hocap_hf_tree.json").read_text())
    info = yaml.safe_load((official / "hocap_info.yaml").read_text())
    splits = json.loads((official / "hocap_hpe.json").read_text())
    selected = select_archives(tree, info, splits, subjects)
    for name in ("models.tar", "calibration.tar"):
        item = next(item for item in tree if item["path"] == name)
        selected.append({"name": name, "size": item["size"],
                         "sha256": item["lfs"]["oid"], "url": BASE_URL + name})
    total = sum(item["size"] for item in selected)
    if total > MAX_BYTES - 100_000_000:
        raise ValueError("Plan exceeds 15GB including metadata/Range allowance")
    result = {
        "dataset": "HO-Cap", "repository": REPOSITORY, "revision": REVISION,
        "license": "CC-BY-4.0", "selected": selected, "planned_bytes": total,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "selection_rule": "smallest complete training archive per requested subject; no model scores",
        "split_warning": "HF automatic train is not the official split; use official config indices",
        "training_started": False, "test_reference_downloaded": False,
    }
    write_json(root / "download_plan.json", result)
    return result


def read_range(url: str, start: int, size: int, total: int, timeout: float = 30) -> bytes:
    if size <= 0 or start < 0 or start + size > total:
        raise ValueError("Range outside archive")
    request = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{start + size - 1}", "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        expected = f"bytes {start}-{start + size - 1}/{total}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not honor exact Range; refusing full archive")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("Truncated or oversized Range response")
    return data


def fetch_selected_labels(root: Path, seconds: int = 600, plan_name: str = "download_plan.json") -> dict:
    selection = json.loads((root / plan_name).read_text())["selected"]
    captures = {item["name"].removesuffix(".tar") for item in selection if "subject_id" in item}
    tree = json.loads((root / "official/hocap_hf_tree.json").read_text())
    archive = next(item for item in tree if item["path"] == "sequence_labels.tar")
    url = BASE_URL + "sequence_labels.tar"
    deadline = time.monotonic() + min(seconds, MAX_SECONDS)
    result = {"archive_url": url, "archive_sha256_publisher": archive["lfs"]["oid"],
              "archive_size": archive["size"], "range_bytes_received": 0, "members": [],
              "test_payloads_downloaded": False, "status": "running"}
    offset = 0
    destination = root / "labels_selected"
    destination.mkdir(exist_ok=True)
    try:
        while offset + 512 <= archive["size"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Label acquisition time budget exhausted")
            header = read_range(url, offset, 512, archive["size"])
            result["range_bytes_received"] += len(header)
            if header == bytes(512):
                break
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            name = safe_name(member.name)
            if member.isfile() and any(name.startswith(capture + ".") for capture in captures):
                data = read_range(url, offset + 512, member.size, archive["size"])
                result["range_bytes_received"] += len(data)
                output = destination / name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(data)
                result["members"].append({"name": name, "bytes": len(data),
                                          "sha256": hashlib.sha256(data).hexdigest(),
                                          "archive_offset": offset + 512})
                print("label", name, member.size, flush=True)
            offset += 512 + math.ceil(member.size / 512) * 512
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(root / "labels_receipt.json", result)
    return result


def fast_labels(root: Path, seconds: int, plan_name: str) -> dict:
    import requests

    selected = json.loads((root / plan_name).read_text())["selected"]
    captures = {item["name"].removesuffix(".tar") for item in selected if "subject_id" in item}
    tree = json.loads((root / "official/hocap_hf_tree.json").read_text())
    archive = next(item for item in tree if item["path"] == "sequence_labels.tar")
    stable_url = BASE_URL + "sequence_labels.tar"
    current_url = stable_url
    deadline = time.monotonic() + seconds
    result = {"source": stable_url, "members": [], "range_bytes_received": 0, "status": "running",
              "test_payloads_downloaded": False, "index": []}
    session = requests.Session()

    def get_bytes(offset, size):
        nonlocal current_url
        if time.monotonic() >= deadline:
            raise TimeoutError("Labels deadline exceeded")
        expected = f'bytes {offset}-{offset + size - 1}/{archive["size"]}'
        response = session.get(current_url, headers={"Range": f"bytes={offset}-{offset + size - 1}"},
                               timeout=min(20, max(1, deadline - time.monotonic())))
        response.raise_for_status()
        current_url = response.url
        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Incorrect label Range response")
        data = response.content
        if len(data) != size:
            raise ValueError("Incomplete label Range")
        result["range_bytes_received"] += len(data)
        return data

    offset = 0
    destination = root / "labels_selected"
    destination.mkdir(exist_ok=True)
    try:
        while offset + 512 <= archive["size"]:
            header = get_bytes(offset, 512)
            if header == bytes(512):
                break
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            if member.type == tarfile.XHDTYPE:
                padded = math.ceil(member.size / 512) * 512
                combined = get_bytes(offset + 512, padded + 512)
                offset += 512 + padded
                member = tarfile.TarInfo.frombuf(combined[-512:], "utf-8", "strict")
            name = safe_name(member.name)
            result["index"].append({"name": name, "offset": offset + 512, "bytes": member.size})
            if member.isfile() and any(name.startswith(capture + ".") for capture in captures):
                output = destination / name
                if output.exists() and output.stat().st_size == member.size:
                    data = output.read_bytes()
                else:
                    data = get_bytes(offset + 512, member.size)
                    output.write_bytes(data)
                result["members"].append({"name": name, "bytes": len(data),
                                          "sha256": hashlib.sha256(data).hexdigest(),
                                          "archive_offset": offset + 512})
                print("label", name, flush=True)
            offset += 512 + math.ceil(member.size / 512) * 512
            result["next_offset"] = offset
            if len(result["index"]) % 16 == 0:
                write_json(root / "labels_v2_receipt.json", result)
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        session.close()
    write_json(root / "labels_v2_receipt.json", result)
    return {"status": result["status"], "members": len(result["members"]), "error": result.get("error")}


def download(root: Path, seconds: int) -> dict:
    manifest = json.loads((root / "download_plan.json").read_text())
    files = manifest["selected"]
    if sum(item["size"] for item in files) > MAX_BYTES - 100_000_000:
        raise ValueError("Download budget exceeded")
    if shutil.disk_usage(root).free < 2 * manifest["planned_bytes"] + 1_000_000_000:
        raise ValueError("Insufficient space for archives and extraction")
    archives = root / "archives"
    archives.mkdir(exist_ok=True)
    remaining = []
    for item in files:
        path = archives / safe_name(item["name"])
        if path.exists():
            if path.stat().st_size == item["size"] and sha256_file(path) == item["sha256"]:
                continue
            raise ValueError(f"Existing incomplete/mismatched archive; inspect before retry: {path}")
        remaining.append(item)
    input_file = root / "aria2_inputs.txt"
    input_file.write_text("".join(
        f'{item["url"]}\n out={item["name"]}\n checksum=sha-256={item["sha256"]}\n'
        for item in remaining
    ))
    command = ["aria2c", f"--input-file={input_file}", f"--dir={archives}",
               "--max-concurrent-downloads=3", "--max-connection-per-server=4", "--split=4",
               "--file-allocation=none", "--allow-overwrite=false", "--auto-file-renaming=false",
               "--max-tries=1", "--continue=false", "--console-log-level=warn",
               "--summary-interval=20", "--download-result=full", "--check-integrity=true"]
    started = time.monotonic()
    status = "complete"
    returncode = 0
    if remaining:
        with (root / "aria2_download.log").open("w") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=min(seconds, MAX_SECONDS), check=False)
                returncode = process.returncode
                if returncode:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status = "time_budget_exhausted"
    receipts = []
    for item in files:
        path = archives / item["name"]
        size = path.stat().st_size if path.exists() else 0
        digest = sha256_file(path) if size == item["size"] else None
        receipts.append({**item, "local_bytes": size, "local_sha256": digest,
                         "verified": size == item["size"] and digest == item["sha256"]})
    result = {"status": status, "returncode": returncode, "elapsed_seconds": time.monotonic() - started,
              "files": receipts, "training_started": False}
    write_json(root / "download_receipt.json", result)
    return result


def continuity(frame_ids: list[int], fps: float, minimum_seconds: float = 10) -> dict:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS requires a positive, documented source")
    unique = sorted(set(frame_ids))
    contiguous = bool(unique) and len(unique) == len(frame_ids) and unique == list(range(unique[0], unique[-1] + 1))
    span = (len(unique) - 1) / fps if unique else 0
    return {"frame_count": len(frame_ids), "first_frame": unique[0] if unique else None,
            "last_frame": unique[-1] if unique else None, "contiguous": contiguous,
            "duration_seconds": span, "duration_convention": "(last_frame-first_frame)/fps",
            "eligible_long_continuous": contiguous and span >= minimum_seconds}


class BoundedReader:
    def __init__(self, stream, limit: int, deadline: float):
        self.stream = stream
        self.limit = limit
        self.deadline = deadline
        self.received = 0

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Acquisition time budget exhausted")
        remaining = self.limit - self.received
        if remaining <= 0:
            raise ValueError("Camera byte budget exhausted")
        data = self.stream.read(min(size if size >= 0 else remaining, remaining))
        self.received += len(data)
        return data


def extract_first_camera(stream, destination: Path, deadline: float, archive_offset: int = 0) -> dict:
    camera = None
    files = []
    next_camera_seen = False
    frames = {"color.jpg": [], "depth.png": [], "label.npy": []}
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if time.monotonic() >= deadline:
                raise TimeoutError("Camera acquisition deadline exceeded")
            safe_name(member.name)
            match = MEMBER_PATTERN.fullmatch(member.name)
            if not member.isfile() or not match:
                raise ValueError(f"Unexpected HO-Cap member: {member.name}")
            if camera is None:
                camera = match[1]
            if match[1] != camera:
                next_camera_seen = True
                break
            frame = int(match[2])
            modality = match[3]
            if frame in frames[modality]:
                raise ValueError("Duplicate source frame")
            if member.size > 10_000_000:
                raise ValueError("Unexpectedly large individual frame")
            payload = archive.extractfile(member).read()
            if len(payload) != member.size:
                raise ValueError("Truncated frame")
            output = destination / camera / member.name
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and sha256_file(output) != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"Refusing to replace different source bytes: {output}")
            output.write_bytes(payload)
            frames[modality].append(frame)
            files.append({"path": output.relative_to(destination).as_posix(),
                          "archive_member": member.name, "archive_offset": archive_offset + member.offset_data,
                          "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    unique = sorted(frames["color.jpg"])
    complete = (bool(unique) and unique == list(range(len(unique)))
                and all(sorted(values) == unique for values in frames.values()) and next_camera_seen)
    return {"camera_id": camera, "view": "third_person_fixed" if camera and camera.isdigit() else "first_person",
            "frame_count": len(unique), "first_frame": unique[0] if unique else None,
            "last_frame": unique[-1] if unique else None, "contiguous_all_modalities": complete,
            "next_camera_boundary_seen": next_camera_seen, "files": files}


def cameras(root: Path, seconds: int, plan_name: str = "download_plan.json") -> dict:
    selected = json.loads((root / plan_name).read_text())["selected"]
    selected = [item for item in selected if "subject_id" in item]
    limit = 350_000_000
    if len(selected) * limit > MAX_BYTES - 5_000_000_000:
        raise ValueError("Camera plan exceeds download budget")
    if shutil.disk_usage(root).free < len(selected) * limit + 1_000_000_000:
        raise ValueError("Insufficient disk space")
    deadline = time.monotonic() + min(seconds, MAX_SECONDS)

    def fetch(item):
        destination = root / "captures" / item["subject_id"] / item["acquisition_id"]
        destination.mkdir(parents=True, exist_ok=True)
        previous_path = destination / "receipt.json"
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            if previous.get("status") == "complete" and all(
                (destination / asset["path"]).is_file()
                and (destination / asset["path"]).stat().st_size == asset["bytes"]
                and sha256_file(destination / asset["path"]) == asset["sha256"]
                for asset in previous.get("files", [])
            ) and previous.get("files"):
                return previous
            raise ValueError(f"Existing incomplete camera; inspect before retry: {destination}")
        size = min(limit, item["size"])
        request = urllib.request.Request(item["url"], headers={
            "Range": f"bytes=0-{size - 1}", "Accept-Encoding": "identity",
        })
        reader = None
        result = {**item, "quality_status": "candidate", "full_archive_sha_verified": False}
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f'bytes 0-{size - 1}/{item["size"]}':
                    raise ValueError("Server ignored exact camera prefix Range")
                reader = BoundedReader(response, size, deadline)
                result.update(extract_first_camera(reader, destination, deadline))
            result["status"] = "complete" if result["contiguous_all_modalities"] else "incomplete"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
        result["network_bytes_read"] = reader.received if reader else 0
        write_json(destination / "receipt.json", result)
        print(item["name"], result["status"], result.get("frame_count"), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, selected))
    result = {"dataset": "HO-Cap", "records": records, "training_started": False,
              "test_reference_downloaded": False,
              "network_bytes_read": sum(item["network_bytes_read"] for item in records)}
    suffix = "" if plan_name == "download_plan.json" else "_" + Path(plan_name).stem
    write_json(root / f"camera_receipts{suffix}.json", result)
    return result


def inventory(root: Path) -> dict:
    receipts = json.loads((root / "download_receipt.json").read_text())
    sequences = []
    for item in receipts["files"]:
        if not item["verified"] or "subject_id" not in item:
            continue
        streams = {}
        with tarfile.open(root / "archives" / item["name"], "r:") as archive:
            for member in archive:
                safe_name(member.name)
                match = MEMBER_PATTERN.fullmatch(member.name)
                if match:
                    camera, frame, extension = match.groups()
                    streams.setdefault(camera, {}).setdefault(extension, []).append(int(frame))
        sequences.append({**item, "streams": {
            camera: {extension: {"frame_count": len(frames), "first_frame": min(frames),
                                  "last_frame": max(frames),
                                  "contiguous": len(frames) == len(set(frames)) == max(frames) - min(frames) + 1}
                     for extension, frames in types.items()}
            for camera, types in streams.items()
        }, "quality_status": "candidate", "fps": None,
            "missing_verification": ["source FPS", "3D GT accuracy", "human visual inspection"]})
    result = {"dataset": "HO-Cap", "sequences": sequences, "training_started": False}
    write_json(root / "archive_inventory.json", result)
    return result


def tar_headers(data: bytes, offset: int) -> list:
    headers = []
    for match in HEADER_PATTERN.finditer(data):
        position = match.start()
        absolute = offset + position
        if absolute % 512 or position + 512 > len(data):
            continue
        try:
            member = tarfile.TarInfo.frombuf(data[position:position + 512], "utf-8", "strict")
        except (tarfile.HeaderError, UnicodeError):
            continue
        identity = MEMBER_PATTERN.fullmatch(member.name)
        if identity and member.isfile():
            headers.append({"offset": absolute, "name": member.name, "camera": identity[1],
                            "frame": int(identity[2]), "modality": identity[3], "size": member.size})
    return headers


def locate_camera_start(read_at, total: int, camera: str, window: int = 262144) -> int:
    lower = 0
    upper = total
    for iteration in range(40):
        if upper - lower <= 2 * window:
            offset = lower // 512 * 512
            size = min(total - offset, upper - offset + window)
        else:
            offset = ((lower + upper) // 2) // 512 * 512
            size = min(window, total - offset)
        headers = tar_headers(read_at(offset, size), offset)
        if not headers:
            raise ValueError("No valid TAR headers in bounded probe; refusing full archive scan")
        target = next((item for item in headers if item["camera"] == camera
                       and item["frame"] == 0 and item["modality"] == "color.jpg"), None)
        if target:
            return target["offset"]
        earlier = [item for item in headers if item["camera"] < camera]
        later = [item for item in headers if item["camera"] >= camera]
        new_lower = max(lower, earlier[-1]["offset"] + 512) if earlier else lower
        new_upper = min(upper, later[0]["offset"]) if later else upper
        if new_lower == lower and new_upper == upper:
            raise ValueError("Camera index search made no progress")
        lower, upper = new_lower, new_upper
    raise ValueError("Camera index probe limit exhausted")


class NetworkBudget:
    def __init__(self, limit: int, deadline: float):
        self.limit = limit
        self.deadline = deadline
        self.reserved = 0
        self.lock = threading.Lock()

    def reserve(self, size: int) -> None:
        with self.lock:
            if size <= 0 or self.reserved + size > self.limit:
                raise ValueError("Shared network request budget exhausted")
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Shared acquisition deadline exceeded")
            self.reserved += size


def local_index_range(path: Path, offset: int, size: int) -> bytes | None:
    if not path.is_file() or offset < 4096 or offset + size + 4096 > path.stat().st_size:
        return None
    descriptor = os.open(path, os.O_RDONLY)
    try:
        start = os.lseek(descriptor, offset - 4096, os.SEEK_DATA)
        end = os.lseek(descriptor, start, os.SEEK_HOLE)
        if start > offset - 4096 or end < offset + size + 4096:
            return None
        data = os.pread(descriptor, size, offset)
        return data if len(data) == size else None
    except OSError:
        return None
    finally:
        os.close(descriptor)


def reusable_camera(destination: Path, item: dict, camera: str) -> dict | None:
    path = destination / "receipt.json"
    if not path.is_file():
        return None
    previous = json.loads(path.read_text())
    if (previous.get("status") != "complete" or previous.get("camera_id") != camera
            or previous.get("url") != item["url"] or not previous.get("files")):
        raise ValueError("Existing primary receipt has incompatible identity or status")
    for asset in previous["files"]:
        asset_path = destination / safe_name(asset["path"])
        if not asset_path.is_file() or asset_path.stat().st_size != asset["bytes"] or sha256_file(asset_path) != asset["sha256"]:
            raise ValueError(f"Existing primary payload fails SHA verification: {asset_path}")
    return {**previous, "reused_existing": True, "network_bytes_read": 0,
            "network_bytes_previously_read": previous["network_bytes_read"],
            "camera_selection_basis": CAMERA_SELECTION_BASIS}


def primary_cameras(root: Path, seconds: int, camera: str = "105322251564", extend: bool = False) -> dict:
    import yaml

    if camera != "105322251564":
        raise ValueError("Only the RGB-reviewed primary camera is authorized here")
    plan_name = "download_plan_v2.json" if extend else "download_plan.json"
    selected = json.loads((root / plan_name).read_text())["selected"]
    selected = [item for item in selected if "subject_id" in item]
    prior_name = "primary_download_receipt.json" if extend else "acquisition_report.json"
    prior = json.loads((root / prior_name).read_text())
    prior_bound = prior["cumulative_network_upper_bound_bytes" if extend else "network_transfer_conservative_upper_bound_bytes"]
    overall_limit = 25_000_000_000 if extend else MAX_BYTES
    limit = min(6_000_000_000 if extend else 3_800_000_000, overall_limit - prior_bound)
    deadline = time.monotonic() + min(seconds, 1200 if extend else 720)
    if shutil.disk_usage(root).free < limit + 1_000_000_000:
        raise ValueError("Insufficient disk space")
    receipt_path = root / ("primary_extension_receipt.json" if extend else "primary_download_receipt.json")
    if receipt_path.exists():
        raise ValueError("Primary run already recorded; reconcile it before issuing a new bounded run")
    budget = NetworkBudget(limit, deadline)
    records = []
    started = time.monotonic()

    def fetch(item):
        destination = root / "captures_primary" / item["subject_id"] / item["acquisition_id"]
        destination.mkdir(parents=True, exist_ok=True)
        result = {**item, "requested_camera": camera, "status": "running", "index_probes": [],
                  "network_bytes_read": 0, "local_index_bytes_reused": 0,
                  "full_archive_sha_verified": False,
                  "camera_selection_basis": CAMERA_SELECTION_BASIS}
        if any(destination.iterdir()):
            try:
                reused = reusable_camera(destination, item, camera)
                if reused is not None:
                    return reused
            except ValueError as error:
                result.update(status="failed", error=str(error))
                return result
            result.update(status="failed", error="Existing unreceipted primary output; refusing overwrite")
            return result
        reader = None

        def read_at(offset, size):
            local = local_index_range(root / "archives" / item["name"], offset, size)
            if local is not None and tar_headers(local, offset):
                result["local_index_bytes_reused"] += len(local)
                result["index_probes"].append({"offset": offset, "bytes": size, "source": "local_sparse_headers_only"})
                return local
            budget.reserve(size)
            payload = read_range(item["url"], offset, size, item["size"],
                                 timeout=min(30, max(1, deadline - time.monotonic())))
            result["network_bytes_read"] += len(payload)
            result["index_probes"].append({"offset": offset, "bytes": size, "source": "http_range"})
            return payload

        try:
            metadata = yaml.safe_load((root / "labels_selected" / f'{item["name"][:-4]}.meta.yaml').read_text())
            offset = locate_camera_start(read_at, item["size"], camera)
            result["camera_header_offset"] = offset
            budget.reserve(512)
            verified = read_range(item["url"], offset, 512, item["size"])
            result["network_bytes_read"] += len(verified)
            header = tarfile.TarInfo.frombuf(verified, "utf-8", "strict")
            if header.name != f"{camera}-000000.color.jpg":
                raise ValueError("Remote verification disagrees with camera index")
            if extend:
                cameras_ordered = sorted(metadata["realsense"]["serials"])
                next_camera = cameras_ordered[cameras_ordered.index(camera) + 1]
                next_offset = locate_camera_start(read_at, item["size"], next_camera)
                result["next_camera_header_offset"] = next_offset
                size = next_offset + 512 - offset
                if size <= 512 or size > 350_000_000:
                    raise ValueError("Primary camera interval outside allowed per-capture bounds")
            else:
                size = min(350_000_000, item["size"] - offset)
            budget.reserve(size)
            request = urllib.request.Request(item["url"], headers={
                "Range": f"bytes={offset}-{offset + size - 1}", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=min(30, max(1, deadline - time.monotonic()))) as response:
                expected = f'bytes {offset}-{offset + size - 1}/{item["size"]}'
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Primary stream Range not honored")
                reader = BoundedReader(response, size, deadline)
                result.update(extract_first_camera(reader, destination, deadline, archive_offset=offset))
            result["metadata_frame_count"] = metadata["num_frames"]
            complete = (result["camera_id"] == camera and result["contiguous_all_modalities"]
                        and result["frame_count"] == metadata["num_frames"])
            result["status"] = "complete" if complete else "incomplete"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            result["network_bytes_read"] += reader.received if reader else 0
        write_json(destination / "receipt.json", result)
        print(item["name"], result["status"], result.get("frame_count"), result.get("error", ""), flush=True)
        return result

    first = fetch(selected[0])
    records.append(first)
    if first["status"] == "complete":
        candidate_manifest(root, primary=True)
        with ThreadPoolExecutor(max_workers=4 if extend else 3) as pool:
            records.extend(pool.map(fetch, selected[1:]))
    result = {"camera_id": camera, "records": records,
              "status": "complete" if len(records) == len(selected) and all(item["status"] == "complete" for item in records) else "incomplete",
              "elapsed_seconds": time.monotonic() - started,
              "network_bytes_read": sum(item["network_bytes_read"] for item in records),
              "reserved_response_bytes_upper_bound": budget.reserved,
              "additional_network_limit_bytes": limit,
              "overall_network_limit_bytes": overall_limit, "plan": plan_name,
              "cumulative_network_upper_bound_bytes": prior_bound + budget.reserved,
              "training_started": False, "test_reference_downloaded": False,
              "existing_037_outputs_modified": False}
    write_json(receipt_path, result)
    candidate_manifest(root, primary=True)
    return {key: value for key, value in result.items() if key != "records"}


def candidate_manifest(root: Path, primary: bool = False) -> dict:
    import yaml

    evidence_path = root / "fps_evidence.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    fps = evidence.get("fps") if evidence.get("capture_fps_verified") is True else None
    records = []
    capture_folder = "captures_primary" if primary else "captures"
    for receipt_path in sorted((root / capture_folder).glob("*/*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt["status"] != "complete":
            continue
        subject = receipt["subject_id"]
        acquisition = receipt["acquisition_id"]
        camera = receipt["camera_id"]
        prefix = f"{subject}-{acquisition}"
        metadata_path = root / "labels_selected" / f"{prefix}.meta.yaml"
        metadata = yaml.safe_load(metadata_path.read_text()) if metadata_path.exists() else {}
        assets = []
        for item in receipt["files"]:
            path = receipt_path.parent / item["path"]
            role = "rgb" if path.name.endswith("color.jpg") else "depth" if path.name.endswith("depth.png") else "hand_object_reference"
            assets.append({"role": role, "path": path.relative_to(root).as_posix(),
                           "sha256": item["sha256"], "size_bytes": item["bytes"]})
        extras = [(path, "metadata" if path.name.endswith("meta.yaml") else
                   "hand_reference" if ".poses_m." in path.name else
                   "object_reference" if ".poses_o." in path.name else "camera_pose_reference")
                  for path in sorted((root / "labels_selected").glob(prefix + ".*"))]
        extras += [(root / "calibration" / f"intrinsics.{camera}.yaml", "intrinsics"),
                   (root / "calibration" / "extrinsics.extrinsics_20231014.yaml", "extrinsics")]
        for object_id in metadata.get("object_ids", []):
            extras.extend((path, "mesh" if path.suffix == ".obj" else "mesh_material")
                          for path in sorted((root / "models").glob(object_id + ".*")))
        for path, role in extras:
            if path.is_file():
                assets.append({"role": role, "path": path.relative_to(root).as_posix(),
                               "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        frames = receipt["frame_count"]
        reasons = ["Human inspection pending", "Coordinate/unit validation pending",
                   "Official references are semi-automatically optimized, not independent mocap",
                   "No model observations generated; data acquisition only",
                   "MANO model files require separate licensed access if mesh reconstruction is needed"]
        if camera == "037522251142":
            reasons.append("Primary reviewer found poor hand visibility in this camera; not a qualified hand-view")
        if fps is None:
            reasons.append("Capture FPS not established; do not assume 30 FPS")
        if metadata.get("num_frames") != frames:
            reasons.append("Source metadata frame count unavailable or mismatched")
        roles = {asset["role"] for asset in assets}
        records.append({
            "sequence_id": f"hocap_{subject}_{acquisition}_{camera}", "dataset": "HO-Cap",
            "acquisition_id": acquisition, "subject_id": subject, "scene_id": None,
            "object_ids": metadata.get("object_ids", []),
            "action_labels": [f'hocap_task_{metadata["task_id"]}'] if "task_id" in metadata else [],
            "camera_id": camera, "view": receipt["view"], "frame_count": frames,
            "duration_seconds": (frames - 1) / fps if fps else None, "fps": fps,
            "timestamp_source": evidence if fps else None, "human_verified": False,
            "nominal_fps_evidence": evidence if evidence else None,
            "nominal_fps": evidence.get("nominal_fps"),
            "assets": assets, "capabilities": {
                "rgb": "rgb" in roles, "depth": "depth" in roles,
                "intrinsics": "intrinsics" in roles, "mesh": "mesh" in roles,
                "hand_metric_gt": False, "object_pose_gt": False,
                "hand_observations": False, "object_observations": False,
                "camera_motion_compensated": False,
                "hand_reference_available": "hand_reference" in roles,
                "object_reference_available": "object_reference" in roles,
            }, "quality": {"decode": "pending", "timestamps": "pending",
                            "calibration": "pending", "coordinates": "pending",
                            "annotation_alignment": "pending", "visual_review": "fail" if camera == "037522251142" else "pending"},
            "split": None, "phase": "candidate", "limitations": reasons,
            "source_official_split": receipt["official_split"], "license": "CC-BY-4.0",
            "source_revision": REVISION, "source_archive": receipt["url"],
            "original_num_frames": metadata.get("num_frames"),
        })
    result = {"schema_version": "human_pose_dataset_v1", "records": records,
              "training_allowed": False, "phase": "candidate"}
    filename = "primary_candidates.json" if primary else "hocap_candidates.json"
    write_json(root / filename, result)
    return {"manifest": str(root / filename), "records": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "labels", "fast-labels", "download", "inventory", "cameras", "candidate", "primary", "primary-extend", "primary-candidate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["subject_1", "subject_2", "subject_3", "subject_4", "subject_6", "subject_7"])
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--plan", default="download_plan.json")
    args = parser.parse_args()
    if not 0 < args.seconds <= MAX_SECONDS:
        parser.error("seconds must be between 1 and 1500")
    root = args.root.resolve()
    if args.command == "plan":
        result = plan(root, args.subjects)
    elif args.command == "labels":
        result = fetch_selected_labels(root, args.seconds, args.plan)
    elif args.command == "fast-labels":
        result = fast_labels(root, args.seconds, args.plan)
    elif args.command == "download":
        result = download(root, args.seconds)
    elif args.command == "cameras":
        result = cameras(root, args.seconds, args.plan)
    elif args.command == "candidate":
        result = candidate_manifest(root)
    elif args.command == "primary":
        result = primary_cameras(root, args.seconds)
    elif args.command == "primary-extend":
        result = primary_cameras(root, args.seconds, extend=True)
    elif args.command == "primary-candidate":
        result = candidate_manifest(root, primary=True)
    else:
        result = inventory(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
