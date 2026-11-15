"""Build an offline, deterministic fetch manifest for staged EgoDemo annotations.

Run directly with Python; this module does not import the legacy sample builder.
Readiness checks JSON contents and binary container signatures, not decoded frames
or parquet schemas. Official semantic events are never converted into HOI labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from uuid import UUID


VARIANTS = ("EgoProStandard", "EgoProStandard-body")
CONTRACT_VERSION = "2026-10-22"
UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
LFS_VERSION = "version https://git-lfs.github.com/spec/v1"
REQUIRED_ROLES = (
    "data", "info", "episodes", "tasks", "annotation", "bad_frame_ratio", "head_left_video",
)


class ManifestError(ValueError):
    """The local inventory is ambiguous, malformed, or unsafe to use."""


def _relative_path(value: str) -> str:
    if not value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ManifestError(f"Invalid repository-relative path: {value!r}")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ManifestError(f"Unsafe repository-relative path: {value!r}")
    return path.as_posix()


def _local_path(root: Path, relative: str) -> Path:
    path = root / _relative_path(relative)
    try:
        if not path.resolve().is_relative_to(root):
            raise ManifestError(f"Path escapes dataset root {root}: {relative}")
    except RuntimeError as error:
        raise ManifestError(f"Symlink cycle at {relative}") from error
    return path


def _files_under(root: Path, relative: str | None = None) -> list[str]:
    directory = root if relative is None else _local_path(root, relative)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ManifestError(f"Expected directory: {directory}")
    result = []

    def walk_error(error: OSError) -> None:
        raise error

    for current, directories, filenames in os.walk(directory, onerror=walk_error):
        directories.sort()
        for name in directories:
            child = Path(current) / name
            _local_path(root, child.relative_to(root).as_posix())
            if child.is_symlink():
                raise ManifestError(f"Directory symlinks are not supported: {child}")
        for name in sorted(filenames):
            relative_file = (Path(current) / name).relative_to(root).as_posix()
            _local_path(root, relative_file)
            result.append(relative_file)
    return sorted(result)


def _is_placeholder(value: object) -> bool:
    if value in (None, {}, [], ""):
        return True
    if isinstance(value, dict):
        status = str(value.get("status", "")).lower().replace("-", "_").replace(" ", "_")
        return (
            value.get("placeholder") is True
            or value.get("downloaded") is False
            or status in {"placeholder", "not_downloaded", "pending", "missing"}
        )
    return False


def inspect_file(root: Path, relative: str, role: str) -> dict:
    """Describe a file without mistaking LFS pointers or placeholders for payloads."""
    path = _local_path(root, relative)
    record = {
        "role": role,
        "path": _relative_path(relative),
        "status": "missing",
        "downloaded": False,
        "ready": False,
        "local_size": None,
        "expected_size": None,
        "expected_sha256": None,
        "validation": None,
    }
    if not path.exists():
        return record
    if not path.is_file():
        raise ManifestError(f"Expected a regular file: {relative}")
    size = path.stat().st_size
    record["local_size"] = size
    if not size:
        record.update(status="placeholder", reason="empty_file")
        return record
    with path.open("rb") as stream:
        prefix = stream.read(8192)
        if prefix.lstrip().startswith((b"version https://git-lfs", b"oid sha256:")):
            try:
                lines = prefix.decode("ascii").strip().splitlines()
                hashes = [line[11:] for line in lines if line.startswith("oid sha256:")]
                sizes = [line[5:] for line in lines if line.startswith("size ")]
                valid = (
                    size <= len(prefix)
                    and lines[0] == LFS_VERSION
                    and len(hashes) == len(sizes) == 1
                    and re.fullmatch(r"[0-9a-fA-F]{64}", hashes[0]) is not None
                    and re.fullmatch(r"[0-9]+", sizes[0]) is not None
                    and all(
                        line == LFS_VERSION or line.startswith(("oid sha256:", "size ", "ext-"))
                        for line in lines
                    )
                )
            except UnicodeDecodeError:
                valid = False
            if valid:
                record.update(
                    status="lfs_pointer", expected_size=int(sizes[0]),
                    expected_sha256=hashes[0].lower(), validation="git_lfs_pointer",
                )
            else:
                record.update(status="invalid", reason="malformed_lfs_pointer")
            return record
        normalized = prefix.strip().lower()
        if normalized.startswith((b"placeholder", b"todo", b"not downloaded", b"not_downloaded")):
            record.update(status="placeholder", reason="text_placeholder")
            return record
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            stream.seek(max(0, size - 8))
            footer = stream.read(8)
            valid = (
                size > 12 and prefix.startswith(b"PAR1") and footer[4:] == b"PAR1"
                and 0 < int.from_bytes(footer[:4], "little") <= size - 12
            )
            validation = "parquet_container_signature_only"
        elif suffix == ".mp4":
            valid = (
                size >= 24 and prefix[4:8] == b"ftyp"
                and 16 <= int.from_bytes(prefix[:4], "big") <= size
            )
            validation = "mp4_container_signature_only"
        elif suffix == ".json":
            stream.seek(0)
            try:
                value = json.load(stream)
            except (ValueError, UnicodeDecodeError):
                value = None
                valid = False
            else:
                if _is_placeholder(value):
                    record.update(status="placeholder", reason="json_placeholder")
                    return record
                valid = isinstance(value, (dict, list))
            validation = "json_syntax_only"
        elif suffix == ".csv":
            stream.seek(0)
            try:
                stream.read().decode("utf-8-sig")
                valid = True
            except UnicodeDecodeError:
                valid = False
            validation = "csv_header_checked_on_load"
        else:
            valid = False
            validation = "unsupported_format"
    record.update(
        status="payload" if valid else "invalid", downloaded=valid, ready=valid,
        validation=validation,
    )
    if not valid:
        record["reason"] = "invalid_or_incomplete_payload"
    return record


def _episode_uuid(value: str) -> str:
    if not UUID_PATTERN.fullmatch(value):
        raise ManifestError(f"Invalid source_episode_uuid: {value!r}")
    return str(UUID(value))


def _number(value: object, field: str, integral: bool = False) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ManifestError(f"Invalid {field}: {value!r}") from error
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ManifestError(f"Invalid {field}: {value!r}")
    if integral and not number.is_integer():
        raise ManifestError(f"Nonintegral {field}: {value!r}")
    return int(number) if integral else number


def _boolean(value: str | None, field: str) -> bool | None:
    if not value:
        return None
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ManifestError(f"Invalid {field}: {value!r}")
    return value.lower() in {"true", "1"}


def _read_rows(root: Path) -> tuple[dict, list[dict], int]:
    rows = {}
    sources = []
    duplicate_rows = 0
    for variant in VARIANTS:
        relative = f"{variant}/manifests/train.csv"
        source = inspect_file(root, relative, "source_manifest")
        sources.append(source)
        if not source["ready"]:
            continue
        with _local_path(root, relative).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames or []
            required = {"source_episode_uuid", "task_name", "lerobot_path"}
            if not required.issubset(fields) or len(fields) != len(set(fields)):
                raise ManifestError(f"Missing or duplicate CSV columns in {relative}")
            for line, original in enumerate(reader, 2):
                if None in original or any(value is None for value in original.values()):
                    raise ManifestError(f"Malformed CSV row at {relative}:{line}")
                row = {key: value.strip() for key, value in original.items()}
                episode_uuid = _episode_uuid(row["source_episode_uuid"])
                if not row["task_name"] or row.get("variant", variant) != variant:
                    raise ManifestError(f"Invalid task or variant at {relative}:{line}")
                lerobot_path = _relative_path(row["lerobot_path"])
                parts = PurePosixPath(lerobot_path).parts
                if parts[0] != variant or _episode_uuid(parts[-1]) != episode_uuid:
                    raise ManifestError(f"Variant/UUID disagrees with lerobot_path at {relative}:{line}")
                _local_path(root, lerobot_path)
                row.update(
                    source_episode_uuid=episode_uuid, variant=variant, lerobot_path=lerobot_path,
                    fps=_number(row.get("fps"), "fps"),
                    frame_count=_number(row.get("frame_count"), "frame_count", integral=True),
                    duration_s=_number(row.get("duration_s"), "duration_s"),
                    has_hand_pose=_boolean(row.get("has_hand_pose"), "has_hand_pose"),
                    has_semantic_annotations=_boolean(
                        row.get("has_semantic_annotations"), "has_semantic_annotations",
                    ),
                )
                if episode_uuid in rows:
                    if rows[episode_uuid] != row:
                        raise ManifestError(f"Conflicting duplicate UUID {episode_uuid} at {relative}:{line}")
                    duplicate_rows += 1
                else:
                    rows[episode_uuid] = row
    if not any(source["ready"] for source in sources):
        raise ManifestError("No readable annotated train.csv manifests; metadata is missing or placeholder-only")
    return rows, sources, duplicate_rows


def _describe_episode(root: Path, row: dict, raw_videos: list[dict]) -> dict:
    base = row["lerobot_path"]
    paths = {
        "data": [path for path in _files_under(root, f"{base}/data") if path.endswith(".parquet")],
        "info": [f"{base}/meta/info.json"],
        "episodes": [
            path for path in _files_under(root, f"{base}/meta/episodes") if path.endswith(".parquet")
        ],
        "tasks": [f"{base}/meta/tasks.parquet"],
        "annotation": [f"{base}/annotation.json"],
        "bad_frame_ratio": [f"{base}/bad_frame_ratio.json"],
        "head_left_video": [
            path for path in _files_under(root, f"{base}/videos/observation.images.head_left")
            if path.endswith(".mp4")
        ],
    }
    files = [inspect_file(root, path, role) for role in REQUIRED_ROLES for path in paths[role]]
    info_record = next(record for record in files if record["role"] == "info")
    info = {}
    if info_record["ready"]:
        with _local_path(root, info_record["path"]).open(encoding="utf-8") as stream:
            info = json.load(stream)
        if not isinstance(info, dict):
            raise ManifestError(f"meta/info.json must contain an object: {base}")
    fields = {field: row[field] for field in ("fps", "frame_count", "duration_s")}
    field_sources = {field: "train.csv" if value is not None else None for field, value in fields.items()}
    for field, info_field in (("fps", "fps"), ("frame_count", "total_frames")):
        if fields[field] is None and info.get(info_field) is not None:
            fields[field] = _number(info[info_field], field, integral=field == "frame_count")
            field_sources[field] = f"meta/info.json:{info_field}"
    if fields["duration_s"] is None and fields["fps"] and fields["frame_count"]:
        fields["duration_s"] = fields["frame_count"] / fields["fps"]
        field_sources["duration_s"] = "derived:frame_count/fps"
    missing_roles = [role for role in REQUIRED_ROLES if not paths[role]]
    missing_metadata = [field for field, value in fields.items() if value is None]
    unavailable_roles = sorted({record["role"] for record in files if not record["ready"]})
    return {
        **{field: row[field] for field in (
            "source_episode_uuid", "variant", "task_name", "lerobot_path",
            "has_hand_pose", "has_semantic_annotations",
        )},
        **fields,
        "metadata_sources": field_sources,
        "annotation_path": f"{base}/annotation.json",
        "raw_videos": raw_videos,
        "files": files,
        "readiness": {
            "ready_for_staging": not (missing_roles or unavailable_roles or missing_metadata),
            "fetch_plan_complete": not missing_roles,
            "missing_file_roles": missing_roles,
            "unavailable_payload_roles": unavailable_roles,
            "missing_metadata_fields": missing_metadata,
            "training_ready": False,
        },
    }


def build_manifest(dataset_root: str | Path, raw_root: str | Path, limit: int = 10) -> dict:
    """Select UUID-unique episodes round-robin over sorted tasks, then sorted UUIDs."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ManifestError("limit must be a positive integer")
    dataset_root, raw_root = Path(dataset_root).resolve(), Path(raw_root).resolve()
    for root in (dataset_root, raw_root):
        if not root.is_dir():
            raise ManifestError(f"Dataset root is not a directory: {root}")
    rows, sources, duplicate_rows = _read_rows(dataset_root)
    raw_inventory = {}
    ignored_videos = []
    for relative in _files_under(raw_root):
        parts = PurePosixPath(relative).parts
        if not relative.lower().endswith(".mp4"):
            continue
        if len(parts) < 3 or not UUID_PATTERN.fullmatch(parts[-3]):
            ignored_videos.append(relative)
            continue
        episode_uuid = _episode_uuid(parts[-3])
        video = inspect_file(raw_root, relative, "raw_video")
        video["camera"] = parts[-2]
        raw_inventory.setdefault(episode_uuid, []).append(video)
    matched = sorted(set(raw_inventory) & set(rows))
    tasks = {}
    for episode_uuid in matched:
        tasks.setdefault(rows[episode_uuid]["task_name"], []).append(episode_uuid)
    selected_uuids = []
    for index in range(max((len(episodes) for episodes in tasks.values()), default=0)):
        for task in sorted(tasks):
            if index < len(tasks[task]) and len(selected_uuids) < limit:
                selected_uuids.append(tasks[task][index])
        if len(selected_uuids) >= limit:
            break
    selected = [
        _describe_episode(dataset_root, rows[episode_uuid], raw_inventory[episode_uuid])
        for episode_uuid in selected_uuids
    ]
    selected_set = set(selected_uuids)
    fetch_files = sorted({
        record["path"] for episode in selected for record in episode["files"] if not record["ready"]
    })
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "dataset_root": str(dataset_root),
        "raw_root": str(raw_root),
        "selection": {"strategy": "sorted_task_round_robin_then_uuid", "limit": limit, "count": len(selected)},
        "annotation_semantics": {
            "role": "staged_semantic_events",
            "hoi_ground_truth": False,
            "automatic_label_mapping": False,
            "events": "Preserve original subtask_annotations descriptions and time intervals for human review.",
            "coordinates": "Preserve world-space poses; camera transforms require explicit rotation conventions.",
            "alignment": "UUID matching only; raw and annotated frames are not interchangeable.",
            "readiness_scope": "Local metadata and container signatures only; import must validate schemas and timing.",
        },
        "inventory": {
            "source_manifests": sources,
            "manifest_inventory_complete": all(source["ready"] for source in sources),
            "manifest_episode_count": len(rows),
            "identical_duplicate_rows": duplicate_rows,
            "raw_episode_count": len(raw_inventory),
            "raw_video_count": sum(len(videos) for videos in raw_inventory.values()),
            "raw_payload_video_count": sum(
                video["ready"] for videos in raw_inventory.values() for video in videos
            ),
            "matched_episode_count": len(matched),
            "coverage_ratio": len(matched) / len(raw_inventory) if raw_inventory else None,
            "unmatched_raw_uuids": sorted(set(raw_inventory) - set(rows)),
            "annotated_only_episode_count": len(set(rows) - set(raw_inventory)),
            "ignored_raw_videos": ignored_videos,
            "by_variant": {
                variant: {
                    "manifest_episodes": sum(row["variant"] == variant for row in rows.values()),
                    "matched_episodes": sum(rows[episode_uuid]["variant"] == variant for episode_uuid in matched),
                    "selected_episodes": sum(episode["variant"] == variant for episode in selected),
                }
                for variant in VARIANTS
            },
            "by_task": {
                task: {"matched_episodes": len(episodes), "selected_episodes": len(set(episodes) & selected_set)}
                for task, episodes in sorted(tasks.items())
            },
        },
        "selected_episodes": selected,
        "fetch_files": fetch_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--limit", default=10, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    temporary_path = None
    try:
        manifest = build_manifest(args.dataset_root, args.raw_root, args.limit)
        output = args.output.resolve()
        if output.exists() and any(
            output.is_relative_to(root.resolve()) for root in (args.dataset_root, args.raw_root)
        ):
            raise ManifestError("Refusing to overwrite an existing file inside an input dataset root")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output)
    except (ManifestError, OSError, csv.Error, UnicodeError) as error:
        parser.error(str(error))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
