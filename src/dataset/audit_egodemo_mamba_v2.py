"""Independent full-shard/time/split audit; passing is not annotation approval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def masked_geometry(values, valid, name):
    require(valid.dtype == np.bool_, name + " mask must be boolean")
    require(values.shape[:valid.ndim] == valid.shape, name + " mask shape mismatch")
    require(np.isfinite(values[valid]).all(), name + " valid coordinates are nonfinite")
    require(np.isnan(values[~valid]).all(), name + " invalid coordinates must be NaN")


def se3(values, valid, name):
    masked_geometry(values, valid, name)
    require(values.shape[-2:] == (4, 4), name + " pose dimensions")
    values = values[valid]
    if not len(values):
        return
    rotation = values[..., :3, :3]
    require(np.max(np.abs(rotation.swapaxes(-1, -2) @ rotation - np.eye(3))) < 1e-3,
            name + " rotation orthogonality")
    require(np.max(np.abs(np.linalg.det(rotation) - 1)) < 1e-3, name + " rotation determinant")
    require(np.max(np.abs(values[..., 3, :] - [0, 0, 0, 1])) < 1e-5, name + " homogeneous row")


def audit_windows(rows, timestamps, config):
    """Check every actual target and also missing, duplicate, or extra windows."""
    width, stride = config["window_frames"], config["stride_frames"]
    horizon = config["horizon_seconds"]
    tolerance = np.median(np.diff(timestamps)) * config["max_horizon_overshoot_median_dt"]
    actual = {}
    for row in rows:
        start, end, anchor, target = (row[key] for key in
                                    ["start_frame", "end_frame", "anchor_frame", "target_frame"])
        require(all(type(value) is int for value in [start, end, anchor, target]), "noninteger window index")
        require(0 <= start < end <= len(timestamps) and end - start == width and start % stride == 0,
                "invalid observation bounds")
        require(anchor == end - 1 and end <= target < len(timestamps), "future target/anchor bounds")
        desired = timestamps[anchor] + horizon
        require(timestamps[target] >= desired, "target is earlier than requested horizon")
        require(target == 0 or timestamps[target - 1] < desired, "target is not first eligible timestamp")
        require(timestamps[target] - desired <= tolerance + 1e-9, "target overshoot across timestamp gap")
        require(abs(row["target_timestamp"] - timestamps[target]) < 1e-9, "target timestamp mismatch")
        require(abs(row["actual_horizon_s"] - (timestamps[target] - timestamps[anchor])) < 1e-9,
                "actual horizon mismatch")
        require(start not in actual, "duplicate observation window")
        actual[start] = target
    expected = {}
    for start in range(0, len(timestamps) - width + 1, stride):
        desired = timestamps[start + width - 1] + horizon
        eligible = int(np.searchsorted(timestamps, desired))
        if eligible < len(timestamps) and timestamps[eligible] - desired <= tolerance + 1e-9:
            expected[start] = eligible
    require(actual == expected, "missing or extra native-timestamp windows")


def audit_sparse(arrays, frame_count, object_count):
    sf, so, sp = (arrays[key] for key in ["surface_frame_index", "surface_object_index", "surface_point_index"])
    xyz, ptr = arrays["surface_world"], arrays["surface_frame_ptr"]
    keys, owners = arrays["point_keys"], arrays["point_object_index"]
    size = len(sf)
    require(so.shape == sp.shape == (size,) and xyz.shape == (size, 3), "sparse row dimensions")
    require(ptr.shape == (frame_count + 1,) and ptr[0] == 0 and ptr[-1] == size, "sparse frame pointers")
    require(len(keys) == len(set(keys.tolist())) and owners.shape == (len(keys),), "point identity table")
    require(all("/" in key for key in keys), "point key must include source tracklet")
    require(np.isfinite(xyz).all(), "sparse rows contain invalid coordinates")
    require(((0 <= sf) & (sf < frame_count)).all(), "sparse frame out of range")
    require(((0 <= so) & (so < object_count)).all(), "sparse object out of range")
    require(((0 <= sp) & (sp < len(keys))).all(), "sparse point out of range")
    require(((0 <= owners) & (owners < object_count)).all(), "point owner out of range")
    require(np.array_equal(owners[sp], so), "point key moves between object identities")
    require(np.array_equal(np.diff(ptr), np.bincount(sf, minlength=frame_count)), "sparse pointer counts mismatch")
    require(not size or np.all(np.diff(sf) >= 0), "sparse frames unsorted")
    # Encode each (frame, point) exactly, detecting duplicate physical point rows.
    codes = sf.astype(np.int64) * max(len(keys), 1) + sp
    require(len(np.unique(codes)) == size, "duplicate frame/point observation")
    observed = arrays["object_pose_valid"].copy()
    observed[sf, so] = True
    require(np.array_equal(observed, arrays["object_observed"]), "object observed mask inconsistent")


def expected_selection(arrays, start, end, max_objects=8, max_points=16):
    """Independent selection using observation bounds, never target availability."""
    visible = arrays["object_observed"][start:end]
    candidates = np.flatnonzero(visible.any(axis=0)).tolist()
    candidates.sort(key=lambda i: (int(np.flatnonzero(visible[:, i])[0]), str(arrays["object_ids"][i])))
    objects = candidates[:max_objects]
    lo, hi = arrays["surface_frame_ptr"][[start, end]]
    points = []
    for object_index in objects:
        selected = arrays["surface_object_index"][lo:hi] == object_index
        indexes, first = np.unique(arrays["surface_point_index"][lo:hi][selected], return_index=True)
        first_frames = arrays["surface_frame_index"][lo:hi][selected][first]
        ordered = sorted(zip(indexes.tolist(), first_frames.tolist()),
                         key=lambda item: (item[1], str(arrays["point_keys"][item[0]])))
        points.append([item[0] for item in ordered[:max_points]])
    return objects, points


def audit_surface_source(arrays, metadata):
    """Compare every sparse exported point with its immutable native source key."""
    track_names = sorted({key.rsplit("/", 1)[0] for key in arrays["point_keys"]})
    track_index = {name: index for index, name in enumerate(track_names)}
    point_track = np.asarray([track_index[key.rsplit("/", 1)[0]] for key in arrays["point_keys"]])
    row_track = point_track[arrays["surface_point_index"]]
    for name, track in track_index.items():
        rows = np.flatnonzero(row_track == track)
        with np.load(metadata["source_files"]["track:" + name]["path"], allow_pickle=False) as z:
            local_ids = {int(value): index for index, value in enumerate(z["point_ids"])}
            local_points = np.full(len(arrays["point_keys"]), -1, dtype=np.int32)
            for point in np.flatnonzero(point_track == track):
                local_points[point] = local_ids[int(arrays["point_keys"][point].rsplit("/", 1)[1])]
            source_frames = np.searchsorted(z["frame_index"], arrays["surface_frame_index"][rows])
            require(np.array_equal(z["frame_index"][source_frames], arrays["surface_frame_index"][rows]), "surface source frame mismatch")
            source_points = local_points[arrays["surface_point_index"][rows]]
            require(z["point_valid"][source_frames, source_points].all(), "invalid source point exported")
            expected = z["points_world"][source_frames, source_points].astype(np.float32)
            require(np.array_equal(expected, arrays["surface_world"][rows]), "surface source coordinates changed")
            valid = z["point_valid"] & np.isfinite(z["points_world"]).all(axis=-1)
            require(len(rows) == int(valid.sum()), "retained track lost valid native point observations")


def audit_loaded_window(sample, arrays, row, max_objects=8, max_points=16):
    """Check actual loader output with independently selected historical identities."""
    sample_arrays = {key: value.detach().cpu().numpy() for key, value in sample.items() if key != "metadata"}
    metadata = sample["metadata"]
    start, end, target = (row[key] for key in ["start_frame", "end_frame", "target_frame"])
    objects, points = expected_selection(arrays, start, end, max_objects, max_points)
    require(metadata["object_indices"] == objects + [-1] * (max_objects - len(objects)), "loader used nonhistorical object selection")
    expected_points = np.full((max_objects, max_points), -1, dtype=np.int64)
    for slot, indexes in enumerate(points):
        expected_points[slot, :len(indexes)] = indexes
    require(np.array_equal(metadata["point_indices"], expected_points), "loader used nonhistorical surface key selection")
    for values, mask in [("hand_seq", "hand_valid"), ("obj_pose_seq", "obj_valid"),
                         ("surface_seq", "surface_valid"), ("target_hand", "target_hand_valid"),
                         ("target_obj_pose", "target_obj_valid"), ("target_surface", "target_surface_valid")]:
        masked_geometry(sample_arrays[values], sample_arrays[mask], values)
    require(not sample_arrays["event_label_valid"].any(), "pending event loss mask enabled")
    transform = np.linalg.inv(arrays["camera_to_world"][end - 1])
    for name, source in [("hand_seq", arrays["hands_world"][start:end]), ("target_hand", arrays["hands_world"][target])]:
        expected = source @ transform[:3, :3].T + transform[:3, 3]
        require(np.allclose(sample_arrays[name], expected, rtol=1e-5, atol=2e-5, equal_nan=True), "loader hand anchor transform mismatch")
    for slot, obj in enumerate(objects):
        for name, source in [("obj_pose_seq", arrays["object_pose_world"][start:end, obj]),
                             ("target_obj_pose", arrays["object_pose_world"][target, obj])]:
            expected = transform @ source
            actual = sample_arrays[name][:, slot] if name == "obj_pose_seq" else sample_arrays[name][slot]
            require(np.allclose(actual, expected, rtol=1e-5, atol=2e-5, equal_nan=True), "loader object anchor transform mismatch")
        for pslot, point in enumerate(points[slot]):
            for name, frames in [("surface_seq", range(start, end)), ("target_surface", [target])]:
                expected = np.full((len(frames), 3), np.nan)
                for index, frame in enumerate(frames):
                    lo, hi = arrays["surface_frame_ptr"][[frame, frame + 1]]
                    matches = np.flatnonzero(arrays["surface_point_index"][lo:hi] == point)
                    if len(matches):
                        xyz = arrays["surface_world"][lo + matches[0]]
                        expected[index] = transform[:3, :3] @ xyz + transform[:3, 3]
                actual = sample_arrays[name][:, slot, pslot] if name == "surface_seq" else sample_arrays[name][slot, pslot][None]
                require(np.allclose(actual, expected, rtol=1e-5, atol=2e-5, equal_nan=True), "loader point key/anchor mismatch")


def audit_dataset(root, source_pool, *, check_loader=False):
    root, source_pool = Path(root), Path(source_pool)
    index = read(root / "dataset_index.json")
    split_path = root / index["split_manifest_path"]
    split, split_sha = read(split_path), digest(split_path)
    selection_path = source_pool / "selection_frozen.json"
    selection, source_sha = read(selection_path), digest(selection_path)
    source_rows = {row["source_episode_uuid"]: row for row in selection["records"]}
    require(index["schema_version"] == "egodemo_mamba_v2", "schema version mismatch")
    require(index["source_selection_sha256"] == split["source_selection_sha256"] == source_sha,
            "source selection SHA mismatch")
    require(index["split_manifest_sha256"] == split_sha, "frozen split SHA mismatch")
    require(set(split["assignments"]) == set(source_rows), "split UUID coverage mismatch")
    require(index["training_allowed"] is False and index["allowed_as_model_input"] is False,
            "dataset bypasses noncausal/unreviewed training gate")
    require(index["human_approved"] is False and index["full_annotation_complete"] is False,
            "unreviewed export claims approval")
    vocab = read(root / index["vocab_path"])
    require(digest(root / index["vocab_path"]) == index["vocab_sha256"], "vocabulary SHA mismatch")
    bindings = {row["sequence_id"]: row for row in split["records"]}
    identifiers = defaultdict(set)
    for uid, source in source_rows.items():
        name = split["assignments"][uid]
        for key in ["environment_id", "scene_id"]:
            if source.get(key):
                identifiers[(key, source[key])].add(name)
        for asset in source["planned_assets"]:
            if asset["role"] in {"data", "head_left_video"}:
                identifiers[(asset["role"], asset["expected_sha256"])].add(name)
        if source.get("previously_downloaded") or bindings[uid]["previous_split_membership"]:
            require(name == "train", "previously exposed source leaked to new holdout")
        for filename, key in [("sequence.npz", "staging_sequence_sha256"), ("meta.json", "staging_metadata_sha256")]:
            require(digest(source_pool / "staging" / uid / filename) == bindings[uid][key], "staged source SHA mismatch")
    require(all(len(names) == 1 for names in identifiers.values()), "environment/scene/content split leakage")
    flattened = [uid for group in split["groups"] for uid in group["sequence_ids"]]
    require(len(flattened) == len(set(flattened)) and set(flattened) == set(source_rows), "invalid split groups")
    for group in split["groups"]:
        require(all(split["assignments"][uid] == group["split"] and split["sequence_group"][uid] == group["group_id"]
                    for uid in group["sequence_ids"]), "group membership mismatch")
    windows = defaultdict(list)
    counts = Counter()
    loader_datasets = {}
    if check_loader:
        from src.mamba.egodemo_v2 import EgoDemoV2Dataset
        try:
            EgoDemoV2Dataset(root, diagnostic=False)
        except PermissionError:
            pass
        else:
            raise ValueError("loader permits formal access to pending/noncausal data")
    for name, summary in index["splits"].items():
        path = root / summary["window_path"]
        require(digest(path) == summary["sha256"], "window index SHA mismatch")
        for line_index, line in enumerate(path.read_text().splitlines()):
            row = json.loads(line)
            require(row["split"] == name == split["assignments"][row["sequence_id"]], "window wrong split")
            require(row["source_selection_sha256"] == source_sha and row["split_manifest_sha256"] == split_sha,
                    "window provenance SHA mismatch")
            windows[row["sequence_id"]].append((row, name, line_index))
            counts[name] += 1
        require(counts[name] == summary["window_count"], "window count mismatch")
        if check_loader:
            loader_datasets[name] = EgoDemoV2Dataset(root, split=name, diagnostic=True)
    entries = index["sequences"]
    require(len({row["sequence_id"] for row in entries}) == len(entries), "duplicate export UUID")
    hashes, results = {}, []
    for entry in entries:
        uid = entry["sequence_id"]
        try:
            metadata, events = read(root / entry["metadata_path"]), read(root / entry["events_path"])
            receipt = read(root / "sequences" / uid / "receipt.json")
            for stem in ["sequence", "metadata", "events"]:
                require(digest(root / entry[stem + "_path"]) == entry["sha256"][stem], "export artifact SHA mismatch")
            require(metadata["source_selection_sha256"] == source_sha, "sequence source SHA mismatch")
            for flag in ["training_allowed", "human_approved", "full_annotation_complete", "allowed_as_model_input"]:
                require(metadata[flag] is False, "metadata approval/input gate bypass")
            require(metadata["annotation_mode"] == "offline_noncausal", "geometry causal status misrepresented")
            require(metadata["hand_order"] == ["left", "right"] and metadata["units"] == "meters", "hand/coordinate contract")
            require(metadata["source_files"] == receipt["source_files"], "receipt source provenance mismatch")
            for role, source in metadata["source_files"].items():
                path = source["path"]
                if path not in hashes:
                    hashes[path] = digest(path)
                require(hashes[path] == source["sha256"], "refined source content SHA mismatch: " + role)
            with np.load(root / entry["sequence_path"], allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
            times, n = arrays["timestamps"], entry["frame_count"]
            m = len(arrays["object_ids"])
            require(n == source_rows[uid]["manifest_frame_count"] and times.shape == (n,), "native frame coverage")
            require(np.isfinite(times).all() and (np.diff(times) > 0).all(), "native timestamps invalid")
            require(np.array_equal(arrays["frame_index"], np.arange(n)), "native frame IDs mismatch")
            require(arrays["hands_world"].shape == (n, 2, 21, 3), "hand dimensions")
            require(arrays["hand_valid"].shape == (n, 2), "hand mask dimensions")
            masked_geometry(arrays["hands_world"], arrays["hand_valid"], "hand")
            require(arrays["object_pose_world"].shape == (n, m, 4, 4), "object pose dimensions")
            require(arrays["object_pose_valid"].shape == arrays["object_observed"].shape == (n, m), "object mask dimensions")
            se3(arrays["object_pose_world"], arrays["object_pose_valid"], "object")
            se3(arrays["camera_to_world"], np.ones(n, bool), "camera")
            require(m == len(set(arrays["object_ids"].tolist())) == len(metadata["entities"]), "entity ID ambiguity")
            require([e["id"] for e in metadata["entities"]] == arrays["object_ids"].tolist(), "entity ID order mismatch")
            require(not arrays["object_pose_valid"][:, arrays["object_kind"] == 2].any(), "deformable whole-object 6D")
            audit_sparse(arrays, n, m)
            audit_surface_source(arrays, metadata)
            if metadata["metric_geometry_allowed"] is False:
                require(not arrays["object_pose_valid"].any() and not len(arrays["surface_world"]), "quarantined metric geometry leaked")
            # Compare the entire hand/time source, and all filtered camera poses transformed independently.
            with np.load(metadata["source_files"]["hands"]["path"], allow_pickle=False) as source:
                require(np.array_equal(times, source["timestamps"]), "timestamp changed from original hand source")
                require(np.array_equal(arrays["hand_valid"], source["hand_valid"]), "hand validity changed")
                require(np.allclose(arrays["hands_world"], source["hands_world"], atol=1e-7, equal_nan=True), "hand coordinates changed")
                require(np.array_equal(arrays["camera_to_world"], source["camera_to_world"]), "camera trajectory changed")
            with np.load(metadata["source_files"]["objects"]["path"], allow_pickle=False) as source:
                expected_valid = source["pose_valid"].T.copy()
                for j, entity in enumerate(metadata["entities"]):
                    if not metadata["metric_geometry_allowed"] or not entity["metric_geometry_allowed"]:
                        expected_valid[:, j] = False
                require(np.array_equal(expected_valid, arrays["object_pose_valid"]), "filtered pose validity changed")
                f, j = np.nonzero(expected_valid)
                expected = arrays["camera_to_world"][f] @ source["poses"][j, f]
                require(np.allclose(expected, arrays["object_pose_world"][f, j], atol=2e-5, rtol=1e-5), "object world transform mismatch")
            for event in events["records"]:
                require(event["label_valid"] is False and event["review_status"] == "pending", "pending event promoted to gold")
                require(event["weak_positive"] is True, "weak candidate flag missing")
                require(event["type"] in vocab[event["layer"]]["codes"], "event absent from vocabulary")
                require(0 <= event["start_frame"] < event["end_frame"] <= n, "event outside native timeline")
                require(-1 <= event["object_index"] < m and -1 <= event["secondary_object_index"] < m, "event object index invalid")
                if event["layer"] == "operation":
                    require(event["subject"] == event["object_index"] == -1, "source semantic interval invented identity")
            require(events["gold_label_count"] == 0 and events["training_allowed"] is False, "event supervision gate bypass")
            rows = [row for row, _, _ in windows[uid]]
            for row in rows:
                require(row["sequence_sha256"] == entry["sha256"]["sequence"], "window/shard content mismatch")
                for key in ["sequence_path", "metadata_path", "events_path"]:
                    require(row[key] == entry[key], "window refers to a different sequence artifact")
            audit_windows(rows, times, index["window_config"])
            require(len(rows) == entry["window_count"], "sequence window count mismatch")
            result = {"sequence_id": uid, "status": "passed", "native_frames": n, "windows": len(rows),
                      "pose_observations": int(arrays["object_pose_valid"].sum()),
                      "surface_observations": len(arrays["surface_world"]), "events": len(events["records"])}
            if check_loader and rows:
                # Exercise the richest observation window in every acquisition,
                # including all-missing/quarantined sequences without dropping them.
                chosen = max(windows[uid], key=lambda item: int(arrays["surface_frame_ptr"][item[0]["end_frame"]]
                             - arrays["surface_frame_ptr"][item[0]["start_frame"]]))
                row, name, line_index = chosen
                sample = loader_datasets[name][line_index]
                audit_loaded_window(sample, arrays, row)
                result["loader_window_checked"] = row["window_id"]
            results.append(result)
        except Exception as error:
            results.append({"sequence_id": uid, "status": "failed", "error": str(error), "error_type": type(error).__name__})
    bad = [row for row in results if row["status"] != "passed"]
    complete = set(row["sequence_id"] for row in entries) == set(source_rows) and not index["is_partial_export"]
    return {"schema_version": "egodemo_mamba_v2_audit", "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_selection_sha256": source_sha, "split_manifest_sha256": split_sha,
            "dataset_index_sha256": digest(root / "dataset_index.json"), "expected_sequences": len(source_rows),
            "checked_sequences": len(results), "passed": len(results) - len(bad), "failed": len(bad),
            "checked_windows": sum(counts.values()), "windows_by_split": dict(counts),
            "checked_source_files": len(hashes), "records": results,
            "loader_samples_checked": sum("loader_window_checked" in row for row in results),
            "format_ready": complete and not bad, "formal_training_ready": False, "training_allowed": False,
            "independent_annotation_accuracy": None, "gold_label_count": 0,
            "subject_disjoint_verified": False, "scene_environment_id_disjoint": True,
            "residual_risks": split["residual_risks"],
            "scope_zh": "全序列来源/数组/坐标/掩码/原生时间窗口与分组一致性；不表示独立姿态准确度、人工批准或正式训练准入。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-loader", action="store_true")
    args = parser.parse_args()
    report = audit_dataset(args.dataset, args.source_pool, check_loader=args.check_loader)
    output = args.output or args.dataset / "independent_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "residual_risks"}}, ensure_ascii=False))
    print(json.dumps([row for row in report["records"] if row["status"] != "passed"][:10], ensure_ascii=False))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
