"""Read-only native-frame checks; numerical validity is not annotation approval.

No inference, interpolation, label editing or training occurs here. Missing
observations and unreviewed semantics remain pending; outlier thresholds only
create warnings. V3 exclusions are applied to temporary masks, never sources.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLDS = {
    "hand_speed_mps": 3.0,
    "bone_relative_change": .25,
    "object_speed_mps": 3.0,
    "object_rotation_deg_per_s": 720.0,
    "se3_atol": 1e-3,
}
HAND_IDS = ("left_hand", "right_hand", "both_hands")
# Official 21-joint ordering: wrist followed by five four-joint fingers.
BONES = [(0, i) for i in (1, 5, 9, 13, 17)] + [
    (i, i + 1) for start in (1, 5, 9, 13, 17) for i in range(start, start + 3)]
COUNT_METRICS = (
    "native_frames", "hand_valid_frames_left", "hand_valid_frames_right",
    "hand_missing_frames_left", "hand_missing_frames_right", "object_count",
    "raw_pose_observations", "effective_pose_observations", "raw_surface_observations",
    "effective_surface_observations", "geometry_excluded_pose_observations",
    "geometry_excluded_surface_observations", "frames_with_effective_geometry",
    "frames_without_effective_geometry", "nonrigid_object_count", "nonrigid_pose_not_applicable_objects",
    "continuous_reviewed_frames", "declared_reviewed_frames", "semantic_pending_frames",
    "known_state_positive_frame_cells", "known_state_negative_frame_cells", "transition_events",
    "censored_transition_events", "hand_speed_warning_frames", "bone_change_warning_frames",
    "object_speed_warning_observations", "object_rotation_warning_observations",
    "review_entities_without_base_geometry", "source_files_checked", "review_evidence_pages_checked",
)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _gallery_bindings(value):
    if isinstance(value, dict):
        if isinstance(value.get("gallery_path"), str):
            yield value["gallery_path"], value.get("gallery_sha256")
        for child in value.values():
            yield from _gallery_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _gallery_bindings(child)


def sequence_dependencies(base_root, base_row, annotation_path=None, review_path=None):
    """Enumerate every input read, including missing evidence paths.

    Deliberately does not require valid JSON: the orchestrator can fingerprint
    partial/missing sources and preserve failures for retry. No data are written.
    """
    root = Path(base_root)
    paths = [root / base_row[k] for k in ("sequence_path", "metadata_path", "events_path") if base_row.get(k)]
    paths += [Path(p) for p in (annotation_path, review_path) if p is not None]
    if annotation_path is not None:
        try:
            paths += [Path(annotation_path).parent.parent / name for name, _ in _gallery_bindings(_read(annotation_path))]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if review_path is not None:
        evidence_root = Path(review_path).parent.parent
        try:
            review = _read(review_path)
            for binding in review.get("evidence_manifests", []):
                if not isinstance(binding, dict):
                    continue
                if isinstance(binding.get("path"), str):
                    manifest_path = evidence_root / binding["path"]
                    paths.append(manifest_path)
                    try:
                        manifest = _read(manifest_path)
                        paths += [evidence_root / page["path"] for page in manifest.get("pages", [])
                                  if isinstance(page, dict) and isinstance(page.get("path"), str)]
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                for name in binding.get("reviewed_page_paths", []):
                    if isinstance(name, str):
                        paths.append(evidence_root / name)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return sorted({p.absolute() for p in paths}, key=str)


def _range(lo, hi, count):
    return type(lo) is int and type(hi) is int and 0 <= lo < hi <= count


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


class _Result:
    def __init__(self, row):
        count = row.get("frame_count", 0)
        count = count if type(count) is int and count >= 0 else 0
        self.data = {"sequence_id": row.get("sequence_id"), "split": row.get("split"),
            "frame_count": count, "checks": [], "metrics": dict.fromkeys(COUNT_METRICS, 0), "issues": []}
        self.metrics.update(native_frames=count, semantic_pending_frames=count,
            full_annotation_complete=False, whole_object_6d_certified=False,
            automatic_semantic_approval=False, source_mutated=False)

    @property
    def metrics(self):
        return self.data["metrics"]

    def check(self, code, status, message):
        self.data["checks"].append({"code": code, "status": status, "message_zh": message})

    def issue(self, code, severity, layer, message, action, lo=0, hi=None, entity=None):
        issue = {"code": code, "severity": severity, "layer": layer,
            "start_frame": lo, "end_frame": self.data["frame_count"] if hi is None else hi,
            "entity_id": entity, "message_zh": message, "next_action_zh": action}
        self.data["issues"].append(issue)
        return issue

    def fail(self, code, message, layer="integrity", lo=0, hi=None, entity=None):
        self.check(code, "fail", message)
        self.issue(code, "error", layer, message, "核对对应源文件和接口契约；保留原文件，在独立修订版本纠正。", lo, hi, entity)

    def gaps(self, code, mask, layer, message, entity=None):
        for lo, hi in _runs(mask):
            self.issue(code, "pending", layer, message, "打开该原生帧段的图像证据，补充可观测性或明确未知原因。", lo, hi, entity)

    def outliers(self, code, mask, message, entity=None):
        for lo, hi in _runs(mask):
            self.issue(code, "warning", "geometry", message,
                "逐帧检查投影、身份和真实快速运动；阈值不自动否决动作或改写数值。", lo, hi, entity)

    def finish(self):
        # Always retain this gap even when structural errors stop later checks.
        self.check("joint_annotation_acceptance", "pending",
            "自动数值检查不认证当前工件身份、整物体6D精度或全层标注完成。")
        self.issue("joint_annotation_acceptance", "pending", "joint_annotation",
            "数值有效不等于交互工件几何完整；整物体位姿、身份和未知标注仍需独立确认。",
            "对照实际交互实体核验几何归属、刚体范围和独立位姿精度；保持现有准入范围。")
        self.metrics["automated_checks_passed"] = not any(x["status"] == "fail" for x in self.data["checks"])
        self.metrics["issue_counts"] = dict(Counter(x["severity"] for x in self.data["issues"]))
        return self.data


def _document(path, uid, count, result, code):
    if path is None:
        return None
    try:
        doc = _read(path)
    except (OSError, ValueError, TypeError) as error:
        result.fail(code, f"无法读取指定标注文件：{error}")
        return None
    if doc.get("sequence_id") != uid or doc.get("frame_count", count) != count:
        result.fail(code, "标注采集ID或原生帧数与数值分片不一致。")
        return None
    return doc


def _annotation_evidence(annotation, path, result):
    if annotation is None:
        return
    missing = 0
    for name, expected in set(_gallery_bindings(annotation)):
        try:
            if not isinstance(expected, str) or _sha(Path(path).parent.parent / name) != expected:
                result.fail("annotation_evidence", "V3视觉证据页缺少内容SHA或与绑定不一致。", "semantic")
        except (FileNotFoundError, PermissionError):
            missing += 1
        except OSError as error:
            result.fail("annotation_evidence", f"无法读取V3证据页：{error}", "semantic")
    if missing:
        result.check("annotation_evidence", "pending", f"{missing}个V3视觉证据文件未提供，保留来源记录但无法复验图像。")
        result.issue("annotation_evidence_missing", "pending", "semantic", "V3抽样证据文件未齐；抽样审核不等于连续审核。", "恢复绑定的证据页后重新运行，只读取原内容，不替换相邻帧。")


def _shape_checks(a, count, result):
    required = {"timestamps", "frame_index", "hands_world", "hand_valid", "camera_to_world",
        "object_ids", "object_kind", "object_pose_world", "object_pose_valid", "object_observed",
        "surface_frame_index", "surface_object_index", "surface_point_index", "surface_world",
        "surface_frame_ptr", "point_keys", "point_object_index"}
    missing = sorted(required - set(a))
    if missing:
        result.fail("array_schema", "数值分片缺少字段：" + ", ".join(missing))
        return False
    N, S, M = a["object_ids"].size, a["surface_frame_index"].size, a["point_keys"].size
    shapes = {"timestamps": (count,), "frame_index": (count,), "hands_world": (count, 2, 21, 3),
        "hand_valid": (count, 2), "camera_to_world": (count, 4, 4), "object_ids": (N,),
        "object_kind": (N,), "object_pose_world": (count, N, 4, 4), "object_pose_valid": (count, N),
        "object_observed": (count, N), "surface_frame_index": (S,), "surface_object_index": (S,),
        "surface_point_index": (S,), "surface_world": (S, 3), "surface_frame_ptr": (count + 1,),
        "point_keys": (M,), "point_object_index": (M,)}
    bad = [f"{k}: {a[k].shape} != {v}" for k, v in shapes.items() if a[k].shape != v]
    boolean = ("hand_valid", "object_pose_valid", "object_observed")
    integers = ("frame_index", "object_kind", "surface_frame_index", "surface_object_index",
        "surface_point_index", "surface_frame_ptr", "point_object_index")
    floating = ("timestamps", "hands_world", "camera_to_world", "object_pose_world", "surface_world")
    bad += [k + "必须为bool" for k in boolean if a[k].dtype != np.bool_]
    bad += [k + "必须为整数" for k in integers if a[k].dtype.kind not in "iu"]
    bad += [k + "必须为浮点数" for k in floating if a[k].dtype.kind != "f"]
    bad += [k + "必须为字符串" for k in ("object_ids", "point_keys") if a[k].dtype.kind not in "US"]
    if bad:
        result.fail("array_schema", "；".join(bad))
        return False
    result.check("array_schema", "pass", "所有原生数组字段、形状和基本类型一致。")
    return True


def _se3_valid(poses, atol):
    finite = np.isfinite(poses).all(axis=(-2, -1))
    good = np.zeros(poses.shape[:-2], dtype=bool)
    p = poses[finite].astype(np.float64)
    R = p[:, :3, :3]
    if len(p):
        good[finite] = (np.max(np.abs(p[:, 3] - [0, 0, 0, 1]), axis=1) <= atol) & (
            np.max(np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)), axis=(1, 2)) <= atol) & (
            np.abs(np.linalg.det(R) - 1) <= atol)
    return good


def _geometry(a, metadata, annotation, thresholds, result):
    T, N = len(a["timestamps"]), len(a["object_ids"])
    ids = a["object_ids"].astype(str).tolist()
    result.metrics["object_count"] = N
    identity_ok = len(set(ids)) == N and all(ids) and np.isin(a["object_kind"], [0, 1, 2, 3]).all()
    if not identity_ok:
        result.fail("object_identity", "物体ID重复/为空，或材料类型不属于冻结编码。", "entity")
    else:
        result.check("object_identity", "pass", "物体ID唯一；仅检查片段键，不推定跨遮挡身份。")
    entities = {e.get("entity_id", e.get("id")): e for e in metadata.get("entities", [])}
    if metadata.get("entities") is not None and set(entities) != set(ids):
        result.fail("metadata_identity", "元信息实体集合与NPZ物体ID不一致。", "entity")
    pose = a["object_pose_world"]
    raw = a["object_pose_valid"]
    result.metrics["raw_pose_observations"] = int(raw.sum())
    structural = _se3_valid(pose, thresholds["se3_atol"])
    if (raw & ~structural).any() or not np.isnan(pose[~raw]).all():
        result.fail("object_se3", "有效物体位姿不满足有限SE(3)，或无效位姿未保持NaN。", "geometry")
    else:
        result.check("object_se3", "pass", "有效候选位姿满足SE(3)；无效位姿保持NaN。")
    if not _se3_valid(a["camera_to_world"], thresholds["se3_atol"]).all():
        result.fail("camera_se3", "至少一个原生相机变换不满足有限SE(3)。", "geometry")
    else:
        result.check("camera_se3", "pass", "原生相机变换满足SE(3)数值约束。")
    nonrigid = np.isin(a["object_kind"], [2, 3])
    result.metrics["nonrigid_object_count"] = int(nonrigid.sum())
    result.metrics["nonrigid_pose_not_applicable_objects"] = int(nonrigid.sum())
    illegal_nonrigid = []
    for j in np.flatnonzero(nonrigid):
        scope = entities.get(ids[j], {}).get("pose_scope", "unknown")
        if raw[:, j].any() and scope not in ("rigid_part", "rigid_part_candidate", "surface_only", "local_surface"):
            illegal_nonrigid.append(ids[j])
    if illegal_nonrigid:
        result.fail("nonrigid_pose_scope", "非刚体记录了整物体或未说明局部范围的6D：" + ", ".join(illegal_nonrigid), "geometry")
    else:
        result.check("nonrigid_pose_scope", "pass", "非刚体整物体6D不适用；局部刚体候选不能替代整体。")

    sf, so, sp = (a[k] for k in ("surface_frame_index", "surface_object_index", "surface_point_index"))
    ptr, keys, owners = a["surface_frame_ptr"], a["point_keys"].astype(str), a["point_object_index"]
    S, M = len(sf), len(keys)
    result.metrics["raw_surface_observations"] = S
    if metadata.get("metric_geometry_allowed") is False and (raw.any() or S):
        result.fail("calibration_geometry_gate", "标定/质量门禁禁止米制几何，但分片仍含位姿或表面点。", "geometry")
    bounds = (np.all((0 <= sf) & (sf < T)) and np.all((0 <= so) & (so < N))
        and np.all((0 <= sp) & (sp < M)) and np.all((0 <= owners) & (owners < N)))
    surface_ok = bool(bounds)
    if bounds:
        expected_ptr = np.r_[0, np.cumsum(np.bincount(sf.astype(np.int64), minlength=T))]
        surface_ok = bool(np.array_equal(ptr, expected_ptr) and np.all(sf[1:] >= sf[:-1])
            and np.array_equal(owners[sp], so) and len(set(keys.tolist())) == M and all(keys)
            and np.isfinite(a["surface_world"]).all())
        # Each persistent tracklet/point key is observed at most once per frame.
        if S:
            pairs = np.rec.fromarrays([sf, sp])
            surface_ok &= len(np.unique(pairs)) == S
    if not surface_ok:
        result.fail("surface_csr", "表面CSR指针、原生帧/物体/点索引、点归属或有限坐标不一致。", "geometry")
    else:
        result.check("surface_csr", "pass", "表面点CSR及稳定点键逐帧一致，无越界或重复观测。")
        observed = raw.copy()
        observed[sf, so] = True
        if not np.array_equal(observed, a["object_observed"]):
            result.fail("object_observed_mask", "候选可见几何mask与位姿/表面观测不一致。", "geometry")
        else:
            result.check("object_observed_mask", "pass", "几何观测mask与候选数组一致；不代表语义存在标签。")

    excluded = np.zeros((T, N), bool)
    bad_exclusions = False
    for row in (annotation or {}).get("geometry_exclusions", []):
        identity = row.get("object_id")
        scope = row.get("scope", row.get("geometry_exclusion_scope"))
        if scope == "entity_until_reidentified":
            lo, hi = 0, T
        elif scope in ("frame_range", "specific_frames", "specific_frame", "frames"):
            if "frames" in row or "evidence_frames" in row:
                frames = row.get("frames", row.get("evidence_frames"))
                if identity not in ids or not isinstance(frames, list) or any(type(f) is not int or not 0 <= f < T for f in frames):
                    bad_exclusions = True
                    continue
                excluded[frames, ids.index(identity)] = True
                continue
            lo, hi = row.get("start_frame"), row.get("end_frame")
        else:
            bad_exclusions = True
            continue
        if identity not in ids or not _range(lo, hi, T):
            bad_exclusions = True
            continue
        excluded[lo:hi, ids.index(identity)] = True
    if bad_exclusions:
        result.fail("geometry_exclusions", "几何排除含未知实体、未知scope或非法帧范围；未猜测扩大排除范围。", "geometry")
    elif annotation is None:
        result.check("geometry_exclusions", "pending", "没有V3几何复核层，无法确认已有候选是否被视觉否决。")
    else:
        result.check("geometry_exclusions", "pass", "V3显式排除已作用于临时mask，源文件未修改。")
    effective = raw & structural & ~excluded
    result.metrics.update(effective_pose_observations=int(effective.sum()),
        geometry_excluded_pose_observations=int((raw & excluded).sum()))
    present = effective.copy()
    if surface_ok:
        keep = ~excluded[sf, so]
        result.metrics.update(effective_surface_observations=int(keep.sum()),
            geometry_excluded_surface_observations=int((~keep).sum()))
        present[sf[keep], so[keep]] = True
    result.metrics["frames_with_effective_geometry"] = int(present.any(1).sum())
    result.metrics["frames_without_effective_geometry"] = T - int(present.any(1).sum())
    result.gaps("scene_geometry_missing", ~present.any(1), "geometry", "该段没有保留的场景几何候选，不能由缺测推断对象不存在。")
    for j, identity in enumerate(ids):
        if a["object_kind"][j] == 1:
            result.gaps("rigid_pose_missing", ~effective[:, j], "geometry", "该刚体候选缺少有效位姿；也可能尚未出现或已遮挡。", identity)
    result.check("geometry_observability", "pending", "几何覆盖按排除后的候选统计；与当前实际工件的对应仍需确认。")
    return effective, ids


def _motion(a, effective, ids, thresholds, result):
    times, hands, hv = a["timestamps"], a["hands_world"], a["hand_valid"]
    T = len(times)
    dt = np.diff(times)
    if not np.isfinite(times).all() or np.any(dt <= 0):
        result.check("motion_outliers", "pending", "时间轴错误，无法可靠计算速度或角速度。")
        return
    safe_hv = hv & np.isfinite(hands).all(axis=(2, 3))
    for j, identity in enumerate(HAND_IDS[:2]):
        adjacent = safe_hv[:-1, j] & safe_hv[1:, j]
        speed = np.linalg.norm(np.diff(hands[:, j, 0].astype(np.float64), axis=0), axis=1) / dt
        high = np.r_[False, adjacent & (speed > thresholds["hand_speed_mps"])]
        result.metrics["hand_speed_warning_frames"] += int(high.sum())
        result.outliers("hand_speed", high, "手腕相邻有效帧速度超过配置阈值。", identity)
        lengths = np.stack([np.linalg.norm(hands[:, j, v] - hands[:, j, u], axis=1) for u, v in BONES], axis=1)
        if safe_hv[:, j].any():
            median = np.median(lengths[safe_hv[:, j]], axis=0)
            variation = np.abs(lengths - median) / np.maximum(median, 1e-6)
            high = safe_hv[:, j] & ((variation > thresholds["bone_relative_change"]) | (lengths <= 1e-6)).any(1)
            result.metrics["bone_change_warning_frames"] += int(high.sum())
            result.outliers("hand_bone_change", high, "手部骨长相对本段有效帧中位数异常或骨段近零；须确认关节顺序和姿态。", identity)
    pose = a["object_pose_world"].astype(np.float64)
    for j, identity in enumerate(ids):
        adjacent = effective[:-1, j] & effective[1:, j]
        speed = np.linalg.norm(np.diff(pose[:, j, :3, 3], axis=0), axis=1) / dt
        high = np.r_[False, adjacent & (speed > thresholds["object_speed_mps"])]
        result.metrics["object_speed_warning_observations"] += int(high.sum())
        result.outliers("object_translation_jump", high, "物体相邻有效帧平移速度超过配置阈值；不跨缺测帧计算。", identity)
        R = pose[:, j, :3, :3]
        cosine = (np.einsum("tij,tij->t", R[:-1], R[1:]) - 1) / 2
        angular = np.degrees(np.arccos(np.clip(cosine, -1, 1))) / dt
        high = np.r_[False, adjacent & (angular > thresholds["object_rotation_deg_per_s"])]
        result.metrics["object_rotation_warning_observations"] += int(high.sum())
        result.outliers("object_rotation_jump", high, "物体相邻有效帧旋转角速度超过配置阈值；未插值补全旋转。", identity)
    total = sum(result.metrics[k] for k in ("hand_speed_warning_frames", "bone_change_warning_frames",
        "object_speed_warning_observations", "object_rotation_warning_observations"))
    result.check("motion_outliers", "warn" if total else "pass", f"检测到{total}个手/物体数值异常帧项；仅为复核候选。")


def _review_evidence(review, review_path, T, result):
    declared = np.zeros(T, bool)
    for span in review.get("reviewed_ranges", []):
        pair = [span.get("start_frame"), span.get("end_frame")] if isinstance(span, dict) else span
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not _range(*pair, T):
            result.fail("reviewed_ranges", "连续审核范围不是合法原生半开区间。", "semantic")
            continue
        declared[pair[0]:pair[1]] = True
    result.metrics["declared_reviewed_frames"] = int(declared.sum())
    root = Path(review_path).parent.parent
    covered = np.zeros(T, bool)
    missing = False
    for binding in review.get("evidence_manifests", []):
        try:
            path = root / binding["path"]
            if _sha(path) != binding["sha256"]:
                raise ValueError("证据manifest SHA不一致")
            doc = _read(path)
            if doc.get("sequence_id") != result.data["sequence_id"]:
                raise ValueError("证据采集ID不一致")
            pages = {p["path"]: p for p in doc["pages"]}
            chosen, hashes = binding["reviewed_page_paths"], binding["reviewed_page_sha256"]
            if len(chosen) != len(hashes) or len(set(chosen)) != len(chosen):
                raise ValueError("实际审核页及SHA列表不一致")
            for name, expected in zip(chosen, hashes):
                row = pages.get(name)
                if row is None or row["sha256"] != expected or _sha(root / name) != expected:
                    raise ValueError("实际审核页内容或SHA不一致")
                frames = row["frame_indices"]
                if any(type(f) is not int or not 0 <= f < T for f in frames):
                    raise ValueError("证据页帧索引越界")
                covered[frames] = True
                result.metrics["review_evidence_pages_checked"] += 1
        except (FileNotFoundError, PermissionError):
            missing = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.fail("review_evidence", f"连续证据绑定错误：{error}", "semantic")
    actual = declared & covered
    if (review.get("annotation_source") != "assistant_visual_review"
            or not review.get("reviewer_agent") or not review.get("reviewed_at")):
        result.fail("review_provenance", "连续审核缺少实际审核来源、审核者或时间；生成联系表本身不算已审核。", "semantic")
        actual[:] = False
    uncovered = declared & ~covered
    if uncovered.any() or missing:
        result.check("review_evidence", "pending", "部分声明审核帧缺少可核验的实际审核页；不计入已完成覆盖。")
        result.gaps("review_evidence_missing", uncovered, "semantic", "声明审核范围尚未绑定可读取且SHA一致的证据页。")
    elif declared.any():
        result.check("review_evidence", "pass", "已保存审核范围由实际审核页及SHA覆盖；程序没有新增视觉审核。")
    else:
        result.check("review_evidence", "pending", "没有可确认的连续原生帧审核范围。")
    return actual, declared


def _review(review, path, annotation, ids, T, result):
    if review is None:
        result.check("semantic_review", "pending", "没有连续审核文件，全部原生帧仍待语义复核。")
        result.gaps("semantic_review_missing", np.ones(T, bool), "semantic", "无连续逐帧审核记录；不能用源文本或数值规则补成正负标签。")
        return
    actual, declared = _review_evidence(review, path, T, result)
    result.metrics.update(continuous_reviewed_frames=int(actual.sum()), semantic_pending_frames=T - int(actual.sum()))
    result.gaps("semantic_review_missing", ~actual, "semantic", "该原生帧段没有可核验的连续审核覆盖。")
    entities = review.get("entities", [])
    names = [e.get("entity_id", e.get("id")) for e in entities]
    if any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names):
        result.fail("review_entity_ids", "审核实体ID为空或重复。", "entity")
    known = set(x for x in names if isinstance(x, str)) | set(HAND_IDS)
    unmapped = known - set(ids or []) - set(HAND_IDS)
    result.metrics["review_entities_without_base_geometry"] = len(unmapped)
    if unmapped:
        result.check("review_entity_geometry", "pending", f"{len(unmapped)}个审核实体没有同ID的基础几何，未按名称或局部部件强制绑定。")
    for e in entities:
        for part in e.get("source_geometry_parts", []):
            if ids is not None and part.get("source_object_id") not in ids:
                result.fail("review_geometry_reference", "局部几何引用不是基础数值分片的物体ID。", "entity", entity=e.get("entity_id", e.get("id")))
    if review.get("annotation_source") == "assistant_visual_review" and review.get("human_label_valid") is True:
        result.fail("review_provenance", "助手审核被错误标为human有效。", "semantic")
    aliases = {}
    for alias in review.get("visual_identity_aliases", []):
        src, dst = alias.get("alias_id"), alias.get("canonical_entity_id")
        if not isinstance(src, str) or not src.startswith("visual_") or dst not in known or not str(dst).startswith("visual_"):
            result.fail("review_alias", "视觉别名引用非法，或试图与旧几何ID合并。", "entity")
        elif not _range(alias.get("start_frame"), alias.get("end_frame"), T):
            result.fail("review_alias", "视觉别名缺少有效范围，不能宣称全段同一身份。", "entity")
        else:
            aliases[src] = (dst, alias["start_frame"], alias["end_frame"])
            known.add(src)
    def canonical(identity, frame):
        value = aliases.get(identity)
        return value[0] if value is not None and value[1] <= frame < value[2] else identity
    cells, record_ids = {}, set()
    bad_before = sum(c["status"] == "fail" for c in result.data["checks"])
    for transition, records in ((False, review.get("state_segments", [])), (True, review.get("transition_events", []))):
        for r in records:
            identity = r.get("object_id")
            record_id = r.get("id")
            if record_id is not None:
                if record_id in record_ids:
                    result.fail("review_record_ids", "审核记录ID重复。", "semantic", entity=identity)
                record_ids.add(record_id)
            if transition:
                result.metrics["transition_events"] += 1
                span = r.get("boundary_frame_range", [])
            else:
                span = [r.get("start_frame"), r.get("end_frame")]
            if not isinstance(span, (list, tuple)) or len(span) != 2 or not _range(*span, T):
                result.fail("review_label_range", "状态或转换区间越界/为空/非原生整数帧。", "semantic", entity=identity)
                continue
            lo, hi = span
            if r.get("layer") not in ("operation", "relation", "cooperation", "transition") or not isinstance(r.get("type"), str) or not r["type"]:
                result.fail("review_label_type", "监督层或类别键缺失/非法。", "semantic", lo, hi, identity)
            if transition != (r.get("layer") == "transition"):
                result.fail("review_label_type", "转换与持续状态存放在错误的记录层。", "semantic", lo, hi, identity)
            refs = [identity, r.get("secondary_object_id")]
            refs += list(r.get("participant_object_ids", []))
            if r.get("subject") in HAND_IDS:
                refs.append(r["subject"])
            elif r.get("subject") not in ("object", None, "unknown"):
                refs.append(r.get("subject"))
            if any(x is not None and x not in known for x in refs):
                result.fail("review_entity_reference", "交互记录引用了未声明的实体。", "entity", lo, hi, identity)
            if identity is None or r.get("subject") in (None, "unknown"):
                result.issue("review_participants_unknown", "pending", "entity", "交互主体或客体未明确，不能自动绑定到近邻物体。", "对照证据明确实体和方向，无法判定则保留未知。", lo, hi, identity)
            silver = r.get("silver_label_valid") is True
            evidence = r.get("evidence_frames", [])
            valid_evidence = isinstance(evidence, list) and all(type(f) is int and 0 <= f < T for f in evidence)
            if not valid_evidence:
                result.fail("review_label_evidence", "监督证据帧不是合法原生帧号。", "semantic", lo, hi, identity)
                continue
            ef = set(evidence)
            if silver and (r.get("annotation_source") != "assistant_visual_review" or r.get("human_label_valid") is True):
                result.fail("review_provenance", "银标来源或human标志与助手审核契约不符。", "semantic", lo, hi, identity)
            if ef and not declared[list(ef)].all():
                result.fail("review_label_evidence", "标签证据超出声明连续审核范围。", "semantic", lo, hi, identity)
            if transition:
                censored = r.get("censored")
                if type(censored) is not bool:
                    result.fail("transition_censoring", "转换记录缺少显式删失布尔标记。", "transition", lo, hi, identity)
                elif censored:
                    result.metrics["censored_transition_events"] += 1
                    issue = result.issue("transition_censored", "pending", "transition", "转换边界删失，仅保留不确定范围。", "补看可见前后状态；不要自动压为精确帧或预测时间窗正例。", lo, hi, identity)
                    issue.update(source_record_id=r.get("id"), event_type=r.get("type"),
                        subject=r.get("subject"), secondary_object_id=r.get("secondary_object_id"),
                        source_reason_zh=r.get("reason_zh"))
                if r.get("frame_resolved") is True and (hi - lo != 1 or censored is not False):
                    result.fail("transition_precision", "范围或删失转换被错误标为精确单帧。", "transition", lo, hi, identity)
                if silver and not ef:
                    result.fail("transition_evidence", "银标转换没有实际证据帧。", "transition", lo, hi, identity)
                if r.get("occurrence_confirmed") is True and censored is False:
                    before, after = r.get("before_evidence_frame"), r.get("after_evidence_frame")
                    if not (type(before) is int and type(after) is int and before < lo <= after < hi
                            and before in ef and after in ef and r.get("before_after_observed") is True):
                        result.fail("transition_before_after", "发生监督缺少包住边界范围的明确前后状态证据。", "transition", lo, hi, identity)
                continue
            value = r.get("value")
            if not silver:
                result.issue("state_unknown", "pending", "semantic", "状态尚无有效银标；未解释为负例。", "查看已绑定证据后明确0/1或保留未知。", lo, hi, identity)
                continue
            if type(value) is not int or value not in (0, 1):
                result.fail("state_value", "有效状态必须是明确整数0或1，未知不能当负例。", "semantic", lo, hi, identity)
                continue
            frames = set(range(lo, hi))
            if not frames <= ef or not declared[lo:hi].all():
                result.fail("state_evidence_range", "持续状态覆盖了未实际提供证据的中间帧。", "semantic", lo, hi, identity)
                continue
            for f in range(lo, hi):
                key = (f, r.get("layer"), r.get("type"), r.get("subject"), canonical(identity, f),
                    canonical(r.get("secondary_object_id"), f))
                if key in cells and cells[key] != value:
                    result.fail("state_conflict", "同一原生帧有向关系同时出现明确正例和负例。", "semantic", f, f + 1, identity)
                else:
                    cells[key] = value
    result.metrics["known_state_positive_frame_cells"] = int(sum(v == 1 and actual[k[0]] for k, v in cells.items()))
    result.metrics["known_state_negative_frame_cells"] = int(sum(v == 0 and actual[k[0]] for k, v in cells.items()))
    if sum(c["status"] == "fail" for c in result.data["checks"]) == bad_before:
        result.check("review_label_integrity", "pass", "已存状态/转换的引用、范围、正负一致性和删失字段已检查；未审标签仍未知。")


def review_sequence(base_root, base_row, annotation_path=None, review_path=None, thresholds=None):
    """Return JSON-safe checks, metrics and issues without changing any source.

    ``base_row`` is a V2 dataset_index sequence record; hashes are verified when
    supplied. ``review_path`` points to a V4/V5 dense_reviews/<uid>.json. Evidence
    paths resolve against that review's dataset root. Missing evidence is
    pending, mismatched evidence is an integrity failure. All time ranges are
    native half-open intervals.
    """
    result = _Result(base_row)
    settings = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if set(settings) != set(DEFAULT_THRESHOLDS) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in settings.values()):
        result.fail("threshold_config", "数值阈值包含未知键，或不是有限正数。")
        return result.finish()
    result.metrics["thresholds"] = settings
    root = Path(base_root)
    uid, count = result.data["sequence_id"], result.data["frame_count"]
    annotation = _document(annotation_path, uid, count, result, "annotation_binding")
    review = _document(review_path, uid, count, result, "review_binding")
    _annotation_evidence(annotation, annotation_path, result)
    for doc in (annotation, review):
        if doc is not None and "split" in doc and doc["split"] != base_row.get("split"):
            result.fail("annotation_split", "审核标注的采集划分与冻结基础划分不一致。")
    if annotation is not None and review is not None:
        expected = review.get("source_bindings", {}).get("v3_annotation_sha256")
        if expected is not None and expected != _sha(annotation_path):
            result.fail("review_binding", "连续审核绑定的V3标注SHA与实际文件不一致。")
    events = None
    try:
        seq_path = root / base_row["sequence_path"]
        metadata_path = root / base_row["metadata_path"]
        source_paths = [("sequence", seq_path), ("metadata", metadata_path)]
        if base_row.get("events_path"):
            source_paths.append(("events", root / base_row["events_path"]))
        for role, path in source_paths:
            digest = _sha(path)
            expected = base_row.get("sha256", {}).get(role, base_row.get(role + "_sha256"))
            if expected is not None and digest != expected:
                result.fail("source_sha256", f"{role}文件与冻结SHA不一致。")
            result.metrics["source_files_checked"] += 1
        metadata = _read(metadata_path)
        if base_row.get("events_path"):
            events = _read(root / base_row["events_path"])
        with np.load(seq_path, allow_pickle=False) as archive:
            a = {k: archive[k] for k in archive.files}
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.fail("sequence_files", f"无法读取完整V2分片：{error}")
        _review(review, review_path, annotation, None, count, result)
        return result.finish()
    if metadata.get("sequence_id") != uid or metadata.get("frame_count") != count:
        result.fail("metadata_timeline", "元信息采集ID或帧数与冻结清单不一致。")
    if "hand_order" in metadata and metadata["hand_order"] != ["left", "right"]:
        result.fail("hand_order", "元信息左右手顺序不符合固定接口；检查器不会暗中翻转手别。", "hand")
    if annotation is not None:
        bindings = annotation.get("source_bindings", {})
        for key, role in (("base_sequence_sha256", "sequence"), ("base_metadata_sha256", "metadata")):
            expected = base_row.get("sha256", {}).get(role, base_row.get(role + "_sha256"))
            if key in bindings and expected is not None and bindings[key] != expected:
                result.fail("annotation_binding", "V3标注绑定的源SHA与当前数值分片不符。")
    if not _shape_checks(a, count, result):
        _review(review, review_path, annotation, None, count, result)
        return result.finish()
    if events is not None:
        invalid_events = False
        records = events.get("records")
        if not isinstance(records, list):
            invalid_events = True
        else:
            N = len(a["object_ids"])
            for r in records:
                if not isinstance(r, dict) or not _range(r.get("start_frame"), r.get("end_frame"), count):
                    invalid_events = True
                    continue
                for key in ("object_index", "secondary_object_index"):
                    j = r.get(key, -1)
                    if type(j) is not int or not -1 <= j < N:
                        invalid_events = True
                if r.get("label_valid") is not False:
                    invalid_events = True
        if invalid_events:
            result.fail("source_event_candidates", "V2弱候选的范围/客体索引非法，或候选被错误升级为有效标签。", "semantic")
        else:
            result.check("source_event_candidates", "pass", "V2事件来源SHA与候选范围已核；弱候选未计入连续监督。")
    times = a["timestamps"]
    if count == 0 or not np.array_equal(a["frame_index"], np.arange(count)) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        result.fail("native_timeline", "原生时间轴缺帧/重复、时间非有限或未严格递增。")
    else:
        result.check("native_timeline", "pass", "逐原生帧索引连续、时间戳有限且严格递增。")
    hands, hv = a["hands_world"], a["hand_valid"]
    if not np.isfinite(hands[hv]).all() or not np.isnan(hands[~hv]).all():
        result.fail("hand_mask_finite", "有效手关节含非有限值，或无效手没有保持NaN。", "hand")
    else:
        result.check("hand_mask_finite", "pass", "左右手mask与21点有限值/NaN一致，未自动纠正手别。")
    for j, side in enumerate(("left", "right")):
        result.metrics["hand_valid_frames_" + side] = int(hv[:, j].sum())
        result.metrics["hand_missing_frames_" + side] = count - int(hv[:, j].sum())
        result.gaps("hand_missing", ~hv[:, j], "hand", "官方手姿态缺测，未插值补造。", HAND_IDS[j])
    effective, ids = _geometry(a, metadata, annotation, settings, result)
    _motion(a, effective, ids, settings, result)
    _review(review, review_path, annotation, ids, count, result)
    return result.finish()
