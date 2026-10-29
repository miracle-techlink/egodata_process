"""Calibrated native hand exports and traceable, non-gold semantic candidates.

Never fabricates object poses, contact events, hand identities or precise verb
boundaries from a coarse source sentence. Object reconstruction is a separate
stage. Run with --pool (frozen 300-source pool) --output (new private directory).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from src.dataset.check_camera_rig import quaternion_matrices
from src.dataset.fusion_geometry import world_to_camera, project_pinhole


# Vocabulary size is NOT a measured class count. Only literal source phrases
# propose operations; ambiguous, negated and unmapped sentences remain reviewable.
RULES = [
    ('insert', '插入/装入', r'\b(insert\w*|slot\w*)\b'),
    ('extract', '取出/拆出', r'\b(remov\w*|extract\w*|take\s+out|pull\s+out)\b'),
    ('plug', '插接', r'\bplug(?:s|ged|ging)?\s+(?:\w+\s+){0,4}in(?:to)?\b'),
    ('unplug', '拔接', r'\b(unplug\w*|disconnect\w*)\b'),
    ('tighten', '旋紧', r'\b(tighten\w*|screw\s+(?:\w+\s+){0,3}(?:in|on))\b'),
    ('loosen', '旋松', r'\b(loosen\w*|unscrew\w*)\b'),
    ('rotate', '转动', r'\b(rotat\w*|twist\w*|turn(?:s|ing|ed)?\s+(?:the\s+)?(?:lid|cap|knob|handle))\b'),
    ('open', '打开', r'\bopen(?:s|ed|ing)?\b'),
    ('close', '合上', r'\bclos(?:e|es|ed|ing)\b'),
    ('push', '推', r'\bpush\w*\b'),
    ('pull', '拉', r'\bpull\w*\b'),
    ('press', '按压', r'\b(press\w*|punch(?:es|ed|ing)?)\b'),
    ('cut', '剪切', r'\b(cut(?:s|ting)?|snip\w*)\b'),
    ('tear', '撕开', r'\b(tear(?:s|ing)?|tore|rip(?:s|ped|ping)?)\b'),
    ('peel', '揭除/剥离', r'\bpeel\w*\b'),
    ('attach', '贴附/连接', r'\b(attach\w*|affix\w*|stick(?:s|ing)?\s+(?:the\s+)?(?:label|tape)|apply\s+(?:the\s+)?(?:label|tape))\b'),
    ('seal', '封口', r'\bseal(?:s|ed|ing)?\b'),
    ('fold', '折叠', r'\bfold(?:s|ed|ing)?\b'),
    ('unfold', '展开', r'\bunfold\w*\b'),
    ('wind', '绕卷/缠绕', r'\b(coil\w*|wind(?:s|ing)?|wrap(?:s|ped|ping)?)\b'),
    ('unwind', '解卷/解缠', r'\b(uncoil\w*|unwind\w*|unwrap\w*|untangle\w*)\b'),
    ('tie', '系紧/打结', r'\b(tie(?:s|d)?|tying|knot(?:s|ted|ting)?)\b'),
    ('untie', '解开', r'\b(untie\w*|untying|unfasten\w*)\b'),
    ('pour', '倾倒', r'\bpour\w*\b'),
    ('fill', '填充/补充', r'\b(refill\w*|fill(?:s|ed|ing)?|replenish\w*)\b'),
    ('stir', '搅拌/混合', r'\b(stir(?:s|red|ring)?|mix(?:es|ed|ing)?)\b'),
    ('wipe', '擦拭', r'\b(wip\w*|eras\w*|scrub\w*)\b'),
    ('scan', '扫描', r'\bscan(?:s|ned|ning)?\b'),
    ('align', '对齐/调整', r'\b(align\w*|adjust\w*|straighten\w*)\b'),
    ('stack', '堆叠', r'\bstack(?:s|ed|ing)?\b'),
    ('separate', '分离/分拣', r'\b(separat\w*|sort(?:s|ed|ing)?)\b'),
    ('hang', '挂放', r'\bhang(?:s|ing)?\b'),
    ('shake', '摇动', r'\b(shak\w*|shook)\b'),
    ('squeeze', '挤压', r'\bsqueez\w*\b'),
    ('grasp', '拿取', r'\b(grasp\w*|grab\w*|pick\w*\s+up)\b'),
    ('place', '放置', r'\b(plac(?:e|es|ed|ing)|put(?:s|ting)?|set\s+down)\b'),
    ('transfer', '移动/递送', r'\b(mov(?:e|es|ed|ing)|transfer\w*|deliver\w*|pass\w*\s+(?:the|a))\b'),
    ('hold', '持握/扶稳', r'\b(hold\w*|stabiliz\w*|steady\w*)\b'),
    ('reach', '伸手接近', r'\breach\w*\b'),
    ('inspect', '检查', r'\b(inspect\w*|check\w*|examin\w*)\b'),
    ('cap', '盖帽', r'\bcap(?:s|ped|ping)?\s+(?:the|a|both|each)\b'),
    ('uncap', '取下帽盖', r'\buncap\w*\b'),
    ('flip', '翻转', r'\bflip\w*\b'),
    ('discard', '丢弃', r'\b(throw\w*|discard\w*)\b'),
    ('spread', '铺开', r'\bspread\w*\b'),
    ('collect', '收集/聚拢', r'\b(collect\w*|gather\w*)\b'),
    ('dispense', '滴取/滴注', r'\b(draw\s+liquid|drop\s+it)\b'),
    ('assemble', '组装/安装', r'\b(assembl\w*|install\w*)\b'),
    ('arrange', '整理/排列', r'\b(arrang\w*|organiz\w*|organis\w*)\b'),
]
OPERATIONS = {key: name for key, name, _ in RULES}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def operation_candidates(description):
    result = []
    for code, name, pattern in RULES:
        matches = list(re.finditer(pattern, description, flags=re.I))
        if matches:
            result.append({'operation_type': code, 'operation_name_zh': name,
                           'text_evidence': [{'text': m.group(), 'span': list(m.span())} for m in matches]})
    return result


def source_intervals(events, timestamps):
    timestamps = np.asarray(timestamps, dtype=float)
    if len(timestamps) < 2 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Native timestamps must be finite, ordered and nontrivial')
    duration = float(timestamps[-1] + np.median(np.diff(timestamps)))
    rows, candidates = [], []
    for index, event in enumerate(events['subtask_annotations']):
        start, end = float(event['start_time']), float(event['end_time'])
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration + .1:
            raise ValueError('Source event outside native timeline')
        a, b = np.searchsorted(timestamps, [start, end], side='left').tolist()
        if a >= b:
            raise ValueError('Source interval contains no native frames')
        identity = f'source_{index:04d}'
        row = {'source_interval_id': identity, 'source_event_id': event['subtask_id'],
               'start_seconds': start, 'end_seconds': end,
               'start_frame': a, 'end_frame': b, 'description_original': event['description'],
               'source_annotation': event, 'review_status': 'pending'}
        rows.append(row)
        for proposed in operation_candidates(event['description']):
            candidates.append({**proposed, 'candidate_id': f'operation_{len(candidates):05d}',
                               'source_interval_id': identity, 'possible_start_frame': a,
                               'possible_end_frame': b, 'subject': None, 'object_id': None,
                               'temporal_precision': 'source_interval_only',
                               'label_source': 'source_text_rule_candidate', 'review_status': 'pending',
                               'label_valid': False})
    return rows, candidates


def export_one(pool_string, output_string, record):
    import pyarrow.parquet as pq
    pool, output = Path(pool_string), Path(output_string)
    uid = record['source_episode_uuid']
    dest = output / 'sequences' / uid
    if (dest / 'receipt.json').exists():
        receipt = json.loads((dest / 'receipt.json').read_text())
        if all(digest(dest / x['file']) == x['sha256'] for x in receipt['artifacts']):
            return receipt
        raise ValueError('Existing output failed checksum; use a new output directory')
    check = json.loads((pool / 'checks' / f'{uid}.json').read_text())
    if check.get('status') != 'passed':
        raise ValueError('Source sequence has not passed the expansion audit')
    source = pool / 'staging' / uid
    with np.load(source / 'sequence.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    t = arrays['timestamps']
    count = len(t)
    if not np.array_equal(arrays['frame_index'], np.arange(count)):
        raise ValueError('Source frame index differs from native video')
    by_role = {a['role']: a for a in record['planned_assets']}
    paths = {role: pool / 'assets' / a['source_path'] for role, a in by_role.items()}
    for role in ['episodes', 'info', 'data', 'annotation']:
        if digest(paths[role]) != by_role[role]['expected_sha256']:
            raise ValueError(f'{role} hash mismatch')
    rows = pq.read_table(paths['episodes']).to_pylist()
    if len(rows) != 1 or rows[0]['episode_uuid'] != uid or rows[0]['length'] != count:
        raise ValueError('Calibration UUID or frame count mismatch')
    calib = np.asarray(rows[0]['camera_intrinsics/head_left'], dtype=float)
    if calib.shape != (8,) or not np.isfinite(calib).all() or np.any(calib[4:] != 0) or np.any(calib[:2] <= 0):
        raise ValueError('Unsupported or invalid native calibration')
    K = np.array([[calib[0], 0, calib[2]], [0, calib[1], calib[3]], [0, 0, 1]])
    info = json.loads(paths['info'].read_text())
    height, width, _ = info['features']['observation.images.head_left']['shape']
    hands = np.stack([arrays['left_hand_world'], arrays['right_hand_world']], axis=1)
    valid = arrays['hand_valid'].astype(bool)
    if not np.array_equal(valid, np.isfinite(hands).all(axis=(2, 3))):
        raise ValueError('Source hand mask inconsistent with finite positions')
    camera = world_to_camera(hands, arrays['head_left_camera_position'], arrays['head_left_camera_rotation'])
    projection = project_pinhole(camera, dict(zip(['fx', 'fy', 'cx', 'cy'], calib[:4])), width, height)
    c2w = np.repeat(np.eye(4)[None], count, axis=0)
    c2w[:, :3, :3] = quaternion_matrices(arrays['head_left_camera_rotation'])
    c2w[:, :3, 3] = arrays['head_left_camera_position']
    names = ['observation.state.hand_left_world_rotation', 'observation.state.hand_right_world_rotation']
    raw = pq.read_table(paths['data'], columns=names)
    rotation_raw = np.stack([np.asarray(raw[n].to_pylist(), dtype=np.float32).reshape(count, 21, 4) for n in names], axis=1)
    norms = np.linalg.norm(rotation_raw, axis=-1)
    rotation_valid = np.isfinite(rotation_raw).all(axis=-1) & (np.abs(norms-1) < .05) & valid[..., None]
    rotation = np.full_like(rotation_raw, np.nan)
    np.divide(rotation_raw, norms[..., None], out=rotation, where=rotation_valid[..., None])
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'hands.npz.partial').open('wb') as f:
        np.savez_compressed(f, timestamps=t, frame_index=arrays['frame_index'], hands_world=hands,
                            hands_camera=camera.astype(np.float32), hand_uv=projection['points_uv'].astype(np.float32),
                            hand_valid=valid, joint_in_image=projection['inside_image'],
                            joint_rotation_world_wxyz=rotation, joint_rotation_world_wxyz_raw=rotation_raw,
                            joint_rotation_valid=rotation_valid, camera_to_world=c2w, K=K)
    (dest / 'hands.npz.partial').replace(dest / 'hands.npz')
    events = json.loads((source / 'events.json').read_text())
    intervals, candidates = source_intervals(events, t)
    annotations = {'schema_version': 'egodemo_joint_completion_v1', 'sequence_id': uid,
                   'frame_count': count, 'hand_order': ['left', 'right'],
                   'annotation_mode': 'offline_noncausal', 'allowed_as_model_input': False,
                   'training_allowed': False, 'source_semantic_intervals': intervals,
                   'operation_candidates': candidates, 'relations': [], 'transitions': [], 'cooperations': [],
                   'unresolved': ['需视觉绑定稳定物体ID', '需定位各操作边界', '需依据接触证据标注关系与协作'],
                   'source_bindings': {'events_sha256': digest(source / 'events.json'),
                                       'sequence_sha256': digest(source / 'sequence.npz')}}
    save(dest / 'interactions.json', annotations)
    speed = np.linalg.norm(np.diff(hands[:, :, 0], axis=0), axis=-1) / np.diff(t)[:, None]
    metadata = {'sequence_id': uid, 'task_name': record['task_name'], 'frame_count': count,
                'video_path': str(paths['head_left_video']), 'video_sha256': by_role['head_left_video']['expected_sha256'],
                'calibration_path': str(paths['episodes']), 'calibration_sha256': digest(paths['episodes']),
                'calibration_field': 'camera_intrinsics/head_left', 'K': K.tolist(), 'image_size': [width, height],
                'coordinate_convention': 'OpenCV; C2W quaternion wxyz; meters as declared by source',
                'hand_source': 'official annotation, not newly inferred', 'hand_order': ['left', 'right'],
                'hand_valid_counts': valid.sum(axis=0).tolist(),
                'all_joints_in_image_counts': projection['inside_image'].all(axis=-1).sum(axis=0).tolist(),
                'rotation_valid_joint_counts': rotation_valid.sum(axis=(0, 2)).tolist(),
                'wrist_speed_above_3m_s': (speed > 3).sum(axis=0).tolist(),
                'independent_hand_accuracy': None, 'object_pose_status': 'pending_reconstruction',
                'object_pose_files': [], 'object_pose_accuracy': None, 'training_allowed': False,
                'source_check_sha256': digest(pool / 'checks' / f'{uid}.json'),
                'source_selection_sha256': digest(pool / 'selection_frozen.json'),
                'source_metadata_sha256': digest(source / 'meta.json')}
    save(dest / 'metadata.json', metadata)
    receipt = {'sequence_id': uid, 'status': 'hands_and_semantic_candidates_exported',
               'frame_count': count, 'hand_valid_counts': valid.sum(axis=0).tolist(),
               'source_event_count': len(intervals), 'operation_candidate_count': len(candidates),
               'operation_counts': dict(Counter(x['operation_type'] for x in candidates)),
               'unmapped_source_event_count': sum(not operation_candidates(x['description_original']) for x in intervals),
               'object_pose_sequence_complete': False, 'human_approved': False,
               'artifacts': [{'file': f, 'sha256': digest(dest / f)} for f in ['hands.npz', 'interactions.json', 'metadata.json']]}
    save(dest / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    selection = json.loads((args.pool / 'selection_frozen.json').read_text())
    records = selection['records'][:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / 'operation_vocabulary.json', {'vocabulary': OPERATIONS, 'not_observed_class_count': True})
    receipts, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(export_one, str(args.pool), str(args.output), r): r['source_episode_uuid'] for r in records}
        for future in as_completed(jobs):
            try:
                receipts.append(future.result())
            except Exception as error:
                errors.append({'sequence_id': jobs[future], 'error': str(error)})
            save(args.output / 'progress.json', {'done': len(receipts), 'total': len(records), 'errors': errors})
            print(json.dumps({'done': len(receipts), 'total': len(records), 'errors': len(errors)}), flush=True)
    counts = Counter()
    for receipt in receipts:
        counts.update(receipt['operation_counts'])
    summary = {'schema_version': 'egodemo_joint_completion_v1', 'source_sequences': len(records),
               'hand_sequences_exported': len(receipts), 'frames_exported': sum(x['frame_count'] for x in receipts),
               'source_semantic_intervals': sum(x['source_event_count'] for x in receipts),
               'operation_candidates': sum(counts.values()), 'observed_candidate_operation_types': len(counts),
               'operation_counts': dict(counts), 'unmapped_source_events': sum(x['unmapped_source_event_count'] for x in receipts),
               'object_pose_sequences_complete': 0, 'human_approved_interaction_sequences': 0,
               'training_allowed': False, 'errors': errors,
               'definition': 'Operation types are text-supported pending candidates, not verified interaction classes.'}
    save(args.output / 'summary.json', summary)
    with (args.output / '操作候选统计.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['操作编码', '操作中文名称', '原文候选数量', '审核状态'])
        for key, count in counts.most_common():
            writer.writerow([key, OPERATIONS[key], count, '待视觉审核'])
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
