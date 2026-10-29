"""Recompute native-frame observables; proximity is never a contact label."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def velocity(values, valid, timestamps):
    """Never differentiate across missing frames or timestamp gaps."""
    values = np.asarray(values)
    out = np.full(values.shape, np.nan, np.float32)
    dt = np.diff(timestamps)
    if len(dt):
        valid_pair = valid[1:] & valid[:-1]
        valid_pair &= (dt <= 1.5 * np.median(dt)).reshape((-1,) + (1,) * (valid.ndim - 1))
        raw = np.diff(values, axis=0) / dt.reshape((-1,) + (1,) * (values.ndim - 1))
        out[1:][valid_pair] = raw[valid_pair]
    return out


def native_surface_distance(hands, hand_valid, surface_frames, surface_objects, xyz,
                            object_count, excluded=(), min_points=8, block_size=32768):
    """Minimum distance to observed surface, independently per native object."""
    frames = len(hands)
    counts = np.zeros((frames, object_count), np.int32)
    distances = np.full((frames * object_count, 2), np.inf, np.float32)
    excluded = np.asarray(list(excluded), dtype=np.int64)
    if len(xyz):
        if (np.any(surface_frames < 0) or np.any(surface_frames >= frames)
                or np.any(surface_objects < 0) or np.any(surface_objects >= object_count)
                or not np.isfinite(xyz).all()):
            raise ValueError('Invalid source surface observation')
        np.add.at(counts, (surface_frames, surface_objects), 1)
        for begin in range(0, len(xyz), block_size):
            stop = min(begin + block_size, len(xyz))
            f, o = surface_frames[begin:stop], surface_objects[begin:stop]
            delta = xyz[begin:stop, None, None] - hands[f]
            squared = np.einsum('bhjd,bhjd->bhj', delta, delta)
            squared[~np.isfinite(squared)] = np.inf
            nearest = np.sqrt(squared.min(axis=2))
            flat_index = f * object_count + o
            for hand in range(2):
                np.minimum.at(distances[:, hand], flat_index, nearest[:, hand])
    distances = distances.reshape(frames, object_count, 2)
    known = (counts[:, :, None] >= min_points) & hand_valid[:, None, :] & np.isfinite(distances)
    if len(excluded):
        known[:, excluded] = False
        counts[:, excluded] = 0
    distances[~known] = np.nan
    return distances, known, counts


def sparse_frame_records(records, frame_count):
    pairs = sorted({(int(f), i) for i, r in enumerate(records) for f in r.get('evidence_frames', [])})
    if any(not 0 <= f < frame_count for f, _ in pairs):
        raise ValueError('Silver record lies outside native sequence')
    ptr = np.zeros(frame_count + 1, np.int64)
    for f, _ in pairs:
        ptr[f + 1] += 1
    return np.cumsum(ptr), np.asarray([i for _, i in pairs], np.int32)


def proximity_boundaries(distance, observed, timestamps, object_ids, threshold=.03):
    result = []
    near = distance < threshold
    dt = np.diff(timestamps)
    contiguous = dt <= 1.5 * np.median(dt) if len(dt) else np.empty(0, bool)
    for obj, identity in enumerate(object_ids):
        for hand in range(2):
            valid_pair = observed[1:, obj, hand] & observed[:-1, obj, hand] & contiguous
            changed = valid_pair & (near[1:, obj, hand] != near[:-1, obj, hand])
            for frame in np.flatnonzero(changed) + 1:
                result.append({'subject': ['left_hand', 'right_hand'][hand], 'object_id': str(identity),
                               'candidate_type': 'proximity_enter' if near[frame, obj, hand] else 'proximity_exit',
                               'candidate_boundary_frame': int(frame),
                               'suggested_review_range': [max(0, int(frame)-12), min(len(timestamps), int(frame)+13)],
                               'distance_before_m': float(distance[frame-1, obj, hand]),
                               'distance_after_m': float(distance[frame, obj, hand]),
                               'annotation_source': 'geometry_rule', 'review_status': 'pending',
                               'silver_label_valid': False, 'human_label_valid': False,
                               'reason_zh': '仅三维接近阈值变化，须连续看图判定，不是接触或释放事件。'})
    return result


def build_one(base_root, v3_root, output_root, entry, annotation_entry):
    base_root, v3_root, output_root = map(Path, (base_root, v3_root, output_root))
    uid = entry['sequence_id']
    source = base_root / entry['sequence_path']
    apath = v3_root / annotation_entry['annotation_path']
    if sha(source) != entry['sha256']['sequence'] or sha(apath) != annotation_entry['sha256']:
        raise ValueError('Frozen source hash changed')
    annotation = read(apath)
    with np.load(source, allow_pickle=False) as z:
        times = z['timestamps']
        frame_index = z['frame_index']
        hands, hand_valid = z['hands_world'], z['hand_valid'].copy()
        ids = z['object_ids']
        pose, pose_valid = z['object_pose_world'], z['object_pose_valid'].copy()
        if not np.array_equal(frame_index, np.arange(entry['frame_count'])) or not np.all(np.diff(times)>0):
            raise ValueError('Missing or unordered native frame')
        excluded_ids = {e['object_id'] for e in annotation['geometry_exclusions']}
        excluded = [i for i, identity in enumerate(ids) if identity in excluded_ids]
        pose_valid[:, excluded] = False
        hand_valid &= np.isfinite(hands).all(axis=(-1,-2))
        dist, known, count = native_surface_distance(hands, hand_valid,
            z['surface_frame_index'], z['surface_object_index'], z['surface_world'], len(ids), excluded)
        hv = velocity(hands[:,:,0], hand_valid, times)
        ov = velocity(pose[:,:,:3,3], pose_valid, times)
        relative_speed = np.linalg.norm(ov[:,:,None] - hv[:,None], axis=-1).astype(np.float32)
        ptr, record_indices = sparse_frame_records(annotation['silver_event_records'], len(times))
        arrays = dict(frame_index=frame_index, timestamps=times, hand_valid=hand_valid,
                      hand_speed_mps=np.linalg.norm(hv, axis=-1).astype(np.float32), object_ids=ids,
                      pose_valid=pose_valid, surface_point_count=count, nearest_hand_surface_m=dist,
                      distance_observed=known, near_surface_candidate=known & (dist < .03),
                      object_speed_mps=np.linalg.norm(ov,axis=-1).astype(np.float32),
                      relative_speed_mps=relative_speed, v3_silver_frame_ptr=ptr,
                      v3_silver_record_index=record_indices)
    destination = output_root / 'dense_sequences' / uid
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'observables.npz', **arrays)
    candidates = proximity_boundaries(dist, known, times, ids)
    save(destination / 'boundary_candidates.json', {'records': candidates, 'training_allowed': False})
    counts = dict(frames_indexed=len(times), frames_with_any_pose=int(pose_valid.any(axis=1).sum()),
                  frames_with_any_surface=int((count>0).any(axis=1).sum()),
                  frame_object_hand_distances=int(known.sum()),
                  proximity_candidate_values=int(arrays['near_surface_candidate'].sum()),
                  candidate_boundary_count=len(candidates), v3_labeled_native_frames=int((np.diff(ptr)>0).sum()))
    metadata = {'schema_version': 'assistant_annotation_v4_dense_observables', 'sequence_id': uid,
                'split': entry['split'], 'source_bindings': {'base_sequence_sha256': sha(source),
                'v3_annotation_sha256': sha(apath), 'builder_sha256': sha(__file__)},
                'geometry_excluded_object_ids': sorted(excluded_ids),
                'thresholds': {'minimum_surface_points': 8, 'proximity_m': .03},
                'counts': counts, 'dense_index_complete': True, 'continuous_visual_review_complete': False,
                'full_annotation_complete': False, 'training_allowed': False,
                'annotation_mode': 'offline_noncausal',
                'scope_zh': '真实原生观测的逐帧数值与边界候选；不把距离或数值索引齐全当物理关系确认。',
                'sha256': {'observables': sha(destination/'observables.npz'),
                           'boundary_candidates': sha(destination/'boundary_candidates.json')}}
    save(destination/'metadata.json', metadata)
    return {'sequence_id': uid, **counts, 'metadata_sha256': sha(destination/'metadata.json')}


def build(base_root, v3_root, output_root, workers=4):
    base_root, v3_root, output_root = map(Path, (base_root, v3_root, output_root))
    base, overlay = read(base_root/'dataset_index.json'), read(v3_root/'mamba_overlay_index.json')
    if sha(base_root/'dataset_index.json') != overlay['base_dataset_index_sha256']:
        raise ValueError('V3 overlay belongs to different source')
    annotations = {r['sequence_id']: r for r in overlay['sequences']}
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_one, base_root, v3_root, output_root, entry, annotations[entry['sequence_id']]): entry['sequence_id']
                   for entry in base['sequences']}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%25==0:
                print(json.dumps({'processed':len(rows),'total':len(futures)}),flush=True)
    totals = Counter()
    for row in rows:
        totals.update({key:value for key,value in row.items() if isinstance(value,int)})
    report = {'schema_version':'assistant_annotation_v4_dense_observables_delivery',
              'sequences':len(rows), 'totals':dict(totals), 'records':sorted(rows,key=lambda x:x['sequence_id']),
              'base_index_sha256':sha(base_root/'dataset_index.json'),
              'v3_overlay_sha256':sha(v3_root/'mamba_overlay_index.json'),
              'training_allowed':False,'full_annotation_complete':False}
    save(output_root/'dense_observables_report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root',required=True);p.add_argument('--v3-root',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();r=build(a.base_root,a.v3_root,a.output_root,a.workers)
    print(json.dumps({'sequences':r['sequences'],'totals':r['totals']}))


if __name__=='__main__':
    main()
