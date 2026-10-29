"""Causal short-horizon 2D tracking from explicitly timed annotation seeds.

Each seed belongs to its current frame. No smoothing, backward interpolation,
future detection or semantic label is used. This is annotated initialization,
not autonomous detection, and does not produce 3D object poses.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda: s.read(4 * 1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_seeds(doc, frame_count):
    if doc['schema_version'] != 'assistant_annotation_v5_visual_seeds':
        raise ValueError('Wrong seed schema')
    w, h = doc['native_image_size']
    seen = set()
    for row in doc['seeds']:
        f, identity = row['frame_index'], row['entity_id']
        if type(f) is not int or not 0 <= f < frame_count or not isinstance(identity, str) or not identity.startswith('visual_'):
            raise ValueError('Invalid native seed or visual identity')
        x0, y0, x1, y1 = row['bbox_xyxy_native']
        if not np.isfinite([x0, y0, x1, y1]).all() or not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
            raise ValueError('Seed box outside native image')
        if (f, identity) in seen:
            raise ValueError('Duplicate seed for the same entity and frame')
        seen.add((f, identity))


def project_hands(hands_world, camera_to_world, K, valid):
    R = camera_to_world[:, :3, :3]
    t = camera_to_world[:, :3, 3]
    camera = np.einsum('fhji,fik->fhjk', hands_world - t[:, None, None], R)
    uvz = camera @ np.asarray(K).T
    good = valid & np.isfinite(camera).all(axis=(2, 3)) & (camera[..., 2] > 1e-5).all(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = uvz[..., :2] / uvz[..., 2:]
    uv[~good] = np.nan
    return uv, good


class PrefixTracker:
    """State contains only previous/current pixels; age limit is fixed upfront."""
    def __init__(self, *, scale=0.5, max_seed_age=30, minimum_points=6):
        if not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError('Tracking scale must be finite and in (0,1]')
        if type(max_seed_age) is not int or max_seed_age < 1:
            raise ValueError('Maximum seed age must be a positive integer')
        if type(minimum_points) is not int or minimum_points < 3:
            raise ValueError('At least three tracked points are required')
        self.scale = scale
        self.max_seed_age = max_seed_age
        self.minimum_points = minimum_points
        self.states = {}
        self.previous = None
        self.last_frame = -1
        self.image_shape = None

    def _points(self, gray, corners, hand_uv):
        mask = np.zeros(gray.shape, np.uint8)
        polygon = np.round(corners).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        for point in np.asarray(hand_uv).reshape(-1, 2):
            if np.isfinite(point).all():
                cv2.circle(mask, tuple(np.round(point * self.scale).astype(int)), 5, 0, -1)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=90, qualityLevel=.015,
            minDistance=4, mask=mask, blockSize=5)
        return points

    def step(self, frame_index, bgr, seeds, hand_uv):
        if type(frame_index) is not int or frame_index != self.last_frame + 1:
            raise ValueError('Tracker requires a consecutive native prefix')
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError('Tracker input must be native uint8 BGR')
        if self.image_shape is not None and bgr.shape != self.image_shape:
            raise ValueError('Native image dimensions changed inside a sequence')
        width, height = bgr.shape[1], bgr.shape[0]
        # Validate before touching any tracker state: a future seed must not
        # move last_frame or silently override another current-frame seed.
        validate_seeds({'schema_version': 'assistant_annotation_v5_visual_seeds',
                        'native_image_size': [width, height], 'seeds': seeds}, frame_index + 1)
        if any(row['frame_index'] != frame_index for row in seeds):
            raise ValueError('Seed belongs to a different current frame')
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None,
                          fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        reset = {row['entity_id']: row for row in seeds}
        result = {}
        for identity, state in list(self.states.items()):
            if identity in reset:
                continue
            points = state['points']
            if (frame_index - state['seed_frame'] >= self.max_seed_age
                    or points is None or len(points) < self.minimum_points):
                del self.states[identity]
                continue
            current, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None,
                winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01))
            if current is None or ok is None:
                del self.states[identity]
                continue
            back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, current, None,
                winSize=(21, 21), maxLevel=3)
            if back is None or reverse_ok is None:
                del self.states[identity]
                continue
            good = ((ok[:, 0] == 1) & (reverse_ok[:, 0] == 1)
                    & (np.linalg.norm(back[:, 0] - points[:, 0], axis=-1) <= 1.5))
            old, new = points[good, 0], current[good, 0]
            if len(old) < self.minimum_points:
                del self.states[identity]
                continue
            cv2.setRNGSeed(17)
            affine, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                ransacReprojThreshold=2.5, maxIters=1000, confidence=.99)
            if affine is None or inliers is None or not np.isfinite(affine).all() or inliers.sum() < self.minimum_points or inliers.mean() < .55:
                del self.states[identity]
                continue
            scale = np.sqrt(np.linalg.det(affine[:, :2]))
            if not .75 <= scale <= 1.33 or np.median(np.linalg.norm(new - old, axis=-1)) > 80:
                del self.states[identity]
                continue
            corners = state['corners'] @ affine[:, :2].T + affine[:, 2]
            box = np.r_[corners.min(0), corners.max(0)] / self.scale
            unclipped_area = np.prod(box[2:] - box[:2])
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if np.prod(box[2:] - box[:2]) < max(64, .5 * unclipped_area):
                del self.states[identity]
                continue
            # Preserve actual point identity until an explicit new seed. Fresh
            # corner detection inside an estimated box would allow background
            # or an occluding hand to replenish a lost object's track.
            retained = new[inliers[:, 0].astype(bool)]
            inside = ((retained[:, 0] >= 0) & (retained[:, 0] < gray.shape[1])
                      & (retained[:, 1] >= 0) & (retained[:, 1] < gray.shape[0]))
            for hand_point in np.asarray(hand_uv).reshape(-1, 2):
                if np.isfinite(hand_point).all():
                    inside &= np.linalg.norm(retained - hand_point * self.scale, axis=-1) > 5
            next_points = retained[inside, None].astype(np.float32)
            if len(next_points) < self.minimum_points:
                del self.states[identity]
                continue
            state.update(corners=corners, points=next_points)
            result[identity] = (box, state['seed_frame'], len(next_points), 2)
        for identity, row in reset.items():
            if row['frame_index'] != frame_index:
                raise ValueError('Seed belongs to a different current frame')
            x0, y0, x1, y1 = row['bbox_xyxy_native']
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32) * self.scale
            points = self._points(gray, corners, hand_uv)
            self.states[identity] = {'corners': corners, 'points': points, 'seed_frame': frame_index}
            result[identity] = (np.array([x0, y0, x1, y1], np.float32), frame_index,
                                0 if points is None else len(points), 1)
        self.previous = gray
        self.last_frame = frame_index
        self.image_shape = bgr.shape
        return result


def track_sequence(base_root, refine_root, root, uid, *, output_dir=None, limit=None):
    base_root, refine_root, root = map(Path, (base_root, refine_root, root))
    folder = root / 'visual_observations' / uid
    seeds_path = folder / 'seeds.json'
    seeds = read(seeds_path)
    source_meta_path = refine_root / 'sequences' / uid / 'metadata.json'
    source_meta = read(source_meta_path)
    count = source_meta['frame_count'] if limit is None else min(limit, source_meta['frame_count'])
    validate_seeds(seeds, source_meta['frame_count'])
    if seeds['sequence_id'] != uid or seeds['native_image_size'] != source_meta['image_size']:
        raise ValueError('Seed acquisition or image dimensions differ')
    for row in seeds['seeds']:
        manifest_path = root / row['evidence_manifest_path']
        if sha(manifest_path) != row['evidence_manifest_sha256']:
            raise ValueError('Seed evidence changed')
        evidence = read(manifest_path)
        if row['frame_index'] not in {v['frame_index'] for v in evidence['frames']}:
            raise ValueError('Seed current frame absent from evidence')
    video_path = Path(source_meta['video_path'])
    if sha(video_path) != source_meta['video_sha256']:
        raise ValueError('Native video source changed')
    sequence_path = base_root / 'sequences' / uid / 'sequence.npz'
    with np.load(sequence_path, allow_pickle=False) as seq:
        uv, hand_valid = project_hands(seq['hands_world'][:count], seq['camera_to_world'][:count],
                                      source_meta['K'], seq['hand_valid'][:count])
        timestamps = seq['timestamps'][:count].copy()
    identity_order = sorted({r['entity_id'] for r in seeds['seeds']})
    slots = {v: i for i, v in enumerate(identity_order)}
    by_frame = defaultdict(list)
    for row in seeds['seeds']:
        if row['frame_index'] < count:
            by_frame[row['frame_index']].append(row)
    shape = (count, len(slots))
    boxes = np.full((*shape, 4), np.nan, np.float32)
    valid = np.zeros(shape, bool)
    rgb_stats = np.full((*shape, 6), np.nan, np.float32)
    seed_at = np.full(shape, -1, np.int32)
    inlier_count = np.zeros(shape, np.int16)
    source_code = np.zeros(shape, np.uint8)
    tracker = PrefixTracker()
    cap = cv2.VideoCapture(str(video_path))
    cv2.setNumThreads(1)
    try:
        for f in range(count):
            ok, bgr = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (f + 1)) > .1:
                raise ValueError('Native decode skipped/reordered a frame')
            result = tracker.step(f, bgr, by_frame[f], uv[f])
            for identity, (box, seed, inliers, source) in result.items():
                j = slots[identity]
                x0, y0, x1, y1 = np.round(box).astype(int)
                crop = bgr[y0:y1, x0:x1, ::-1].reshape(-1, 3).astype(np.float32) / 255
                if len(crop) == 0:
                    continue
                boxes[f, j], valid[f, j] = box, True
                rgb_stats[f, j] = np.r_[crop.mean(0), crop.std(0)]
                seed_at[f, j], inlier_count[f, j], source_code[f, j] = seed, inliers, source
    finally:
        cap.release()
    destination = Path(output_dir) if output_dir else folder
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / 'observations.npz'
    if arrays_path.exists():
        raise FileExistsError('Frozen observations already exist; choose a new output directory')
    np.savez_compressed(arrays_path, frame_index=np.arange(count, dtype=np.int32), timestamps=timestamps,
        object_ids=np.asarray(identity_order), boxes_xyxy_native=boxes, object_valid=valid, rgb_stats=rgb_stats,
        last_seed_frame=seed_at, tracking_inliers=inlier_count, source_code=source_code,
        hand_uv_native=uv.astype(np.float32), hand_projection_valid=hand_valid,
        max_frame_accessed=np.arange(count, dtype=np.int32))
    meta = {'schema_version':'assistant_annotation_v5_visual_observations', 'sequence_id':uid,
        'created_at':datetime.now(timezone.utc).isoformat(), 'native_image_size':source_meta['image_size'],
        'frame_count':count, 'object_ids':identity_order,
        'observation_mode':'causal_replay_with_annotated_initialization',
        'source_code_definition':{'0':'missing','1':'current_frame_visual_seed','2':'past_current_optical_flow_estimate'},
        'counts':{'seed_observations':int((source_code == 1).sum()), 'flow_observations':int((source_code == 2).sum()),
                  'valid_object_frames':int(valid.sum()), 'frames_any_object':int(valid.any(1).sum())},
        'tracking_config':{'scale':.5,'max_seed_age':30,'minimum_points':6,'backward_flow_max_px':1.5,
                           'point_replenishment':False,'point_loss_stops_identity':True},
        'source_bindings':{'seeds_sha256':sha(seeds_path), 'base_sequence_sha256':sha(sequence_path),
            'source_metadata_sha256':sha(source_meta_path),'video_sha256':source_meta['video_sha256'],
            'tracker_sha256':sha(__file__)},
        'observations_sha256':sha(arrays_path), 'visual_tracking_review_status':'pending',
        'whole_object_6d_certified':False,'human_label_valid':False,'training_allowed':False,
        'scope_zh':'当前帧手工可见框初始化；仅过去与当前帧短时光流，固定30帧过期。轨迹须独立叠框查看后才纳入训练子集。框内RGB可能含遮挡手/背景，不是分割特征或整物体6D。'}
    save(destination/'metadata.json',meta)
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root',required=True);p.add_argument('--refine-root',required=True)
    p.add_argument('--root',required=True);p.add_argument('--sequence-id',required=True)
    p.add_argument('--output-dir');p.add_argument('--limit',type=int)
    a=p.parse_args()
    m=track_sequence(a.base_root,a.refine_root,a.root,a.sequence_id,output_dir=a.output_dir,limit=a.limit)
    print(json.dumps({'sequence_id':m['sequence_id'],'counts':m['counts']},ensure_ascii=False))

if __name__=='__main__':main()
