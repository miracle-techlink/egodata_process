"""Loopback-only review of private EgoDemo candidates, with versioned drafts."""
import argparse
import base64
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Lock
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

from src.dataset.complete_egodemo import OPERATIONS, digest, save
from src.annotation.quality_bindings import bind_quality_to_tracks
from src.annotation.refine_semantics import REFINED_OPERATIONS,semantic_directory

KINDS = {'rigid': '刚体', 'deformable': '非刚体', 'articulated': '铰接 / 分部件', 'unknown': '待确认'}
TYPES = {
    'operation': REFINED_OPERATIONS,
    'relation': {'contact': '接触', 'grasp_control': '抓持控制', 'co_motion': '共同运动',
                 'supported_by': '受到支撑', 'stabilizes': '稳定工件', 'acts_on': '工具作用于工件'},
    'transition': {'contact_onset': '首次接触', 'grasp_onset': '建立抓持', 'operation_onset': '开始操作',
                   'support_gained': '建立支撑', 'support_lost': '失去支撑', 'release': '最终脱离'},
    'cooperation': {'single_hand': '单手操作', 'bimanual_joint': '双手共同操作',
                    'complementary': '一手操作、一手稳定', 'handover': '手间传递', 'tool_mediated': '工具介导'},
}


def clean_array(value):
    a = np.asarray(value)
    return np.where(np.isfinite(a), a, None).tolist()


def validate_draft(draft, sequence_id, source_sha, count, tracklet_ids, image_size=None):
    if draft.get('sequence_id') != sequence_id or draft.get('source_selection_sha256') != source_sha:
        raise ValueError('来源不匹配')
    if not isinstance(draft.get('entities'), list) or not isinstance(draft.get('interactions'), list):
        raise ValueError('实体或交互格式错误')
    ids, used = set(), set()
    for entity in draft['entities']:
        identity = entity.get('id', '')
        if not re.fullmatch(r'obj_[A-Za-z0-9_]{1,64}', identity) or identity in ids:
            raise ValueError('物体ID重复或格式不符（使用obj_开头）')
        ids.add(identity)
        if not isinstance(entity.get('name_zh'), str) or not entity['name_zh'].strip():
            raise ValueError('物体缺少中文名称')
        if entity.get('rigidity') not in KINDS or entity.get('role') not in {'tool', 'workpiece', 'support', 'other', 'unknown'}:
            raise ValueError('物体类别或角色无效')
        for track in entity.get('tracklets', []):
            if track not in tracklet_ids or track in used:
                raise ValueError('轨迹不存在或被分配给多个物体')
            used.add(track)
        for seed in entity.get('seed_polygons', []):
            frame = seed.get('frame_index')
            points = np.asarray(seed.get('points_native', []), dtype=float)
            if (type(frame) is not int or not 0 <= frame < count or points.ndim != 2 or
                    points.shape[1] != 2 or not 3 <= len(points) <= 200 or not np.isfinite(points).all()):
                raise ValueError('物体种子多边形或帧号无效')
            if np.any(points < 0) or (image_size is not None and np.any(points >= np.asarray(image_size))):
                raise ValueError('种子点超出原图范围')
            if cv2.contourArea(points.astype(np.float32)) < 25:
                raise ValueError('种子区域过小')
            seed.update(review_status='pending', label_valid=False)
    for interaction in draft['interactions']:
        layer = interaction.get('layer')
        if layer not in TYPES or interaction.get('type') not in TYPES[layer]:
            raise ValueError('交互层或类型无效')
        if interaction.get('subject') not in {'left_hand', 'right_hand', 'both_hands', 'object'}:
            raise ValueError('交互主体无效')
        if interaction.get('object_id') not in ids or (interaction.get('secondary_object_id') and interaction['secondary_object_id'] not in ids):
            raise ValueError('请先建立交互涉及的物体ID')
        a, b = interaction.get('start_frame'), interaction.get('end_frame')
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= count:
            raise ValueError('交互帧区间无效')
    # Never trust client-supplied acceptance flags.
    for row in draft['entities'] + draft['interactions']:
        row.update(review_status='pending', label_valid=False)
    draft.update(review_status='pending', training_allowed=False)
    return draft


class ReviewStore:
    def __init__(self, root, draft_root=None):
        self.root = root
        self.draft_root = draft_root or root/'review_drafts'
        self.pool = root.parent / 'egodemo_expansion_300_20261023'
        selection = self.pool / 'selection_frozen.json'
        self.sha = digest(selection)
        self.records = {r['source_episode_uuid']: r for r in json.loads(selection.read_text())['records']}
        titles=root/'task_titles_zh.json'
        self.titles=json.loads(titles.read_text()) if titles.exists() else {}
        names=root/'display_names_zh.json'
        self.display_names=json.loads(names.read_text())['translations'] if names.exists() else {}
        self.lock = Lock()

    def check(self, uid):
        if uid not in self.records:
            raise ValueError('未知采集ID')

    def index(self):
        rows = []
        for uid, r in self.records.items():
            metadata = json.loads((self.root / 'sequences' / uid / 'metadata.json').read_text())
            receipt = self.root / 'dense_tracklets_v4' / uid / 'receipt.json'
            track = json.loads(receipt.read_text()) if receipt.exists() else {}
            joined = self.root/'joint_annotations'/uid/'receipt.json'
            joint = json.loads(joined.read_text()) if joined.exists() else {}
            rows.append({'id': uid, 'task': self.titles.get(r['task_name'],r['task_name']),
                         'task_original':r['task_name'], 'frames': metadata['frame_count'],
                         'stereo_checked': (self.root / 'stereo_checks' / (uid+'.json')).exists(),
                         'observations_ready': (self.root / 'object_observations_v2' / uid / 'receipt.json').exists(),
                         'tracks_ready': receipt.exists(), 'tracklets': track.get('tracklets', 0),
                         'pose_frames': joint.get('pose_frames', 0),
                         'hand_near_pose_frames': joint.get('hand_near_pose_frames', 0),
                         'raw_pose_frames': track.get('object_pose_frames', 0),
                         'proposal_ready': (self.root/'joint_annotations'/uid/'proposal.json').exists(),
                         'draft_saved': (self.draft_root / (uid+'.json')).exists()})
        return {'records': rows, 'source_sha256': self.sha, 'kinds': KINDS, 'types': TYPES}

    def sequence(self, uid):
        self.check(uid)
        meta = json.loads((self.root / 'sequences' / uid / 'metadata.json').read_text())
        events = []
        for path in sorted(semantic_directory(self.root,uid).glob('chunk_*.json')):
            if '.error.' not in path.name:
                events.extend(json.loads(path.read_text())['records'])
        path = self.root / 'dense_tracklets_v4' / uid / 'tracklets.json'
        tracks = json.loads(path.read_text()) if path.exists() else {'records': [], 'relation_candidates': []}
        for row in tracks['records']:
            row['name_zh']=self.display_names.get(row['name_zh'],row['name_zh'])
        quality = self.root/'visual_quality_overrides.json'
        if quality.exists():
            overrides = bind_quality_to_tracks([r for r in json.loads(quality.read_text())['records']
                                               if r['sequence_id']==uid],tracks['records'])
            for row in tracks['records']:
                row['visual_quality'] = overrides.get(row['tracklet_id'])
                if row['visual_quality'] and row['visual_quality'].get('correction'):
                    row.update(row['visual_quality']['correction'])
        draft = self.draft_root / (uid+'.json')
        proposal = self.root/'joint_annotations'/uid/'proposal.json'
        prepared = json.loads(proposal.read_text()) if proposal.exists() else None
        return {'sequence_id': uid, 'task': self.titles.get(self.records[uid]['task_name'],self.records[uid]['task_name']), 'metadata': meta,
                'events': events, 'tracks': tracks,
                'draft': json.loads(draft.read_text()) if draft.exists() else prepared or {
                    'schema_version': 'egodemo_review_draft_v1', 'sequence_id': uid,
                    'source_selection_sha256': self.sha, 'revision': 0, 'reviewer': '',
                    'entities': [], 'interactions': [], 'review_status': 'pending', 'training_allowed': False}}

    def geometry(self, uid):
        paths=[self.root/'dense_tracklets_v4'/uid/'tracklets.json',
               self.root/'joint_annotations'/uid/'objects.npz',self.root/'visual_quality_overrides.json']
        signature=tuple((p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else None for p in paths)
        return self._geometry(uid,signature)

    @lru_cache(maxsize=2)
    def _geometry(self, uid, signature):
        with np.load(self.root / 'sequences' / uid / 'hands.npz') as z:
            hands = {k: z[k] for k in ['hand_uv', 'timestamps', 'K']}
        tracks = []
        folder = self.root / 'dense_tracklets_v4' / uid
        manifest = folder / 'tracklets.json'
        if manifest.exists():
            for row in self.sequence(uid)['tracks']['records']:
                with np.load(folder / (row['tracklet_id']+'.npz')) as z:
                    tracks.append((row, {k: z[k] for k in ['frame_index', 'points_uv', 'point_valid', 'object_pose', 'object_pose_valid']}))
        aligned_path=self.root/'joint_annotations'/uid/'objects.npz'
        aligned=None
        if aligned_path.exists():
            with np.load(aligned_path) as z:aligned={k:z[k] for k in ['object_ids','poses','pose_valid']}
        return hands, tracks, aligned

    def frame(self, uid, index):
        self.check(uid)
        meta = json.loads((self.root / 'sequences' / uid / 'metadata.json').read_text())
        if not 0 <= index < meta['frame_count']:
            raise ValueError('帧号越界')
        cap = cv2.VideoCapture(meta['video_path'])
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError('视频帧解码失败')
        native_h, native_w = frame.shape[:2]
        frame = cv2.resize(frame, (960, round(native_h*960/native_w)))
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise ValueError('图像编码失败')
        hands, tracks, aligned = self.geometry(uid)
        proposed=self.root/'joint_annotations'/uid/'proposal.json'
        entities=json.loads(proposed.read_text())['entities'] if proposed.exists() else []
        track_to_entity={track:e['id'] for e in entities for track in e['tracklets']}
        aligned_ids={str(identity):i for i,identity in enumerate(aligned['object_ids'])} if aligned else {}
        emitted=set()
        objects = []
        for row, z in tracks:
            j = np.searchsorted(z['frame_index'], index)
            if j >= len(z['frame_index']) or z['frame_index'][j] != index:
                continue
            axes = None
            rejected = (row.get('visual_quality') or {}).get('status')=='rejected'
            pose=None
            entity=track_to_entity.get(row['tracklet_id'])
            if aligned is not None:
                slot=aligned_ids.get(entity)
                if slot is not None and entity not in emitted and aligned['pose_valid'][slot,index] and not rejected:
                    pose=aligned['poses'][slot,index];emitted.add(entity)
            elif z['object_pose_valid'][j] and not rejected:
                pose=z['object_pose'][j]
            if pose is not None:
                p = np.vstack([np.zeros(3), np.eye(3)*.05]) @ pose[:3, :3].T + pose[:3, 3]
                p = p @ hands['K'].T
                axes = clean_array(p[:, :2]/p[:, 2:])
            objects.append({'id': row['tracklet_id'], 'name_zh': row['name_zh'], 'axes': axes, 'rejected':rejected,
                            'points': clean_array(z['points_uv'][j, 0, z['point_valid'][j]])})
        return {'jpeg': base64.b64encode(jpg).decode(), 'width': native_w, 'height': native_h,
                'frame': index, 'seconds': float(hands['timestamps'][index]),
                'hands': clean_array(hands['hand_uv'][index]), 'objects': objects}

    def write(self, uid, draft):
        self.check(uid)
        current = self.sequence(uid)
        validated = validate_draft(draft, uid, self.sha, current['metadata']['frame_count'],
                                   {r['tracklet_id'] for r in current['tracks']['records']}, current['metadata']['image_size'])
        proposal=self.root/'joint_annotations'/uid/'proposal.json'
        if proposal.exists():
            prepared=json.loads(proposal.read_text())
            for key in ['proposal_source_sha256','association_version','proposal_quality_sha256','proposal_calibration_sha256','proposal_calibration_recovery_sha256']:
                if validated.get(key)!=prepared.get(key):
                    raise ValueError('后台提案已更新，请先导出当前草稿，再刷新核对物体编号')
        with self.lock:
            path = self.draft_root / (uid+'.json')
            old = json.loads(path.read_text()) if path.exists() else {'revision': 0}
            if validated.get('revision') != old['revision']:
                raise ValueError('草稿已被其他窗口修改，请刷新后合并')
            if path.exists():
                save(self.draft_root / 'history' / uid / f"revision_{old['revision']:06d}.json", old)
            validated.update(revision=old['revision']+1, saved_at=datetime.now(timezone.utc).isoformat())
            save(path, validated)
        return validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18787)
    parser.add_argument('--draft-root', type=Path)
    args = parser.parse_args()
    store = ReviewStore(args.root, args.draft_root)
    cv2.setNumThreads(1)

    class Handler(BaseHTTPRequestHandler):
        def send(self, value, status=200, mime='application/json; charset=utf-8'):
            content = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if parsed.path == '/':
                    return self.send((Path(__file__).parent/'egodemo_review.html').read_bytes(), mime='text/html; charset=utf-8')
                if parsed.path == '/api/index':
                    return self.send(store.index())
                uid = query.get('id', [''])[0]
                if parsed.path == '/api/sequence':
                    store._geometry.cache_clear()
                    return self.send(store.sequence(uid))
                if parsed.path == '/api/frame':
                    return self.send(store.frame(uid, int(query.get('frame', ['0'])[0])))
                self.send({'error': '地址不存在'}, 404)
            except (ValueError, KeyError, OSError) as error:
                self.send({'error': str(error)}, 400)

        def do_POST(self):
            try:
                if urlparse(self.path).path != '/api/draft' or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('无效保存请求')
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    raise ValueError('禁止跨站写入')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('草稿大小不符')
                draft = json.loads(self.rfile.read(length))
                self.send(store.write(draft['sequence_id'], draft))
            except (ValueError, KeyError, OSError) as error:
                self.send({'error': str(error)}, 400)

    print(f'私有审核入口：http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
