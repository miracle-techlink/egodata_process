"""Render offline entity/pose/relation annotations without changing source estimates."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.hand_pose.online_hands import HAND_CONNECTIONS


SCHEMA = 'pose_relation_annotation_v1'
PREDICATES = {
    'approaching': '接近', 'contact': '接触', 'grasp_control': '抓持控制',
    'co_motion': '共同运动', 'supported_by': '受到支撑',
    'stabilizes': '稳定固定', 'acts_on': '作用于',
}
EVENTS = {
    'contact_onset': '开始接触', 'grasp_onset': '建立抓持',
    'motion_onset': '开始操作', 'support_gained': '建立支撑',
    'support_lost': '离开支撑', 'release': '最终脱离',
}
MODES = {
    'single_hand_manipulation': '单手操作', 'bimanual_joint': '双手共同操作',
    'bimanual_complementary': '双手分工配合', 'handover': '交接',
    'tool_mediated': '工具介导操作',
}
STATUSES = {'pending', 'accepted', 'rejected', 'needs_revision'}
SOURCES = {'human', 'assistant_visual_review', 'kinematic_candidate', 'vlm'}
STATUS_NAMES = {'pending': '待审', 'accepted': '已审核',
                'needs_revision': '需修改', 'rejected': '已拒绝'}
COLORS = [(255, 195, 92), (111, 193, 255), (208, 157, 247), (247, 133, 151)]
LEFT, RIGHT = (86, 222, 173), (255, 179, 100)
INK, MUTED, BG, PANEL = (238, 243, 249), (151, 169, 190), (17, 25, 38), (26, 38, 54)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _interval(record, count):
    a, b = record.get('start_frame'), record.get('end_frame')
    if not _int(a) or not _int(b) or not 0 <= a < b <= count:
        raise ValueError('Invalid native-frame interval')
    evidence = record.get('evidence_frames')
    if (not isinstance(evidence, list) or not evidence
            or any(not _int(f) or not a <= f < b for f in evidence)):
        raise ValueError('Evidence must lie within the interval')
    status = record.get('review_status')
    if status not in STATUSES or record.get('label_source') not in SOURCES:
        raise ValueError('Unknown review status or label source')
    if status == 'accepted' and (record['label_source'] != 'human'
            or not record.get('reviewer') or not record.get('reviewed_at')):
        raise ValueError('Accepted annotations need named human review')


def validate_annotations(data, object_ids, frame_count):
    if data.get('schema_version') != SCHEMA or not data.get('sequence_id'):
        raise ValueError('Unknown schema or missing sequence ID')
    if data.get('frame_count') != frame_count:
        raise ValueError('Frame count differs from source')
    if data.get('allowed_as_model_input') is not False:
        raise ValueError('Offline annotations cannot be model input')
    if data.get('annotation_mode') != 'offline_noncausal':
        raise ValueError('Expected offline annotation provenance')
    entities = {}
    for e in data['entities']:
        identity = e.get('entity_id')
        if not isinstance(identity, str) or not identity or identity in entities:
            raise ValueError('Entity IDs must be nonempty and unique')
        if e.get('kind') not in {'hand', 'object', 'context'} or not e.get('display_name'):
            raise ValueError('Invalid entity kind or name')
        if e['kind'] == 'hand' and (identity not in {'left_hand', 'right_hand'}
                or e.get('side') != identity.split('_')[0]):
            raise ValueError('Hand side must match its anatomical entity ID')
        entities[identity] = e
    if {e['entity_id'] for e in entities.values() if e['kind'] == 'hand'} != {'left_hand', 'right_hand'}:
        raise ValueError('Both anatomical hand entities must be declared')
    if {e['entity_id'] for e in entities.values() if e['kind'] == 'object'} != set(object_ids):
        raise ValueError('Object entities must match the original pose IDs')
    identifiers, relations = set(), {}
    for collection, key in [('relations', 'relation_id'), ('transitions', 'event_id'),
                            ('cooperations', 'cooperation_id'), ('task_semantics', 'task_id')]:
        for record in data.get(collection, []):
            identity = record.get(key)
            if not isinstance(identity, str) or not identity or identity in identifiers:
                raise ValueError('Annotation IDs must be unique')
            identifiers.add(identity)
            _interval(record, frame_count)
            if collection in {'relations', 'transitions'}:
                if record.get('source') not in entities or record.get('target') not in entities:
                    raise ValueError('Relation endpoints must be registered entities')
                if record['source'] == record['target']:
                    raise ValueError('Self relations are not supported')
                vocab, field = (PREDICATES, 'predicate') if collection == 'relations' else (EVENTS, 'event_type')
                if record.get(field) not in vocab:
                    raise ValueError('Unknown relation or event type')
                if collection == 'relations':
                    source_kind = entities[record['source']]['kind']
                    target_kind = entities[record['target']]['kind']
                    if record[field] == 'supported_by':
                        valid_pair = source_kind == 'object' and target_kind in {'object', 'context'}
                    elif record[field] == 'acts_on':
                        valid_pair = source_kind in {'hand', 'object'} and target_kind == 'object'
                    else:
                        valid_pair = source_kind == 'hand' and target_kind == 'object'
                    if not valid_pair:
                        raise ValueError('Invalid entity kinds for this relation')
                    relations[identity] = record
            if collection == 'cooperations':
                if record.get('mode') not in MODES:
                    raise ValueError('Unknown cooperation mode')
                participants = record.get('participants', [])
                ids = [p.get('entity_id') for p in participants]
                if (len(ids) < 2 or len(ids) != len(set(ids)) or any(i not in entities for i in ids)
                        or any(not p.get('role') for p in participants)):
                    raise ValueError('Cooperation needs unique entities and explicit roles')
                hands = {i for i in ids if entities[i]['kind'] == 'hand'}
                if record['mode'] in {'bimanual_joint', 'bimanual_complementary', 'handover'} and len(hands) != 2:
                    raise ValueError('Bimanual cooperation requires both hands')
                if record['mode'] == 'single_hand_manipulation' and len(hands) != 1:
                    raise ValueError('Single-hand cooperation needs exactly one hand')
                if not record.get('relation_ids'):
                    raise ValueError('Cooperation needs supporting relations')
                for reference in record['relation_ids']:
                    r = relations.get(reference)
                    if r is None or r['source'] not in ids or r['target'] not in ids:
                        raise ValueError('Cooperation relation is missing or has different participants')
                    if r['end_frame'] <= record['start_frame'] or r['start_frame'] >= record['end_frame']:
                        raise ValueError('Cooperation and relation intervals do not overlap')
                    if record['review_status'] == 'accepted' and r['review_status'] != 'accepted':
                        raise ValueError('Accepted cooperation cannot depend on pending relations')
            if collection == 'task_semantics' and not record.get('description'):
                raise ValueError('Task semantics need a description')
    if any(not _int(f) or not 0 <= f < frame_count for f in data.get('hand_review_frames', [])):
        raise ValueError('Invalid handedness review frame')
    return entities


def frame_graph(data, frame_id, observed):
    if not _int(frame_id) or not 0 <= frame_id < data['frame_count']:
        raise ValueError('Frame outside annotation timeline')
    graph = {'frame_id': frame_id, 'sequence_id': data['sequence_id'],
             'annotation_mode': 'offline_noncausal', 'allowed_as_model_input': False,
             'training_allowed': False, 'observed_entities': observed,
             'handedness_review_required': frame_id in data.get('hand_review_frames', [])}
    for name in ['relations', 'transitions', 'cooperations', 'task_semantics']:
        records = [dict(r) for r in data.get(name, []) if r['start_frame'] <= frame_id < r['end_frame']
                   and r['review_status'] != 'rejected']
        if name == 'relations':
            for record in records:
                record['source_observed'] = bool(observed.get(record['source'], False))
                record['target_observed'] = bool(observed.get(record['target'], False))
        graph[name] = records
    active_relations = {r['relation_id'] for r in graph['relations']}
    graph['cooperations'] = [c for c in graph['cooperations']
                             if active_relations.intersection(c['relation_ids'])]
    return graph


def _dashed(draw, a, b, color, width=3):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b-a))
    if length < 1:
        return
    direction = (b-a) / length
    for start in np.arange(0, length, 14):
        draw.line([tuple(a+direction*start), tuple(a+direction*min(start+8, length))], fill=color, width=width)


def _arrow(draw, a, b, color, dashed=True, card_endpoints=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    distance = float(np.linalg.norm(b-a))
    if distance < 1:
        return
    direction = (b-a)/distance
    if card_endpoints:
        offset = min(124/max(abs(direction[0]), 1e-9), 36/max(abs(direction[1]), 1e-9))
        if 2*offset >= distance:
            return
        a, b = a+direction*offset, b-direction*offset
    if dashed:
        _dashed(draw, a, b, color)
    else:
        draw.line([tuple(a), tuple(b)], fill=color, width=3)
    normal = np.array([-direction[1], direction[0]])
    draw.polygon([tuple(b),tuple(b-direction*10+normal*5),tuple(b-direction*10-normal*5)],fill=color)


class Painter:
    def __init__(self, font):
        self.fonts = {n: ImageFont.truetype(str(font), n) for n in [15, 17, 19, 21, 24, 31]}

    def text(self, draw, xy, text, size=19, color=INK):
        draw.text(xy, str(text), font=self.fonts[size], fill=color)

    def tag(self, draw, xy, text, color):
        font = self.fonts[19]
        box = draw.textbbox((0, 0), text, font=font)
        x, y = xy
        draw.rounded_rectangle((x-7, y-3, x+box[2]+9, y+29), radius=5, fill=(20, 30, 43), outline=color)
        self.text(draw, (x, y), text, color=color)


def _project(points, pose, intrinsics):
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        return None
    pixels = camera @ intrinsics.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    return pixels if np.isfinite(pixels).all() and np.abs(pixels).max() < 1e6 else None


def render_frame(bgr, packed, index, manifest, data, entities, bounds, painter):
    height, width = bgr.shape[:2]
    canvas = Image.new('RGB', (1600, 900), BG)
    canvas.paste(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((960, 720)), (24, 80))
    draw = ImageDraw.Draw(canvas)
    pixel = lambda p: (24 + float(p[0])*960/width, 80 + float(p[1])*720/height)
    painter.text(draw, (24, 15), '手—物协作标注', 31)
    painter.text(draw, (300, 29), 'HO-Cap · 第三人称 · 真实姿态估计 + 关系候选', 19, MUTED)
    painter.text(draw, (1280, 26), f'原生帧 {index:03d} / {data["frame_count"]-1}', 21)
    observed, anchors = {}, {}
    object_ids = packed['object_ids'].tolist()
    intrinsics = np.asarray(manifest['intrinsics'])
    for n, identity in enumerate(object_ids):
        pose = packed['object_poses'][n, index]
        valid = bool(packed['object_valid'][n, index] and np.isfinite(pose).all())
        observed[identity] = valid
        if not valid:
            continue
        low, high = bounds[identity]
        corners = np.array(list(itertools.product(*np.stack([low, high]).T)))
        size = float(min(high-low)*0.65)
        points = np.vstack([corners, np.zeros((1, 3)), np.eye(3)*size])
        projected = _project(points, pose, intrinsics)
        if projected is None:
            continue
        projected = [pixel(p) for p in projected]
        color = COLORS[n % len(COLORS)]
        for a in range(8):
            for bit in (1, 2, 4):
                b = a ^ bit
                if a < b:
                    draw.line([projected[a], projected[b]], fill=color, width=2)
        for axis, c in enumerate([(255, 80, 80), (80, 230, 130), (95, 160, 255)]):
            draw.line([projected[8], projected[9+axis]], fill=c, width=4)
            painter.text(draw, projected[9+axis], 'XYZ'[axis], 15, c)
        anchors[identity] = projected[8]
        location = entities[identity].get('label_anchor', [0.02, 0.08+n*0.12])
        label_xy = (24+960*location[0], 80+720*location[1])
        draw.line([projected[8], label_xy], fill=color, width=1)
        painter.tag(draw, label_xy, entities[identity]['display_name']+'  '+identity, color)
    for side, color in [('left', LEFT), ('right', RIGHT)]:
        identity = side+'_hand'
        values = packed[identity][index]
        valid = bool(np.isfinite(values).all())
        observed[identity] = valid
        if not valid:
            continue
        points = [pixel(p[:2]*[width, height]) for p in values]
        for a, b in HAND_CONNECTIONS:
            draw.line([points[a], points[b]], fill=color, width=3)
        for x, y in points:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        anchors[identity] = tuple(np.mean(np.array(points)[[0,5,9,13,17]], axis=0))
        x, y = points[0]
        label = entities[identity]['display_name']+' '+('H-L' if side=='left' else 'H-R')
        if index in data.get('hand_review_frames', []):
            label += ' · 手别待审'
        painter.tag(draw, (min(735, max(35,x+20)), min(755,max(90,y+28))), label, color)
    for e in entities.values():
        if e['kind'] == 'context':
            observed[e['entity_id']] = False  # A semantic context entity has no measured pose.
            location = e.get('image_anchor')
            if location:
                anchors[e['entity_id']] = (24+960*location[0],80+720*location[1])
    graph = frame_graph(data, index, observed)
    for r in graph['relations']:
        if r['source'] in anchors and r['target'] in anchors:
            color = (86, 222, 173) if r['review_status']=='accepted' else (255, 208, 118)
            a, b = anchors[r['source']], anchors[r['target']]
            _arrow(draw,a,b,color,dashed=r['review_status']!='accepted')
    draw.rounded_rectangle((1008,80,1576,800),radius=14,fill=PANEL)
    painter.text(draw,(1032,97),'实体关系图',24)
    painter.text(draw,(1286,104),'虚线：待审核候选',17,(255,208,118))
    positions={'left_hand':(1150,178),'right_hand':(1434,178)}
    for n, identity in enumerate(object_ids):
        positions[identity]=(1150+284*(n%2),292+98*(n//2))
    for identity,e in entities.items():
        if e['kind']=='context':
            positions[identity]=(1292,492)
    for r in graph['relations']:
        a,b=positions.get(r['source']),positions.get(r['target'])
        if a and b:
            color=LEFT if r['review_status']=='accepted' else (255,208,118)
            dashed=r['review_status']!='accepted'
            if entities[r['target']]['kind']=='context':
                # Route through the column gutter, never through another object card.
                start=(a[0]+(124 if a[0]<b[0] else -124),a[1])
                elbow=(b[0],a[1])
                end=(b[0],b[1]-36)
                if dashed:
                    _dashed(draw,start,elbow,color)
                else:
                    draw.line([start,elbow],fill=color,width=3)
                _arrow(draw,elbow,end,color,dashed=dashed)
            else:
                _arrow(draw,a,b,color,dashed=dashed,card_endpoints=True)
    for identity,(x,y) in positions.items():
        e=entities[identity]
        color=LEFT if identity=='left_hand' else RIGHT if identity=='right_hand' else MUTED
        if e['kind']=='object':
            color=COLORS[object_ids.index(identity)%4]
        draw.rounded_rectangle((x-123,y-35,x+123,y+35),radius=8,fill=BG,outline=color,width=2)
        painter.text(draw,(x-110,y-28),e['display_name'],21,color)
        state=('有姿态' if observed.get(identity) else '未检出' if e['kind']=='hand'
               else '位姿缺测' if e['kind']=='object' else '环境参照 · 无6D')
        painter.text(draw,(x-110,y+5),identity+' · '+state,15,MUTED)
    painter.text(draw,(1032,540),'当前关系',21)
    if not graph['relations']:
        painter.text(draw,(1032,578),'当前无已标注关系；不等同无交互',19,MUTED)
    for n,r in enumerate(graph['relations'][:4]):
        subject=entities[r['source']]['display_name']
        target=entities[r['target']]['display_name']
        painter.text(draw,(1032,576+n*34),subject+' → '+PREDICATES[r['predicate']]+' → '+target,19)
        painter.text(draw,(1490,576+n*34),STATUS_NAMES[r['review_status']],17,(255,208,118))
    modes='、'.join(MODES[r['mode']] for r in graph['cooperations']) or '尚未标注'
    painter.text(draw,(1032,728),'协作模式：'+modes,21)
    painter.text(draw,(1032,763),'关系按区间保存，可同时存在多条边',17,MUTED)
    x0, x1, y = 24, 984, 839
    draw.rounded_rectangle((x0,y,x1,y+8),radius=4,fill=PANEL)
    for e in data.get('transitions',[]):
        a=x0+(x1-x0)*e['start_frame']/data['frame_count']
        b=x0+(x1-x0)*e['end_frame']/data['frame_count']
        draw.rectangle((a,y,b,y+8),fill=(255,195,92))
    cursor=x0+(x1-x0)*index/max(1,data['frame_count']-1)
    draw.ellipse((cursor-6,y-3,cursor+6,y+11),fill=INK)
    active=' / '.join(EVENTS[e['event_type']]+'（边界待审）' for e in graph['transitions'])
    painter.text(draw,(24,805),active or '状态转换区间见黄色时间条；左右手为被摄者解剖左右',17,MUTED)
    painter.text(draw,(24,864),'姿态为模型估计；关系为视觉候选，未经人工金标准审核',17,MUTED)
    source_time=packed['source_timestamp_s'][index]
    source_known=bool(packed['source_timestamp_valid'][index])
    graph['source_timestamp_s']=float(source_time) if source_known and np.isfinite(source_time) else None
    graph['source_timestamp_valid']=source_known
    painter.text(draw,(1032,826),'源时间：'+(f'{source_time:.3f}s' if source_known else '待确认 · 按原生帧标注'),19)
    if graph['handedness_review_required']:
        painter.text(draw,(1032,860),'本帧手别异常：保留模型输出，需复核',17,(255,179,100))
    else:
        painter.text(draw,(1032,860),'手部相对深度与物体米制坐标分别保留',17,MUTED)
    return cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR),graph


def render(manifest_path, sequence_path, annotations_path, output_dir, font, display_fps=30):
    import trimesh

    if not math.isfinite(display_fps) or display_fps <= 0:
        raise ValueError('Invalid display rate')
    manifest=json.loads(Path(manifest_path).read_text())
    data=json.loads(Path(annotations_path).read_text())
    for name,path in [('scene_manifest',manifest_path),('sequence',sequence_path)]:
        if data['source_bindings'][name]['sha256'] != sha(path):
            raise ValueError('Source binding mismatch: '+name)
    with np.load(sequence_path,allow_pickle=False) as archive:
        packed={k:archive[k] for k in archive.files}
    count=len(packed['frame_ids'])
    if not np.array_equal(packed['frame_ids'],np.arange(count)) or manifest['source_frame_count']!=count:
        raise ValueError('Source frames do not align')
    if [f['frame_id'] for f in manifest['frames']] != list(range(count)):
        raise ValueError('Manifest frames are not native ordered frames')
    ids=packed['object_ids'].tolist()
    if [o['object_id'] for o in manifest['objects']] != ids:
        raise ValueError('Manifest and pose object order differ')
    identity='_'.join(manifest[k] for k in ['subject_id','acquisition_id','camera_id'])
    if data['sequence_id']!=identity:
        raise ValueError('Annotation sequence identity differs from manifest')
    entities=validate_annotations(data,ids,count)
    if len(ids)>4 or sum(e['kind']=='context' for e in entities.values())>1:
        raise ValueError('This review layout supports up to four objects and one context entity')
    bounds={}
    for obj in manifest['objects']:
        if sha(obj['mesh']['path'])!=obj['mesh']['sha256']:
            raise ValueError('Mesh hash changed')
        mesh=trimesh.load(obj['mesh']['path'],process=False,skip_materials=True)
        bounds[obj['object_id']]=np.asarray(mesh.bounds)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    (out/'previews').mkdir()
    painter=Painter(font)
    writer=cv2.VideoWriter(str(out/'overlay_mp4v.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),display_fps,(1600,900))
    if not writer.isOpened():
        raise RuntimeError('Cannot open video encoder')
    previews=set(np.linspace(0,count-1,max(3,int(math.ceil(count*.05)))).astype(int))
    previews.update(data.get('preview_frames',[]))
    try:
        with (out/'frame_graphs.jsonl').open('w',encoding='utf-8') as stream:
            for i,frame in enumerate(manifest['frames']):
                asset=frame['rgb']
                if sha(asset['path'])!=asset['sha256']:
                    raise ValueError('RGB source changed')
                bgr=cv2.imread(asset['path'])
                if bgr is None or list(bgr.shape[:2])!=manifest['image_shape']:
                    raise ValueError('Invalid source image')
                result,graph=render_frame(bgr,packed,i,manifest,data,entities,bounds,painter)
                writer.write(result)
                stream.write(json.dumps(graph,ensure_ascii=False,allow_nan=False)+'\n')
                if i in previews and not cv2.imwrite(str(out/'previews'/f'{i:06d}.jpg'),result):
                    raise RuntimeError('Cannot write preview')
                if (i+1)%100==0:
                    print(f'Rendered {i+1}/{count}',flush=True)
    finally:
        writer.release()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(out/'overlay_mp4v.mp4'),
                    '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',
                    str(out/'overlay.mp4')],check=True)
    cap=cv2.VideoCapture(str(out/'overlay.mp4')); decoded=0
    try:
        while True:
            ok,im=cap.read()
            if not ok:
                break
            if im.shape[:2]!=(900,1600):
                raise ValueError('Output geometry changed')
            decoded+=1
    finally:
        cap.release()
    if decoded!=count:
        raise ValueError('Output video is incomplete')
    json_write(out/'annotations.json',data)
    with (out/'review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['record_type','record_id','start_frame','end_frame','source','target','label','review_status','evidence_frames','notes']
        csv_writer=csv.DictWriter(stream,fields);csv_writer.writeheader()
        for name,key,label in [('relations','relation_id','predicate'),('transitions','event_id','event_type'),
                               ('cooperations','cooperation_id','mode'),('task_semantics','task_id','description')]:
            for r in data.get(name,[]):
                csv_writer.writerow({'record_type':name,'record_id':r[key],
                    **{k:r.get(k,'') for k in ['start_frame','end_frame','source','target','review_status','notes']},
                    'label':r[label],'evidence_frames':json.dumps(r['evidence_frames'])})
    report={'schema_version':SCHEMA,'frames':count,'decoded_video_frames':decoded,'display_fps':display_fps,
            'source_timestamp_known':bool(packed['source_timestamp_valid'].all()),
            'training_allowed':False,'allowed_as_model_input':False,'accuracy':None,
            'relations':len(data['relations']),'cooperations':len(data['cooperations']),
            'hand_review_frames':data.get('hand_review_frames',[]),
            'inputs':{name:{'path':str(Path(path).resolve()),'sha256':sha(path)} for name,path in
                     [('scene_manifest',manifest_path),('sequence',sequence_path),('annotations',annotations_path),('renderer',__file__)]},
            'outputs':[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size}
                       for p in sorted(out.rglob('*')) if p.is_file()]}
    json_write(out/'manifest.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','sequence','annotations','output-dir','font']:
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--display-fps',type=float,default=30)
    a=parser.parse_args()
    r=render(a.manifest,a.sequence,a.annotations,a.output_dir,a.font,a.display_fps)
    print(json.dumps({'frames':r['frames'],'decoded_video_frames':r['decoded_video_frames'],
                      'output_dir':str(a.output_dir),'training_allowed':False}))


if __name__=='__main__':
    main()
