"""Build the Chinese, browser-local pose/relation review bundle from verified sources."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import trimesh

from src.annotation.pose_relations import (
    EVENTS, MODES, PREDICATES, HAND_CONNECTIONS, _project,
    json_write, sha, validate_annotations,
)


def build(manifest_path, sequence_path, annotation_path, output):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty; keep earlier review bundles')
    m = json.loads(Path(manifest_path).read_text())
    a = json.loads(Path(annotation_path).read_text())
    for key, path in [('scene_manifest', manifest_path), ('sequence', sequence_path)]:
        if a['source_bindings'][key]['sha256'] != sha(path):
            raise ValueError('Source hash mismatch: ' + key)
    with np.load(sequence_path, allow_pickle=False) as z:
        z = dict(z)
    count = len(z['frame_ids'])
    ids = z['object_ids'].tolist()
    validate_annotations(a, ids, count)
    if (not np.array_equal(z['frame_ids'], np.arange(count)) or
            [f['frame_id'] for f in m['frames']] != list(range(count)) or
            [o['object_id'] for o in m['objects']] != ids or
            a['sequence_id'] != '_'.join(m[k] for k in ['subject_id','acquisition_id','camera_id'])):
        raise ValueError('Source alignment mismatch')
    out.mkdir(parents=True, exist_ok=True)
    (out/'frames').mkdir()
    points = {}
    for obj in m['objects']:
        if sha(obj['mesh']['path']) != obj['mesh']['sha256']:
            raise ValueError('Mesh changed')
        lo, hi = trimesh.load(obj['mesh']['path'], process=False, skip_materials=True).bounds
        points[obj['object_id']] = np.vstack([
            np.array(list(itertools.product(*np.stack([lo,hi]).T))),
            np.zeros((1,3)), np.eye(3)*float(min(hi-lo)*.65)])
    frames = []
    h,w = m['image_shape']
    for i,f in enumerate(m['frames']):
        source = Path(f['rgb']['path'])
        if sha(source) != f['rgb']['sha256']:
            raise ValueError('RGB changed')
        image = cv2.imread(str(source))
        if image is None or list(image.shape[:2]) != [h,w]:
            raise ValueError('Invalid frame')
        shutil.copyfile(source, out/'frames'/f'{i:06d}.jpg')
        pose = {'hands': {}, 'objects': {}}
        for side in ['left_hand','right_hand']:
            v=z[side][i]
            pose['hands'][side] = np.round(v[:,:2]*[w,h],3).tolist() if np.isfinite(v).all() else None
        for n, identity in enumerate(ids):
            p = _project(points[identity], z['object_poses'][n,i], np.asarray(m['intrinsics'])) if z['object_valid'][n,i] else None
            pose['objects'][identity] = np.round(p,3).tolist() if p is not None else None
        frames.append(pose)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-framerate','30','-i',str(out/'frames/%06d.jpg'),
                    '-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],check=True)
    cap = cv2.VideoCapture(str(out/'video.mp4'))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != count:
        raise ValueError('Video frame count mismatch')
    json_write(out/'data.json', {'version':'pose_relation_review_web_v1', 'annotation_sha256':sha(annotation_path),
        'annotations':a, 'frame_count':count, 'display_fps':30, 'width':w, 'height':h,
        'predicates':PREDICATES, 'events':EVENTS, 'modes':MODES, 'hand_connections':list(HAND_CONNECTIONS),
        'poses':frames})
    for source in (Path(__file__).parent/'web_assets').iterdir():
        if source.is_file():
            shutil.copyfile(source,out/source.name)
    json_write(out/'bundle-manifest.json', {'frames':count,'decoded_frames':decoded,
        'annotation_sha256':sha(annotation_path),'sequence_sha256':sha(sequence_path),
        'builder_sha256':sha(__file__),'outputs':[{'path':str(p.relative_to(out)), 'sha256':sha(p)}
            for p in sorted(out.rglob('*')) if p.is_file()]})
    return {'frames':count,'decoded_frames':decoded,'output':str(out)}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['manifest','sequence','annotations','output']:
        p.add_argument('--'+k,required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(build(args.manifest,args.sequence,args.annotations,args.output)))
