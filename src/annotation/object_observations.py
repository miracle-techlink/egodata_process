"""Private image-grounded object/mask/stereo candidates for later association.

Runs real Grounding DINO and SAM. It does not turn per-image detections into
verified stable identities or invent poses for textureless/deformable objects.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.dataset.complete_egodemo import digest, save
from src.dataset.check_camera_rig import quaternion_matrices
from src.pose6d.stereo_reference import make_reference, relative_camera, stereo_features
from src.annotation.source_semantics import rigidity_candidate


class Segmenter:
    def __init__(self, model_root):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        sys.path.insert(0, str(model_root / 'sam-vendor'))
        from segment_anything import sam_model_registry, SamPredictor
        self.torch = torch
        torch.set_num_threads(2)
        self.processor = AutoProcessor.from_pretrained(str(model_root / 'grounding-dino-tiny'), local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(str(model_root / 'grounding-dino-tiny'), local_files_only=True).to('cuda').eval()
        self.sam = SamPredictor(sam_model_registry['vit_b'](checkpoint=str(model_root / 'sam_vit_b_01ec64.pth')).to('cuda').eval())

    def predict(self, rgb, queries, focus_boxes=()):
        torch = self.torch
        all_boxes, all_scores, labels = [], [], []
        regions = [(0, 0, rgb.shape[1], rgb.shape[0])] + list(focus_boxes)
        for x0,y0,x1,y1 in regions:
            crop = rgb[y0:y1,x0:x1]
            if min(crop.shape[:2]) < 32: continue
            for start in range(0,len(queries),10):
                text = '. '.join(queries[start:start+10]) + '.'
                inputs = self.processor(images=Image.fromarray(crop), text=text, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    outputs = self.detector(**inputs)
                result = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=.35,
                           text_threshold=.25, target_sizes=[crop.shape[:2]])[0]
                boxes=result['boxes'].clone()
                boxes[:,[0,2]]+=x0;boxes[:,[1,3]]+=y0
                all_boxes.append(boxes);all_scores.append(result['scores']);labels.extend(result['text_labels'])
        if not all_boxes:return []
        boxes=torch.cat(all_boxes);all_scores_tensor=torch.cat(all_scores)
        from torchvision.ops import nms
        keep = nms(boxes, all_scores_tensor, .65).cpu().numpy().tolist()[:30]
        if not keep:
            return []
        self.sam.set_image(rgb)
        observations = []
        for i in keep:
            box = boxes[i].detach().cpu().numpy()
            masks, mask_scores, _ = self.sam.predict(box=box, multimask_output=False)
            mask = masks[0]
            if mask.sum() < 64:
                continue
            observations.append({'bbox': box.tolist(), 'detector_score': float(all_scores_tensor[i]),
                                 'sam_score': float(mask_scores[0]), 'label': labels[i], 'mask': mask})
        return observations


def camera_matrices(source, side):
    count = len(source['timestamps'])
    matrices = np.tile(np.eye(4), (count, 1, 1))
    matrices[:, :3, :3] = quaternion_matrices(source[f'head_{side}_camera_rotation'])
    matrices[:, :3, 3] = source[f'head_{side}_camera_position']
    return matrices


def run_sequence(root, pool, record, segmenter, pilot=False):
    uid = record['source_episode_uuid']
    output_name = 'object_pilot_v2' if pilot else 'object_observations_v2'
    destination = root / output_name / uid
    if (destination / 'receipt.json').exists():
        return json.loads((destination / 'receipt.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    native = pool / 'staging' / uid
    source = np.load(native / 'sequence.npz', allow_pickle=False)
    hand_data = np.load(root / 'sequences' / uid / 'hands.npz', allow_pickle=False)
    metadata = json.loads((root / 'sequences' / uid / 'metadata.json').read_text())
    events = json.loads((native / 'events.json').read_text())['subtask_annotations']
    source_assets = {a['role']: pool / 'assets' / a['source_path'] for a in record['planned_assets']}
    calibration = json.loads((root / 'stereo_calibration.json').read_text())[uid]
    if digest(source_assets['episodes']) != calibration['source_sha256'] or calibration['frame_count'] != len(source['timestamps']):
        raise ValueError('Native stereo calibration binding mismatch')
    K = {}
    for side in ['left', 'right']:
        values = np.array(calibration[side])
        if values.shape != (8,) or np.any(values[4:] != 0):
            raise ValueError('Unsupported calibration')
        K[side] = np.array([[values[0], 0, values[2]], [0, values[1], values[3]], [0, 0, 1.]])
    c2w = {s: camera_matrices(source, s) for s in ['left', 'right']}
    right_video = root / 'stereo' / (uid + '_right.mp4')
    stereo_record = next(r for r in json.loads((root / 'stereo_plan.json').read_text())['blobs'] if r['sequence_id'] == uid)
    if not right_video.exists() or digest(right_video) != stereo_record['sha256']:
        raise ValueError('Right video payload absent or unverified')
    left_cap, right_cap = cv2.VideoCapture(metadata['video_path']), cv2.VideoCapture(str(right_video))
    semantic_rows = []
    for path in sorted((root / 'semantic_drafts' / uid).glob('chunk_*.json')):
        if '.error.' not in path.name:
            semantic_rows.extend(json.loads(path.read_text())['records'])
    if not pilot and len(semantic_rows) != len(events):
        raise ValueError('Source semantic draft incomplete; retry after translation')
    queries = []
    translations = {}
    for row in semantic_rows:
        for mention in row['object_mentions']:
            phrase = mention['source_phrase'].lower()
            if phrase in {'table', 'desk', 'shelf', 'drawer', 'hand', 'hands'}:
                continue
            if phrase not in queries:
                queries.append(phrase)
            translations[phrase] = mention['name_zh']
    if pilot:
        task = record['task_name']
        queries = (['cardboard box', 'barcode scanner'] if 'Scanning' in task else
                   ['marker', 'pen cap'] if 'Markers' in task else ['plastic jar', 'lid'])
        translations.update({'cardboard box':'纸箱','barcode scanner':'扫码器','marker':'白板笔','pen cap':'笔帽','plastic jar':'塑料罐','lid':'盖子'})
        frame_ids = [int(len(source['timestamps'])*r) for r in [.2,.5,.8]]
    else:
        frame_ids = sorted(set(int(np.searchsorted(source['timestamps'], (e['start_time']+e['end_time'])/2)) for e in events))
    if not queries:
        raise ValueError('No grounded object noun phrases')
    # At most 256 text tokens in this detector; per-event vocabulary will be
    # grouped if a sequence contains many distinct qualified noun phrases.
    font = ImageFont.truetype(str(pool / 'tools/NotoSansCJK-Regular.ttc'), 22)
    total, triangulated, frames = 0, 0, []
    for frame_index in frame_ids:
        frames_bgr = []
        for cap in [left_cap, right_cap]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok: raise ValueError('Native video frame decode failed')
            frames_bgr.append(frame)
        rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)
        focus_boxes = []
        hand_mask = np.zeros(rgb.shape[:2], np.uint8)
        equipment_mask = np.zeros(rgb.shape[:2], np.uint8)
        for h in range(2):
            uv = hand_data['hand_uv'][frame_index,h]
            uv = uv[np.isfinite(uv).all(axis=1)]
            if len(uv)<3:continue
            a=uv.min(axis=0)-180;b=uv.max(axis=0)+180
            x0,y0=np.maximum(a,0).astype(int);x1,y1=np.minimum(b,[rgb.shape[1],rgb.shape[0]]).astype(int)
            if x1>x0 and y1>y0:focus_boxes.append((x0,y0,x1,y1))
            cv2.fillConvexPoly(hand_mask,cv2.convexHull(uv.astype(np.int32)),255)
            original_uv=hand_data['hand_uv'][frame_index,h]
            if np.isfinite(original_uv[[0,9]]).all():
                palm=original_uv[9]-original_uv[0]
                centre=original_uv[0]-.8*palm
                radius=int(np.clip(np.linalg.norm(palm)*.65,25,130))
                cv2.circle(equipment_mask,tuple(centre.astype(int)),radius,255,-1)
        hand_mask=cv2.dilate(hand_mask,np.ones((9,9),np.uint8))
        observations = segmenter.predict(rgb, queries, focus_boxes)
        features = stereo_features(*frames_bgr) if observations else None
        frame_dir = destination / f'frame_{frame_index:06d}'
        frame_dir.mkdir(exist_ok=True)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        records = []
        for i, observation in enumerate(observations):
            identity = f'frame{frame_index:06d}_instance{i:02d}'
            mask = observation.pop('mask')
            equipment_overlap=float(np.mean(equipment_mask[mask]>0))
            if equipment_overlap>.35:
                # Instrumentation worn proximal to the wrist is context, not a
                # task target. Its area remains in raw images for human review.
                continue
            cv2.imwrite(str(frame_dir / (identity + '_mask.png')), mask.astype(np.uint8)*255)
            label = observation['label']
            zh = translations.get(label, ' / '.join(v for k,v in translations.items() if k in label) or label)
            row = {**observation, 'observation_id':identity, 'stable_object_id':None, 'name_zh':zh,
                   'mask_file':identity+'_mask.png', 'mask_area':int(mask.sum()),
                   'rigidity_candidate':rigidity_candidate(label, record['task_name']),
                   'review_status':'pending', 'triangulation_status':'not_attempted', 'reference_points':0}
            row['wrist_equipment_overlap_fraction']=equipment_overlap
            try:
                surface_mask = mask & (hand_mask == 0) & (equipment_mask == 0)
                row['hand_excluded_pixels'] = int(mask.sum()-surface_mask.sum())
                ref = make_reference(*frames_bgr, None, K['left'], K['right'],
                        relative_camera(c2w['left'][frame_index],c2w['right'][frame_index]), pixel_mask=surface_mask, features=features)
                np.savez_compressed(frame_dir / (identity + '_reference.npz'),
                                    **{k:v for k,v in ref.items() if isinstance(v,np.ndarray)})
                row.update(triangulation_status='estimated_reference', reference_points=len(ref['points_object']),
                           reference_origin_camera=ref['origin_camera'].tolist(),
                           reference_reprojection_median_px=float(np.median(ref['stereo_reprojection_error_px'])))
                triangulated += 1
            except ValueError as error:
                row['triangulation_status'] = 'insufficient_geometry'
                row['geometry_reason'] = str(error)
            color = [(35,220,160),(255,177,50),(80,170,255),(240,100,160)][i%4]
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # PIL draws readable Chinese; contours show actual mask geometry.
            for contour in contours:
                if len(contour)>2:draw.line([tuple(p[0]) for p in contour]+[tuple(contour[0,0])],fill=color,width=3)
            x,y,_,_=row['bbox'];draw.text((max(0,x),max(0,y-28)),f'{i:02d} {zh} 待审',font=font,fill=color)
            records.append(row)
        image.thumbnail((1280,970));image.save(frame_dir/'review.jpg',quality=88)
        save(frame_dir/'observations.json',{'sequence_id':uid,'frame_index':frame_index,
            'timestamp_seconds':float(source['timestamps'][frame_index]),'observations':records,
            'allowed_as_model_input':False,'training_allowed':False})
        total += len(records);frames.append(frame_index)
        print(json.dumps({'sequence':uid,'frame':frame_index,'objects':len(records),'triangulated_total':triangulated}),flush=True)
    left_cap.release();right_cap.release()
    receipt={'sequence_id':uid,'sampled_frames':frames,'object_observations':total,'stereo_references':triangulated,
             'stable_object_identity_review':'pending','full_sequence_pose_complete':False,'training_allowed':False,
             'source_left_sha256':metadata['video_sha256'],'source_right_sha256':stereo_record['sha256']}
    save(destination/'receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    root=args.root;pool=root.parent/'egodemo_expansion_300_20261023'
    records=json.loads((pool/'selection_frozen.json').read_text())['records']
    if not 0<=args.shard_index<args.shards:raise ValueError('Invalid shard index')
    records=records[args.shard_index::args.shards]
    if args.pilot:
        selected={r['sequence_id'] for r in json.loads((root/'pilot_selection.json').read_text())}
        records=[r for r in records if r['source_episode_uuid'] in selected]
    segmenter=Segmenter(root/'models')
    results,errors=[],[]
    output_name='object_pilot_v2' if args.pilot else 'object_observations_v2'
    suffix='' if args.shards==1 else f'_shard_{args.shard_index:02d}'
    pending=list(records)
    while pending:
        waiting=[]
        for record in pending:
            try:results.append(run_sequence(root,pool,record,segmenter,args.pilot))
            except Exception as error:
                message=str(error)
                if args.watch and ('Right video payload absent' in message or 'semantic draft incomplete' in message):
                    waiting.append(record)
                else:
                    errors.append({'sequence_id':record['source_episode_uuid'],'error':message})
                    print(json.dumps(errors[-1]),flush=True)
            save(root/output_name/f'progress{suffix}.json',{'done':len(results),'total':len(records),'errors':errors,'waiting':len(waiting)})
        pending=waiting
        if pending:time.sleep(20)
    save(root/output_name/f'summary{suffix}.json',{'processed_sequences':len(results),'results':results,'errors':errors,
         'full_sequence_pose_complete':0,'training_allowed':False})


if __name__=='__main__':main()
