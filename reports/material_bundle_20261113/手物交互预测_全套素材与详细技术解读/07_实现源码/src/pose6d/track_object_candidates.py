"""Dense native-frame stereo tracklets from reviewed-or-pending surface seeds.

Outputs remain offline candidates. A tracklet ID is not a resolved whole-video
object identity. Nonrigid points never become whole-object rigid poses.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
from pathlib import Path
import time
import cv2
import numpy as np

from src.annotation.object_observations import camera_matrices
from src.dataset.complete_egodemo import save,digest
from src.pose6d.nonrigid_tracks import follow_points,stereo_points
from src.pose6d.stereo_reference import relative_camera
from src.pose6d.rigid_fit import robust_rigid_fit
from src.pose6d.observation_exclusions import capture_exclusion_mask


def distinct_points(points,limit=64):
    selected=[]
    for i,p in enumerate(points):
        if not selected or np.min(np.linalg.norm(points[selected]-p,axis=1))>=3:
            selected.append(i)
            if len(selected)>=limit:break
    return np.asarray(selected,dtype=int)


def runs(mask,min_length=3):
    changes=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)) if b-a>=min_length]


def track_sequence(root_string,uid):
    cv2.setNumThreads(2)
    root=Path(root_string);pool=root.parent/'egodemo_expansion_300_20261023'
    dest=root/'dense_tracklets_v4'/uid
    if (dest/'receipt.json').exists():return json.loads((dest/'receipt.json').read_text())
    observations=root/'object_observations_v2'/uid
    if not (observations/'receipt.json').exists():raise ValueError('Object observations not ready')
    source=np.load(pool/'staging'/uid/'sequence.npz')
    timing=json.loads((root/'stereo_checks'/(uid+'.json')).read_text())
    if timing.get('status')!='passed' or timing['frame_count']!=len(source['timestamps']):
        raise ValueError('Stereo native timeline not verified')
    hands=np.stack([source['left_hand_world'],source['right_hand_world']],axis=1)
    metadata=json.loads((root/'sequences'/uid/'metadata.json').read_text())
    hand_projection=np.load(root/'sequences'/uid/'hands.npz')['hand_uv']
    seed_video=cv2.VideoCapture(metadata['video_path'])
    calib=json.loads((root/'stereo_calibration.json').read_text())[uid]
    K={s:np.array([[v[0],0,v[2]],[0,v[1],v[3]],[0,0,1.]]) for s,v in calib.items() if s in ['left','right']}
    c2w={s:camera_matrices(source,s) for s in ['left','right']}
    seeds=[]
    for path in sorted(observations.glob('frame_*/observations.json')):
        data=json.loads(path.read_text())
        seed_video.set(1,data['frame_index']);ok,seed_image=seed_video.read()
        if not ok:raise ValueError('Seed image decode failed')
        excluded=capture_exclusion_mask(seed_image,hand_projection[data['frame_index']])
        for observation in data['observations']:
            if re.fullmatch(r'(?:the )?(?:stack|pile|bunch|group|items|objects|materials)',observation['label'],re.I):
                continue
            refpath=path.parent/(observation['observation_id']+'_reference.npz')
            if observation['triangulation_status']!='estimated_reference' or not refpath.exists():continue
            with np.load(refpath) as ref:
                xy=np.rint(ref['left_uv']).astype(int)
                xy[:,0]=np.clip(xy[:,0],0,excluded.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,excluded.shape[0]-1)
                allowed=np.flatnonzero(excluded[xy[:,1],xy[:,0]]==0)
                ids=allowed[distinct_points(ref['left_uv'][allowed])]
                if len(ids)<8:continue
                reference_points=ref['points_object'][ids]
                reference_points=reference_points-np.median(reference_points,axis=0)
                seeds.append({'start':data['frame_index'],'name':observation['name_zh'],
                    'rigidity':observation['rigidity_candidate'],'label':observation['label'],'observation_id':observation['observation_id'],
                    'reference_sha256':digest(refpath),'left':ref['left_uv'][ids].astype(np.float32),
                    'right':ref['right_uv'][ids].astype(np.float32),'points_object':reference_points,
                    'frames':[],'xyz':[],'world':[],'uv':[],'valid':[],'surface_pose':[], 'fit':[], 'near':[]})
    seed_video.release()
    dest.mkdir(parents=True,exist_ok=True)
    if not seeds:
        receipt={'sequence_id':uid,'status':'no_observable_stereo_surface','tracklets':0,'point_frames':0,
                 'object_pose_frames':0,'full_annotation_complete':False,'training_allowed':False}
        save(dest/'receipt.json',receipt);return receipt
    seeds.sort(key=lambda s:s['start'])
    right_path=root/'stereo'/(uid+'_right.mp4')
    caps=[cv2.VideoCapture(metadata['video_path']),cv2.VideoCapture(str(right_path))]
    native_size=tuple(metadata['image_size'])
    image_size=(native_size[0]//2,native_size[1]//2)
    scale=np.array([image_size[0]/native_size[0],image_size[1]/native_size[1]],np.float32)
    for s in seeds:
        s['points']=[s['left']*scale,s['right']*scale]
        s['alive']=[np.ones(len(s['left']),bool),np.ones(len(s['left']),bool)]
        s['lost_stereo']=np.zeros(len(s['left']),np.int16)
    active=[];next_seed=0;previous=None
    index=seeds[0]['start'];last=len(source['timestamps'])
    for cap in caps:cap.set(1,index)
    while index<last:
        if not active and next_seed<len(seeds) and index<seeds[next_seed]['start']:
            index=seeds[next_seed]['start']
            for cap in caps:cap.set(1,index)
            previous=None
        if not active and next_seed>=len(seeds):break
        images=[]
        for cap in caps:
            ok,bgr=cap.read()
            if not ok:raise ValueError('Native stereo decode failed')
            images.append(cv2.cvtColor(cv2.resize(bgr,image_size),cv2.COLOR_BGR2GRAY))
        if active and previous is not None:
            for side in range(2):
                lengths=[len(s['points'][side]) for s in active]
                points=np.concatenate([s['points'][side] for s in active])
                alive=np.concatenate([s['alive'][side] for s in active])
                points,alive=follow_points(previous[side],images[side],points,alive,fb_limit=.9)
                offset=0
                for s,length in zip(active,lengths):
                    s['points'][side]=points[offset:offset+length];s['alive'][side]=alive[offset:offset+length];offset+=length
        while next_seed<len(seeds) and seeds[next_seed]['start']==index:
            active.append(seeds[next_seed]);next_seed+=1
        rel=relative_camera(c2w['left'][index],c2w['right'][index])
        retained=[]
        for s in active:
            uv=[p/scale for p in s['points']]
            xyz,valid,_=stereo_points(*uv,K['left'],K['right'],rel)
            s['lost_stereo']=np.where(valid,0,s['lost_stereo']+1)
            for side in range(2):s['alive'][side]&=s['lost_stereo']<5
            world=xyz@c2w['left'][index,:3,:3].T+c2w['left'][index,:3,3]
            pose,fit=robust_rigid_fit(s['points_object'],xyz,trials=20)
            near=[]
            for hand in hands[index]:
                observed=hand[np.isfinite(hand).all(axis=1)]
                near.append(float(np.min(np.linalg.norm(observed[:,None]-world[valid][None],axis=-1))) if len(observed) and valid.any() else np.nan)
            s['frames'].append(index);s['xyz'].append(xyz.astype(np.float32));s['world'].append(world.astype(np.float32))
            s['uv'].append(np.stack(uv).astype(np.float32));s['valid'].append(valid)
            s['surface_pose'].append(pose.astype(np.float32) if pose is not None else np.full((4,4),np.nan,np.float32))
            s['fit'].append(fit);s['near'].append(near)
            if (s['alive'][0]&s['alive'][1]).sum()>=4:retained.append(s)
        active=retained;previous=images;index+=1
    for cap in caps:cap.release()
    track_records=[];relation_candidates=[]
    for number,s in enumerate(seeds):
        if not s['frames']:continue
        identity=f'tracklet_{number:04d}'
        frame_ids=np.array(s['frames']);valid=np.array(s['valid']);surface_pose=np.array(s['surface_pose'])
        near=np.array(s['near'],np.float32)
        pose_valid=np.isfinite(surface_pose).all(axis=(1,2))
        # Local rigid surface consistency does not establish whole-object rigidity.
        # Container masks may contain texture on contents or seen through the
        # wall. Without an opacity review these cannot define the container pose.
        transparent=bool(re.search(r'transparent|glass|\b(cup|jar|bottle)s?\b|透明|玻璃|杯|罐|瓶',s['name']+' '+s['label'],re.I))
        whole_rigid_candidate=s['rigidity']=='rigid' and not transparent
        temporally_supported=np.zeros_like(pose_valid)
        for a,b in runs(pose_valid,min_length=5):temporally_supported[a:b]=True
        whole_valid=temporally_supported if whole_rigid_candidate else np.zeros_like(pose_valid)
        object_pose=surface_pose.copy();object_pose[~whole_valid]=np.nan
        np.savez_compressed(dest/(identity+'.npz'),frame_index=frame_ids,timestamps=source['timestamps'][frame_ids],
            reference_points_object=s['points_object'],
            points_uv=np.array(s['uv']),points_camera=np.array(s['xyz']),points_world=np.array(s['world']),
            point_valid=valid,point_ids=np.arange(valid.shape[1]),local_surface_pose=surface_pose,
            object_pose=object_pose,object_pose_valid=whole_valid,
            nearest_hand_surface_m=near)
        record={'tracklet_id':identity,'stable_object_id':None,'name_zh':s['name'],'rigidity_candidate':s['rigidity'],
                'source_observation_id':s['observation_id'],'source_reference_sha256':s['reference_sha256'],
                'start_frame':int(frame_ids[0]),'end_frame':int(frame_ids[-1]+1),'frame_count':len(frame_ids),
                'surface_point_count':valid.shape[1],'valid_point_frames':int(valid.any(axis=1).sum()),
                'local_surface_pose_frames':int(pose_valid.sum()),
                'object_pose_frames':int(whole_valid.sum()),
                'transparent_surface_ambiguous':transparent,
                'pose_frame_definition':'initial camera axes, initial visible surface median point',
                'annotation_mode':'offline_noncausal','allowed_as_model_input':False,'review_status':'pending',
                'quality':'coverage_only','independent_accuracy':None,'sha256':digest(dest/(identity+'.npz'))}
        track_records.append(record)
        # These are proximity observations, never accepted contact/grasp labels.
        for side in range(2):
            for a,b in runs(np.isfinite(near[:,side])&(near[:,side]<.03)):
                relation_candidates.append({'subject':['left_hand','right_hand'][side], 'target_tracklet':identity,
                    'relation_candidate':'near_visible_surface','display_name_zh':'接近可见表面（接触待核）',
                    'start_frame':int(frame_ids[a]),'end_frame':int(frame_ids[b-1]+1),
                    'review_status':'pending','label_valid':False})
    save(dest/'tracklets.json',{'sequence_id':uid,'records':track_records,'relation_candidates':relation_candidates,
                               'identity_merge_required':True,'training_allowed':False})
    receipt={'sequence_id':uid,'status':'candidate_tracklets_exported','tracklets':len(track_records),
             'point_frames':sum(r['valid_point_frames'] for r in track_records),
             'object_pose_frames':sum(r['object_pose_frames'] for r in track_records),
             'proximity_intervals':len(relation_candidates),'full_annotation_complete':False,'training_allowed':False}
    save(dest/'receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--sequence');parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();root=args.root
    if args.sequence:
        print(json.dumps(track_sequence(str(root),args.sequence)));return
    known=set();errors=[];results=[]
    while True:
        ready=[p.parent.name for p in (root/'object_observations_v2').glob('*/receipt.json')
               if p.parent.name not in known and (root/'stereo_checks'/(p.parent.name+'.json')).exists()]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures={executor.submit(track_sequence,str(root),uid):uid for uid in ready}
            for future in as_completed(futures):
                uid=futures[future];known.add(uid)
                try:results.append(future.result())
                except Exception as error:errors.append({'sequence_id':uid,'error':str(error)})
                save(root/'dense_tracklets_v4'/'progress.json',{'done':len(results),'errors':errors,'results':results})
                print(json.dumps({'done':len(results),'errors':len(errors)}),flush=True)
        if not args.watch or len(known)==300:break
        time.sleep(20)


if __name__=='__main__':main()
