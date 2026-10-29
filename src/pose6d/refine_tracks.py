"""Bidirectional native-frame metric tracks from learned stereo references.

Uses observed pixel correspondences on both cameras. Masks and material names
remain hypotheses, and no unobserved pose is filled or interpolated.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import cv2
import numpy as np

from src.annotation.object_observations import camera_matrices
from src.dataset.complete_egodemo import save,digest
from src.pose6d.nonrigid_tracks import follow_points,stereo_points
from src.pose6d.stereo_reference import relative_camera
from src.pose6d.rigid_fit import robust_rigid_fit_batched as robust_rigid_fit
from src.pose6d.track_object_candidates import distinct_points,runs
from src.pose6d.refine_stereo_calibration import load_candidate_calibration


def process(root_string,uid,force=False):
    cv2.setNumThreads(2)
    root=Path(root_string);old=root.parent/'egodemo_joint_300_20261023';pool=root.parent/'egodemo_expansion_300_20261023'
    dest=root/'native_tracks'/uid
    if not force and (dest/'receipt.json').exists():return json.loads((dest/'receipt.json').read_text())
    dest.mkdir(parents=True,exist_ok=True)
    source=np.load(pool/'staging'/uid/'sequence.npz');count=len(source['timestamps'])
    metadata=json.loads((old/'sequences'/uid/'metadata.json').read_text())
    recovered_calibration,recovery_sha=load_candidate_calibration(root,uid,metadata['calibration_sha256'])
    calib=json.loads((old/'stereo_calibration.json').read_text())[uid]
    ks=[np.array([[calib[s][0],0,calib[s][2]],[0,calib[s][1],calib[s][3]],[0,0,1.]]) for s in ['left','right']]
    c2w=[camera_matrices(source,s) for s in ['left','right']]
    hands=np.stack([source['left_hand_world'],source['right_hand_world']],axis=1)
    seeds=[]
    for path in sorted((root/'dense_seeds'/uid).glob('frame_*/observations.json')):
        data=json.loads(path.read_text())
        for obs in data['observations']:
            refpath=path.parent/(obs['observation_id']+'_reference.npz')
            if obs['triangulation_status']!='estimated_reference':continue
            if digest(refpath)!=obs['reference_sha256']:raise ValueError('Reference hash mismatch')
            with np.load(refpath) as z:
                ids=distinct_points(z['left_uv'],96)
                seeds.append({'reference_frame':data['frame_index'],'row':obs,
                    'points_object':z['points_object'][ids],
                    'uv_initial':[z['left_uv'][ids].astype(np.float32),z['right_uv'][ids].astype(np.float32)],
                    'frame':[],'xyz':[],'uv':[],'valid':[],'pose':[],'near':[],'inliers':[]})
    if not seeds:
        receipt={'sequence_id':uid,'tracklets':0,'point_frames':0,'object_pose_frames':0,'status':'no_trackable_surface',
                 'stereo_calibration_recovery_sha256':recovery_sha,
                 'full_annotation_complete':False,'training_allowed':False}
        save(dest/'tracklets.json',{'sequence_id':uid,'records':[]});save(dest/'receipt.json',receipt);return receipt
    started=time.time();size=(metadata['image_size'][0]//2,metadata['image_size'][1]//2)
    scale=np.array(size)/metadata['image_size']
    # Bound RAM by one acquisition per worker. Both directions reuse these
    # decoded frames instead of inaccurate backwards random seeking.
    gray=[]
    for path in [metadata['video_path'],str(old/'stereo'/(uid+'_right.mp4'))]:
        cap=cv2.VideoCapture(path);frames=np.empty((count,size[1],size[0]),np.uint8)
        try:
            for i in range(count):
                ok,image=cap.read()
                if not ok:raise ValueError('Incomplete native video decode')
                frames[i]=cv2.cvtColor(cv2.resize(image,size),cv2.COLOR_BGR2GRAY)
        finally:cap.release()
        gray.append(frames)
    for direction in [1,-1]:
        ordered=sorted(seeds,key=lambda x:x['reference_frame'],reverse=direction<0)
        for s in seeds:
            s['uv_current']=[x*scale for x in s['uv_initial']]
            s['alive']=[np.ones(len(s['points_object']),bool) for _ in range(2)]
            s['stereo_lost']=np.zeros(len(s['points_object']),np.int16)
        active=[];next_seed=0;index=ordered[0]['reference_frame'];previous=None
        while 0<=index<count:
            if not active:
                if next_seed>=len(ordered):break
                index=ordered[next_seed]['reference_frame'];previous=None
            if active and previous is not None:
                lengths=[len(s['points_object']) for s in active]
                for side in range(2):
                    uv=np.concatenate([s['uv_current'][side] for s in active])
                    alive=np.concatenate([s['alive'][side] for s in active])
                    uv,alive=follow_points(gray[side][previous],gray[side][index],uv,alive,fb_limit=.8)
                    at=0
                    for s,n in zip(active,lengths):
                        s['uv_current'][side]=uv[at:at+n];s['alive'][side]=alive[at:at+n];at+=n
            while next_seed<len(ordered) and ordered[next_seed]['reference_frame']==index:
                active.append(ordered[next_seed]);next_seed+=1
            rel=relative_camera(c2w[0][index],c2w[1][index]) if recovered_calibration is None else recovered_calibration
            retained=[]
            for s in active:
                uv=[p/scale for p in s['uv_current']]
                xyz,valid,_=stereo_points(*uv,*ks,rel)
                valid &= s['alive'][0]&s['alive'][1]
                xyz[~valid]=np.nan
                s['stereo_lost']=np.where(valid,0,s['stereo_lost']+1)
                for side in range(2):s['alive'][side]&=s['stereo_lost']<4
                # Store the shared seed only once; retain the same object frame
                # in both temporal directions.
                if direction>0 or index!=s['reference_frame']:
                    pose,fit=robust_rigid_fit(s['points_object'],xyz,trials=20)
                    world=xyz@c2w[0][index,:3,:3].T+c2w[0][index,:3,3]
                    near=[]
                    for hand in hands[index]:
                        observed=hand[np.isfinite(hand).all(axis=1)]
                        near.append(float(np.min(np.linalg.norm(observed[:,None]-world[valid][None],axis=-1)))
                                    if len(observed) and valid.any() else np.nan)
                    s['frame'].append(index);s['uv'].append(np.stack(uv).astype(np.float32));s['xyz'].append(xyz.astype(np.float32))
                    s['valid'].append(valid);s['near'].append(near);s['inliers'].append(fit['inliers'])
                    s['pose'].append(pose.astype(np.float32) if pose is not None else np.full((4,4),np.nan,np.float32))
                if (s['alive'][0]&s['alive'][1]).sum()>=4:retained.append(s)
            active=retained;previous=index;index+=direction
    records=[]
    for i,s in enumerate(seeds):
        order=np.argsort(s['frame']);frame=np.array(s['frame'])[order];valid=np.array(s['valid'])[order]
        pose=np.array(s['pose'])[order];surface_valid=np.isfinite(pose).all(axis=(1,2));whole=np.zeros_like(surface_valid)
        if s['row']['whole_rigid_candidate']:
            # Gaps cannot be bridged by a compressed index-space run.
            for a,b in runs(surface_valid,min_length=5):
                breaks=np.r_[a,np.flatnonzero(np.diff(frame[a:b])!=1)+a+1,b]
                for lo,hi in zip(breaks[:-1],breaks[1:]):
                    if hi-lo>=5:whole[lo:hi]=True
        object_pose=pose.copy();object_pose[~whole]=np.nan
        xyz=np.array(s['xyz'])[order]
        world=np.einsum('tij,tnj->tni',c2w[0][frame,:3,:3],xyz)+c2w[0][frame,None,:3,3]
        identity=f'tracklet_{i:04d}';path=dest/(identity+'.npz')
        np.savez_compressed(path,frame_index=frame,timestamps=source['timestamps'][frame],
            reference_points_object=s['points_object'],points_uv=np.array(s['uv'])[order],
            points_camera=xyz,points_world=world,point_valid=valid,point_ids=np.arange(valid.shape[1]),
            local_surface_pose=pose,object_pose=object_pose,object_pose_valid=whole,
            nearest_hand_surface_m=np.array(s['near'],np.float32)[order],inlier_count=np.array(s['inliers'])[order])
        records.append({'tracklet_id':identity,'stable_object_id':None,'reference_frame':s['reference_frame'],
            'name_zh':s['row']['name_zh'],'label':s['row']['label'],'rigidity_candidate':s['row']['rigidity_candidate'],
            'material_basis':s['row']['material_basis'],'surface_ownership_ambiguous':s['row']['surface_ownership_ambiguous'],
            'pose_scope':s['row'].get('pose_scope','whole_object' if s['row']['whole_rigid_candidate'] else 'surface_only'),
            'seed_source':s['row'].get('seed_source','grounded_mask'),
            'source_observation_id':s['row']['observation_id'],'source_reference_sha256':s['row']['reference_sha256'],
            'start_frame':int(frame[0]),'end_frame':int(frame[-1]+1),'frame_count':len(frame),
            'valid_point_frames':int(valid.any(axis=1).sum()),'object_pose_frames':int(whole.sum()),
            'pose_frame_definition':'seed camera axes, median visible surface origin; both directions share one gauge',
            'annotation_mode':'offline_noncausal','review_status':'pending','allowed_as_model_input':False,
            'independent_accuracy':None,'sha256':digest(path)})
    save(dest/'tracklets.json',{'sequence_id':uid,'records':records,'identity_merge_required':True,'training_allowed':False})
    receipt={'sequence_id':uid,'tracklets':len(records),'point_frames':sum(x['valid_point_frames'] for x in records),
             'object_pose_frames':sum(x['object_pose_frames'] for x in records),'seconds':time.time()-started,
             'status':'bidirectional_candidates_exported','full_annotation_complete':False,'training_allowed':False}
    receipt['stereo_calibration_recovery_sha256']=recovery_sha
    save(dest/'receipt.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--sequence',action='append')
    parser.add_argument('--watch',action='store_true');parser.add_argument('--force',action='store_true');args=parser.parse_args()
    if args.force and not args.sequence:parser.error('--force requires explicit --sequence selection')
    seen=set();results=[];errors=[]
    while True:
        ready=[p.parent.name for p in (args.root/'dense_seeds').glob('*/receipt.json')
               if p.parent.name not in seen and (not args.sequence or p.parent.name in args.sequence)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(process,str(args.root),uid,args.force):uid for uid in ready}
            for job in as_completed(jobs):
                uid=jobs[job];seen.add(uid)
                try:results.append(job.result())
                except Exception as error:
                    errors.append({'sequence_id':uid,'error':str(error)})
                    import traceback;traceback.print_exc()
                save(args.root/'track_progress.json',{'done':len(results),'errors':errors,'results':results})
                print(json.dumps({'done':len(results),'errors':len(errors),'last':results[-1] if results else None}),flush=True)
        if not args.watch or len(seen)>=300:break
        time.sleep(10)


if __name__=='__main__':main()
