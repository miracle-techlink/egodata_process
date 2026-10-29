"""Reconstruct all frozen EgoDemo object observations with calibrated stereo.

This stage measures surfaces, preserves rejected observations and never promotes
an object name or a successful file export into a verified annotation.
"""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import cv2
import numpy as np

from src.annotation.object_observations import camera_matrices
from src.annotation.source_semantics import rigidity_candidate
from src.dataset.complete_egodemo import digest, save
from src.pose6d.refine_backends import MetricStereo, Rectification
from src.pose6d.stereo_reference import relative_camera
from src.pose6d.observation_exclusions import capture_exclusion_mask
from src.pose6d.refine_stereo_calibration import load_candidate_calibration


def material_profile(label, context):
    """Explain the effective material hypothesis without editing source labels."""
    text=label.lower()
    kind=rigidity_candidate(text,context)
    basis='existing_material_rule'
    if re.search(r'\b(zip tie|rope|bubble wrap|film|chain|sponge|straws?|gloves?)\b',text):
        kind,basis='deformable','flexible_material_noun'
    elif re.search(r'\b(hole punch|tongs|pliers|glasses)\b',text):
        kind,basis='articulated','moving_parts_noun'
    elif re.search(r'\b(box|boxes|carton|cartons|book|books)\b',text):
        if re.search(r'\b(fold|unfold|flatten|collapse|tear|open|close|assemble)\w*\b',context,re.I):
            kind,basis='articulated','task_changes_parts_or_shape'
        else:
            kind,basis='rigid','rigid_motion_hypothesis_for_transport_only'
    elif re.search(r'\b(tray|basket|card holder|kettle|eraser|hair dryer|plug|charger|ruler|cube|clip|staple)\w*\b',text):
        kind,basis='rigid','rigid_item_noun'
    ambiguous=bool(re.search(r'\b(transparent|glass|cup|jar|bottle)\w*\b',text))
    collective=bool(re.search(r'\b(stack|pile|bunch|group|first two|last two|items|objects|materials)\b',text))
    # Multiple model phrases concatenated by grounding cannot establish which
    # physical item owns the mask (e.g. "black marker blue marker").
    nouns=re.findall(r'\b(box|book|marker|bottle|container|package|holder|lid|cap|pen)s?\b',text)
    combined=len(nouns)>1 and len(set(nouns))<len(nouns)
    return {'rigidity_candidate':kind,'material_basis':basis,
            'surface_ownership_ambiguous':ambiguous or collective or combined,
            'whole_rigid_candidate':kind=='rigid' and not (ambiguous or collective or combined)}


def run_sequence(root,old,pool,uid,stereo):
    dest=root/'dense_seeds'/uid
    meta=json.loads((old/'sequences'/uid/'metadata.json').read_text())
    recovered_calibration,recovery_sha=load_candidate_calibration(root,uid,meta['calibration_sha256'])
    manual_paths=sorted((root/'manual_observations'/uid).glob('frame_*/observations.json'))
    manual_sources={str(p.relative_to(root)):digest(p) for p in manual_paths}
    if (dest/'receipt.json').exists():
        previous=json.loads((dest/'receipt.json').read_text())
        if (previous.get('manual_sources',{})==manual_sources and previous.get('stereo_calibration_recovery_sha256')==recovery_sha
                and (not manual_paths or previous.get('manual_backend_version')==2)):return previous
    dest.mkdir(parents=True,exist_ok=True)
    meta=json.loads((old/'sequences'/uid/'metadata.json').read_text())
    source=np.load(pool/'staging'/uid/'sequence.npz')
    timeline=json.loads((old/'stereo_checks'/(uid+'.json')).read_text())
    if timeline['status']!='passed' or timeline['frame_count']!=len(source['timestamps']):
        raise ValueError('Stereo timeline mismatch')
    hands=np.load(old/'sequences'/uid/'hands.npz')['hand_uv']
    calib=json.loads((old/'stereo_calibration.json').read_text())[uid]
    ks=[np.array([[calib[s][0],0,calib[s][2]],[0,calib[s][1],calib[s][3]],[0,0,1.]]) for s in ['left','right']]
    cameras=[camera_matrices(source,s) for s in ['left','right']]
    events=json.loads((pool/'staging'/uid/'events.json').read_text())['subtask_annotations']
    rejected=set()
    quality=old/'visual_quality_overrides.json'
    if quality.exists():
        rejects=[r for r in json.loads(quality.read_text())['records'] if r['sequence_id']==uid and r['status']=='rejected']
        trackfile=old/'dense_tracklets_v4'/uid/'tracklets.json'
        if rejects and trackfile.exists():
            tracks={r['tracklet_id']:r for r in json.loads(trackfile.read_text())['records']}
            rejected={tracks[r['tracklet_id']]['source_observation_id'] for r in rejects}
    caps=[cv2.VideoCapture(meta['video_path']),cv2.VideoCapture(str(old/'stereo'/(uid+'_right.mp4')))]
    results=[];started=time.time()
    try:
        for path in sorted((old/'object_observations_v2'/uid).glob('frame_*/observations.json'))+manual_paths:
            original=json.loads(path.read_text());frame=original['frame_index']
            folder=dest/path.parent.name;folder.mkdir(exist_ok=True)
            images=[]
            for cap in caps:
                cap.set(1,frame);ok,im=cap.read()
                if not ok:raise ValueError('Frame decode failed')
                images.append(im)
            rel=relative_camera(cameras[0][frame],cameras[1][frame]) if recovered_calibration is None else recovered_calibration
            manual_frame=path in manual_paths
            rect=Rectification(*ks,rel,tuple(meta['image_size']),width=1280 if manual_frame else 768)
            depth=stereo.estimate(rect,*images)
            excluded=capture_exclusion_mask(images[0],hands[frame])
            context=' '.join(e['description'] for e in events if e['start_time']<=source['timestamps'][frame]<=e['end_time'])
            observations=[]
            for obs in original['observations']:
                identity=obs['observation_id'];row={**obs,**material_profile(obs['label'],context)}
                manual=obs.get('seed_source')=='assistant_visual_polygon'
                if manual:
                    # A viewed surface polygon is a seed, never a gold mask or
                    # an invented detector confidence. Preserve explicit scope.
                    row.update(obs['material_profile'])
                row.update(source_observation_sha256=digest(path),reference_points=0,
                           stereo_calibration_recovery_sha256=recovery_sha,
                           reference_backend='c_fast_foundationstereo_lr_checked',
                           triangulation_status='insufficient_geometry',review_status='pending')
                mask=cv2.imread(str(path.parent/obs['mask_file']),cv2.IMREAD_GRAYSCALE)
                if mask is None:raise ValueError('Missing original object mask')
                if identity in rejected:
                    row['geometry_reason']='prior_assistant_visual_rejection';observations.append(row);continue
                if (not manual and obs['sam_score']<.75) or mask.astype(bool).mean()>.4:
                    row['geometry_reason']='weak_or_background_mask';observations.append(row);continue
                allowed=(mask>0) if manual else (mask>0)&(excluded==0)
                allowed=cv2.erode(allowed.astype(np.uint8),np.ones((7,7),np.uint8))
                gray=cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
                points=cv2.goodFeaturesToTrack(gray,maxCorners=160,qualityLevel=.003 if manual else .01,
                                              minDistance=3 if manual else 8,mask=allowed*255)
                row['feature_sampling']={'quality_level':.003 if manual else .01,'min_distance_native_px':3 if manual else 8}
                if points is None:
                    row['geometry_reason']='no_trackable_surface_corners';observations.append(row);continue
                uv=points[:,0]
                xyz,valid=rect.sample_xyz(uv,depth['xyz_camera'],depth['valid'])
                if valid.sum()<12:
                    row['geometry_reason']='fewer_than_12_lr_consistent_points';observations.append(row);continue
                origin=np.median(xyz[valid],axis=0)
                valid &= np.linalg.norm(xyz-origin,axis=1)<.35
                xyz,uv=xyz[valid],uv[valid]
                if len(xyz)<12:
                    row['geometry_reason']='incoherent_surface';observations.append(row);continue
                right_xyz=xyz@rel[:3,:3].T+rel[:3,3]
                right_uv=right_xyz@ks[1].T;right_uv=right_uv[:,:2]/right_uv[:,2:]
                finite=(right_uv[:,0]>0)&(right_uv[:,0]<meta['image_size'][0]-1)&(right_uv[:,1]>0)&(right_uv[:,1]<meta['image_size'][1]-1)
                xyz,uv,right_uv=xyz[finite],uv[finite],right_uv[finite]
                if len(xyz)<12:
                    row['geometry_reason']='outside_right_camera';observations.append(row);continue
                origin=np.median(xyz,axis=0)
                reference=folder/(identity+'_reference.npz')
                np.savez_compressed(reference,points_object=(xyz-origin).astype(np.float32),left_uv=uv,
                    right_uv=right_uv,origin_camera=origin,reference_rotation=np.eye(3),
                    depth_source='calibrated_stereo_model_estimate',reference_frame=frame)
                row.update(reference_points=len(xyz),triangulation_status='estimated_reference',
                    reference_sha256=digest(reference),reference_origin_camera=origin.tolist(),
                    mask_file=str(path.parent/obs['mask_file']),independent_accuracy=None,
                    geometry_reason=None)
                observations.append(row)
            save(folder/'observations.json',{**original,'observations':observations,
                'model_sha256':'7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214',
                'stereo_baseline_m':rect.baseline_m,'stereo_width':rect.size[0],'left_right_threshold_px':1.5})
            # Persist representative depth evidence without storing dense maps for every image.
            valid=depth['valid'];z=depth['xyz_camera'][...,2]
            display=cv2.applyColorMap(np.uint8(np.nan_to_num(np.clip(z/1.5,0,1))*255),cv2.COLORMAP_TURBO)
            display[~valid]=0;cv2.imwrite(str(folder/'depth.jpg'),display)
            results.extend(observations)
    finally:
        for cap in caps:cap.release()
    good=[r for r in results if r['triangulation_status']=='estimated_reference']
    receipt={'sequence_id':uid,'observations':len(results),'references':len(good),
             'rigid_references':sum(r['whole_rigid_candidate'] for r in good),
             'reference_points':sum(r['reference_points'] for r in good),'seconds':time.time()-started,
             'manual_sources':manual_sources,'manual_backend_version':2 if manual_paths else None,
             'stereo_calibration_recovery_sha256':recovery_sha,
             'full_annotation_complete':False,'training_allowed':False}
    save(dest/'receipt.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sequence',action='append');args=parser.parse_args()
    root=args.root;old=root.parent/'egodemo_joint_300_20261023';pool=root.parent/'egodemo_expansion_300_20261023'
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(2)
    stereo=MetricStereo(root/'models')
    records=json.loads((pool/'selection_frozen.json').read_text())['records']
    results=[];errors=[]
    for record in records:
        uid=record['source_episode_uuid']
        if args.sequence and uid not in args.sequence:continue
        try:results.append(run_sequence(root,old,pool,uid,stereo))
        except Exception as error:
            errors.append({'sequence_id':uid,'error':str(error)})
            import traceback;traceback.print_exc()
        save(root/'seed_progress.json',{'done':len(results),'errors':errors,'results':results})
        print(json.dumps({'done':len(results),'errors':len(errors),'last':results[-1] if results else None}),flush=True)


if __name__=='__main__':main()
