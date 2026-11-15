"""Evidence-based identity/gauge links and reviewable physical-event proposals.

Names alone never merge two instances. An accepted computational association
requires simultaneous spatial overlap and a constant reference-frame transform.
All proposals remain machine estimates awaiting visual review.
"""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path
import re
import time
import numpy as np
from scipy.spatial import cKDTree

from src.dataset.complete_egodemo import save,digest
from src.pose6d.track_object_candidates import runs
from src.annotation.quality_bindings import bind_quality_to_tracks
from src.annotation.refine_semantics import semantic_directory

ASSOCIATION_VERSION='geometric_overlap_v10_surface_frames_all'


def angle_degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))


def overlap_link(a,b,min_overlap=15):
    """Return b-object -> a-object transform only with concurrent evidence."""
    common,ia,ib=np.intersect1d(a['frame_index'],b['frame_index'],return_indices=True)
    good=a['object_pose_valid'][ia]&b['object_pose_valid'][ib]
    common,ia,ib=common[good],ia[good],ib[good]
    if len(common)<min_overlap or common[-1]-common[0]<15:return None
    chosen=np.linspace(0,len(common)-1,min(7,len(common))).astype(int)
    transforms=[];fractions=[]
    for j in chosen:
        x=a['points_camera'][ia[j]][a['point_valid'][ia[j]]]
        y=b['points_camera'][ib[j]][b['point_valid'][ib[j]]]
        if min(len(x),len(y))<8:return None
        # Geometrically adjacent objects cannot be merged just because they
        # are both stationary and therefore have a constant relative pose.
        distances=cKDTree(x).query(y)[0]
        reverse=cKDTree(y).query(x)[0]
        fractions.append(min(float(np.mean(distances<.008)),float(np.mean(reverse<.008))))
        transforms.append(np.linalg.inv(a['object_pose'][ia[j]])@b['object_pose'][ib[j]])
    if sum(f>=.30 for f in fractions)<max(3,len(fractions)-1):return None
    ref=transforms[len(transforms)//2]
    translation=max(float(np.linalg.norm(t[:3,3]-ref[:3,3])) for t in transforms)
    rotation=max(angle_degrees(t[:3,:3]@ref[:3,:3].T) for t in transforms)
    if translation>.012 or rotation>8:return None
    return {'b_to_a':ref.tolist(),'evidence_frames':common[chosen].tolist(),
            'min_surface_overlap_fraction':min(fractions),
            'max_reference_translation_difference_m':translation,
            'max_reference_rotation_difference_deg':rotation}


def compatible(a,b):
    if a['surface_ownership_ambiguous'] or b['surface_ownership_ambiguous']:return False
    if a['rigidity_candidate']!=b['rigidity_candidate']:return False
    # Keep strict lexical compatibility; false negatives are explicit fragments.
    normalize=lambda s:re.sub(r'\b(the|a|an|another|first|second|third|fourth|last)\b|[^a-z ]','',s.lower()).strip()
    return normalize(a['label'])==normalize(b['label'])


def native_runs(frame,mask,min_length=3):
    for a,b in runs(mask,min_length=min_length):
        cuts=np.r_[a,np.flatnonzero(np.diff(frame[a:b])!=1)+a+1,b]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if hi-lo>=min_length:yield int(lo),int(hi)


def make_proposals(identity,z,total_frames):
    result=[];frames=z['frame_index'];near=z['nearest_hand_surface_m']
    for hand in range(2):
        known=np.isfinite(near[:,hand])&(z['point_valid'].sum(axis=1)>=8)
        for lo,hi in native_runs(frames,known&(near[:,hand]<.03),min_length=5):
            start,end=int(frames[lo]),int(frames[hi-1]+1)
            before=lo-1 if lo else None;after=hi if hi<len(frames) else None
            onset_known=(before is not None and known[before] and frames[before]+1==start and near[before,hand]>=.03)
            offset_known=(after is not None and known[after] and frames[after]==end and near[after,hand]>=.03)
            evidence={'method':'visible_surface_joint_distance','distance_threshold_m':.03,
                      'minimum_distance_m':float(np.min(near[lo:hi,hand])),
                      'minimum_visible_points':int(z['point_valid'][lo:hi].sum(axis=1).min()),
                      'direct_contact_observed':False,'start_censored':not onset_known,'end_censored':not offset_known}
            common={'subject':['left_hand','right_hand'][hand],'object_id':identity,'secondary_object_id':None,
                    'start_frame':start,'end_frame':end,'review_status':'pending','label_valid':False,
                    'label_source':'geometry_proposal','evidence':evidence,
                    'note':'关节接近可见表面；接触尚需看图确认，不能据此认定抓持。'}
            result.append({**common,'layer':'relation','type':'contact'})
            if onset_known:
                result.append({**common,'layer':'transition','type':'contact_onset','end_frame':start+1,
                               'boundary_frame_range':[start-1,start+1]})
            # Crossing a distance threshold is not proof of grasp release.
            # Keep the end as a proximity boundary, not a false release event.
    return result


def co_motion_mask(object_world,wrist_world,near,valid,window=12):
    """Observed joint motion, not a proof of contact or intent."""
    result=np.zeros(len(valid),bool)
    known=valid & np.isfinite(object_world).all(axis=1) & np.isfinite(wrist_world).all(axis=1) & np.isfinite(near)
    for t in range(window-1,len(valid)):
        lo=t-window+1
        if not known[lo:t+1].all() or np.mean(near[lo:t+1]<.10)<.75:continue
        obj=object_world[lo:t+1];hand=wrist_world[lo:t+1]
        relative=obj-hand
        variation=np.max(np.linalg.norm(relative-np.median(relative,axis=0),axis=1))
        moved=min(np.linalg.norm(obj[-1]-obj[0]),np.linalg.norm(hand[-1]-hand[0]))
        if variation<.015 and moved>.025:result[t]=True
    return result


def motion_proposals(identity,world_pose,pose_valid,nearest,hands_world):
    output=[];masks=[];frames=np.arange(len(pose_valid))
    for hand in range(2):
        mask=co_motion_mask(world_pose[:,:3,3],hands_world[:,hand,0],nearest[:,hand],pose_valid)
        masks.append(mask)
        for lo,hi in native_runs(frames,mask,min_length=5):
            base={'subject':['left_hand','right_hand'][hand],'object_id':identity,'secondary_object_id':None,
                  'start_frame':lo,'end_frame':hi,'review_status':'pending','label_valid':False,
                  'label_source':'geometry_proposal',
                  'evidence':{'method':'joint_translation_with_stable_relative_offset','trailing_window_frames':12,
                              'relative_variation_threshold_m':.015,'minimum_displacement_m':.025,
                              'direct_grasp_observed':False},
                  'note':'手腕与物体共同移动，手—物相对位置较稳定；仍需核对是否实际抓持。'}
            output.append({**base,'layer':'relation','type':'co_motion'})
            output.append({**base,'layer':'relation','type':'grasp_control'})
            output.append({**base,'layer':'transition','type':'grasp_onset','end_frame':lo+1,
                           'boundary_frame_range':[max(0,lo-12),lo+1]})
    for lo,hi in native_runs(frames,masks[0]&masks[1],min_length=5):
        output.append({'layer':'cooperation','type':'bimanual_joint','subject':'both_hands',
                       'object_id':identity,'secondary_object_id':None,'start_frame':lo,'end_frame':hi,
                       'review_status':'pending','label_valid':False,'label_source':'geometry_proposal',
                       'evidence':{'method':'both_hand_co_motion_overlap','direct_cooperation_verified':False},
                       'note':'两只手同时满足共同运动条件，双手协作方式待视觉确认。'})
    return output


def assemble(root,uid):
    old=root.parent/'egodemo_joint_300_20261023';pool=root.parent/'egodemo_expansion_300_20261023'
    out=root/'joint_annotations'/uid;out.mkdir(parents=True,exist_ok=True)
    manifest=root/'native_tracks'/uid/'tracklets.json';source_sha=digest(manifest)
    receipt=out/'receipt.json'
    quality_path=root/'visual_quality_overrides.json'
    quality_sha=digest(quality_path) if quality_path.exists() else None
    names_path=root/'display_names_zh.json'
    names_sha=digest(names_path) if names_path.exists() else None
    calibration_path=root/'pixel_calibration_audit'/(uid+'.json')
    calibration_sha=digest(calibration_path) if calibration_path.exists() else None
    calibration=json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    calibration_status=calibration.get('status','not_yet_audited')
    metric_allowed=calibration_status!='inconsistent_on_sampled_frames'
    recovery_path=root/'calibration_recovery_trials'/(uid+'.json')
    recovery_sha=None
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text());candidate_sha=digest(recovery_path)
        track_receipt=json.loads((manifest.parent/'receipt.json').read_text())
        seed_receipt=json.loads((root/'dense_seeds'/uid/'receipt.json').read_text())
        if (recovery.get('usable_for_candidate_reconstruction') and recovery['pixel_evidence_sha256']==calibration_sha
                and seed_receipt.get('stereo_calibration_recovery_sha256')==candidate_sha
                and track_receipt.get('stereo_calibration_recovery_sha256')==candidate_sha):
            recovery_sha=candidate_sha;calibration_status='image_refined_with_heldout_checks';metric_allowed=True
    translated=json.loads(names_path.read_text())['translations'] if names_path.exists() else {}
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if (prior.get('tracks_sha256')==source_sha and prior.get('association_version')==ASSOCIATION_VERSION
                and prior.get('quality_overrides_sha256')==quality_sha):
            if (prior.get('display_names_sha256')==names_sha and prior.get('pixel_calibration_sha256')==calibration_sha
                    and prior.get('calibration_recovery_sha256')==recovery_sha):return prior
    all_rows=json.loads(manifest.read_text())['records']
    # A collection of several objects is scene evidence, not a single entity.
    reviewed=bind_quality_to_tracks([r for r in json.loads(quality_path.read_text())['records']
              if r['sequence_id']==uid],all_rows) if quality_path.exists() else {}
    quality={k:r for k,r in reviewed.items() if r['status']=='rejected'}
    rows=[r for r in all_rows if not re.fullmatch(r'(?:the )?(?:stack|pile|bunch|group|items|objects|materials)',r['label'],re.I)
          and quality.get(r['tracklet_id'],{}).get('scope')!='all_geometry_and_interactions']
    arrays={r['tracklet_id']:dict(np.load(manifest.parent/(r['tracklet_id']+'.npz'))) for r in rows}
    count=json.loads((old/'sequences'/uid/'metadata.json').read_text())['frame_count']
    with np.load(old/'sequences'/uid/'hands.npz') as hand_data:
        hands_world=hand_data['hands_world'];camera_to_world=hand_data['camera_to_world'];timestamps=hand_data['timestamps']
    source_intervals=json.loads((pool/'staging'/uid/'events.json').read_text())['subtask_annotations']
    for row in rows:
        z=arrays[row['tracklet_id']]
        row['name_zh']=translated.get(row['name_zh'],row['name_zh'])
        correction=reviewed.get(row['tracklet_id'],{}).get('correction')
        if correction:
            row.update(correction)
            row['material_basis']='assistant_sampled_component_correction'
        if row.get('pose_scope')=='rigid_part':
            possible=np.isfinite(z['local_surface_pose']).all(axis=(1,2))
            valid_part=np.zeros_like(possible)
            for lo,hi in native_runs(z['frame_index'],possible,min_length=5):valid_part[lo:hi]=True
            z['object_pose']=z['local_surface_pose'].copy();z['object_pose_valid']=valid_part
            z['object_pose'][~valid_part]=np.nan
        plural=bool(re.search(r'\b(markers|pens|lids|caps|clips|staples|boxes|books|bottles|jars|bowls|cups)\b',row['label'],re.I))
        row['physical_target_ambiguous']=plural or row['surface_ownership_ambiguous']
        if not metric_allowed:
            # Preserve raw failed geometry separately, but do not derive a
            # physical identity/interaction from an unsupported camera rig.
            z['object_pose_valid'][:]=False;z['object_pose'][:]=np.nan
            z['point_valid'][:]=False;z['nearest_hand_surface_m'][:]=np.nan
        if row['tracklet_id'] in quality:
            z['object_pose_valid'][:]=False;z['object_pose'][:]=np.nan
        elif plural:
            # A collection (e.g. a clump of staples) carries a group pose while
            # its tracked surface moves coherently. The per-frame rigid-fit
            # residual already NaNs frames where the collection is disturbed;
            # keep the >=5-consecutive-frame support requirement. Identity
            # stays ambiguous and review stays pending: this is a candidate
            # group frame, not a certified whole-object 6D pose.
            z['object_pose']=z['local_surface_pose'].copy()
            group=np.isfinite(z['object_pose']).all(axis=(1,2))
            supported=np.zeros_like(group)
            for lo,hi in native_runs(z['frame_index'],group,min_length=5):supported[lo:hi]=True
            z['object_pose_valid']=supported
            z['object_pose'][~supported]=np.nan
            row['pose_scope']='group_quasi_rigid'
        elif row.get('pose_scope')=='surface_only' or (row.get('pose_scope') is None
                and row['rigidity_candidate'] in ('deformable','unknown','articulated')):
            # Deformable/unknown-rigidity surfaces still carry a tracked-
            # surface frame while the visible patch moves coherently. This is
            # the frame of the tracked surface patch, never a certified
            # whole-object pose; identity and review flags stay pending.
            z['object_pose']=z['local_surface_pose'].copy()
            surface=np.isfinite(z['object_pose']).all(axis=(1,2))
            supported=np.zeros_like(surface)
            for lo,hi in native_runs(z['frame_index'],surface,min_length=5):supported[lo:hi]=True
            z['object_pose_valid']=supported
            z['object_pose'][~supported]=np.nan
            row['pose_scope']='deformable_surface_frame'
        if row['material_basis']=='rigid_motion_hypothesis_for_transport_only':
            reference_seconds=timestamps[row['reference_frame']]
            intervals=[e for e in source_intervals if e['start_time']<=reference_seconds<=e['end_time']]
            allowed=np.zeros(len(z['frame_index']),bool)
            for event in intervals:
                t=timestamps[z['frame_index']];allowed|=(t>=event['start_time'])&(t<event['end_time'])
            z['object_pose_valid']&=allowed;z['object_pose'][~z['object_pose_valid']]=np.nan
    neighbors={r['tracklet_id']:[] for r in rows};links=[]
    for a,b in combinations(rows,2):
        if not compatible(a,b):continue
        if min(a['end_frame'],b['end_frame'])-max(a['start_frame'],b['start_frame'])<15:continue
        link=overlap_link(arrays[a['tracklet_id']],arrays[b['tracklet_id']])
        if link is None:continue
        transform=np.asarray(link['b_to_a'])
        links.append({'a':a['tracklet_id'],'b':b['tracklet_id'],**link,'status':'computed_pending_visual'})
        neighbors[a['tracklet_id']].append((b['tracklet_id'],transform))
        neighbors[b['tracklet_id']].append((a['tracklet_id'],np.linalg.inv(transform)))
    by_id={r['tracklet_id']:r for r in rows};visited=set();entities=[];proposals=[];poses=[];valids=[];conflicts=[]
    for row in sorted(rows,key=lambda x:(x['reference_frame'],x['tracklet_id'])):
        first=row['tracklet_id']
        if first in visited:continue
        transforms={first:np.eye(4)};queue=[first];visited.add(first)
        while queue:
            current=queue.pop(0)
            for other,other_to_current in neighbors[current]:
                if other not in transforms:
                    transforms[other]=transforms[current]@other_to_current
                    visited.add(other);queue.append(other)
        identity=f'obj_{len(entities)+1:03d}'
        pose=np.full((count,4,4),np.nan,np.float32);valid=np.zeros(count,bool);score=np.zeros(count,np.int16)
        conflict=np.zeros(count,bool);nearest=np.full((count,2),np.nan,np.float32)
        for track,t_to_object in transforms.items():
            z=arrays[track];frames=z['frame_index'];ok=z['object_pose_valid']
            aligned=z['object_pose'][ok]@np.linalg.inv(t_to_object);ids=frames[ok]
            overlap=valid[ids]
            if overlap.any():
                old_pose=pose[ids[overlap]];new_pose=aligned[overlap]
                for f,p,q in zip(ids[overlap],old_pose,new_pose):
                    if np.linalg.norm(p[:3,3]-q[:3,3])>.025 or angle_degrees(p[:3,:3]@q[:3,:3].T)>15:
                        conflict[f]=True
            better=z['inlier_count'][ok]>score[ids]
            pose[ids[better]]=aligned[better];valid[ids[better]]=True;score[ids[better]]=z['inlier_count'][ok][better]
            if not by_id[track]['physical_target_ambiguous']:
                proposals.extend(make_proposals(identity,z,count))
            observed_near=z['nearest_hand_surface_m'].copy()
            observed_near[z['point_valid'].sum(axis=1)<8]=np.nan
            nearest[frames]=np.fmin(nearest[frames],observed_near)
        pose[conflict]=np.nan;valid[conflict]=False
        world_pose=camera_to_world@pose
        proposals.extend(motion_proposals(identity,world_pose,valid,nearest,hands_world))
        entities.append({'id':identity,'name_zh':row['name_zh'],'label':row['label'],
            'rigidity':row['rigidity_candidate'],'role':'unknown','tracklets':list(transforms),
            'pose_scope':row.get('pose_scope','whole_object_candidate'),
            'physical_target_ambiguous':row['physical_target_ambiguous'],
            'metric_geometry_allowed':metric_allowed,'pixel_calibration_status':calibration_status,
            'identity_scope':'observed_continuous_track_and_verified_geometric_overlap',
            'identity_review_status':'pending_visual','cross_occlusion_identity_verified':False,
            'object_frame_reference_track':first,
            'track_to_object_transforms':{k:v.tolist() for k,v in transforms.items()},
            'pose_valid_frames':int(valid.sum()),'pose_conflict_frames':int(conflict.sum()),
            'hand_near_pose_frames':int((valid&(np.nan_to_num(nearest,nan=10).min(axis=1)<.10)).sum()),
            'hand_near_point_frames':int((np.nan_to_num(nearest,nan=10).min(axis=1)<.10).sum()),
            'review_status':'pending','label_valid':False})
        poses.append(pose);valids.append(valid);conflicts.append(conflict)
    # Deduplicate proposals generated by overlapping linked tracklets.
    dedup={}
    for row in proposals:
        key=tuple(row[k] for k in ['object_id','subject','layer','type','start_frame','end_frame'])
        dedup[key]=row
    proposals=list(dedup.values())
    source_events=[]
    for path in sorted(semantic_directory(root,uid).glob('chunk_*.json')):
        if '.error.' not in path.name:source_events.extend(json.loads(path.read_text())['records'])
    poses=np.asarray(poses,np.float32).reshape(len(entities),count,4,4)
    valid=np.asarray(valids,bool).reshape(len(entities),count)
    np.savez_compressed(out/'objects.partial.npz',object_ids=np.array([e['id'] for e in entities],dtype='U32'),
                        frame_index=np.arange(count),poses=poses,pose_valid=valid)
    (out/'objects.partial.npz').replace(out/'objects.npz')
    draft={'schema_version':'egodemo_review_draft_v1','sequence_id':uid,
           'proposal_source_sha256':source_sha,'association_version':ASSOCIATION_VERSION,
           'proposal_quality_sha256':quality_sha,
           'proposal_calibration_sha256':calibration_sha,
           'proposal_calibration_recovery_sha256':recovery_sha,
           'source_selection_sha256':digest(pool/'selection_frozen.json'),'revision':0,'reviewer':'',
           'entities':entities,'interactions':proposals,'source_semantic_intervals':source_events,
           'origin':'automatically_prepared_review_proposal','review_status':'pending','training_allowed':False}
    save(out/'proposal.json',draft);save(out/'identity_links.json',{'links':links})
    result={'sequence_id':uid,'tracks_sha256':source_sha,'association_version':ASSOCIATION_VERSION,
            'entities':len(entities),'identity_links':len(links),
            'quality_overrides_sha256':quality_sha,
            'display_names_sha256':names_sha,
            'pixel_calibration_sha256':calibration_sha,'pixel_calibration_status':calibration_status,
            'metric_geometry_allowed':metric_allowed,
            'calibration_recovery_sha256':recovery_sha,
            'pose_frames':int(valid.sum()),'unique_pose_frames':int(valid.any(axis=0).sum()),
            'proximity_proposals':sum(r['layer']=='relation' for r in proposals),
            'onset_proposals':sum(r['layer']=='transition' for r in proposals),'source_intervals':len(source_events),
            'hand_near_pose_frames':sum(e['hand_near_pose_frames'] for e in entities),
            'hand_near_point_frames':sum(e['hand_near_point_frames'] for e in entities),
            'rigid_part_pose_frames':sum(e['pose_valid_frames'] for e in entities if e['pose_scope']=='rigid_part'),
            'group_quasi_rigid_pose_frames':sum(e['pose_valid_frames'] for e in entities if e['pose_scope']=='group_quasi_rigid'),
            'deformable_surface_frame_frames':sum(e['pose_valid_frames'] for e in entities if e['pose_scope']=='deformable_surface_frame'),
            'co_motion_proposals':sum(r['type']=='co_motion' for r in proposals),
            'grasp_proposals':sum(r['type']=='grasp_control' for r in proposals),
            'cooperation_proposals':sum(r['layer']=='cooperation' for r in proposals),
            'human_approved':False,'full_annotation_complete':False,'training_allowed':False}
    save(receipt,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();seen=set();results=[];errors=[]
    while True:
        ready=[p.parent.name for p in (args.root/'native_tracks').glob('*/receipt.json') if p.parent.name not in seen]
        for uid in ready:
            try:results.append(assemble(args.root,uid))
            except Exception as e:
                errors.append({'sequence_id':uid,'error':str(e)})
                import traceback;traceback.print_exc()
            seen.add(uid);save(args.root/'association_progress.json',{'done':len(results),'errors':errors,'results':results})
            print(json.dumps({'done':len(results),'errors':len(errors),'last':results[-1] if results else None}),flush=True)
        if not args.watch or len(seen)>=300:break
        time.sleep(10)


if __name__=='__main__':main()
