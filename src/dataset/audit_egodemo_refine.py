"""Audit every refined acquisition; engineering consistency is not accuracy."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from src.annotation.egodemo_review import TYPES
from src.annotation.quality_bindings import bind_quality_to_tracks
from src.dataset.complete_egodemo import digest,save


def se3_error(poses,valid):
    if not np.isnan(poses[~valid]).all():raise ValueError('Invalid poses must be NaN')
    p=poses[valid]
    if not len(p):return 0.
    if not np.isfinite(p).all():raise ValueError('Nonfinite valid pose')
    r=p[:,:3,:3]
    err=max(float(np.max(np.abs(r.swapaxes(1,2)@r-np.eye(3)))),
            float(np.max(np.abs(np.linalg.det(r)-1))),
            float(np.max(np.abs(p[:,3]-[0,0,0,1]))))
    if err>1e-3:raise ValueError(f'Invalid SE3: {err}')
    return err


def audit_sequence(root,rec,selection_sha,quality_sha):
    uid=rec['source_episode_uuid'];count=rec['manifest_frame_count']
    folder=root/'native_tracks'/uid;joined=root/'joint_annotations'/uid
    manifest=json.loads((folder/'tracklets.json').read_text())
    proposal=json.loads((joined/'proposal.json').read_text())
    receipt=json.loads((joined/'receipt.json').read_text())
    assert receipt['tracks_sha256']==digest(folder/'tracklets.json'),'Track provenance mismatch'
    assert receipt['quality_overrides_sha256']==quality_sha,'Stale quality filtering'
    assert proposal['source_selection_sha256']==selection_sha,'Source selection mismatch'
    assert proposal['proposal_source_sha256']==receipt['tracks_sha256'],'Proposal provenance mismatch'
    calibration_path=root/'pixel_calibration_audit'/(uid+'.json')
    if calibration_path.exists():
        assert receipt.get('pixel_calibration_sha256')==digest(calibration_path),'Stale pixel calibration gate'
    if receipt.get('calibration_recovery_sha256'):
        recovery_path=root/'calibration_recovery_trials'/(uid+'.json')
        assert digest(recovery_path)==receipt['calibration_recovery_sha256'],'Calibration recovery hash mismatch'
        recovery=json.loads(recovery_path.read_text())
        assert recovery['usable_for_candidate_reconstruction'] and len(recovery['additional_heldout_checks'])==3
        assert json.loads((folder/'receipt.json').read_text())['stereo_calibration_recovery_sha256']==digest(recovery_path),'Geometry used different calibration'
    assert proposal['training_allowed'] is False and receipt['training_allowed'] is False,'Training gate bypass'
    with np.load(root/'sequences'/uid/'hands.npz') as h:
        assert h['hands_world'].shape==(count,2,21,3)
        assert h['hands_camera'].shape==(count,2,21,3)
        assert h['hand_uv'].shape==(count,2,21,2)
        times=h['timestamps'];assert np.isfinite(times).all() and (np.diff(times)>0).all()
        c2w=h['camera_to_world'];k=h['K']
        xyz=h['hands_camera'];uv=h['hand_uv'];p=xyz@k.T
        expected=p[...,:2]/p[...,2:]
        known=np.isfinite(expected).all(axis=-1)&np.isfinite(uv).all(axis=-1)
        hand_reprojection=float(np.max(np.abs(expected[known]-uv[known]))) if known.any() else 0.
        assert hand_reprojection<.02,'Hand camera/UV inconsistency'
        world=np.einsum('tij,thnj->thni',c2w[:,:3,:3],xyz)+c2w[:,None,None,:3,3]
        known=np.isfinite(world)&np.isfinite(h['hands_world'])
        assert np.max(np.abs(world[known]-h['hands_world'][known]))<1e-4,'Hand world transform mismatch'
    max_pose_error=0.;max_point_transform_error=0.;track_ids=set();track_hashes={}
    retained_tracks={track for e in proposal['entities'] for track in e['tracklets']}
    surface_observed=np.zeros(count,bool)
    for row in manifest['records']:
        track=row['tracklet_id'];assert track not in track_ids;track_ids.add(track)
        path=folder/(track+'.npz');sha=digest(path);assert sha==row['sha256'],'Track file hash mismatch'
        track_hashes[track]=sha
        with np.load(path) as z:
            f=z['frame_index'];v=z['point_valid'];xyz=z['points_camera']
            assert len(f)>0 and (np.diff(f)>0).all() and 0<=f[0]<=f[-1]<count,'Invalid track timeline'
            assert np.allclose(z['timestamps'],times[f],rtol=0,atol=1e-6),'Timestamp mismatch'
            assert np.isfinite(xyz[v]).all() and np.isnan(xyz[~v]).all(),'Point validity mismatch'
            assert z['points_uv'].shape==(len(f),2,v.shape[1],2)
            assert len(np.unique(z['point_ids']))==v.shape[1],'Duplicate point IDs'
            if track in retained_tracks and receipt.get('metric_geometry_allowed',True):
                surface_observed[f]|=v.sum(axis=1)>=8
            expected=np.einsum('tij,tnj->tni',c2w[f,:3,:3],xyz)+c2w[f,None,:3,3]
            error=float(np.max(np.abs(expected[v]-z['points_world'][v]))) if v.any() else 0.
            assert error<1e-4,'Point world transform mismatch'
            max_point_transform_error=max(max_point_transform_error,error)
            max_pose_error=max(max_pose_error,se3_error(z['object_pose'],z['object_pose_valid']))
    entities=proposal['entities'];ids=[e['id'] for e in entities]
    assert len(ids)==len(set(ids)),'Duplicate entity IDs'
    assigned=[]
    for entity in entities:
        assigned+=entity['tracklets']
        assert entity['label_valid'] is False and entity['review_status']=='pending'
        assert entity['cross_occlusion_identity_verified'] is False
        assert set(entity['tracklets'])<=track_ids,'Entity references missing track'
    assert len(assigned)==len(set(assigned)),'Track assigned to multiple objects'
    quality=json.loads((root/'visual_quality_overrides.json').read_text())['records']
    bound=bind_quality_to_tracks([r for r in quality if r['sequence_id']==uid],manifest['records'])
    excluded={track for track,r in bound.items() if r['status']=='rejected' and r['scope']=='all_geometry_and_interactions'}
    assert not excluded.intersection(assigned),'Rejected source observation leaked into entity inventory'
    with np.load(joined/'objects.npz') as z:
        assert z['object_ids'].tolist()==ids and z['poses'].shape==(len(ids),count,4,4)
        assert np.array_equal(z['frame_index'],np.arange(count))
        max_pose_error=max(max_pose_error,se3_error(z['poses'].reshape(-1,4,4),z['pose_valid'].ravel()))
        assert int(z['pose_valid'].sum())==receipt['pose_frames']
        if receipt.get('metric_geometry_allowed') is False:
            assert not z['pose_valid'].any() and not proposal['interactions'],'Unsupported calibration leaked physical labels'
        for i,entity in enumerate(entities):
            assert int(z['pose_valid'][i].sum())==entity['pose_valid_frames']
            if entity['rigidity']=='deformable':assert not z['pose_valid'][i].any(),'Nonrigid whole pose fabricated'
    for event in proposal['interactions']:
        assert event['object_id'] in ids and event['type'] in TYPES[event['layer']]
        assert 0<=event['start_frame']<event['end_frame']<=count,'Interaction outside source timeline'
        assert event['review_status']=='pending' and event['label_valid'] is False
        if 'boundary_frame_range' in event:
            a,b=event['boundary_frame_range'];assert 0<=a<b<=count,'Boundary uncertainty outside timeline'
    for event in proposal['source_semantic_intervals']:
        assert 0<=event['start_frame']<event['end_frame']<=count
        assert event['operations'],'Source operation was not mapped'
        for operation in event['operations']:
            assert operation['label_valid'] is False and operation['object_id'] is None
            for evidence in operation['text_evidence']:
                a,b=evidence['span'];assert event['description_original'][a:b]==evidence['text'],'Source quote mismatch'
    return {'sequence_id':uid,'status':'passed','native_frames':count,'tracklets':len(track_ids),
            'entities':len(ids),'interactions':len(proposal['interactions']),
            'unique_video_frames_with_8_surface_points':int(surface_observed.sum()),
            'surface_observation_fraction':float(surface_observed.mean()),
            'unique_video_frames_with_pose':receipt['unique_pose_frames'],
            'max_se3_error':max_pose_error,'max_hand_projection_difference_px':hand_reprojection,
            'max_point_world_difference_m':max_point_transform_error,
            'tracks_manifest_sha256':receipt['tracks_sha256'],'objects_sha256':digest(joined/'objects.npz'),
            'proposal_sha256':digest(joined/'proposal.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--available-only',action='store_true');a=p.parse_args()
    pool=a.root.parent/'egodemo_expansion_300_20261023';selection=pool/'selection_frozen.json'
    rows=json.loads(selection.read_text())['records'];sha=digest(selection)
    quality=digest(a.root/'visual_quality_overrides.json');results=[]
    for rec in rows:
        uid=rec['source_episode_uuid']
        if a.available_only and not (a.root/'joint_annotations'/uid/'receipt.json').exists():continue
        try:results.append(audit_sequence(a.root,rec,sha,quality))
        except Exception as e:results.append({'sequence_id':uid,'status':'failed','error':str(e),'error_type':type(e).__name__})
    bad=[r for r in results if r['status']!='passed']
    out={'generated_at':datetime.now(timezone.utc).isoformat(),'expected_sequences':len(rows),
         'checked_sequences':len(results),'passed':len(results)-len(bad),'failed':len(bad),
         'source_selection_sha256':sha,'quality_overrides_sha256':quality,'records':results,
         'engineering_audit_complete':len(results)==len(rows) and not bad,
         'annotation_accuracy':None,'full_annotation_complete':False,'training_allowed':False,
         'scope_zh':'仅审计来源、数组、坐标、时轴、身份引用、事件边界范围与有效标记；不表示视觉准确度或人工通过。'}
    save(a.root/('engineering_audit_partial.json' if a.available_only else 'engineering_audit.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k!='records'},ensure_ascii=False));print(json.dumps(bad[:10],ensure_ascii=False))
    if bad:raise SystemExit(1)

if __name__=='__main__':main()
