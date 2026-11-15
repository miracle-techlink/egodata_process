"""Structural audit of candidate geometry. This is not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from src.dataset.complete_egodemo import digest, save


def audit(root, uid):
    folder=root/'dense_tracklets_v4'/uid
    meta=json.loads((root/'sequences'/uid/'metadata.json').read_text())
    receipt=json.loads((folder/'receipt.json').read_text())
    if not receipt['tracklets']:
        return {'sequence_id':uid,'status':'passed_structure','tracklets':0,'pose_frames':0,'accuracy':None}
    data=json.loads((folder/'tracklets.json').read_text())
    source_t=np.load(root/'sequences'/uid/'hands.npz')['timestamps']
    total=0;warnings=[]
    for row in data['records']:
        path=folder/(row['tracklet_id']+'.npz')
        if digest(path)!=row['sha256']:raise ValueError('Tracklet payload hash mismatch')
        with np.load(path) as z:
            f=z['frame_index'];v=z['point_valid'];p=z['object_pose'];pv=z['object_pose_valid']
            if not len(f) or f.min()<0 or f.max()>=meta['frame_count'] or np.any(np.diff(f)!=1):
                raise ValueError('Native frame alignment invalid')
            if not np.array_equal(z['timestamps'],source_t[f]):raise ValueError('Native timestamp mismatch')
            if not np.array_equal(v,np.isfinite(z['points_camera']).all(axis=-1)):
                raise ValueError('Point validity mismatch')
            if np.any(np.isfinite(z['points_camera'][~v])):raise ValueError('Invalid points are not NaN')
            if not np.array_equal(pv,np.isfinite(p).all(axis=(1,2))):raise ValueError('Pose validity mismatch')
            if np.any(np.isfinite(p[~pv])):raise ValueError('Invalid poses are not NaN')
            if row['rigidity_candidate']!='rigid' and pv.any():raise ValueError('Nonrigid whole pose')
            if row['transparent_surface_ambiguous'] and pv.any():raise ValueError('Unverified container pose')
            if pv.any():
                good=p[pv];r=good[:,:3,:3]
                if not np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-4):raise ValueError('Nonorthogonal rotation')
                if not np.allclose(np.linalg.det(r),1,atol=1e-4):raise ValueError('Improper rotation')
                if not np.allclose(good[:,3],[0,0,0,1]):raise ValueError('Invalid homogeneous row')
                adjacent=pv[:-1]&pv[1:]
                if adjacent.any():
                    dt=np.diff(z['timestamps'])[adjacent]
                    velocity=np.linalg.norm(np.diff(p[:,:3,3],axis=0)[adjacent],axis=-1)/dt
                    if np.any(velocity>5):warnings.append({'tracklet_id':row['tracklet_id'],'camera_relative_speed_over_5m_s':int((velocity>5).sum())})
            total+=int(pv.sum())
    return {'sequence_id':uid,'status':'passed_structure','tracklets':len(data['records']),
            'pose_frames':total,'warnings':warnings,'accuracy':None,'training_allowed':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    done={};errors=[]
    while True:
        for receipt in sorted((a.root/'dense_tracklets_v4').glob('*/receipt.json')):
            uid=receipt.parent.name
            if uid in done:continue
            try:
                result=audit(a.root,uid);save(a.root/'geometry_checks'/(uid+'.json'),result);done[uid]=True
            except Exception as e:
                done[uid]=False;errors.append({'sequence_id':uid,'error':str(e)})
            save(a.root/'geometry_audit.json',{'checked':len(done),'passed':sum(done.values()),'errors':errors,
                 'independent_accuracy':None,'training_allowed':False})
        print(json.dumps({'checked':len(done),'errors':errors}),flush=True)
        if not a.watch or len(done)==300:break
        time.sleep(30)


if __name__=='__main__':main()
