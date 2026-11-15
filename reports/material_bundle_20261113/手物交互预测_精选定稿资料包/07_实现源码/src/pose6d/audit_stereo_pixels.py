"""Independent image-correspondence audit of supplied stereo calibration."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from src.annotation.object_observations import camera_matrices
from src.dataset.complete_egodemo import save,digest
from src.pose6d.stereo_reference import relative_camera


def pair_evidence(left,right,ks,relative,size=(960,728)):
    scale=np.array(size)/[left.shape[1],left.shape[0]]
    sift=cv2.SIFT_create(nfeatures=1600)
    k1,d1=sift.detectAndCompute(cv2.cvtColor(cv2.resize(left,size),cv2.COLOR_BGR2GRAY),None)
    k2,d2=sift.detectAndCompute(cv2.cvtColor(cv2.resize(right,size),cv2.COLOR_BGR2GRAY),None)
    if d1 is None or d2 is None:return {'status':'insufficient_matches','matches':0}
    matcher=cv2.BFMatcher();forward=matcher.knnMatch(d1,d2,k=2);reverse=matcher.knnMatch(d2,d1,k=2)
    back={a.queryIdx:a.trainIdx for a,b in reverse if a.distance<.7*b.distance}
    matches=[a for a,b in forward if a.distance<.7*b.distance and back.get(a.trainIdx)==a.queryIdx]
    if len(matches)<30:return {'status':'insufficient_matches','matches':len(matches)}
    a=np.array([k1[m.queryIdx].pt for m in matches])/scale;b=np.array([k2[m.trainIdx].pt for m in matches])/scale
    cv2.setRNGSeed(0)
    estimated,inliers=cv2.findFundamentalMat(a,b,cv2.USAC_MAGSAC,1.5,.999,10000)
    if inliers is None or inliers.sum()<30:return {'status':'insufficient_independent_inliers','matches':len(matches)}
    a,b=a[inliers.ravel()>0],b[inliers.ravel()>0]
    t=relative[:3,3];cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    fundamental=np.linalg.inv(ks[1]).T@cross@relative[:3,:3]@np.linalg.inv(ks[0])
    ah=np.c_[a,np.ones(len(a))];bh=np.c_[b,np.ones(len(b))];lb=ah@fundamental.T;la=bh@fundamental
    residual=np.abs(np.sum(bh*lb,axis=1))
    error=.5*residual*(1/np.linalg.norm(lb[:,:2],axis=1)+1/np.linalg.norm(la[:,:2],axis=1))
    hull=cv2.contourArea(cv2.convexHull(a.astype(np.float32)))/(left.shape[0]*left.shape[1])
    return {'status':'measured','matches':len(matches),'independent_inliers':len(a),
            'native_epipolar_median_px':float(np.median(error)),'native_epipolar_p90_px':float(np.percentile(error,90)),
            'native_below_2px_fraction':float(np.mean(error<2)),'left_coverage_fraction':hull,
            'left_uv':a.tolist(),'right_uv':b.tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sequence',action='append');a=p.parse_args()
    r=a.root;old=r.parent/'egodemo_joint_300_20261023';pool=r.parent/'egodemo_expansion_300_20261023'
    rows=json.loads((pool/'selection_frozen.json').read_text())['records'];calib=json.loads((old/'stereo_calibration.json').read_text())
    cv2.setNumThreads(1);results=[]
    for record in rows:
        uid=record['source_episode_uuid']
        if a.sequence and uid not in a.sequence:continue
        path=r/'pixel_calibration_audit'/(uid+'.json')
        if path.exists():results.append(json.loads(path.read_text()));continue
        try:
            meta=json.loads((old/'sequences'/uid/'metadata.json').read_text());source=np.load(pool/'staging'/uid/'sequence.npz')
            cameras=[camera_matrices(source,s) for s in ['left','right']]
            ks=[np.array([[calib[uid][s][0],0,calib[uid][s][2]],[0,calib[uid][s][1],calib[uid][s][3]],[0,0,1.]]) for s in ['left','right']]
            caps=[cv2.VideoCapture(meta['video_path']),cv2.VideoCapture(str(old/'stereo'/(uid+'_right.mp4')))];frames=[]
            try:
                for f in np.linspace(0,meta['frame_count']-1,5).astype(int)[1:-1]:
                    images=[]
                    for cap in caps:
                        cap.set(1,int(f));ok,im=cap.read()
                        if not ok:raise ValueError('Cannot decode stereo frame')
                        images.append(im)
                    frames.append({'frame_index':int(f),**pair_evidence(*images,ks,relative_camera(cameras[0][f],cameras[1][f]))})
            finally:
                for cap in caps:cap.release()
                source.close()
            measured=[x for x in frames if x['status']=='measured' and x['left_coverage_fraction']>.10]
            medians=[x['native_epipolar_median_px'] for x in measured]
            # This flags an unsupported calibration; it is not an automatic
            # replacement calibration or an annotation acceptance criterion.
            status='consistent_on_sampled_frames' if len(medians)>=2 and max(medians)<2 else ('inconsistent_on_sampled_frames' if len(medians)>=2 and np.median(medians)>4 else 'needs_review')
            result={'sequence_id':uid,'status':status,'frames':frames,'source_calibration_sha256':meta['calibration_sha256'],
                    'method':'mutual_SIFT_independent_MAGSAC_matches_vs_supplied_F','sampled_only':True,'independent_pose_accuracy':None}
        except Exception as e:result={'sequence_id':uid,'status':'error','error':str(e)}
        save(path,result);results.append(result)
        print(json.dumps({'done':len(results),'id':uid,'status':result['status']},ensure_ascii=False),flush=True)
    summary={'sequences':len(results),'statuses':{s:sum(r['status']==s for r in results) for s in sorted(set(r['status'] for r in results))},
             'records':[{k:v for k,v in r.items() if k!='frames'} for r in results],'sampled_only':True,'training_allowed':False}
    save(r/'pixel_calibration_audit_summary.json',summary)

if __name__=='__main__':main()
