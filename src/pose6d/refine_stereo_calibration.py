"""Recover a candidate stereo extrinsic using held-out image correspondences.

Original calibration is immutable. Metric scale comes from its physical baseline;
this is an image-based estimate, not an independently measured camera pose.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from src.dataset.complete_egodemo import save,digest
from src.annotation.object_observations import camera_matrices
from src.pose6d.stereo_reference import relative_camera,triangulate
from src.pose6d.audit_stereo_pixels import pair_evidence


def fundamental(ks,rel):
 t=rel[:3,3];cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
 return np.linalg.inv(ks[1]).T@cross@rel[:3,:3]@np.linalg.inv(ks[0])


def residual(a,b,f):
 ah=np.c_[a,np.ones(len(a))];bh=np.c_[b,np.ones(len(b))];lb=ah@f.T;la=bh@f
 return np.sum(bh*lb,axis=1)/np.sqrt(np.sum(lb[:,:2]**2,axis=1)+np.sum(la[:,:2]**2,axis=1))


def refine(root,uid):
 old=root.parent/'egodemo_joint_300_20261023';pool=root.parent/'egodemo_expansion_300_20261023'
 path=root/'pixel_calibration_audit'/(uid+'.json');evidence=json.loads(path.read_text());frames=evidence['frames']
 if len(frames)!=3 or any(f['status']!='measured' or f['independent_inliers']<80 or f['left_coverage_fraction']<.2 for f in frames):return {'sequence_id':uid,'status':'insufficient_evidence'}
 with np.load(pool/'staging'/uid/'sequence.npz') as s:
  cams=[camera_matrices(s,side) for side in ['left','right']]
 rel=relative_camera(cams[0][frames[0]['frame_index']],cams[1][frames[0]['frame_index']])
 c=json.loads((old/'stereo_calibration.json').read_text())[uid]
 ks=[np.array([[c[s][0],0,c[s][2]],[0,c[s][1],c[s][3]],[0,0,1.]]) for s in ['left','right']]
 a=np.concatenate([f['left_uv'] for f in frames[:2]]);b=np.concatenate([f['right_uv'] for f in frames[:2]])
 def transform(x):
  result=rel.copy();result[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@rel[:3,:3]
  result[:3,3]=Rotation.from_rotvec([0,x[3],x[4]]).apply(rel[:3,3]);return result
 fit=least_squares(lambda x:residual(a,b,fundamental(ks,transform(x))),np.zeros(5),
                   bounds=([-0.1]*3+[-0.15]*2,[0.1]*3+[0.15]*2),loss='soft_l1',f_scale=.7,max_nfev=200)
 candidate=transform(fit.x);checks=[]
 for j,frame in enumerate(frames):
  aa=np.array(frame['left_uv']);bb=np.array(frame['right_uv']);err=np.abs(residual(aa,bb,fundamental(ks,candidate)))
  xyz,valid,_=triangulate(aa,bb,*ks,candidate)
  checks.append({'frame_index':frame['frame_index'],'split':'fit' if j<2 else 'held_out',
                 'sampson_median_px':float(np.median(err)),'sampson_p90_px':float(np.percentile(err,90)),
                 'triangulation_accepted_fraction':float(valid.mean()),'matches':len(aa)})
 passed=fit.success and all(c['sampson_median_px']<.8 and c['sampson_p90_px']<2 and c['triangulation_accepted_fraction']>.8 for c in checks)
 extra=[]
 if passed:
  meta=json.loads((old/'sequences'/uid/'metadata.json').read_text())
  caps=[cv2.VideoCapture(meta['video_path']),cv2.VideoCapture(str(old/'stereo'/(uid+'_right.mp4')))];cv2.setNumThreads(1)
  try:
   for frame in np.rint(np.array([.10,.45,.90])*(meta['frame_count']-1)).astype(int):
    images=[]
    for cap in caps:
     cap.set(1,int(frame));ok,im=cap.read()
     if not ok:raise ValueError('Cannot decode independent validation frame')
     images.append(im)
    measured=pair_evidence(*images,ks,candidate)
    measured['frame_index']=int(frame);measured['split']='additional_held_out'
    extra.append(measured)
  finally:
   for cap in caps:cap.release()
  passed=all(c['status']=='measured' and c['independent_inliers']>=80 and c['left_coverage_fraction']>.15
             and c['native_epipolar_median_px']<1.5 and c['native_epipolar_p90_px']<3 for c in extra)
 result={'sequence_id':uid,'status':'candidate_passed_heldout_pixel_check' if passed else 'recovery_rejected',
         'right_from_left':candidate.tolist(),'original_right_from_left':rel.tolist(),'parameters_delta':fit.x.tolist(),
         'baseline_m':float(np.linalg.norm(rel[:3,3])),'baseline_source':'original_supplied_metric_camera_rig',
         'source_calibration_sha256':evidence['source_calibration_sha256'],'pixel_evidence_sha256':digest(path),
         'checks':checks,'fit_frames':[frames[0]['frame_index'],frames[1]['frame_index']],
         'additional_heldout_checks':extra,'usable_for_candidate_reconstruction':bool(passed),
         'heldout_frames':[frames[2]['frame_index']],'independent_pose_accuracy':None,'review_status':'pending',
         'note_zh':'只修正右相机相对左相机的估计外参，基线长度沿用源标定；第三采样帧未参与拟合。不改官方文件，不代表独立位姿精度。'}
 save(root/'calibration_recovery_trials'/(uid+'.json'),result);return result


def load_candidate_calibration(root,uid,source_sha):
 path=root/'calibration_recovery_trials'/(uid+'.json')
 if not path.exists():return None,None
 result=json.loads(path.read_text())
 if not result.get('usable_for_candidate_reconstruction'):return None,None
 if result['source_calibration_sha256']!=source_sha:raise ValueError('Calibration recovery source mismatch')
 if result['pixel_evidence_sha256']!=digest(root/'pixel_calibration_audit'/(uid+'.json')):raise ValueError('Calibration recovery evidence changed')
 transform=np.array(result['right_from_left']);assert transform.shape==(4,4) and np.isfinite(transform).all()
 assert len(result['additional_heldout_checks'])==3
 return transform,digest(path)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sequence',action='append',required=True);args=p.parse_args()
 for uid in args.sequence:
  result=refine(args.root,uid);print(json.dumps({k:v for k,v in result.items() if k not in ['right_from_left','original_right_from_left','additional_heldout_checks']},ensure_ascii=False))
