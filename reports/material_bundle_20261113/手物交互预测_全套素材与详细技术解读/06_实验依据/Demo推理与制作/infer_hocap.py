"""Inference-only, cross-dataset visualization; does not modify training assets."""
import sys, json, hashlib, itertools
from pathlib import Path
import numpy as np
import cv2
import torch
import trimesh

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mamba.dataset_v5 import build_features, attach_pose_features
from src.mamba.train_v5 import build_v5_model

OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/full_hocap_20261022'
seq=np.load(OUT/'hocap_sequence.npz')
manifest=json.loads((BASE/'scenes/subject_7_20231022_192832_105322251564/scene_manifest.json').read_text())
checkpoint=OUT/'lstm_final_private.pt'
ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
index=json.loads((OUT/'model_vocabulary.json').read_text())
torch.set_num_threads(2)
model=build_v5_model(ck['config']).eval();model.load_state_dict(ck['model'])
T=len(seq['frame_ids']); N=len(seq['object_ids']); K=np.asarray(manifest['intrinsics'])
poses=seq['object_poses'].transpose(1,0,2,3)
boxes=np.full((T,N,4),np.nan,np.float32);rgb_stats=np.zeros((T,N,6),np.float32)
valid=seq['object_valid'].T.copy() & seq['object_projection_visible'].T
corners=[]
for obj in manifest['objects']:
 mesh=trimesh.load(obj['mesh']['path'],force='mesh')
 bounds=mesh.bounds
 corners.append(np.array(list(itertools.product(*bounds.T))))
writer=cv2.VideoWriter(str(OUT/'raw_rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
for f,frame in enumerate(manifest['frames']):
 im=cv2.imread(frame['rgb']['path']);assert im is not None
 writer.write(im)
 for n in range(N):
  p=corners[n]@poses[f,n,:3,:3].T+poses[f,n,:3,3]
  if not valid[f,n] or np.any(p[:,2]<=0):valid[f,n]=False;continue
  uv=p@K.T;uv=uv[:,:2]/uv[:,2:]
  box=np.r_[uv.min(0),uv.max(0)];box=np.clip(box,[0,0,0,0],[640,480,640,480]);boxes[f,n]=box
  x0,y0,x1,y1=np.round(box).astype(int);crop=im[y0:y1,x0:x1,::-1].reshape(-1,3).astype(np.float32)/255
  if not len(crop):valid[f,n]=False;continue
  rgb_stats[f,n]=np.r_[crop.mean(0),crop.std(0)]
writer.release()
# Anatomical metric hand joints are unavailable: retain only measured 2D landmarks.
# HO-Cap source timestamps are unknown. 30 Hz is explicitly an interface assumption.
sequence={'timestamps':np.arange(T)/30,'hand_valid':np.zeros((T,2),bool),
 'hands_world':np.full((T,2,21,3),np.nan,np.float32),
 'camera_to_world':np.tile(np.eye(4),(T,1,1))}
hand_uv=np.stack([seq['left_hand'][...,:2],seq['right_hand'][...,:2]],axis=1)*[640,480]
obs={'hand_uv_native':hand_uv,'hand_projection_valid':seq['hand_valid'],
 'boxes_xyxy_native':boxes,'rgb_stats':rgb_stats,'source_code':np.zeros((T,N),np.int8)}
obs['source_code'][0]=1
prob=np.full((T,11,11,66),np.nan,np.float32);masks=np.zeros((T,11,11),bool)
last_features=np.zeros((T,11,132),np.float32);seen=np.zeros((T,11),bool)
with torch.inference_mode():
 for end in range(32,T+1):
  b=build_features(sequence,obs,valid,end-32,end,list(range(N)),(640,480))
  b['entity_features']=attach_pose_features(b['entity_features'],(poses,valid),end-32,end,list(range(N)))
  batch={k:torch.from_numpy(b[k]).unsqueeze(0) for k in ['entity_features','entity_valid','entity_kind']}
  out=model(batch)
  prob[end-1]=out['graph_logits'].sigmoid()[0].numpy();masks[end-1]=out['graph_prediction_mask'][0].numpy()
  last_features[end-1]=b['entity_features'][-1];seen[end-1]=b['entity_valid'].any(0)
np.savez_compressed(OUT/'demo_arrays.npz',probabilities=prob,prediction_mask=masks,
 hand_uv=hand_uv,hand_valid=seq['hand_valid'],boxes=boxes,object_valid=valid,
 poses=poses,features_last=last_features,entity_seen=seen)
meta={'dataset':'HO-Cap','source':'https://irvlutd.github.io/HOCap/',
 'license':'CC BY 4.0','frames':T,'fps_playback':30,'source_timestamps_known':False,
 'feature_timebase':'Assumed 30 Hz for interface demonstration; not measured source time',
 'model':'lstm_final','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
 'checkpoint_epoch':ck['epoch'],'model_config':ck['config']['model'],
 'history_frames':32,'training_horizon_seconds':1.0,'demo_horizon_seconds_verified':False,
 'metric_hand_3d_used':False,'object_pose_frame':'OpenCV camera coordinates; demo adapter only',
 'bbox_source':'Projected mesh bounds from saved FoundationPose estimates',
 'rgb_stats_source':'Current RGB crop mean and standard deviation',
 'prediction_source':'Actual checkpoint inference, no retraining, no GT labels',
 'event_types':index['event_types'],'object_ids':seq['object_ids'].tolist(),
 'threshold':0.5,'accuracy':None,'claimed_test_f1':None,
 'training_performed':False,'scope':'Cross-dataset pipeline interface demonstration, not the reported 0.93 experiment',
 'perception_npz_sha256':hashlib.sha256((OUT/'hocap_sequence.npz').read_bytes()).hexdigest()}
(OUT/'demo_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(json.dumps({'output':str(OUT),'frames':T,'predicted_frames':T-31,'model':meta['model'],'checkpoint_epoch':ck['epoch']}))
