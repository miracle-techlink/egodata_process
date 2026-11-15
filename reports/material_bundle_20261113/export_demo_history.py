"""Freeze the exact historical tensors used in the published demo; no training."""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mamba.dataset_v5 import build_features,attach_pose_features
D=ROOT/'reports/demo_update_20261113'
OUT=Path(__file__).resolve().parent/'手物交互预测_全套素材与详细技术解读/06_实验依据/Demo推理与制作'
with np.load(D/'demo_arrays.npz') as data:a={k:data[k] for k in data.files}
with np.load(D/'hocap_sequence.npz') as data:seq={k:data[k] for k in data.files}
T=len(a['hand_uv']);N=4;stats=np.zeros((T,N,6),np.float32)
folder=ROOT/'runs/human_dataset_v1_20261022/acquisition/captures_primary/subject_7/20231022_192832/105322251564'
for f in range(T):
 im=cv2.imread(str(folder/f'105322251564-{f:06d}.color.jpg'));assert im is not None
 for n in range(N):
  if not a['object_valid'][f,n]:continue
  x0,y0,x1,y1=np.round(a['boxes'][f,n]).astype(int);crop=im[y0:y1,x0:x1,::-1].reshape(-1,3).astype(np.float32)/255
  stats[f,n]=np.r_[crop.mean(0),crop.std(0)]
s={'timestamps':np.arange(T)/30,'hand_valid':np.zeros((T,2),bool),'hands_world':np.full((T,2,21,3),np.nan,np.float32),'camera_to_world':np.tile(np.eye(4),(T,1,1))}
obs={'hand_uv_native':a['hand_uv'],'hand_projection_valid':a['hand_valid'],'boxes_xyxy_native':a['boxes'],'rgb_stats':stats,'source_code':np.zeros((T,N),np.int8)};obs['source_code'][0]=1
fs=[];vs=[];ks=[]
for end in range(32,T+1):
 b=build_features(s,obs,a['object_valid'],end-32,end,list(range(4)),(640,480))
 feat=attach_pose_features(b['entity_features'],(a['poses'],a['object_valid']),end-32,end,list(range(4)))
 assert np.array_equal(feat[-1],a['features_last'][end-1]),end
 fs.append(feat);vs.append(b['entity_valid']);ks.append(b['entity_kind'])
np.savez_compressed(OUT/'history_inputs.npz',entity_features=np.asarray(fs),entity_valid=np.asarray(vs),entity_kind=np.asarray(ks),anchor_frames=np.arange(31,T))
print('Frozen 415 exact 32-frame inputs; every saved last-frame feature matches original.')
