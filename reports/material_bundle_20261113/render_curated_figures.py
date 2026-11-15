from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
import trimesh

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
OUT=WORK/'手物交互预测_精选定稿资料包'
FIG=OUT/'04_图片与图表'
E=OUT/'06_实验依据/同帧点云与数值验证';E.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#bc7d00','#087cb0','#24824a','#bb419b']
names=['红盖圆瓶','蓝色纸盒','椭圆罐','粉色包装盒']
z=np.load(OUT/'06_实验依据/Demo推理与制作/demo_arrays.npz')
f=223
m=json.loads((ROOT/'runs/full_hocap_20261022/scenes/subject_7_20231022_192832_105322251564/scene_manifest.json').read_text())
K=np.array(m['intrinsics'])
capture=ROOT/'runs/human_dataset_v1_20261022/acquisition/captures_primary/subject_7/20231022_192832/105322251564'
rgbpath=capture/f'105322251564-{f:06d}.color.jpg';depthpath=capture/f'105322251564-{f:06d}.depth.png'
rgb=np.array(Image.open(rgbpath));depth=np.array(Image.open(depthpath)).astype(float)*m['depth_scale']
y,x=np.mgrid[:480:3,:640:3];d=depth[::3,::3]
valid=(d>.15)&(d<1.5)
points=np.stack([(x-K[0,2])*d/K[0,0],(y-K[1,2])*d/K[1,1],d],axis=-1)[valid]
point_colors=rgb[::3,::3][valid]/255.
verts=[];boxpoints=[];meshmeta=[]
for i,obj in enumerate(m['objects']):
    path=Path(obj['mesh']['path']);mesh=trimesh.load(path,process=False,force='mesh')
    v=np.asarray(mesh.vertices);v=v[::max(1,len(v)//2000)]
    T=z['poses'][f,i];verts.append(v@T[:3,:3].T+T[:3,3])
    lo,hi=mesh.bounds
    corners=np.array([[a,b,c] for a in [lo[0],hi[0]] for b in [lo[1],hi[1]] for c in [lo[2],hi[2]]])
    boxpoints.append(corners@T[:3,:3].T+T[:3,3])
    meshmeta.append({'id':obj['object_id'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
np.savez_compressed(E/'frame223_geometry.npz',rgb=rgb,depth_m=depth.astype('float32'),K=K,cloud_xyz=points,cloud_rgb=point_colors,
                    estimated_poses=z['poses'][f],boxpoints_camera=np.array(boxpoints),**{f'mesh_camera_{i}':v for i,v in enumerate(verts)})
fig=plt.figure(figsize=(12.4,8.0),layout='constrained')
ax=fig.add_subplot(221);ax.imshow(rgb)
edges=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in [1,2,4]]
for n,pts in enumerate(boxpoints):
    uv=pts@K.T;uv=uv[:,:2]/uv[:,2:]
    for a,b in edges:ax.plot(uv[[a,b],0],uv[[a,b],1],c=colors[n],lw=1.2)
    xx,yy=z['boxes'][f,n,:2];ax.text(xx,max(yy-10,10),names[n],fontsize=9,color='white',bbox={'facecolor':colors[n],'edgecolor':'none','pad':2})
ax.set_title('(a) 同帧 RGB + 已估计物体框');ax.set_xlim(0,640);ax.set_ylim(480,0);ax.axis('off')
ax=fig.add_subplot(222);im=ax.imshow(np.ma.masked_where(depth<=0,depth),cmap='viridis',vmin=.4,vmax=1.2)
ax.set_title('(b) 实测深度（无效像素留空）');ax.axis('off');fig.colorbar(im,ax=ax,shrink=.72,label='深度 / m')
def cloud_axis(ax, close=False):
    if close:
        vv=np.concatenate(verts);lo=vv.min(0)-.06;hi=vv.max(0)+.06
    else:lo=np.array([-.5,-.5,.3]);hi=np.array([.5,.5,1.3])
    visible=np.all((points>=lo)&(points<=hi),axis=1)
    q=points[visible]
    ax.scatter(q[:,0],q[:,2],-q[:,1],c=point_colors[visible],s=.65,alpha=.65,rasterized=True)
    for i,v in enumerate(verts):ax.scatter(v[:,0],v[:,2],-v[:,1],c=colors[i],s=1.4,alpha=.8,rasterized=True,label=names[i])
    ax.set_xlabel('相机 X / m');ax.set_ylabel('相机 Z / m');ax.set_zlabel('−相机 Y / m')
    if close:
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[2],hi[2]);ax.set_zlim(-hi[1],-lo[1]);ax.view_init(35,-78)
    else:
        ax.set_xlim(-.5,.5);ax.set_ylim(.3,1.3);ax.set_zlim(-.5,.5);ax.view_init(35,-78)
    ax.set_box_aspect((1,1,.8));ax.tick_params(labelsize=8)
ax=fig.add_subplot(223,projection='3d');cloud_axis(ax);ax.set_title('(c) 深度反投影 + 估计模型顶点')
ax=fig.add_subplot(224,projection='3d');cloud_axis(ax,True);ax.set_title('(d) 四物体局部视图（仅裁剪显示范围）')
ax.legend(loc='upper left',fontsize=8,markerscale=3,framealpha=.8)
fig.suptitle('HO-Cap 同源真人场景 · 第 223 / 445 帧',fontsize=15)
fig.savefig(FIG/'hocap_same_scene_pointcloud.png',dpi=190,bbox_inches='tight',pad_inches=.15);plt.close(fig)

R=z['poses'][:,:,:3,:3].astype('float64')
orth=np.max(np.abs(np.swapaxes(R,-1,-2)@R-np.eye(3)),axis=(-1,-2))
det=np.abs(np.linalg.det(R)-1)
coverage=np.r_[z['object_valid'].sum(0),z['hand_valid'].sum(0)]
fig,axs=plt.subplots(1,3,figsize=(13.4,3.7),layout='constrained')
labels=names+['左手','右手']
axs[0].barh(labels,coverage/446,color=colors+['#306ea2','#768591'])
for i,v in enumerate(coverage):axs[0].text(min(v/446+.02,1.02),i,f'{v}/446',va='center',fontsize=8)
axs[0].set_xlim(0,1.30);axs[0].set_xticks([0,.5,1]);axs[0].set_xlabel('有效输出比例');axs[0].set_title('当前序列输出覆盖')
for i,label in enumerate(['x','y','z']):axs[1].plot(z['poses'][:,0,i,3],label=label,lw=1.3)
axs[1].set_title('红盖圆瓶：相机系平移');axs[1].set_xlabel('帧号');axs[1].set_ylabel('平移 / m');axs[1].legend(ncol=3,fontsize=8)
axs[2].plot(np.max(orth,axis=1)*1e6,label=r'max $|R^T R-I|$',lw=1)
axs[2].plot(np.max(det,axis=1)*1e6,label=r'max $|\det R-1|$',lw=1)
axs[2].set_title('四物体刚体矩阵数值检查');axs[2].set_xlabel('帧号');axs[2].set_ylabel(r'偏差 / $10^{-6}$');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(axis='x' if ax is axs[0] else 'y',alpha=.15)
fig.savefig(FIG/'hocap_perception_validation.pdf');fig.savefig(FIG/'hocap_perception_validation.png',dpi=180);plt.close(fig)
metrics={'frames':446,'object_valid_frames':coverage[:4].tolist(),'hand_valid_frames_left_right':coverage[4:].tolist(),
         'max_rotation_orthogonality_element_error':float(orth.max()),'max_abs_det_minus_one':float(det.max()),
         'scope':'output coverage and numerical consistency only; no independent pose accuracy assessed',
         'pointcloud_frame':223,'depth_scale':m['depth_scale'],'K':K.tolist(),'pointcloud_samples':len(points),
         'depth_filter_m':[.15,1.5],'display_sampling_stride_px':3,'mesh_alignment':'saved estimated pose only; no reference alignment',
         'source_rgb_sha256':hashlib.sha256(rgbpath.read_bytes()).hexdigest(),'source_depth_sha256':hashlib.sha256(depthpath.read_bytes()).hexdigest(),'meshes':meshmeta}
(E/'figure_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print(json.dumps(metrics,ensure_ascii=False))
