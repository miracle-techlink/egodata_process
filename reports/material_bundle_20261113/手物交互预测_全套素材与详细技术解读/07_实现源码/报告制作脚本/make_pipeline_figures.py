"""Draw report figures from saved perception arrays and SSH-read RSI records."""
from pathlib import Path
import csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];ev=root/'evidence';out=root/'figures';p=ev/'perception'
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':9,'axes.unicode_minus':False,'pdf.fonttype':3})
colors=['#267b9d','#d79537','#6875ae','#64956d']
def save(fig,name):
 fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=190,bbox_inches='tight');plt.close(fig)
def clean(ax):
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)

trials=json.loads((p/'performance_trials.json').read_text())['trials']
order=['dexycb_human_full','dexycb_human_fast','dexycb_human_stream','dexycb_human_lite_stream_1','dexycb_human_lite_stream_2','dexycb_human_lite_stream_3']
rows=[next(t for t in trials if t['name']==name) for name in order]
labels=['full\n2次跟踪','full\n1次跟踪','full\n预取','lite\n复测1','lite\n复测2','lite\n复测3']
fig,axes=plt.subplots(1,2,figsize=(7.1,2.7))
for ax,key,title,ref in [(axes[0],'steady_processing_fps','稳态处理能力（FPS）',30),(axes[1],'steady_latency_p95_ms','帧内处理 P95（ms）',1000/30)]:
 vals=[t['timing'][key] for t in rows];ax.bar(range(6),vals,color=[colors[0]]*3+[colors[3]]*3)
 ax.axhline(ref,color='#ba4d42',ls='--',lw=1,label=f'预算线 {ref:.2f}');ax.set_title(title)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.2f}',ha='center',fontsize=7)
 ax.set_xticks(range(6),labels,fontsize=7);ax.set_ylim(0,max(vals)*1.25);ax.legend(frameon=False,fontsize=7);clean(ax)
fig.tight_layout(w_pad=2);save(fig,'perception_runtime')

z=np.load(p/'fp_poses.npz',allow_pickle=False);poses=z['poses'][0].astype(float);rot=poses[:,:3,:3]
orth=np.max(np.abs(rot.transpose(0,2,1)@rot-np.eye(3)),axis=(1,2));det=np.abs(np.linalg.det(rot)-1)
fig,axes=plt.subplots(1,3,figsize=(7.1,2.35))
for j,label in enumerate(['X','Y','Z']):axes[0].plot(poses[:,j,3],label=label,color=colors[j],lw=1)
axes[0].set_ylabel('估计平移（m）');axes[0].legend(ncol=3,fontsize=7,frameon=False)
axes[0].set_title('真实6D输出随帧变化',fontsize=9)
axes[1].plot(orth*1e6,label='正交误差',color=colors[0],lw=.7);axes[1].plot(det*1e6,label='行列式误差',color=colors[1],lw=.7)
axes[1].set_ylabel('矩阵数值偏差 $\times10^{-6}$');axes[1].legend(frameon=False,fontsize=7);axes[1].set_title('旋转矩阵数值检查',fontsize=9)
frames=list(csv.DictReader((p/'rgbd_frames.csv').open()))[1:]
for key,label,color in [('total_ms','帧内处理',colors[0]),('source_read_to_output_ms','读取至写出',colors[1])]:
 vals=[float(row[key]) for row in frames];axes[2].plot(range(1,len(vals)+1),vals,label=label,color=color,lw=1)
axes[2].set_yscale('log');axes[2].axhline(1000/30,ls='--',color='#ba4d42',lw=.7);axes[2].set_ylabel('延迟（ms，对数轴）');axes[2].legend(frameon=False,fontsize=7);axes[2].set_title('启动排队与稳态分别保留',fontsize=9)
for ax in axes:ax.set_xlabel('帧序号');clean(ax)
fig.tight_layout(w_pad=1.3);save(fig,'perception_validation')

shelf=json.loads((p/'shelf_acceptance.json').read_text());cable=json.loads((p/'cable_acceptance.json').read_text())
refine=json.loads((p/'refinement_report.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(7.1,2.55))
for i,d in enumerate([shelf,cable]):
 vals=[d['valid_hand_frames']['left'],d['valid_hand_frames']['right'],d['both_hands_valid_frames']]
 xs=np.arange(3)+(i-.5)*.32;axes[0].bar(xs,np.array(vals)/d['frames']*100,width=.3,color=colors[i],label=f"{'货架标签' if i==0 else '电缆'}：{d['frames']}帧")
 for x,v in zip(xs,vals):axes[0].text(x,v/d['frames']*100-9,str(v),ha='center',fontsize=7)
axes[0].set_xticks(range(3),['左手有效','右手有效','双手均有效']);axes[0].set_ylabel('来源姿态有效覆盖（%）');axes[0].set_ylim(0,122);axes[0].legend(frameon=False,fontsize=7,loc='upper left');axes[0].set_title('手部来源数据检查（不是检测精度）',fontsize=9)
status=refine['pixel_calibration_status_counts'];labs=['抽样一致','修正后检查','待复核','仍不一致'];keys=['consistent_on_sampled_frames','image_refined_with_heldout_checks','needs_review','inconsistent_on_sampled_frames'];vals=[status[k] for k in keys]
axes[1].bar(range(4),vals,color=[colors[0],colors[3],colors[1],'#ba4d42'])
for i,v in enumerate(vals):axes[1].text(i,v+4,str(v),ha='center')
axes[1].set_xticks(range(4),labs,fontsize=8);axes[1].set_ylabel('采集数');axes[1].set_ylim(0,320);axes[1].set_title('300段标定像素一致性复核',fontsize=9)
for ax in axes:clean(ax)
fig.tight_layout(w_pad=2);save(fig,'hand_calibration_coverage')

fig,axes=plt.subplots(1,3,figsize=(7.1,2.55));summary=[];curve=[]
for k,series in enumerate(['rsi_v5','rsi_auto','rsi_final']):
 history=[json.loads(s) for s in (ev/'remote/runs'/series/'history.jsonl').read_text().splitlines() if s.strip()]
 unique={r['name']:r for r in history};ys=[r.get('score',np.nan) for r in history];xs=np.arange(1,len(ys)+1)
 axes[k].plot(xs,ys,'o-',ms=3,lw=.8,label='本条候选评分',color=colors[k]);axes[k].step(xs,np.maximum.accumulate(ys),where='post',lw=1.3,label='迄今最高（非晋升）',color='#ba4d42')
 axes[k].set_title(f'{series}\n{len(history)}条记录 / {len(unique)}个候选',fontsize=9);axes[k].set_ylim(.45,1.06);axes[k].set_xlabel('history记录序号');clean(axes[k])
 if k==0:axes[k].set_ylabel('各系列自身 val micro-F1');axes[k].legend(fontsize=6.5,frameon=False,loc='lower right')
 summary.append({'series':series,'history_records':len(history),'unique_scored_candidates':len(unique),'successful_unique':sum(bool(x.get('ok')) for x in unique.values()),'best_observed_val_f1':max(ys)})
 for i,r in enumerate(history):curve.append({'series':series,'record_index':i+1,'stored_round':r.get('round'),'candidate':r['name'],'ok':r.get('ok'),'val_micro_f1':r.get('score')})
fig.tight_layout(w_pad=1.3);save(fig,'rsi_observed_history')
(ev/'rsi_series_summary.json').write_text(json.dumps(summary,indent=2))
with (ev/'rsi_observed_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
metrics={'pose_frames':len(poses),'orthogonality_max_abs':float(orth.max()),'determinant_deviation_max_abs':float(det.max()),'hand_sequences':[{'frames':d['frames'],'both':d['both_hands_valid_frames']} for d in [shelf,cable]],'calibration_counts':status}
(ev/'perception_metrics_derived.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print('Four evidence-based validation figures generated',metrics)
