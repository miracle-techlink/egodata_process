"""Build publication figures from frozen local evidence; no synthetic results."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[1]
DATA = PROJECT / 'runs/egodemo_training_v5_300_20261024'
BACKUP = Path('/home/nvidia/backup_ego_server/raw/EgoDemo')
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False,
                     'pdf.fonttype': 3, 'font.size': 9})
COLORS = ['#267b9d', '#e29b39', '#6875ae']
index = json.loads((DATA/'training_index.json').read_text())
coverage = json.loads((DATA/'model_checks/real_graph_target_coverage.json').read_text())['splits']
names = {'2a797a4e': '纸盒扫码', '2eb9dcc4': '水壶操作', '99190763': '订书机收纳',
         '9be303ef': '封口贴操作', 'acc3c38d': '手柄归位'}
samples = []
videos = list(BACKUP.glob('EgoProStandard*/lerobot/*/*/videos/observation.images.head_left/chunk-000/file-000.mp4'))
for seq in index['sequences']:
    uid = seq['sequence_id']
    matches = [p for p in videos if uid in str(p)]
    if len(matches) != 1:
        raise ValueError((uid, matches))
    video = matches[0]
    pointer = video.read_text().strip().splitlines()
    if not pointer[0].startswith('version https://git-lfs.github.com'):
        raise ValueError('Expected local pointer record; inspect actual video first')
    frame = 120
    if seq['split'] == 'val':
        source = DATA/'seed_evidence'/uid/'000120.png'
    elif seq['split'] == 'test':
        frame = 156
        source = next((DATA/'dense_evidence'/uid).rglob('000156.png'))
    else:
        source = DATA/'navigation/root_seed_frames'/f'{uid}_000120.jpg'
    bgr = cv2.imread(str(source))
    if bgr is None:
        raise ValueError(f'Cannot read actual saved evidence: {source}')
    filename = f'{uid[:8]}_f{frame:06d}{source.suffix}'
    import shutil
    shutil.copy2(source, OUT/'figures'/filename)
    samples.append({'sequence_id':uid, 'split':seq['split'], 'name':names[uid[:8]],
                    'frame':frame, 'frame_count':seq['frame_count'], 'video':str(video),
                    'video_local_status':'git_lfs_pointer_only', 'video_lfs_oid':pointer[1],
                    'figure':'figures/'+filename,
                    'source_image':str(source),
                    'source_image_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})

fig, axes = plt.subplots(1,5,figsize=(15,2.85))
for ax,s in zip(axes,samples):
    ax.imshow(cv2.cvtColor(cv2.imread(str(OUT/s['figure'])),cv2.COLOR_BGR2RGB))
    ax.set_title(s['name']+' · '+s['split'],fontsize=22)
    ax.set_xlabel(f"{s['sequence_id'][:8]} · 原生帧 {s['frame']}",fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
fig.subplots_adjust(left=.005,right=.995,top=.85,bottom=.12,wspace=.06)
fig.savefig(OUT/'figures/dataset_examples.pdf',bbox_inches='tight',dpi=220)
fig.savefig(OUT/'figures/dataset_examples.png',bbox_inches='tight',dpi=140)
plt.close(fig)

# Existing training numbers are transcribed from the dated local training report.
results = [
    {'model':'Mamba','val_loss':.6293,'val_micro':.7346,'val_seen':.9151,'test_loss':.6634,'test_micro':.4761,'test_seen':.6519,'test_unseen':.4505},
    {'model':'LSTM','val_loss':.6070,'val_micro':.8027,'val_seen':.9349,'test_loss':.6474,'test_micro':.4801,'test_seen':.6565,'test_unseen':.7226},
    {'model':'Transformer','val_loss':.7064,'val_micro':.6442,'val_seen':.8788,'test_loss':.6540,'test_micro':.5609,'test_seen':.7112,'test_unseen':.4708}]
with (OUT/'evidence/results_transcribed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader();w.writerows(results)
fig,axes=plt.subplots(2,1,figsize=(3.55,4.55),sharey=True)
baselines={}
for ax,split,title in zip(axes,['val','test'],['验证集：1 段采集 / 366 窗口','测试集：1 段采集 / 280 窗口']):
    p=coverage[split]['positive_cells']; n=coverage[split]['negative_cells']
    baseline=2*p/(2*p+n); baselines[split]={'positive':p,'negative':n,'all_positive_micro_f1':baseline}
    vals=[r[split+'_micro'] for r in results]
    ax.bar(range(3),vals,color=COLORS,width=.6)
    ax.axhline(baseline,color='#ab4b3b',ls='--',lw=1.5,label=f'全正参照 {baseline:.4f}')
    for i,v in enumerate(vals):ax.text(i,v+.025,f'{v:.4f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),[r['model'] for r in results]);ax.set_title(title,fontsize=9)
    ax.set_ylim(0,1.04);ax.legend(loc='upper right',fontsize=8,frameon=False)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('micro-F1'); axes[1].set_ylabel('micro-F1')
fig.tight_layout();fig.savefig(OUT/'figures/backbone_results.pdf',bbox_inches='tight');fig.savefig(OUT/'figures/backbone_results.png',dpi=150,bbox_inches='tight');plt.close(fig)
(OUT/'evidence/derived_baselines.json').write_text(json.dumps({'formula':'2P/(2P+N)','source':'real_graph_target_coverage.json','values':baselines},indent=2))

fig,axes=plt.subplots(2,1,figsize=(3.55,4.15))
splits=['train','val','test'];x=np.arange(3)
pos=np.array([coverage[s]['positive_cells'] for s in splits]); neg=np.array([coverage[s]['negative_cells'] for s in splits])
axes[0].bar(x,pos,color=COLORS[0],label='明确正例');axes[0].bar(x,neg,bottom=pos,color='#b8c8cf',label='明确负例')
for i in range(3):
    axes[0].text(i,pos[i]/2,str(pos[i]),ha='center',color='white');axes[0].text(i,pos[i]+neg[i]/2,str(neg[i]),ha='center')
axes[0].set_xticks(x,splits);axes[0].set_ylabel('监督单元数');axes[0].legend(fontsize=9)
layers=['operation','relation','transition','cooperation'];labels=['操作','关系','转换','协作']
for i,s in enumerate(splits):
    vals=[coverage[s]['valid_cells_by_layer'].get(k,0) for k in layers]
    axes[1].bar(np.arange(4)+(i-1)*.25,vals,width=.25,color=COLORS[i],label=s)
axes[1].set_xticks(range(4),labels);axes[1].set_ylabel('已知标签单元数');axes[1].legend(fontsize=9)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'figures/supervision.pdf',bbox_inches='tight');plt.close(fig)

# Real test-sequence state annotation timeline, explicitly NOT model predictions.
uid='9be303ef-7491-43be-9915-ddfd51fde7b3'
review=json.loads((DATA/'dense_reviews'/f'{uid}.json').read_text())
rows=[('left_hand','visual_red_roll','contact','左手 → 贴卷：接触'),
      ('right_hand','visual_red_strip','contact','右手 → 贴片：接触'),
      ('left_hand','visual_envelope','contact','左手 → 信封：接触')]
fig,ax=plt.subplots(figsize=(3.55,2.0))
for i,(subject,obj,typ,label) in enumerate(rows):
    ax.broken_barh([(0,405)],(i-.3,.6),facecolors='#e9ecef')
    for seg in review['state_segments']:
        if seg.get('subject')==subject and seg.get('object_id')==obj and seg['type']==typ and seg['layer']=='relation':
            ax.broken_barh([(seg['start_frame'],seg['end_frame']-seg['start_frame'])],(i-.3,.6),facecolors=('#267b9d' if seg['value'] else '#e29b39'))
ax.set_yticks(range(3),[r[3].replace('：接触','') for r in rows]);ax.set_xlim(0,405);ax.set_xlabel('标注版原生帧号')
ax.set_title('蓝=正例，橙=负例，灰=未知',fontsize=9)
ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'figures/annotation_timeline.pdf',bbox_inches='tight');plt.close(fig)

(OUT/'evidence/figure_sources.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2))
print(json.dumps({'figures':4,'raw_frames':len(samples),'baselines':baselines},ensure_ascii=False))
