"""Replot aggregate hand coverage and final calibration status from bundled records."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--font',default='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname=a.font)
plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
e=root/'06_实验依据/报告证据/perception'
hands=[json.loads((e/n).read_text()) for n in ['shelf_acceptance.json','cable_acceptance.json']]
total=sum(x['frames'] for x in hands)
counts=[sum(x['valid_hand_frames'][k] for x in hands) for k in ['left','right']]+[sum(x['both_hands_valid_frames'] for x in hands)]
ref=json.loads((e/'refinement_report.json').read_text());status=ref['pixel_calibration_status_counts']
keys=['consistent_on_sampled_frames','image_refined_with_heldout_checks','needs_review','inconsistent_on_sampled_frames']
cal=[status[k] for k in keys];assert sum(cal)==ref['sequences']==300
fig,axes=plt.subplots(1,2,figsize=(12.4,3.7),layout='constrained')
axes[0].bar(['左手有效','右手有效','双手同时有效'],np.array(counts)/total*100,color=['#2383a5','#268d74','#df9527'])
for i,n in enumerate(counts):axes[0].text(i,n/total*100+2,f'{n}/{total}\n{n/total:.2%}',ha='center',va='bottom',fontsize=10)
axes[0].set_ylim(0,122);axes[0].set_yticks([0,25,50,75,100]);axes[0].set_ylabel('来源姿态有效比例 / %');axes[0].set_title('手部来源覆盖：两采集合并')
axes[1].bar(['直接一致','修正后检查','待复核','不一致'],np.array(cal)/300*100,color=['#2383a5','#268d74','#df9527','#b75b62'])
for i,n in enumerate(cal):axes[1].text(i,n/300*100+2,f'{n}/300\n{n/300:.2%}',ha='center',va='bottom',fontsize=10)
axes[1].set_ylim(0,118);axes[1].set_yticks([0,25,50,75,100]);axes[1].set_ylabel('采集比例 / %');axes[1].set_title('像素标定复核：最终汇总状态')
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.savefig(a.output/'hand_calibration_coverage.pdf');fig.savefig(a.output/'hand_calibration_coverage.png',dpi=180)
print(json.dumps({'total_frames':total,'hand_counts':counts,'calibration_counts':cal,'accuracy_measured':False}))
