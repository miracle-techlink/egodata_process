"""Later-run evidence plots. Different dataset versions are not paired trials."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'Noto Sans CJK SC','pdf.fonttype':3,'font.size':9,'axes.unicode_minus':False})
rows=list(csv.DictReader((root/'evidence/expanded_benchmark_20261026.csv').open()))
fig,ax=plt.subplots(figsize=(3.55,2.7))
x=np.arange(3)
for offset,metric,color,label in [(-.18,'val_f1','#267b9d','验证'),(.18,'test_f1','#d79537','测试')]:
 vals=[float(r[metric]) for r in rows]
 ax.bar(x+offset,vals,width=.34,color=color,label=label)
 for i,v in enumerate(vals):ax.text(i+offset,v+.012,f'{v:.3f}',ha='center',fontsize=8)
ax.set_xticks(x,[r['model'] for r in rows]);ax.set_ylim(0,1.14);ax.set_ylabel('micro-F1')
ax.set_title('扩展冻结版：2026-10-26 评估输出',fontsize=9)
ax.legend(ncol=2,loc='upper left',frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(root/'figures/expanded_results.pdf',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,2.2))
labels=['初始 V5\n10月24日','自动复核试点\n10月25日','扩展冻结版\n10月26日']
values={'Mamba':[.4761,.761,.905],'LSTM':[.4801,.778,.939],'Transformer':[.5609,.797,.939]}
for i,(name,ys) in enumerate(values.items()):
 ax.bar(np.arange(3)+(i-1)*.22,ys,width=.21,label=name,color=['#267b9d','#d79537','#6875ae'][i])
 for j,v in enumerate(ys):ax.text(j+(i-1)*.22,v+.015,f'{v:.3f}',ha='center',fontsize=8)
ax.set_xticks(range(3),labels);ax.set_ylim(0,1.2);ax.set_ylabel('各版本 test micro-F1')
ax.legend(ncol=3,loc='upper left',frameon=False);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(root/'figures/version_history.pdf',bbox_inches='tight');plt.close(fig)
print('Later-run plots generated; no cross-version statistical baseline reused.')
