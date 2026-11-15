from pathlib import Path
import argparse,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--frame',type=int,default=223);args=p.parse_args()
d=ROOT/'06_实验依据/Demo推理与制作';m=json.loads((d/'demo_manifest.json').read_text())
with np.load(d/'demo_arrays.npz',allow_pickle=False) as z:
 print('Arrays:',{k:list(z[k].shape) for k in z.files})
 if not 0<=args.frame<m['frames']:p.error('frame outside available range')
 f=args.frame
 print('Frame',f,'phase',next(x for x in m['visual_review_phases'] if x['start']<=f<=x['end']))
 print('Visual phase is a coarse display annotation, not prediction ground truth.')
 if f<31:print('History not ready; no prediction.');raise SystemExit(0)
 scores=z['probabilities'][f];valid=z['prediction_mask'][f];names=['左手','右手','双手']+m['object_display_names'];rows=[]
 for a in range(3):
  for b in range(3,7):
   if valid[a,b]:
    for k,t in enumerate(m['event_types']):rows.append((float(scores[a,b,k]),names[a],t['name_zh'],names[b]))
 for score,source,relation,target in sorted(rows,reverse=True)[:10]:print(f'{source} → {relation} → {target}: {score:.8f}')
