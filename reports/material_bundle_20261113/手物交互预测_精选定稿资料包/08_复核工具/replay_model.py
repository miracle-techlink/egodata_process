"""Replay saved demo inputs on CPU. Does not train, fetch data or upload files."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'07_实现源码'))
from src.mamba.train_v5 import build_v5_model
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--batch-size',type=int,default=16);args=parser.parse_args()
 if args.batch_size<1:parser.error('batch-size must be positive')
 d=ROOT/'06_实验依据/Demo推理与制作';torch.set_num_threads(2)
 ckpath=d/'lstm_final_private.pt'
 # This is the bundled project checkpoint. Only load trusted local checkpoints.
 ck=torch.load(ckpath,map_location='cpu',weights_only=False)
 model=build_v5_model(ck['config']).eval();model.load_state_dict(ck['model'])
 with np.load(d/'history_inputs.npz',allow_pickle=False) as q:inputs={k:q[k] for k in q.files}
 with np.load(d/'demo_arrays.npz',allow_pickle=False) as q:
  expected=q['probabilities'][inputs['anchor_frames']];expected_mask=q['prediction_mask'][inputs['anchor_frames']]
 outputs=[];masks=[]
 with torch.inference_mode():
  for start in range(0,len(expected),args.batch_size):
   b={k:torch.from_numpy(inputs[k][start:start+args.batch_size]) for k in ['entity_features','entity_valid','entity_kind']}
   result=model(b);outputs.append(result['graph_logits'].sigmoid().numpy());masks.append(result['graph_prediction_mask'].numpy())
 actual=np.concatenate(outputs);mask=np.concatenate(masks)
 error=float(np.max(np.abs(actual-expected)))
 np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)
 assert np.array_equal(mask,expected_mask),'prediction mask mismatch'
 report={'status':'passed','anchors':len(actual),'feature_shape':list(inputs['entity_features'].shape),
  'maximum_absolute_probability_difference':error,'prediction_masks_identical':True,
  'checkpoint_sha256':hashlib.sha256(ckpath.read_bytes()).hexdigest(),'model':ck['config']['model'],
  'epoch':ck['epoch'],'training_performed':False,'scope':'replay equivalence only; not an accuracy evaluation'}
 if args.output:args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
