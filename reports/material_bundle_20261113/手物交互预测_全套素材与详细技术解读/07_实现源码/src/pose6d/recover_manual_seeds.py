"""Refresh completed tracks whose assistant seed reference hashes changed."""
import argparse,json
from pathlib import Path
from src.dataset.complete_egodemo import save
from src.pose6d.refine_tracks import process
from src.annotation.refine_associations import assemble

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();results=[]
 for directory in sorted((a.root/'manual_observations').iterdir()):
  uid=directory.name;folder=a.root/'native_tracks'/uid;manifest=folder/'tracklets.json'
  if not (folder/'receipt.json').exists():continue
  expected={}
  for path in (a.root/'dense_seeds'/uid).glob('frame_*/observations.json'):
   for row in json.loads(path.read_text())['observations']:
    if row['triangulation_status']=='estimated_reference':expected[row['reference_sha256']]=row
  tracks=json.loads(manifest.read_text())
  old={r['source_reference_sha256'] for r in tracks['records']}
  changed=old!=set(expected)
  if changed:
   process(str(a.root),uid,force=True);tracks=json.loads(manifest.read_text())
  patched=False
  for row in tracks['records']:
   original=expected[row['source_reference_sha256']]
   for key in ['pose_scope','seed_source']:
    if key in original and row.get(key)!=original[key]:row[key]=original[key];patched=True
  if patched:save(manifest,tracks)
  receipt=assemble(a.root,uid)
  result={'sequence_id':uid,'retracked':changed,'metadata_updated':patched,'references':len(expected),
          'pose_frames':receipt['pose_frames'],'rigid_part_pose_frames':receipt['rigid_part_pose_frames']}
  results.append(result);print(json.dumps(result),flush=True)
 save(a.root/'manual_recovery_receipt.json',{'records':results,'human_approved':False,'training_allowed':False})
if __name__=='__main__':main()
