"""Publish only the report/video whitelist; never stage the working tree."""
from pathlib import Path
import subprocess,json,base64,hashlib
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/public_technical_zh_20261112'
DEMO=Path(__file__).resolve().parent
repo='repos/miracle-techlink/egodata_process'
branch='reports/technical-report-2026-11-12';tag='technical-report-2026-11-12'
def api(path,method='GET',data=None):
 cmd=['gh','api',repo+'/'+path,'--method',method]
 if data is not None:cmd+=['--input','-']
 p=subprocess.run(cmd,input=json.dumps(data) if data is not None else None,capture_output=True,text=True,check=True)
 return json.loads(p.stdout)
assets=[REPORT/'egodata-technical-report-zh.pdf',DEMO/'site/dist/media/full-pipeline.mp4',
 DEMO/'site/dist/media/hands_and_object_rgbd.mp4',DEMO/'site/dist/media/hocap_four_objects.mp4',
 REPORT/'THIRD_PARTY_NOTICES.txt']
(REPORT/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in assets))
old=api('git/ref/heads/'+branch)['object']['sha']
prior=DEMO/'github_publication.json'
expected=json.loads(prior.read_text())['commit'] if prior.exists() else '8c2d41717844a684a37ffc59d1d1d935729bb434'
assert old==expected,f'Report branch changed: {old}'
base=api('git/commits/'+old)['tree']['sha'];entries=[]
files={
 'egodata-technical-report-zh.pdf':REPORT/'egodata-technical-report-zh.pdf',
 'README.md':REPORT/'PUBLIC_README.md','SHA256SUMS.txt':REPORT/'SHA256SUMS.txt',
 'THIRD_PARTY_NOTICES.txt':REPORT/'THIRD_PARTY_NOTICES.txt',
 'demo_manifest.json':DEMO/'demo_manifest.json'}
for name,p in files.items():
 blob=api('git/blobs','POST',{'encoding':'base64','content':base64.b64encode(p.read_bytes()).decode()})
 entries.append({'path':'publications/2026-11-12/'+name,'mode':'100644','type':'blob','sha':blob['sha']})
tree=api('git/trees','POST',{'base_tree':base,'tree':entries})
commit=api('git/commits','POST',{'message':'docs: add synchronized hand-object pipeline demos and updated report','tree':tree['sha'],'parents':[old]})
api('git/refs/heads/'+branch,'PATCH',{'sha':commit['sha'],'force':False})
subprocess.run(['gh','release','upload',tag,*map(str,assets),str(REPORT/'SHA256SUMS.txt'),
 '--repo','miracle-techlink/egodata_process','--clobber'],check=True)
subprocess.run(['gh','release','edit',tag,'--repo','miracle-techlink/egodata_process',
 '--notes-file',str(REPORT/'release_notes.txt'),'--title','完整Pipeline技术报告 · 连续Demo更新（2026-11-13）'],check=True)
state={'branch':branch,'previous_commit':old,'commit':commit['sha'],'tag_preserved':tag,
 'release_url':'https://github.com/miracle-techlink/egodata_process/releases/tag/'+tag,
 'demo_url':'https://egodata-full-pipeline-demo.miracle-techlink.chatgpt.site',
 'pdf_url':'https://egodata-full-pipeline-demo.miracle-techlink.chatgpt.site/egodata-technical-report-zh.pdf'}
(DEMO/'github_publication.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
