from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'文件清单.json').read_text(encoding='utf-8'))
failed=[]
for row in manifest['files']:
 p=ROOT/row['path']
 if not p.is_file():failed.append((row['path'],'missing'));continue
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:failed.append((row['path'],'sha256 mismatch'))
print(json.dumps({'checked':len(manifest['files']),'failed':failed},ensure_ascii=False,indent=2))
sys.exit(bool(failed))
