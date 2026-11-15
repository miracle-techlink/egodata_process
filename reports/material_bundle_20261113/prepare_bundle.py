from pathlib import Path
import shutil, hashlib, json, csv, re, subprocess, html
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=WORK/'手物交互预测_全套素材与详细技术解读'
OUT.mkdir(exist_ok=True)
records=[];image_hashes={};video_hashes={};aliases=[]
def copy(src,rel,category,rights='授权工作区材料；按来源使用'):
 src=Path(src);target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(src,target)
 records.append({'path':str(rel),'source':str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
 'category':category,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'usage':rights})
 return target
def unique(src,folder,category,seen):
 digest=hashlib.sha256(src.read_bytes()).hexdigest()
 if digest in seen:
  aliases.append({'source':str(src),'included_as':seen[digest]});return seen[digest]
 rel=Path(folder)/src.name
 if (OUT/rel).exists():rel=Path(folder)/(src.stem+'_'+digest[:8]+src.suffix)
 copy(src,rel,category);seen[digest]=str(rel);return str(rel)
full=ROOT/'reports/technical_zh_20261112';public=ROOT/'reports/public_technical_zh_20261112';demo=ROOT/'reports/demo_update_20261113'
copy(public/'egodata-technical-report-zh.pdf','01_技术报告/最新公开技术报告.pdf','当前报告','可公开，第三方图像遵守原许可')
copy(full/'完整技术报告_单栏图文版.pdf','01_技术报告/授权完整版技术报告.pdf','当前报告')
for name in ['手部与6D_三页配图预览.pdf','demo_materials.pdf']:
 if (full/name).exists():copy(full/name,Path('01_技术报告')/name,'配套说明')
# All video versions, including original encodings and browser-compatible encodings.
sources=[full/'media',demo/'site/dist/media',ROOT.parent/'demo']
for folder in sources:
 for src in sorted(folder.glob('*.mp4')):unique(src,'03_视频素材','视频',video_hashes)
unique(demo/'raw_rgb.mp4','03_视频素材','视频',video_hashes)
# Full, editable report figures retain their original names.
for src in sorted((full/'figures').rglob('*')):
 if src.is_file() and src.suffix.lower() in {'.png','.jpg','.jpeg','.svg','.pdf','.tex'}:
  rel=Path('04_图片与图表/当前报告插图')/src.relative_to(full/'figures');copy(src,rel,'图表与绘图源码')
  image_hashes[hashlib.sha256(src.read_bytes()).hexdigest()]=str(rel)
report_dirs=[public,ROOT/'reports/cvpr_zh_full_pipeline_20261112',ROOT/'reports/cvpr_zh_20261112']
for folder in report_dirs:
 for src in sorted(folder.rglob('*')):
  if src.is_file() and src.suffix.lower() in {'.png','.jpg','.jpeg','.svg'}:
   unique(src,'04_图片与图表/历史与补充图片','历史补充图片',image_hashes)
 for src in sorted((folder/'figures').glob('*')):
  if src.suffix.lower() in {'.pdf','.tex'}:unique(src,'04_图片与图表/历史与补充图片','矢量图与源码',image_hashes)
for folder in [demo,ROOT.parent/'demo']:
 for src in sorted(folder.rglob('*')):
  if '.git' in src.parts:continue
  if src.is_file() and src.suffix.lower() in {'.png','.jpg','.jpeg','.svg'}:
   unique(src,'04_图片与图表/Demo与网页素材','Demo与网页素材',image_hashes)
# Saved report source uses the bundled figures by relative path.
for tag,folder in [('授权完整版',full),('公开版',public)]:
 dest=OUT/'05_报告源码'/tag;dest.mkdir(parents=True,exist_ok=True)
 for src in folder.iterdir():
  if src.is_file() and src.suffix in {'.tex','.bib','.bst','.sty'}:
   copy(src,Path('05_报告源码')/tag/src.name,'报告源码')
 for src in (folder/'sections').glob('*.tex'):
  copy(src,Path('05_报告源码')/tag/'sections'/src.name,'报告源码')
 # Shared base figures; public-only figures retain their own source and PDF.
 for src in (folder/'figures').glob('*'):
  if src.suffix in {'.png','.jpg','.pdf','.tex','.svg'}:copy(src,Path('05_报告源码')/tag/'figures'/src.name,'可直接编译的报告依赖')
 (dest/'build.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\nxelatex -interaction=nonstopmode -halt-on-error main.tex\nbibtex main\nxelatex -interaction=nonstopmode -halt-on-error main.tex\nxelatex -interaction=nonstopmode -halt-on-error main.tex\n')
for src in sorted((full/'evidence').rglob('*')):
 if src.is_file() and src.suffix in {'.json','.jsonl','.npz','.csv','.py','.md','.txt','.yaml','.yml'}:
  copy(src,Path('06_实验依据/报告证据')/src.relative_to(full/'evidence'),'原始实验依据')
for src in sorted(demo.iterdir()):
 if src.is_file() and src.suffix in {'.json','.npz','.pt','.py'}:
  # Hosting publisher/credentials are unnecessary to understand or replay the experiment.
  if src.name in {'publish_github.py','preview_server.py'}:continue
  copy(src,Path('06_实验依据/Demo推理与制作')/src.name,'Demo推理与制作')
for src in sorted((ROOT/'src').rglob('*.py')):
 copy(src,Path('07_实现源码/src')/src.relative_to(ROOT/'src'),'实现源码快照')
for name in ['graph_v5_base.yaml','full_hocap_v1.json','lstm_base.yaml','mamba_base.yaml']:
 if (ROOT/'configs'/name).exists():copy(ROOT/'configs'/name,Path('07_实现源码/configs')/name,'实现配置')
for name in ['requirements.txt','pyproject.toml']:
 if (ROOT/name).exists():copy(ROOT/name,Path('07_实现源码')/name,'依赖说明')
for src in (full/'scripts').glob('*.py'):copy(src,Path('07_实现源码/报告制作脚本')/src.name,'原报告制作脚本')
for name in ['HUMAN_DATASET_V1_20261022.md','V5_TRAINING_20261024.md','TRAINING_V5_HANDOFF.md','POSE_FORECAST_PILOT_20261022.md','FOUNDATIONPOSE_REAL_RUN_20261022.md','FUSION_ACCEPTANCE_20261022.md','AUTOMATIC_REVIEW_PIPELINE.md']:
 p=ROOT/'docs'/name
 if p.exists():copy(p,Path('07_实现源码/阶段技术文档')/name,'阶段技术文档')
# Offline copy of the actually deployed page, preserving a self-contained relative tree.
for src in (demo/'site/dist').rglob('*'):
 if src.is_file():copy(src,Path('08_公开网页离线副本')/src.relative_to(demo/'site/dist'),'可离线浏览的公开演示','按THIRD_PARTY_NOTICES使用')
for name in ['cvpr_zh_20261112','cvpr_zh_full_pipeline_20261112']:
 copy(ROOT/'reports'/name/'main.pdf',Path('09_历史报告')/(name+'.pdf'),'历史报告，不作为当前结论')
(OUT/'09_历史报告/版本说明.txt').write_text('本目录保留历史排版与阶段结论，仅供溯源。当前结论和Demo以01_技术报告及02_详细技术解读为准。早期小子集结果不代表最终全量结果。\n')
copy(public/'THIRD_PARTY_NOTICES.txt','第三方来源与许可.txt','来源说明')
(WORK/'copy_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
(WORK/'deduplicated_aliases.json').write_text(json.dumps(aliases,ensure_ascii=False,indent=2))
print(json.dumps({'root':str(OUT),'copied':len(records),'video_versions':len(video_hashes),'unique_graphic_hashes':len(image_hashes),'deduplicated_source_aliases':len(aliases)},ensure_ascii=False))
