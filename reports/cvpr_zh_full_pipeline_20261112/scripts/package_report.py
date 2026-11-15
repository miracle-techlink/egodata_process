"""Build portable full-pipeline report; include source inputs, figures and demos."""
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
archive=root/'完整Pipeline_CVPR中文报告与素材.zip'
files=[root/name for name in ['README.txt','main.tex','main.pdf','完整技术报告_手部与6D配图版.pdf','visual_check.tex','visual_check.pdf','手部与6D_三页配图预览.pdf','demo_materials.tex','demo_materials.pdf','references.bib','cvpr.sty','ieeenat_fullname.bst','build.sh','素材清单.csv']]
for folder,suffixes in [('sections',{'.tex'}),('figures',{'.pdf','.png','.jpg','.tex'}),('media',{'.mp4'}),('evidence',{'.json','.jsonl','.csv','.py','.yaml','.yml','.md','.npz'}),('scripts',{'.py'})]:
 files.extend(p for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in suffixes and '__pycache__' not in p.parts)
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
 for path in sorted(set(files)):bundle.write(path,path.relative_to(root))
print(archive.name,len(set(files)),'files',archive.stat().st_size,'bytes')
