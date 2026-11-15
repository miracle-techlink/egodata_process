"""Package current deliverables; explicitly excludes withdrawn subset artifacts."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
archive = root / 'CVPR中文技术报告_源码与素材.zip'
files = [root / name for name in [
    'README.txt', 'main.tex', 'main.pdf', 'demo_materials.tex', 'demo_materials.pdf',
    'references.bib', 'cvpr.sty', 'ieeenat_fullname.bst', 'build.sh', '素材清单.csv',
]]
for folder, suffixes in [('figures', {'.pdf', '.png', '.tex'}),
                         ('media', {'.mp4'}),
                         ('evidence', {'.json', '.csv', '.py', '.yaml', '.md'}),
                         ('scripts', {'.py'})]:
    files.extend(p for p in (root / folder).iterdir() if p.is_file() and p.suffix in suffixes)
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(files):
        bundle.write(path, path.relative_to(root))
print(f'{archive.name}: {len(files)} files, {archive.stat().st_size} bytes')
