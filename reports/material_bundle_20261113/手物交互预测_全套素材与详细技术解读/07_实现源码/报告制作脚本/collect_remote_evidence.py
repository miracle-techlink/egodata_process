"""Read-only collector for graph training and all known RSI series.

Run where the original files are accessible. Copies records, hashes checkpoints;
never loads model weights, trains, connects to a network or changes source runs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import zipfile

DEFAULT_SERIES = ['rsi_final', 'rsi_auto', 'graph_v5_auto', 'rsi_v5']
RECORDS = {'.json', '.jsonl', '.yaml', '.yml', '.csv', '.log'}
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors'}
DATA_NAMES = {
    'training_index.json', 'training_index_audit.json', 'training_release.json',
    'index.json', 'manifest.json', 'summary.json', 'dataset_index.json',
    'audit.json', 'release.json', 'diagnostic_admission.json', 'vocab.json',
    'type_vocab.json', 'split_manifest.json',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def files_without_symlinks(root):
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(folder) / d).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink():
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--series', nargs='+', default=DEFAULT_SERIES)
    parser.add_argument('--data-root', type=Path, action='append', default=[],
                        help='Explicit run-bound release root; repeat for several versions')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = args.runs_root.resolve()
    if not runs.is_dir():
        raise SystemExit('runs-root does not exist')
    for name in args.series:
        if Path(name).name != name or name in {'.', '..'}:
            raise SystemExit('Each series must be one directory name under runs-root')
    data = [p.resolve() for p in args.data_root]
    for path in data:
        if not path.is_dir():
            raise SystemExit('A data-root does not exist')
    output = args.output.resolve()
    for source in [runs, *data]:
        if source == output or source in output.parents or output in source.parents:
            raise SystemExit('Output must be outside and must not contain any source tree')
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists():
        raise SystemExit('Refusing to overwrite an output directory or zip')
    output.mkdir(parents=True)
    inventory = []

    def collect(path, relative):
        size = path.stat().st_size
        digest = sha256(path)
        record = {'source': str(path), 'path': str(relative), 'bytes': size, 'sha256': digest}
        if path.suffix.lower() in WEIGHTS:
            record['status'] = 'checkpoint_hash_only'
        elif size > 8 * 1024 * 1024:
            record['status'] = 'large_file_hash_only'
        else:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if sha256(destination) != digest:
                raise RuntimeError(f'Source changed while collecting {relative}')
            record['status'] = 'copied'
        inventory.append(record)

    for name in args.series:
        series = runs / name
        if series.is_symlink() or not series.is_dir():
            inventory.append({'series': name, 'status': 'missing_or_symlink_series'})
            continue
        for path in files_without_symlinks(series):
            if path.suffix.lower() in RECORDS | WEIGHTS or path.name in {'REPORT.md', 'README.md'}:
                collect(path, Path('runs') / path.relative_to(runs))
    for number, root in enumerate(data):
        # Only release summaries: video and annotation frame directories are excluded.
        candidates = list(root.iterdir())
        for subdir in ['model_checks', 'audit', 'release']:
            folder = root / subdir
            if folder.is_dir() and not folder.is_symlink():
                candidates.extend(folder.iterdir())
        for path in sorted(candidates):
            if path.is_file() and not path.is_symlink() and path.name in DATA_NAMES:
                collect(path, Path('data') / f'{number:02d}_{root.name}' / path.relative_to(root))
    payload = {'runs_root': str(runs), 'series': args.series,
               'data_roots': [str(p) for p in data], 'files': inventory,
               'note': 'A read-only collection, not a simultaneous frozen snapshot. Inspect run-bound hashes.'}
    (output / 'inventory.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    print(archive)


if __name__ == '__main__':
    main()
