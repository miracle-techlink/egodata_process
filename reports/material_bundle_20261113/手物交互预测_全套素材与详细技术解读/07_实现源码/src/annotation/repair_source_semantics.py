"""Recover structured drafts without accepting hallucinated noun evidence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from src.annotation.source_semantics import parse_draft
from src.dataset.complete_egodemo import digest, save


def repair_response(raw, source_rows, glossary):
    text = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    value, end = json.JSONDecoder().raw_decode(text)
    if not isinstance(value, list):
        raise ValueError('Expected a JSON array')
    by_id = {r['source_interval_id']: r for r in source_rows}
    audit = {'trailing_text_removed': bool(text[end:].strip()), 'dropped_mentions': [], 'dictionary_mentions': []}
    for row in value:
        if row.get('id') not in by_id:
            raise ValueError('Unknown source interval')
        original = by_id[row['id']]['description_original']
        kept = []
        for mention in row.get('objects', []):
            phrase, name = mention.get('source_phrase'), mention.get('name_zh')
            if isinstance(phrase, str) and phrase and phrase.lower() in original.lower() and isinstance(name, str) and name:
                kept.append(mention)
            else:
                audit['dropped_mentions'].append({'id': row['id'], 'value': mention})
        if not kept:
            occupied = []
            for phrase, name in sorted(glossary.items(), key=lambda x: -len(x[0])):
                match = re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', original, re.I)
                if match and not any(match.start() < b and match.end() > a for a, b in occupied):
                    occupied.append(match.span())
                    mention = {'source_phrase': match.group(), 'name_zh': name}
                    kept.append(mention)
                    audit['dictionary_mentions'].append({'id': row['id'], **mention})
        row['objects'] = kept
    return parse_draft(json.dumps(value, ensure_ascii=False), source_rows), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root / 'semantic_drafts'
    names = defaultdict(Counter)
    for path in out.glob('*/chunk_*.json'):
        if '.error.' in path.name:
            continue
        for row in json.loads(path.read_text())['records']:
            for m in row['object_mentions']:
                if len(m['source_phrase']) >= 3:
                    names[m['source_phrase'].lower()][m['name_zh']] += 1
    glossary = {p: c.most_common(1)[0][0] for p, c in names.items()}
    repaired, errors = [], []
    for path in sorted(out.glob('*/*.error.json')):
        dest = path.with_name(path.name.replace('.error', ''))
        if dest.exists():
            continue
        try:
            source = args.root / 'sequences' / path.parent.name / 'interactions.json'
            start = int(path.name.split('_')[1].split('.')[0])
            rows = json.loads(source.read_text())['source_semantic_intervals'][start:start+6]
            failure = json.loads(path.read_text())
            if failure['source_sha256'] != digest(source):
                raise ValueError('Source changed')
            records, audit = repair_response(failure['raw_model_text'], rows, glossary)
            save(dest, {'source_sha256': digest(source), 'sequence_id': path.parent.name,
                        'records': records, 'allowed_as_model_input': False, 'training_allowed': False,
                        'raw_model_text': failure['raw_model_text'], 'repair_audit': audit,
                        'original_error_sha256': digest(path)})
            repaired.append(str(dest))
        except Exception as error:
            errors.append({'file': str(path), 'error': str(error)})
    report = {'repaired': repaired, 'errors': errors, 'training_allowed': False}
    save(out / 'repair_report.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
