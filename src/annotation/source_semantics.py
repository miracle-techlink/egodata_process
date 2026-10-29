"""Source-grounded Chinese semantic drafts; no image inference or gold labels.

Use the existing verified local model only to structure and translate source
annotations. Verbatim evidence is checked in code. Unknown physical state,
identity, anatomy and exact boundaries stay unresolved for visual annotation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import time

from src.dataset.complete_egodemo import OPERATIONS, digest, save, operation_candidates


def rigidity_candidate(phrase, description):
    phrase = phrase.lower()
    if re.search(r'\b(paper\s*clips?|binder\s*clips?)\b', phrase):
        return 'rigid'
    if re.search(r'\b(box|carton)\b', phrase):
        return 'deformable' if re.search(r'\b(fold|unfold|flatten|collapse)\w*\b', description, re.I) else 'unknown'
    if re.search(r'\b(paper|tissue|cable|cord|wire|cloth|towel|bag|tape|label|wrapper|packet|apron|ribbon|lanyard)\w*\b', phrase):
        return 'deformable'
    if re.search(r'\b(stapler|printer|scissors|dispenser|microwave|drawer|blind)\w*\b', phrase):
        return 'articulated'
    if re.search(r'\b(marker|pen|lid|cap|jar|bottle|cup|bowl|scanner|screwdriver|screw|bulb|hanger|spoon|fork|knife|whistle|controller)\w*\b', phrase):
        return 'rigid'
    return 'unknown'


def parse_draft(text, source_rows):
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', stripped)
    value = json.loads(stripped)
    if not isinstance(value, list) or len(value) != len(source_rows):
        raise ValueError('Output must contain each source interval exactly once')
    by_id = {r['source_interval_id']: r for r in source_rows}
    seen, output = set(), []
    for row in value:
        identity = row.get('id')
        if identity not in by_id or identity in seen:
            raise ValueError('Source interval identity mismatch')
        seen.add(identity)
        source = by_id[identity]
        original = source['description_original']
        zh = row.get('zh')
        if not isinstance(zh, str) or not re.search('[\u4e00-\u9fff]', zh):
            raise ValueError('Chinese translation is missing')
        # Import no model-generated actions/material classes as annotations.
        operations = [{'type': c['operation_type'], 'name_zh': c['operation_name_zh'],
                       'source_quote': c['text_evidence'][0]['text'], 'subject': None,
                       'object_id': None, 'label_valid': False}
                      for c in operation_candidates(original)]
        mentions = []
        for mention in row.get('objects', []):
            phrase, name = mention.get('source_phrase'), mention.get('name_zh')
            rigidity = rigidity_candidate(phrase or '', original)
            if (not isinstance(phrase, str) or not phrase or phrase.lower() not in original.lower()
                    or not isinstance(name, str) or not name or rigidity not in {'rigid', 'deformable', 'articulated', 'unknown'}):
                raise ValueError('Invalid source object mention')
            mentions.append({'source_phrase': phrase, 'name_zh': name, 'rigidity_candidate': rigidity,
                             'stable_object_id': None, 'visual_identity_verified': False})
        output.append({**source, 'description_zh': zh, 'operations': operations, 'object_mentions': mentions,
                       'label_source': 'source_text_language_model', 'review_status': 'pending',
                       'label_valid': False, 'temporal_precision': 'source_interval_only',
                       'subject': None, 'object_id': None})
    return output


def prompt(rows):
    instructions = '''你是翻译助手，将每条英文事件准确译为简洁中文，并提取涉及的物体名词。
只输出JSON数组，每个输入id恰好一个元素：
[{"id":"source_0000","zh":"中文译文","objects":[{"source_phrase":"原文连续名词子串","name_zh":"中文名称"}]}]
source_phrase必须逐字来自对应原文，不能改写或补词。省略人、手、房间、桌面等背景，只提取与操作相关的实体。没有则objects为空数组。
不要合并事件，不要推测动作细节、左右手、物体状态。只翻译原文并提取名词。
'''
    return instructions + '\n输入：' + json.dumps([
        {'id': r['source_interval_id'], 'text': r['description_original']} for r in rows], ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    from src.annotation.qwen_backend import QwenBackend
    backend = QwenBackend()
    backend.load()
    torch, model, tokenizer = backend._torch, backend._model, backend._processor.tokenizer
    torch.set_num_threads(6)
    tokenizer.padding_side = 'left'
    out = args.root / 'semantic_drafts'
    out.mkdir(exist_ok=True)
    save(out / 'model.json', backend.model_metadata)
    jobs = []
    for path in sorted((args.root / 'sequences').glob('*/interactions.json'))[:args.limit]:
        data = json.loads(path.read_text())
        rows = data['source_semantic_intervals']
        for start in range(0, len(rows), 6):
            dest = out / path.parent.name / f'chunk_{start:04d}.json'
            if not dest.exists():
                jobs.append((path, rows[start:start+6], dest))
    done, errors, started = 0, [], time.monotonic()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start+args.batch_size]
        texts = [tokenizer.apply_chat_template([{'role':'user','content':[{'type':'text','text':prompt(rows)}]}], tokenize=False, add_generation_prompt=True)
                 for _, rows, _ in batch]
        if any(prompt(rows) not in text for (_, rows, _), text in zip(batch, texts)):
            raise ValueError('Chat template dropped the source text')
        inputs = tokenizer(texts, padding=True, return_tensors='pt').to(backend._input_device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=2400, do_sample=False)
        decoded = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for (path, rows, dest), response in zip(batch, decoded):
            try:
                parsed = parse_draft(response, rows)
                save(dest, {'source_sha256':digest(path), 'sequence_id':path.parent.name, 'records':parsed,
                            'allowed_as_model_input':False, 'training_allowed':False, 'raw_model_text':response})
                done += 1
            except Exception as error:
                error_dest = dest.with_suffix('.error.json')
                save(error_dest, {'source_sha256':digest(path), 'error':str(error), 'raw_model_text':response})
                errors.append({'file':str(error_dest),'error':str(error)})
        report = {'completed_chunks_this_run':done, 'total_chunks_this_run':len(jobs), 'errors':errors,
                  'elapsed_seconds':time.monotonic()-started}
        save(out / 'progress.json', report)
        print(json.dumps({k:v if k!='errors' else len(v) for k,v in report.items()}), flush=True)
    all_rows, counts = [], Counter()
    for path in sorted(out.glob('*/chunk_*.json')):
        if '.error.' in path.name:
            continue
        data = json.loads(path.read_text())
        for row in data['records']:
            all_rows.append({'sequence_id':data['sequence_id'], **row})
            counts.update(op['type'] for op in row['operations'])
    save(out / 'summary.json', {'translated_source_intervals':len(all_rows), 'sequences':len({r['sequence_id'] for r in all_rows}),
         'operation_candidates':sum(counts.values()), 'operation_types':len(counts), 'counts':dict(counts),
         'human_approved':0, 'training_allowed':False, 'errors_this_run':errors})
    with (out / '中文事件候选.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['采集ID','事件ID','开始秒','结束秒','中文事件','操作候选','物体候选','原文','状态'])
        for row in all_rows:
            writer.writerow([row['sequence_id'],row['source_interval_id'],row['start_seconds'],row['end_seconds'],row['description_zh'],
                 '；'.join(op['name_zh'] for op in row['operations']), '；'.join(m['name_zh'] for m in row['object_mentions']),
                 row['description_original'],'待视觉审核'])


if __name__ == '__main__':
    main()
