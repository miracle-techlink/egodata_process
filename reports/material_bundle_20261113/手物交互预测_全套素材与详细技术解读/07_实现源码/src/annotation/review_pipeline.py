"""Read-only, resumable review of the complete acquisition pool.

Exit 0 means automated checks ran without structural errors. It does not mean
visual annotations or geometry are complete; use --require-complete for that gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback
import uuid

ROOT_KEYS = ('base_root', 'v3_root', 'v5_root', 'source_pool')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_config(config, relative_to=None):
    config = dict(config); base = Path(relative_to or '.').resolve()
    require(config.get('base_root'), '配置缺少 base_root')
    for key in ROOT_KEYS:
        if config.get(key):
            path = Path(config[key]).expanduser()
            config[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    defaults = {'expected_sequences': 300, 'workers': 4, 'max_attempts': 2,
                'queue_chunk_frames': 300, 'thresholds': {}}
    for key, value in defaults.items():
        config.setdefault(key, value)
    for key in ('expected_sequences', 'workers', 'max_attempts', 'queue_chunk_frames'):
        require(type(config[key]) is int and config[key] > 0, key + '必须为正整数')
    require(config['workers'] <= 8 and config['max_attempts'] <= 5, 'workers最多8，max_attempts最多5')
    require(isinstance(config['thresholds'], dict), 'thresholds必须是对象')
    for key, value in config['thresholds'].items():
        require(type(value) in (int, float) and 0 < value < float('inf'), '异常阈值须为有限正数: ' + key)
    return config


def validate_output(config, output):
    output = Path(output).resolve()
    for key in ROOT_KEYS:
        if config.get(key):
            source = Path(config[key]).resolve()
            require(output != source and source not in output.parents and output not in source.parents,
                    '运行目录与源数据目录不得包含或重叠: ' + key)
    if output.exists():
        for folder, dirs, files in os.walk(output, followlinks=False):
            require(not any((Path(folder)/name).is_symlink() for name in dirs+files),
                    '运行目录内不允许预置软链接，避免写入源数据')
    return output


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.pipeline.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('同一输出目录已有流水线正在运行') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def file_bindings(paths):
    result = {}
    for raw in sorted({str(Path(p).absolute()) for p in paths}):
        path = Path(raw)
        if not path.exists():
            result[raw] = {'missing': True}
        elif not path.is_file():
            raise ValueError('依赖必须为文件: ' + raw)
        else:
            result[raw] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return result


def fingerprint(bindings, context):
    payload = json.dumps({'inputs': bindings, 'context': context}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def code_bindings():
    src = Path(__file__).resolve().parents[1]
    # Bind the transitive local checking/loader implementation conservatively.
    return file_bindings(p for directory in ('annotation', 'dataset', 'mamba')
                         for p in (src / directory).rglob('*.py'))


def issue(code, message, *, layer='pipeline', severity='error', frame_count=0, uid=None):
    return {'code': code, 'severity': severity, 'layer': layer,
            'sequence_id': uid, 'start_frame': 0, 'end_frame': int(frame_count),
            'entity_id': None, 'message_zh': message,
            'next_action_zh': '修复输入或补齐所需文件后，用相同命令加 --resume 重跑。'}


def failed_report(code, message, uid=None, frame_count=0, split=None):
    return {'sequence_id': uid, 'split': split, 'frame_count': int(frame_count),
            'checks': [{'code': code, 'status': 'fail', 'message_zh': message}],
            'metrics': {'native_frames': int(frame_count), 'full_annotation_complete': False},
            'issues': [issue(code, message, uid=uid, frame_count=frame_count)], 'status': 'fail'}


def report_status(report):
    statuses = {row.get('status') for row in report.get('checks', [])}
    if report.get('status') == 'fail' or 'fail' in statuses or any(
            row.get('severity') == 'error' for row in report.get('issues', [])):
        return 'fail'
    if report.get('status') == 'pending' or 'pending' in statuses or any(
            row.get('severity') == 'pending' for row in report.get('issues', [])):
        return 'pending'
    if 'warn' in statuses or any(r.get('severity') == 'warning' for r in report.get('issues', [])):
        return 'warn'
    return 'pass'


def relative_source(root, relative):
    require(not Path(relative).is_absolute(), '索引文件路径须相对数据根目录')
    path = (Path(root) / relative).resolve()
    require(Path(root).resolve() in path.parents, '索引文件越出数据目录')
    return path


def inventory_dependencies(config):
    root = Path(config['base_root']); paths = [root / 'dataset_index.json']
    try:
        index = read(paths[0])
        paths += [relative_source(root, index[k]) for k in ('vocab_path', 'split_manifest_path')]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    for key, name in [('v3_root', 'mamba_overlay_index.json'), ('v5_root', 'training_index.json'),
                      ('source_pool', 'selection_frozen.json')]:
        if config.get(key): paths.append(Path(config[key]) / name)
    return paths


def inspect_inventory(config):
    root = Path(config['base_root']); index = read(root / 'dataset_index.json')
    checks, issues = [], []
    def check(code, condition, message):
        checks.append({'code': code, 'status': 'pass' if condition else 'fail', 'message_zh': message})
        if not condition: issues.append(issue(code, message))
    rows = index['sequences']; ids = [r['sequence_id'] for r in rows]
    check('acquisition_count', len(rows) == config['expected_sequences'],
          f"要求{config['expected_sequences']}段，索引实际{len(rows)}段")
    check('unique_ids', len(ids) == len(set(ids)), '采集ID必须唯一')
    check('safe_ids', all(isinstance(uid, str) and re.fullmatch(r'[A-Za-z0-9_-]+', uid) for uid in ids),
          '采集ID必须为安全文件名')
    check('schema', index.get('schema_version') == 'egodemo_mamba_v2', 'V2源数据schema核对')
    split_path = relative_source(root, index['split_manifest_path']); split = read(split_path)
    vocab_path = relative_source(root, index['vocab_path'])
    check('split_hash', sha(split_path) == index['split_manifest_sha256'], '冻结划分SHA核对')
    check('vocab_hash', sha(vocab_path) == index['vocab_sha256'], '冻结词表SHA核对')
    assignments = split['assignments']
    check('split_coverage', set(assignments) == set(ids), '划分覆盖所有采集且没有额外采集')
    check('split_assignments', all(r['split'] in {'train','val','test'} and
        assignments.get(r['sequence_id']) == r['split'] for r in rows), '索引与冻结划分逐项一致')
    groups = split.get('groups', [])
    flattened = [uid for group in groups for uid in group['sequence_ids']]
    check('split_groups', len(flattened) == len(set(flattened)) and set(flattened) == set(ids)
          and all(len({assignments.get(uid) for uid in g['sequence_ids']}) <= 1 for g in groups),
          '同一采集分组不可跨划分，分组须完整且不重复')
    if 'sequence_group' in split:
        check('group_assignments', set(split['sequence_group']) == set(ids) and all(
            split['sequence_group'].get(uid) == group['group_id'] and assignments.get(uid) == group.get('split')
            for group in groups for uid in group['sequence_ids']), '分组ID、分组划分及采集划分逐项一致')
    if 'records' in split:
        records = split['records']
        check('split_records', len(records) == len(ids) and {r['sequence_id'] for r in records} == set(ids)
            and all(r['split'] == assignments.get(r['sequence_id']) for r in records), '划分记录覆盖及归属一致')
        check('previous_exposure', all(not r.get('previous_split_membership') or r['split'] == 'train'
            for r in records) and all(not g.get('previously_seen_sequence_ids') or g.get('split') == 'train'
            for g in groups), '既往已用于开发的采集及其连通组只能进入train')
    for row in rows:
        for key in ('sequence_path', 'metadata_path', 'events_path'):
            relative_source(root, row[key])
    if config.get('v3_root'):
        overlay = read(Path(config['v3_root']) / 'mamba_overlay_index.json')
        entries = overlay['sequences']
        check('v3_binding', overlay['base_dataset_index_sha256'] == sha(root / 'dataset_index.json')
            and overlay['base_split_manifest_sha256'] == index['split_manifest_sha256'], 'V3来源/划分绑定')
        check('v3_coverage', len(entries) == len(ids) and {r['sequence_id'] for r in entries} == set(ids),
              'V3标注必须覆盖同一完整采集池')
        check('v3_unique', len({r['sequence_id'] for r in entries}) == len(entries), 'V3采集条目不能重复')
        check('v3_split', all(r.get('split') == assignments.get(r['sequence_id']) for r in entries),
              'V3条目划分与冻结源一致')
        for row in entries: relative_source(config['v3_root'],row['annotation_path'])
    if config.get('v5_root'):
        current = read(Path(config['v5_root']) / 'training_index.json')
        check('v5_binding', current['base_dataset_index_sha256'] == sha(root / 'dataset_index.json')
              and current['split_manifest_sha256'] == index['split_manifest_sha256'], 'V5子集来源与划分绑定')
        check('v5_subset', len({r['sequence_id'] for r in current['sequences']}) == len(current['sequences'])
              and all(r['sequence_id'] in assignments and r['split'] == assignments[r['sequence_id']]
                      for r in current['sequences']), 'V5子集不得改划分或加入池外采集')
    if config.get('source_pool'):
        selection_path = Path(config['source_pool']) / 'selection_frozen.json'
        selection = read(selection_path); selected = selection['records']
        selected_ids = [r['source_episode_uuid'] for r in selected]
        check('source_selection', len(selected_ids) == len(set(selected_ids)) and set(selected_ids) == set(ids)
              and sha(selection_path) == index['source_selection_sha256'] == split['source_selection_sha256'],
              '原始300段选择清单SHA及覆盖核对')
        source_frames={r['source_episode_uuid']:r.get('manifest_frame_count') for r in selected}
        check('source_frames', all(type(r['frame_count']) is int and r['frame_count'] > 0
              and r['frame_count'] == source_frames.get(r['sequence_id']) for r in rows),
              '每段原生帧数必须与原始采集清单一致，不能仅比较帧数总和')
        from src.dataset.egodemo_v2_split import build_groups
        rebuilt = build_groups(selected, selection.get('subject_identity_verified') is True)
        expected_groups = {g['group_id']: set(g['sequence_ids']) for g in rebuilt}
        frozen_groups = {g['group_id']: set(g['sequence_ids']) for g in groups}
        check('source_groups', expected_groups == frozen_groups, '由源记录独立重建的连通组与冻结划分一致')
        check('downloaded_exposure', all(not r.get('previously_downloaded') or
            assignments.get(r['source_episode_uuid']) == 'train' for r in selected), '已下载参与开发的采集必须为train')
        shared = defaultdict(set)
        for row in selected:
            uid = row['source_episode_uuid']; assigned = assignments.get(uid)
            for key in ('environment_id','scene_id'):
                if row.get(key): shared[(key, row[key])].add(assigned)
            for asset in row.get('planned_assets', []):
                if asset['role'] in ('data','head_left_video'):
                    if asset.get('expected_sha256'):
                        shared[(asset['role'], asset['expected_sha256'])].add(assigned)
        check('source_leakage', all(len(values) == 1 for values in shared.values()),
              '相同场景/环境/已知内容SHA不能跨划分；不宣称缺失被试信息的跨被试隔离')
    return {'checks': checks, 'issues': issues, 'metrics': {'acquisitions': len(rows),
            'native_frames': sum(r['frame_count'] for r in rows), 'split_counts': dict(Counter(r['split'] for r in rows))}}


class JobRunner:
    def __init__(self, output, config, resume=False):
        self.output, self.config = Path(output), config
        self.state_path = self.output / 'jobs.json'; self.lock = threading.Lock()
        self.context = {'config': config, 'code': code_bindings()}
        self.state = read(self.state_path) if resume and self.state_path.exists() else {'schema_version':'automatic_review_jobs_v1','jobs':{}}

    def save_job(self, name, row):
        with self.lock:
            self.state['jobs'][name] = row
            atomic_json(self.state_path, self.state)

    def run(self, name, dependencies, function, relative_output, *, failure_context=None):
        path = self.output / relative_output; failure_context = failure_context or {}
        previous = self.state['jobs'].get(name, {})
        started = time.monotonic(); report = None
        for attempt in range(1, self.config['max_attempts'] + 1):
            bindings = {}; signature = None
            try:
                bindings = file_bindings(dependencies())
                signature = fingerprint(bindings, self.context)
                if attempt == 1 and previous.get('status') in {'pass','warn','pending'} and previous.get('fingerprint') == signature:
                    artifacts_ok = all(Path(p).is_file() and sha(p) == digest
                                       for p,digest in previous.get('artifact_sha256',{}).items())
                    if artifacts_ok and path.is_file() and sha(path) == previous.get('report_sha256'):
                        row = {**previous, 'cache_hit':True, 'checked_at':now()}
                        self.save_job(name, row)
                        return read(path)
                self.save_job(name, {'status':'running','attempt':attempt,'fingerprint':signature,
                    'input_bindings':bindings,'started_at':now(),'cache_hit':False})
                report = function()
                after = file_bindings(dependencies())
                require(bindings == after, '执行期间源依赖发生变化，本次结果不可复用')
                artifacts = {}
                for raw in report.get('artifact_paths',[]):
                    artifact = Path(raw)
                    require(artifact.is_file() and not artifact.is_symlink() and
                            self.output.resolve() in artifact.resolve().parents,
                            '检查子产物必须存在且位于本次运行目录内')
                    artifacts[str(artifact)] = sha(artifact)
                report['status'] = report_status(report)
            except Exception as exc:
                artifacts = {}
                report = failed_report('execution_error', f'{type(exc).__name__}: {exc}', **failure_context)
                report['traceback'] = traceback.format_exc()
            report['pipeline_job'] = name
            atomic_json(path, report)
            row = {'status': report['status'], 'attempt':attempt, 'fingerprint':signature,
                'input_bindings':bindings, 'report_path':relative_output, 'report_sha256':sha(path),
                'finished_at':now(),'elapsed_seconds':round(time.monotonic()-started,3),'cache_hit':False}
            row['artifact_sha256'] = artifacts
            self.save_job(name, row)
            if report['status'] != 'fail': break
        return report


def sequence_source_paths(config, row, annotations, reviews):
    from src.annotation.review_checks import sequence_dependencies
    uid = row['sequence_id']; annotation = annotations.get(uid); review = reviews.get(uid)
    paths = list(sequence_dependencies(config['base_root'], row, annotation_path=annotation, review_path=review))
    return paths + inventory_dependencies(config)


def check_sequence(config, row, annotations, annotation_entries, reviews, review_entries):
    from src.annotation.review_checks import review_sequence
    uid = row['sequence_id']; annotation = annotations.get(uid); review = reviews.get(uid)
    require(type(row.get('frame_count')) is int and row['frame_count'] > 0, '采集frame_count缺失或不是正整数')
    require(row.get('split') in {'train','val','test'}, '采集split缺失或非法')
    if config.get('v3_root'):
        require(uid in annotation_entries and annotation is not None, '采集缺少有效V3索引条目或路径')
    if annotation:
        require(sha(annotation) == annotation_entries[uid]['sha256'], 'V3标注在冻结索引之后改变')
    if review:
        require(sha(review) == review_entries[uid]['review_sha256'], 'V5复核记录在冻结索引之后改变')
    return review_sequence(config['base_root'], row, annotation_path=annotation,
                           review_path=review, thresholds=config['thresholds'])


def build_review_queue(issues, chunk_frames):
    """Merge repeated work while retaining references to every original issue."""
    groups = defaultdict(list)
    fields = ('sequence_id', 'code', 'entity_id', 'layer', 'severity', 'split')
    for line, row in enumerate(issues, 1):
        groups[tuple(row.get(key) for key in fields)].append((line, row))
    queue = []

    def emit(lo, hi, sources):
        spans = [(a, min(a + chunk_frames, hi)) for a in range(lo, hi, chunk_frames)] if hi > lo else [(lo, hi)]
        for a, b in spans:
            relevant = [(line, row) for line, row in sources if hi <= lo or
                        int(row.get('start_frame', 0)) < b and int(row.get('end_frame', 0)) > a]
            row = relevant[0][1]
            queue.append({**row, 'start_frame': a, 'end_frame': b,
                'source_issue_lines': sorted(line for line, _ in relevant),
                'messages_zh': list(dict.fromkeys(r.get('message_zh', '') for _, r in relevant)),
                'next_actions_zh': list(dict.fromkeys(r.get('next_action_zh', '') for _, r in relevant)),
                'requires_visual_confirmation': row['severity'] != 'error'})

    for sources in groups.values():
        active = []; lo = hi = None
        for source in sorted(sources, key=lambda item: (int(item[1].get('start_frame', 0)),
                                                       int(item[1].get('end_frame', 0)), item[0])):
            a, b = int(source[1].get('start_frame', 0)), int(source[1].get('end_frame', 0))
            if b <= a:
                emit(a, b, [source])
                continue
            if active and a > hi:
                emit(lo, hi, active)
                active = []
            if not active:
                lo, hi = a, b
            else:
                hi = max(hi, b)
            active.append(source)
        if active:
            emit(lo, hi, active)
    ranking = {'error': 0, 'warning': 1, 'pending': 2}
    queue.sort(key=lambda row: (ranking.get(row['severity'], 3), row.get('sequence_id') or '',
                               row['start_frame'], row['code'], row.get('entity_id') or ''))
    for i, row in enumerate(queue):
        row['priority_rank'] = i + 1
    return queue


def write_reports(output, config, reports, inventory, training, state):
    issues = []; metrics = Counter()
    for report in reports:
        for key, value in report.get('metrics', {}).items():
            if type(value) in (int,float): metrics[key] += value
        for row in report.get('issues', []):
            issues.append({**row, 'sequence_id': report['sequence_id'], 'split': report.get('split')})
    issues += inventory.get('issues', []) + training.get('issues', [])
    ranking = {'error':0,'warning':1,'pending':2}
    issues.sort(key=lambda x:(ranking.get(x['severity'],3),x.get('sequence_id') or '',x.get('start_frame',0),x['code']))
    (output/'issues.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in issues))
    queue = build_review_queue(issues, config['queue_chunk_frames'])
    (output/'review_queue.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in queue))
    statuses=Counter(report_status(r) for r in reports)
    errors=sum(r['severity']=='error' for r in issues)
    all_jobs=list(state['jobs'].values())
    failed_jobs=sum(r['status']=='fail' for r in all_jobs)
    complete=(len(reports)==config['expected_sequences'] and all(
        r.get('metrics',{}).get('full_annotation_complete') is True for r in reports)
        and errors==0 and failed_jobs==0)
    summary={'schema_version':'automatic_review_summary_v1','created_at':now(),
        'expected_sequences':config['expected_sequences'],'processed_sequences':len(reports),
        'execution_complete':len(reports)==config['expected_sequences'] and all(r['status']!='running' for r in all_jobs),
        'automated_checks_passed':errors==0 and failed_jobs==0,
        'sequence_status_counts':dict(statuses),'issue_counts':dict(Counter(r['severity'] for r in issues)),
        'metrics':dict(metrics),'queue_items':len(queue),'cache_hits':sum(bool(r.get('cache_hit')) for r in all_jobs),
        'jobs':len(all_jobs),'full_annotation_complete':complete,'training_stage_status':report_status(training),
        'training_subset_metrics':training.get('metrics',{}),
        'training_release_modified':False,'human_labels_added':0,'optimizer_steps':0,
        'scope_zh':'自动检查数值、时间轴、来源、标注结构与已交付窗口；未知和待视觉确认单列。自动检查通过不认证300段连续全标注或整物体6D。'}
    atomic_json(output/'summary.json',summary)
    fields=['采集ID','划分','原生帧','自动检查状态','连续已复核帧','语义待复核帧','错误数','警告数','待确认数','报告']
    with (output/'复核汇总.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in sorted(reports,key=lambda r:r['sequence_id']):
            m=r.get('metrics',{});counts=Counter(i['severity'] for i in r.get('issues',[]))
            writer.writerow(dict(zip(fields,[r['sequence_id'],r.get('split'),r['frame_count'],report_status(r),
                m.get('continuous_reviewed_frames',0),m.get('semantic_pending_frames',r['frame_count']),
                counts['error'],counts['warning'],counts['pending'],'sequence_reports/'+r['sequence_id']+'.json'])))
    labels={'pass':'自动检查通过','warn':'发现数值异常待核','pending':'存在未完成或待确认项','fail':'检查失败'}
    rows=''.join('<tr><td>'+html.escape(r['sequence_id'])+'</td><td>'+html.escape(str(r.get('split')))+
        '</td><td>'+str(r['frame_count'])+'</td><td>'+labels[report_status(r)]+
        '</td><td><a href="sequence_reports/'+html.escape(r['sequence_id'])+'.json">逐段原因</a></td></tr>' for r in reports)
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>300段自动复核报告</title><style>body{font:16px/1.6 sans-serif;margin:24px;color:#17324a}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccd7e0;text-align:left}pre{white-space:pre-wrap}a{color:#1254a2}.table{overflow:auto}</style>
<h1>自动化复核结果</h1><p>数值检查、待视觉确认和完整标注完成分别统计。此流水线不会自动批准未知标签或修改训练准入。</p>'''
    body+=f'<p>已检查 {len(reports)} / {config["expected_sequences"]} 段；错误 {errors} 项；复核队列 {len(queue)} 项。</p>'
    body+='<p><a href="summary.json">完整统计</a> · <a href="复核汇总.csv">逐段CSV</a> · <a href="review_queue.jsonl">待处理队列</a> · <a href="jobs.json">运行与缓存记录</a></p>'
    body+='<div class="table"><table><thead><tr><th>采集</th><th>划分</th><th>原生帧</th><th>状态</th><th>明细</th></tr></thead><tbody>'+rows+'</tbody></table></div></html>'
    (output/'report.html').write_text(body)
    return summary


def run_pipeline(config, output, resume=False, require_complete=False):
    config=normalized_config(config);output=validate_output(config,output)
    with output_lock(output):
        require(resume or not (output/'jobs.json').exists(), '运行目录已有结果，请使用 --resume 或新的目录')
        previous_config=read(output/'config_resolved.json') if (output/'config_resolved.json').exists() else None
        if previous_config and previous_config!=config:
            # Different settings are allowed, but all cache fingerprints change.
            atomic_json(output/'previous_config.json',previous_config)
        atomic_json(output/'config_resolved.json',config)
        runner=JobRunner(output,config,resume)
        inventory=runner.run('inventory',lambda:inventory_dependencies(config),lambda:inspect_inventory(config), 'inventory.json')
        try: base=read(Path(config['base_root'])/'dataset_index.json'); rows=base['sequences']
        except (OSError,ValueError,KeyError): rows=[]
        valid_rows={}
        for row in rows:
            if not isinstance(row,dict): continue
            uid=row.get('sequence_id')
            if isinstance(uid,str) and re.fullmatch(r'[A-Za-z0-9_-]+',uid) and uid not in valid_rows:
                valid_rows[uid]=row
        annotations={};annotation_entries={};reviews={};review_entries={}
        if config.get('v3_root'):
            try:
                for row in read(Path(config['v3_root'])/'mamba_overlay_index.json')['sequences']:
                    annotation_entries[row['sequence_id']]=row
                    annotations[row['sequence_id']]=relative_source(config['v3_root'],row['annotation_path'])
            except (OSError,ValueError,KeyError): pass
        if config.get('v5_root'):
            try:
                for row in read(Path(config['v5_root'])/'training_index.json')['sequences']:
                    review_entries[row['sequence_id']]=row
                    reviews[row['sequence_id']]=relative_source(config['v5_root'],row['review_path'])
            except (OSError,ValueError,KeyError): pass
        active_jobs = {'inventory'} | {'sequence:'+uid for uid in valid_rows}
        if config.get('v5_root'): active_jobs.add('training_subset')
        runner.state['jobs'] = {k:v for k,v in runner.state['jobs'].items() if k in active_jobs}
        atomic_json(runner.state_path,runner.state)
        reports=[]
        def work(row):
            uid=row['sequence_id']
            frame_count=row.get('frame_count',0)
            if type(frame_count) is not int or frame_count<0: frame_count=0
            return runner.run('sequence:'+uid,lambda:sequence_source_paths(config,row,annotations,reviews),
                lambda:check_sequence(config,row,annotations,annotation_entries,reviews,review_entries),
                'sequence_reports/'+uid+'.json',failure_context={'uid':uid,'frame_count':frame_count,'split':row.get('split')})
        with ThreadPoolExecutor(max_workers=config['workers']) as pool:
            futures={pool.submit(work,row):uid for uid,row in valid_rows.items()}
            for future in as_completed(futures):
                reports.append(future.result())
                if len(reports)%25==0 or len(reports)==len(futures):
                    print(json.dumps({'checked_sequences':len(reports),'total':len(futures)},ensure_ascii=False),flush=True)
        reports.sort(key=lambda r:r['sequence_id'])
        if config.get('v5_root'):
            from src.annotation.review_training import check_training_subset, training_dependencies
            training=runner.run('training_subset',lambda:training_dependencies(config['v5_root']),
                lambda:check_training_subset(config['v5_root'],output/'training_check'), 'training_stage.json')
        else:
            training={'status':'pending','checks':[{'code':'training_not_configured','status':'pending','message_zh':'未配置V5子集'}],
                      'metrics':{},'issues':[issue('training_not_configured','未配置训练子集，未检查窗口与准入。',severity='pending')]}
        current_code=code_bindings()
        if current_code!=runner.context['code']:
            inventory.setdefault('issues',[]).append(issue('code_changed','运行期间检查代码改变，请在代码固定后重跑。'))
            inventory['status']='fail';atomic_json(output/'inventory.json',inventory)
        if file_bindings(inventory_dependencies(config)) != runner.state['jobs']['inventory'].get('input_bindings'):
            inventory.setdefault('issues',[]).append(issue('inventory_changed','运行期间采集清单或划分/词表改变，本次结果不可作为固定版本验收。'))
            inventory['status']='fail';atomic_json(output/'inventory.json',inventory)
        summary=write_reports(output,config,reports,inventory,training,runner.state)
        summary['exit_code']=2 if not summary['automated_checks_passed'] else (3 if require_complete and not summary['full_annotation_complete'] else 0)
        atomic_json(output/'summary.json',summary)
        atomic_json(output/'run_manifest.json',{'schema_version':'automatic_review_run_manifest_v1',
            'config_sha256':sha(output/'config_resolved.json'),'code_bindings':runner.context['code'],
            'reports':{str(p.relative_to(output)):sha(p) for p in [output/'summary.json',output/'jobs.json',output/'issues.jsonl',
                output/'review_queue.jsonl',output/'复核汇总.csv',output/'report.html']}})
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    try:
        config=normalized_config(read(args.config),Path(args.config).resolve().parent)
        report=run_pipeline(config,args.output,args.resume,args.require_complete)
    except (OSError,ValueError,RuntimeError) as exc:
        print(json.dumps({'status':'fail','error':str(exc)},ensure_ascii=False),flush=True);return 2
    print(json.dumps({k:report[k] for k in ('processed_sequences','automated_checks_passed','full_annotation_complete','cache_hits','exit_code')},ensure_ascii=False))
    return report['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
