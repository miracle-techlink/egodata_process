"""Count actual private artifacts, keeping candidates separate from acceptance."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from src.dataset.complete_egodemo import digest, save, OPERATIONS


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def report(root):
    pool = root.parent / 'egodemo_expansion_300_20261023'
    selection = read(pool/'selection_frozen.json')['records']
    records, events, counts, rigidity, issues = [], [], Counter(), Counter(), []
    quality=read(root/'visual_quality_overrides.json',{'records':[]})['records']
    rejected={(r['sequence_id'],r['tracklet_id']) for r in quality if r['status']=='rejected'}
    rejected_pose_frames=0
    for source in selection:
        uid = source['source_episode_uuid']
        meta = read(root/'sequences'/uid/'metadata.json', {})
        native = read(root/'sequences'/uid/'interactions.json', {})
        rows = []
        for path in sorted((root/'semantic_drafts'/uid).glob('chunk_*.json')):
            if '.error.' not in path.name:
                rows.extend(read(path)['records'])
        if len({r['source_interval_id'] for r in rows}) != len(rows):
            issues.append({'sequence_id': uid, 'issue': 'duplicate source interval'})
        for row in rows:
            counts.update(x['type'] for x in row['operations'])
            events.append({'sequence_id': uid, **row})
        stereo = read(root/'stereo_checks'/(uid+'.json'), {})
        obs = read(root/'object_observations_v2'/uid/'receipt.json', {})
        dense = read(root/'dense_tracklets_v4'/uid/'receipt.json', {})
        tracks = read(root/'dense_tracklets_v4'/uid/'tracklets.json', {})
        for track in tracks.get('records', []):
            rigidity[track['rigidity_candidate']] += 1
            if (uid,track['tracklet_id']) in rejected:rejected_pose_frames+=track['object_pose_frames']
            if track['rigidity_candidate'] != 'rigid' and track['object_pose_frames']:
                issues.append({'sequence_id': uid, 'issue': 'nonrigid whole pose present'})
        records.append({'sequence_id': uid, 'task_name': source['task_name'],
                        'frame_count': meta.get('frame_count', 0), 'hands_exported': bool(meta),
                        'hand_valid_counts': meta.get('hand_valid_counts', [0, 0]),
                        'source_intervals': len(native.get('source_semantic_intervals', [])),
                        'translated_intervals': len(rows), 'stereo_passed': stereo.get('status') == 'passed',
                        'object_observations_processed': bool(obs), 'object_observations': obs.get('object_observations', 0),
                        'dense_candidates_processed': bool(dense), 'tracklets': dense.get('tracklets', 0),
                        'point_object_frames': dense.get('point_frames', 0),
                        'pose_object_frames': dense.get('object_pose_frames', 0),
                        'proximity_candidates': dense.get('proximity_intervals', 0),
                        'human_approved': 0, 'full_annotation_complete': False, 'training_allowed': False})
    complete = all(r['hands_exported'] and r['stereo_passed'] and r['object_observations_processed']
                   and r['dense_candidates_processed'] and r['translated_intervals'] == r['source_intervals'] for r in records)
    summary = {'schema_version': 'egodemo_joint_completion_report_v1',
               'updated_at': datetime.now(timezone.utc).isoformat(), 'source_sequences': len(records),
               'source_selection_sha256': digest(pool/'selection_frozen.json'),
               'source_task_names': len({r['task_name'] for r in records}),
               'source_frames': sum(r['frame_count'] for r in records),
               'hands_exported': sum(r['hands_exported'] for r in records),
               'stereo_passed': sum(r['stereo_passed'] for r in records),
               'source_intervals': sum(r['source_intervals'] for r in records),
               'translated_intervals': len(events), 'operation_candidates': sum(counts.values()),
               'observed_operation_types': len(counts), 'operation_counts': dict(counts.most_common()),
               'unmapped_source_intervals': sum(not r['operations'] for r in events),
               'object_observations_processed': sum(r['object_observations_processed'] for r in records),
               'dense_candidates_processed': sum(r['dense_candidates_processed'] for r in records),
               'candidate_tracklets': sum(r['tracklets'] for r in records),
               'candidate_tracklets_by_rigidity': dict(rigidity),
               'pose_object_frames': sum(r['pose_object_frames'] for r in records),
               'assistant_visually_sampled_tracklets':len(quality), 'visually_rejected_tracklets':len(rejected),
               'visually_rejected_pose_object_frames':rejected_pose_frames,
               'pose_candidate_frames_after_known_rejections':sum(r['pose_object_frames'] for r in records)-rejected_pose_frames,
               'sequences_with_pose_candidates': sum(r['pose_object_frames'] > 0 for r in records),
               'point_object_frames': sum(r['point_object_frames'] for r in records),
               'proximity_candidates': sum(r['proximity_candidates'] for r in records),
               'automatic_candidate_processing_complete': complete, 'full_annotation_complete': False,
               'human_approved_sequences': 0, 'independent_pose_accuracy': None, 'training_allowed': False,
               'counting_notes': ['tracklet为连续可见片段，不等于去重物体数',
                    'object_frames为各物体轨迹帧数之和，不能当成唯一视频帧覆盖率',
                    '操作数为原文词组候选，不能当成人工确认的接触/抓持/协作类别数',
                    '非刚体、铰接物、透明/内容物混淆容器不输出整物体刚体6D'],
               'remaining': ['稳定物体身份合并与视觉语义确认', '物体可观测性和姿态独立验证',
                             '接触/抓持/支撑/协作与精确边界人工标注', '因果输入和训练适配'],
               'issues': issues}
    save(root/'joint_report.json', summary)
    save(root/'joint_sequence_status.json', {'records': records, 'training_allowed': False})
    with (root/'联合处理进度.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['采集ID', '任务', '原生帧', '手部导出', '双目时间轴通过', '中文事件数', '物体观测处理',
                    '连续轨迹处理', '候选轨迹数', '6D物体帧数（非唯一视频帧）', '人工通过', '训练可用'])
        for r in records:
            w.writerow([r['sequence_id'], r['task_name'], r['frame_count'], r['hands_exported'], r['stereo_passed'],
                        r['translated_intervals'], r['object_observations_processed'], r['dense_candidates_processed'],
                        r['tracklets'], r['pose_object_frames'], False, False])
    with (root/'中文交互候选.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['采集ID','来源事件ID','开始秒','结束秒','中文草稿','原文','操作候选','物体候选','状态'])
        for r in events:
            w.writerow([r['sequence_id'],r['source_interval_id'],r['start_seconds'],r['end_seconds'],
                        r['description_zh'],r['description_original'],'；'.join(x['name_zh'] for x in r['operations']),
                        '；'.join(x['name_zh'] for x in r['object_mentions']),'待视觉审核'])
    with (root/'操作候选分布.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.writer(f);w.writerow(['操作代码','中文类型','原文命中候选数','是否人工通过'])
        for code, count in counts.most_common():w.writerow([code,OPERATIONS[code],count,False])
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        result=report(args.root)
        print(json.dumps({k:v for k,v in result.items() if k not in ['operation_counts','remaining','counting_notes']},ensure_ascii=False),flush=True)
        if not args.watch or result['automatic_candidate_processing_complete']:break
        time.sleep(45)


if __name__=='__main__':main()
