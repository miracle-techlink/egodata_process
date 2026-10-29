"""Report physical evidence coverage separately from completion/acceptance."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
from src.dataset.complete_egodemo import save,digest
from src.annotation.refine_semantics import semantic_directory


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def report(root):
    pool=root.parent/'egodemo_expansion_300_20261023';old=root.parent/'egodemo_joint_300_20261023'
    records=read(pool/'selection_frozen.json')['records'];titles=read(old/'task_titles_zh.json')
    rows=[];types=Counter();methods=Counter();source_operations=Counter()
    for rec in records:
        uid=rec['source_episode_uuid'];seed=read(root/'dense_seeds'/uid/'receipt.json')
        track=read(root/'native_tracks'/uid/'receipt.json');joint=read(root/'joint_annotations'/uid/'receipt.json')
        proposal=read(root/'joint_annotations'/uid/'proposal.json')
        for row in proposal.get('interactions',[]):types[row['layer']+':'+row['type']]+=1
        for entity in proposal.get('entities',[]):methods[entity['rigidity']]+=1
        semantics=[]
        for p in semantic_directory(root,uid).glob('chunk_*.json'):
            if '.error.' not in p.name:semantics.extend(read(p)['records'])
        for event in semantics:
            for op in event['operations']:source_operations[op['type']]+=1
        rows.append({'sequence_id':uid,'task_name':rec['task_name'],'task_zh':titles.get(rec['task_name'],rec['task_name']),
            'native_frames':rec['manifest_frame_count'],'hands_ready':(old/'sequences'/uid/'hands.npz').exists(),
            'dense_seed_stage_complete':bool(seed),'native_track_stage_complete':bool(track),'association_stage_complete':bool(joint),
            'references':seed.get('references',0),'tracklets':track.get('tracklets',0),
            'raw_pose_object_frames':track.get('object_pose_frames',0),'surface_point_object_frames':track.get('point_frames',0),
            'linked_pose_object_frames':joint.get('pose_frames',0),'linked_unique_pose_frames':joint.get('unique_pose_frames',0),
            'hand_near_pose_object_frames':joint.get('hand_near_pose_frames',0),'hand_near_point_object_frames':joint.get('hand_near_point_frames',0),
            'rigid_part_pose_frames':joint.get('rigid_part_pose_frames',0),'entity_hypotheses':joint.get('entities',0),
            'identity_links':joint.get('identity_links',0),'source_intervals':len(semantics),
            'pixel_calibration_status':joint.get('pixel_calibration_status','not_yet_audited'),
            'metric_geometry_allowed':joint.get('metric_geometry_allowed',False),
            'relation_proposals':sum(r['layer']=='relation' for r in proposal.get('interactions',[])),
            'transition_proposals':sum(r['layer']=='transition' for r in proposal.get('interactions',[])),
            'cooperation_proposals':sum(r['layer']=='cooperation' for r in proposal.get('interactions',[])),
            'human_approved':False,'full_annotation_complete':False,'training_allowed':False})
    quality=read(root/'visual_quality_overrides.json').get('records',[])
    result={'schema_version':'egodemo_refinement_report_v1','generated_at':datetime.now(timezone.utc).isoformat(),
            'source_selection_sha256':digest(pool/'selection_frozen.json'),'sequences':len(rows),
            'tasks':len(set(r['task_name'] for r in rows)),'native_frames':sum(r['native_frames'] for r in rows),
            'hands_ready':sum(r['hands_ready'] for r in rows),
            'dense_seeds_processed':sum(r['dense_seed_stage_complete'] for r in rows),
            'native_tracks_processed':sum(r['native_track_stage_complete'] for r in rows),
            'association_proposals_processed':sum(r['association_stage_complete'] for r in rows),
            'sequences_with_surface_tracks':sum(r['surface_point_object_frames']>0 for r in rows),
            'sequences_with_surface_tracks_not_rejected_by_calibration':sum(r['surface_point_object_frames']>0 and r['metric_geometry_allowed'] for r in rows),
            'pixel_calibration_status_counts':dict(Counter(r['pixel_calibration_status'] for r in rows)),
            'sequences_with_raw_pose_candidates':sum(r['raw_pose_object_frames']>0 for r in rows),
            'sequences_with_filtered_pose_candidates':sum(r['linked_pose_object_frames']>0 for r in rows),
            'sequences_with_hand_near_pose_candidates':sum(r['hand_near_pose_object_frames']>=5 for r in rows),
            'sequences_with_hand_near_surface_candidates':sum(r['hand_near_point_object_frames']>=5 for r in rows),
            'completed_sequences_without_trackable_surface':[r['sequence_id'] for r in rows if r['native_track_stage_complete'] and not r['surface_point_object_frames']],
            'source_intervals':sum(r['source_intervals'] for r in rows),'observed_source_operation_types':len(source_operations),
            'source_operation_candidates':dict(source_operations),'physical_proposal_types':dict(types),
            'entity_material_hypotheses':dict(methods),
            'assistant_sampled_sequences':len(set(r['sequence_id'] for r in quality)),
            'assistant_sampled_tracklets':len(quality),'assistant_rejected_tracklets':sum(r['status']=='rejected' for r in quality),
            'human_approved_sequences':0,'independent_accuracy':None,
            'processing_complete':all(r['association_stage_complete'] for r in rows),
            'full_annotation_complete':False,'training_allowed':False,
            'remaining_requirements_zh':['逐段核对目标类别与跨遮挡稳定身份','修正缺失/漂移的操作对象表面种子',
                '确认接触、抓持、支撑和工具作用的实际边界','完成刚体/部件/非刚体几何质量验收'],
            'counts_note_zh':'原始6D候选可能包括背景物体；手附近仅是距离条件，不等同于实际接触。物体帧计数不可当作唯一视频帧数。'}
    for field in ['references','tracklets','surface_point_object_frames','raw_pose_object_frames','linked_pose_object_frames',
                  'linked_unique_pose_frames','hand_near_pose_object_frames','hand_near_point_object_frames','rigid_part_pose_frames','entity_hypotheses','identity_links']:
        result[field]=sum(r[field] for r in rows)
    save(root/'refinement_report.json',result);save(root/'sequence_completion_matrix.json',{'records':rows})
    headings={'sequence_id':'采集ID','task_zh':'任务','native_frames':'原生帧数','hands_ready':'手部已导出',
              'dense_seed_stage_complete':'双目参考已处理','native_track_stage_complete':'双向追踪已处理',
              'association_stage_complete':'关联提案已处理','references':'三维参考实例候选','tracklets':'轨迹片段',
              'linked_unique_pose_frames':'含6D候选的唯一视频帧','hand_near_pose_object_frames':'手附近6D候选物体帧',
              'hand_near_point_object_frames':'手附近表面候选物体帧','entity_hypotheses':'实例身份提案',
              'identity_links':'有几何证据的轨迹关联','relation_proposals':'持续关系提案','transition_proposals':'转换边界提案',
              'cooperation_proposals':'协作提案','source_intervals':'原文任务区间','human_approved':'人工通过','training_allowed':'允许训练'}
    headings.update(pixel_calibration_status='像素级标定抽查状态',metric_geometry_allowed='未被标定不一致拦截')
    with (root/'第二轮补全进度.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(headings.values())
        for row in rows:w.writerow([row[k] for k in headings])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    result=report(args.root)
    print(json.dumps({k:result[k] for k in ['sequences','dense_seeds_processed','native_tracks_processed',
          'association_proposals_processed','sequences_with_filtered_pose_candidates',
          'sequences_with_hand_near_pose_candidates','full_annotation_complete']},ensure_ascii=False))
