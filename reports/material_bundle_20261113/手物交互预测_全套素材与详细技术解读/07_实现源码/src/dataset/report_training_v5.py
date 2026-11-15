"""Chinese scoped-delivery summary and a retained 300-acquisition gap table."""
import argparse
import csv
import json
from pathlib import Path
from src.mamba.dataset_v5 import read, sha
from src.dataset.deliver_training_v5 import save


def report(root, v4_root):
    root,v4_root=Path(root),Path(v4_root)
    index=read(root/'training_index.json');review=read(root/'review_seed_audit.json')
    audit=read(root/'training_index_audit.json');release=read(root/'training_release.json')
    reviews={r['sequence_id']:r for r in review['records']}
    sequences={r['sequence_id']:r for r in index['sequences']}
    with (v4_root/'逐段连续标注缺口.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        row['V4阶段连续查看帧']=row.pop('新增连续查看帧')
        uid=row['采集ID'];record=reviews.get(uid);seq=sequences.get(uid)
        counts=record['counts'] if record else {}
        n=counts.get('reviewed_frames',0)
        row.update({'连续已查看帧_V5':n,'尚未连续查看帧':int(row['原生帧数'])-n,
            '明确正值数':counts.get('positive_state_values',0),
            '明确负值数':counts.get('negative_state_values',0),
            '有明确状态的帧':counts.get('positive_negative_labeled_frames',0),
            '确认发生物理范围数':counts.get('physical_interaction_confirmed_occurrence_ranges',0),
            '删失物理范围数':counts.get('physical_interaction_censored_ranges',0),
            '物理边界范围数':counts.get('physical_interaction_confirmed_occurrence_ranges',0)+counts.get('physical_interaction_censored_ranges',0),
            '精确物理边界数':0,'精确视觉信号变化数':counts.get('visual_signal_frame_resolved',0),
            'V5交互窗口':seq['window_count'] if seq else 0,
            'V5已接受二维物体帧':seq['observed_entity_frame_count'] if seq else 0,
            '已完成全部层标注':False,'整物体6D通过':False})
    with (root/'逐段连续标注缺口_V5.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'schema_version':'v5_scoped_delivery_report','original_acquisitions':len(rows),
        'original_native_frames':sum(int(r['原生帧数']) for r in rows),
        'continuously_reviewed_native_frames':review['counts']['reviewed_frames'],
        'remaining_unreviewed_native_frames':sum(int(r['尚未连续查看帧']) for r in rows),
        'fully_viewed_acquisitions':sum(int(r['尚未连续查看帧'])==0 for r in rows),
        'all_annotation_layers_complete_frames':0,
        'interaction_subset':index['counts'],'review_counts':review['counts'],
        'accepted_2d_object_frames':sum(r['accepted_object_frames'] for r in audit['sequence_records']),
        'rejected_2d_object_frames':sum(r['rejected_object_frames'] for r in audit['sequence_records']),
        'interaction_training_allowed':release['interaction_training_allowed'],
        'full_annotation_complete':False,'full_joint_training_allowed':False,'whole_object_6d_certified':False,
        'training_index_sha256':sha(root/'training_index.json'),'release_sha256':sha(root/'training_release.json'),
        'gap_csv_sha256':sha(root/'逐段连续标注缺口_V5.csv'),
        'scope_zh':'交付已审核助手银标交互子集；300段整体仍需逐帧证据和可信几何。原V4候选覆盖沿用并明确命名，不计入几何认证。'}
    save(root/'交付状态_V5.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--v4-root',required=True)
    a=p.parse_args();print(json.dumps(report(a.root,a.v4_root),ensure_ascii=False))
