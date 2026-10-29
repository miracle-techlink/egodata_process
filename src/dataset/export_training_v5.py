"""Freeze a portable, explicitly scoped graph-supervised observation subset."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from src.mamba.dataset_v5 import (sha,read,covered_range,approved_observation_mask,select_objects,
    build_features,attach_pose_features,target_edges,HAND_IDS,FEATURE_DIM,vocabulary_entries)


def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def export(base_root,root,*,window_frames=32,horizon_seconds=1.,max_objects=8,minimum_observations=8):
    base_root,root=Path(base_root),Path(root)
    if (root/'training_index.json').exists():raise FileExistsError('Frozen index exists; export into a new version')
    base_index=read(base_root/'dataset_index.json')
    split_path=base_root/base_index['split_manifest_path'];split=read(split_path)
    if sha(split_path)!=base_index['split_manifest_sha256']:raise ValueError('Frozen source split changed')
    base_rows={r['sequence_id']:r for r in base_index['sequences']}
    vocab_path=base_root/base_index['vocab_path'];vocab=read(vocab_path);types=vocabulary_entries(vocab)
    if sha(vocab_path)!=base_index['vocab_sha256']:raise ValueError('Frozen vocabulary changed')
    rows=[];windows={k:[] for k in ['train','val','test']};counts={k:Counter() for k in windows}
    by_type={k:Counter() for k in windows};exclusions=Counter()
    for review_path in sorted((root/'dense_reviews').glob('*.json')):
        uid=review_path.stem;review=read(review_path)
        if uid not in base_rows:raise ValueError('Review is outside original 300 acquisitions')
        original=base_rows[uid];assignment=split['assignments'][uid]
        if assignment!=original['split']:raise ValueError('Source index/split mismatch')
        # v7 rebalance: documented val→train reassignment (see rebalance_v7.json)
        reassign_path=Path('/root/gpufree-data/runs/rebalance_v7.json')
        if reassign_path.is_file() and uid in json.loads(reassign_path.read_text())['reassigned_to_train']:
            assignment='train'
        folder=root/'visual_observations'/uid
        meta=read(folder/'metadata.json');quality_path=folder/'tracking_review.json'
        quality=read(quality_path);obs_path=folder/'observations.npz';seeds_path=folder/'seeds.json'
        if sha(obs_path)!=meta['observations_sha256'] or quality['observations_sha256']!=sha(obs_path):
            raise ValueError('Observation/quality source changed')
        if sha(seeds_path)!=meta['source_bindings']['seeds_sha256']:raise ValueError('Seeds changed after tracking')
        if quality.get('review_status')!='assistant_reviewed':raise ValueError('Visual tracking review is pending')
        source_path=base_root/original['sequence_path']
        if sha(source_path)!=original['sha256']['sequence']:raise ValueError('Base sequence changed')
        with np.load(source_path,allow_pickle=False) as a:
            seq={k:a[k].copy() for k in ['timestamps','hands_world','hand_valid','camera_to_world']}
        with np.load(obs_path,allow_pickle=False) as a:obs={k:a[k].copy() for k in a.files}
        times=seq['timestamps'];frame_count=len(times)
        if len(obs['timestamps'])!=frame_count or not np.array_equal(times,obs['timestamps']):
            raise ValueError('Observation time axis differs')
        if not np.isfinite(times).all() or not (np.diff(times)>0).all():raise ValueError('Invalid native timestamps')
        if not np.array_equal(obs['max_frame_accessed'],np.arange(frame_count)):raise ValueError('Noncausal observation prefix')
        valid=approved_observation_mask(obs,quality)
        hand_folder=root/'sequences'/uid;hand_folder.mkdir(parents=True,exist_ok=True)
        sequence_path=hand_folder/'hands.npz'
        np.savez_compressed(sequence_path,**seq)
        sequence_row={'sequence_id':uid,'split':assignment,'frame_count':frame_count,
            'review_path':str(review_path.relative_to(root)),'review_sha256':sha(review_path),
            'observations_path':str(obs_path.relative_to(root)),'observations_sha256':sha(obs_path),
            'observation_metadata_path':str((folder/'metadata.json').relative_to(root)),
            'observation_metadata_sha256':sha(folder/'metadata.json'),
            'seeds_path':str(seeds_path.relative_to(root)),'seeds_sha256':sha(seeds_path),
            'tracking_review_path':str(quality_path.relative_to(root)),'tracking_review_sha256':sha(quality_path),
            'sequence_path':str(sequence_path.relative_to(root)),'sequence_sha256':sha(sequence_path),
            'base_sequence_sha256':sha(source_path),'native_image_size':meta['native_image_size'],
            'observed_entity_frame_count':int(valid.sum()),'evidence_root':str(root)}
        median_dt=float(np.median(np.diff(times)))
        reviewed=set()
        for span in review['reviewed_ranges']:
            lo,hi=(span['start_frame'],span['end_frame']) if isinstance(span,dict) else span
            reviewed.update(range(lo,hi))
        for target in sorted(reviewed):
            anchor=int(np.searchsorted(times,times[target]-horizon_seconds,side='right')-1)
            while anchor >= 0 and times[target]-times[anchor] < horizon_seconds:
                anchor -= 1
            while anchor+1 < target and times[target]-times[anchor+1] >= horizon_seconds:
                anchor += 1
            start=anchor-window_frames+1;end=anchor+1
            if start<0:exclusions['insufficient_history_or_horizon']+=1;continue
            horizon=float(times[target]-times[anchor])
            if horizon<horizon_seconds or horizon-horizon_seconds>1.5*median_dt+1e-9:
                exclusions['timestamp_horizon_gap']+=1;continue
            if not covered_range(review,start,target+1):exclusions['unreviewed_history_target_gap']+=1;continue
            slots=select_objects(obs,valid,start,end,max_objects,minimum_observations)
            entity_ids=list(HAND_IDS)+[str(obs['object_ids'][j]) for j in slots]+['']*(max_objects-len(slots))
            features=build_features(seq,obs,valid,start,end,slots,meta['native_image_size'],max_objects)
            from src.mamba.pose_features_v6 import PoseTable
            obs_ids=[str(v) for v in obs['object_ids']]
            seed_doc=read(root/'visual_observations'/uid/'seeds.json')
            seed_names={x['entity_id']:x.get('name_zh','') for x in seed_doc['seeds']}
            table=PoseTable(str(Path('/root/gpufree-data/datasets/ego_hoi/egodemo_refine_300_group6d_20261025')),uid,obs_ids,seed_names,frame_count)
            pose_series=(table.world_series(frame_count) if table.ok else None)
            features['entity_features']=attach_pose_features(features['entity_features'],pose_series,start,end,slots)
            values,mask,records,dropped=target_edges(review,entity_ids,types,anchor,target)
            seen=features['entity_valid'].any(0)
            mask &= seen[:,None,None]&seen[None,:,None]
            if not mask.any():exclusions['no_known_graph_labels_on_observed_entities']+=1;continue
            positive=int(((values==1)&mask).sum());negative=int(((values==0)&mask).sum())
            windows[assignment].append({'sequence_id':uid,'start_frame':start,'end_frame':end,
                'anchor_frame':anchor,'target_frame':target,'target_timestamp':float(times[target]),
                'actual_horizon_s':horizon,'known_positive_cells':positive,'known_negative_cells':negative})
            counts[assignment].update(windows=1,known_positive_cells=positive,known_negative_cells=negative,
                windows_with_positive=int(positive>0),windows_with_negative=int(negative>0))
            for k,event in enumerate(types):
                n=int(mask[:,:,k].sum())
                if n:by_type[assignment][event['layer']+':'+event['code']]+=n
        sequence_row['window_count']=sum(w['sequence_id']==uid for w in windows[assignment])
        sequence_row['continuously_reviewed_frames']=len(reviewed);rows.append(sequence_row)
    sources=root/'sources';sources.mkdir(exist_ok=True)
    for source,name in [(base_root/'dataset_index.json','base_dataset_index.json'),(split_path,'split_manifest.json'),(vocab_path,'vocab.json')]:
        shutil.copy2(source,sources/name)
    window_root=root/'graph_windows';window_root.mkdir(exist_ok=True)
    split_rows={}
    for name,data in windows.items():
        path=window_root/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
        split_rows[name]={'path':str(path.relative_to(root)),'sha256':sha(path),'windows':len(data),
                         'sequence_count':sum(r['split']==name and r['window_count']>0 for r in rows)}
    index={'schema_version':'assistant_silver_graph_training_v5','created_at':datetime.now(timezone.utc).isoformat(),
        'original_acquisition_count':len(base_rows),'original_native_frame_count':sum(r['frame_count'] for r in base_rows.values()),
        'base_root':str(base_root),'base_dataset_index_sha256':sha(base_root/'dataset_index.json'),
        'split_manifest_sha256':sha(split_path),'vocab_sha256':sha(vocab_path),'event_types':types,
        'feature_version':'v6_pose','pose_refine_root':'/root/gpufree-data/datasets/ego_hoi/egodemo_refine_300_group6d_20261025','window_config':{'window_frames':window_frames,'horizon_seconds':horizon_seconds,'max_objects':max_objects,
                         'minimum_observations':minimum_observations,'feature_dim':FEATURE_DIM},
        'sequences':rows,'splits':split_rows,'counts':{k:dict(v) for k,v in counts.items()},
        'known_cells_by_type':{k:dict(v) for k,v in by_type.items()},'exclusion_counts':dict(exclusions),
        'admission_required':True,'admission_path':'training_release.json',
        'training_scope':'assistant_silver_interaction_forecasting_with_annotated_2d_observations',
        'whole_object_6d_certified':False,'full_joint_training_allowed':False,'full_annotation_complete':False,
        'blind_evaluation':False,'automatic_perception_evaluated':False,'exporter_sha256':sha(__file__)}
    save(root/'training_index.json',index)
    current={r['sequence_id']:r for r in rows}
    catalog=[]
    for uid,original in base_rows.items():
        active=current.get(uid)
        catalog.append({'sequence_id':uid,'split':original['split'],'native_frames':original['frame_count'],
            'included_in_interaction_subset':active is not None and active['window_count']>0,
            'v5_continuously_reviewed_frames':0 if active is None else active['continuously_reviewed_frames'],
            'full_annotation_complete':False,'full_joint_training_allowed':False})
    save(root/'all_300_acquisitions.json',{'schema_version':'v5_original_pool_retained','acquisitions':catalog})
    return index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-root',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();index=export(a.base_root,a.root)
    print(json.dumps({'splits':index['splits'],'counts':index['counts']},ensure_ascii=False))

if __name__=='__main__':main()
