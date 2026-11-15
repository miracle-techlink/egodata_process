"""Observed whole-instance slots and masked directed future graph targets.

This loader deliberately keeps the accepted scope explicit. 2D annotated
initialization does not certify source offline 6D geometry or all 300 clips.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.mamba.egodemo_v2 import vocabulary_entries

FEATURE_DIM=120
POSE_BLOCK=12  # v6: pos3 + rot6 + pose_valid1 + causal_flag1 + pad1 → dims 120:132
FEATURE_DIM_V6=FEATURE_DIM+POSE_BLOCK
HAND_IDS=('left_hand','right_hand','both_hands')
REQUIRED_ADMISSION_EVIDENCE=frozenset({
    'review_seed_audit.json','training_index_audit.json','delivery_integrity_checks_v5.json',
    'model_checks/real_graph_mamba_smoke.json','model_checks/real_graph_target_coverage.json',
    'audit_sources/model_report_crosscheck_v2.json'})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def covered_range(review,start,end):
    mask=np.zeros(end-start,bool)
    for span in review['reviewed_ranges']:
        lo,hi=(span['start_frame'],span['end_frame']) if isinstance(span,dict) else span
        a,b=max(lo,start),min(hi,end)
        if b>a:mask[a-start:b-start]=True
    return bool(mask.all())


def approved_observation_mask(observations,quality,*,diagnostic=False):
    valid=observations['object_valid'].copy()
    if quality is None:
        if diagnostic:return valid
        raise ValueError('Tracking has not been visually checked')
    accepted=np.zeros_like(valid)
    ids={v:i for i,v in enumerate(observations['object_ids'].tolist())}
    for row in quality.get('accepted_ranges',[]):
        if row['entity_id'] not in ids:raise ValueError('Quality refers to an absent observed identity')
        lo,hi=row['start_frame'],row['end_frame']
        if not 0<=lo<hi<len(valid)+1:raise ValueError('Quality range outside native timeline')
        accepted[lo:hi,ids[row['entity_id']]]=True
    # A rejected interval always wins, including an overlapping accepted range.
    for row in quality.get('rejected_ranges',[]):
        lo,hi=row['start_frame'],row['end_frame']
        accepted[lo:hi,ids[row['entity_id']]]=False
    return valid & accepted


def select_objects(observations,valid,start,end,max_objects=8,minimum_observations=8):
    history=valid[start:end]
    count=history.sum(0)
    first=np.where(history,np.arange(end-start)[:,None],end-start).min(0)
    candidates=[j for j in range(history.shape[1]) if count[j]>=minimum_observations]
    return sorted(candidates,key=lambda j:(int(first[j]),str(observations['object_ids'][j])))[:max_objects]


def _bbox_to_features(boxes,image_size):
    scale=np.asarray([*image_size,*image_size],np.float32)
    norm=boxes/scale
    return np.concatenate(((norm[...,:2]+norm[...,2:])*.5,norm[...,2:]-norm[...,:2]),-1)


def _history_velocity(center,valid,timestamps):
    result=np.zeros_like(center)
    dt=np.diff(timestamps)
    good=valid[1:] & valid[:-1] & (dt>0)[:,None]
    delta=(center[1:]-center[:-1])/dt[:,None,None]
    result[1:]=np.where(good[...,None],delta,0)
    return result


def attach_pose_features(features, pose_blocks, start, end, slots):
    """v6: fill dims 120:132 for object slots from a PoseTable time series.

    pose_blocks: (poses (T,N,4,4), valid (T,N)) world poses keyed by slot order.
    The pose block carries its own causal flag=1 so downstream audits can
    distinguish offline pose candidates from causal 2D observations.
    """
    if pose_blocks is None:
        return np.concatenate([features, np.zeros((*features.shape[:2], POSE_BLOCK), np.float32)], -1)
    poses, pvalid = pose_blocks
    W = features.shape[0]
    block = np.zeros((W, features.shape[1], POSE_BLOCK), np.float32)
    for i in range(len(slots)):
        if i >= poses.shape[1]:
            break
        for w, f in enumerate(range(start, end)):
            if f >= poses.shape[0] or not pvalid[f, i]:
                continue
            m = poses[f, i]
            r = m[:3, :3]
            block[w, 3 + i, 0:3] = m[:3, 3]
            block[w, 3 + i, 3:9] = np.concatenate([r[:, 0], r[:, 1]])
            block[w, 3 + i, 9] = 1.0
            block[w, 3 + i, 10] = 1.0  # offline_pose_candidate flag
    return np.concatenate([features, block], -1)


def build_features(sequence,observations,valid,start,end,slots,image_size,max_objects=8):
    """Only [start,end) influences input; camera anchor is the last history frame."""
    W=end-start;E=3+max_objects
    features=np.zeros((W,E,FEATURE_DIM),np.float32)
    entity_valid=np.zeros((W,E),bool)
    kind=np.asarray([0,1,2]+[3]*max_objects,np.int64)
    times=sequence['timestamps'][start:end]
    hand_valid=sequence['hand_valid'][start:end].copy()
    hands=sequence['hands_world'][start:end]
    anchor=sequence['camera_to_world'][end-1]
    anchored=np.einsum('whji,ik->whjk',hands-anchor[:3,3],anchor[:3,:3])
    hand_valid &= np.isfinite(anchored).all(axis=(2,3))
    uv=observations['hand_uv_native'][start:end]/np.asarray(image_size,np.float32)
    projected=observations['hand_projection_valid'][start:end] & np.isfinite(uv).all(axis=(2,3))
    hand_boxes=np.concatenate((np.nan_to_num(uv,nan=0).min(2),np.nan_to_num(uv,nan=0).max(2)),-1)
    hand_bbox=np.concatenate(((hand_boxes[...,:2]+hand_boxes[...,2:])*.5,hand_boxes[...,2:]-hand_boxes[...,:2]),-1)
    features[:,:2,0:4]=np.where(projected[...,None],hand_bbox,0)
    features[:,:2,4:6]=_history_velocity(features[:,:2,0:2],projected,times)
    features[:,:2,12:54]=np.where(projected[...,None],np.nan_to_num(uv.reshape(W,2,42)),0)
    features[:,:2,54:117]=np.where(hand_valid[...,None],np.nan_to_num(anchored.reshape(W,2,63)),0)
    features[:,:2,117]=projected;features[:,:2,118]=hand_valid
    entity_valid[:,:2]=hand_valid|projected
    both=entity_valid[:,:2].all(1)
    both_projected=projected.all(1)
    union=np.concatenate((hand_boxes[...,:2].min(1),hand_boxes[...,2:].max(1)),-1)
    features[:,2,0:4]=np.where(both_projected[:,None],np.concatenate(((union[:,:2]+union[:,2:])*.5,union[:,2:]-union[:,:2]),-1),0)
    features[:,2,4:6]=_history_velocity(features[:,2:3,0:2],both_projected[:,None],times)[:,0]
    # Both-hands is an aggregate entity, not a fictitious 21-joint skeleton.
    entity_valid[:,2]=both
    for i,j in enumerate(slots):
        at=3+i;current=valid[start:end,j]
        boxes=observations['boxes_xyxy_native'][start:end,j]
        features[:,at,0:4]=np.where(current[:,None],np.nan_to_num(_bbox_to_features(boxes,image_size)),0)
        features[:,at,6:12]=np.where(current[:,None],np.nan_to_num(observations['rgb_stats'][start:end,j]),0)
        features[:,at,119]=np.where(current,observations['source_code'][start:end,j]==1,0)
        entity_valid[:,at]=current
    features[:,3:,4:6]=_history_velocity(features[:,3:,0:2],entity_valid[:,3:],times)
    features[~entity_valid]=0
    anchored[~hand_valid]=np.nan
    return {'entity_features':features,'entity_valid':entity_valid,'entity_kind':kind,
            'hand_seq':anchored.astype(np.float32),'hand_valid':hand_valid,
            'anchor_camera_to_world':anchor.astype(np.float32)}


def target_edges(review,entity_ids,event_types,anchor,target):
    E=len(entity_ids);K=len(event_types)
    values=np.zeros((E,E,K),np.float32);mask=np.zeros((E,E,K),bool)
    slot={v:i for i,v in enumerate(entity_ids) if v};vocab={(v['layer'],v['code']):i for i,v in enumerate(event_types)}
    source_records=[];dropped=Counter()
    for record in review.get('state_segments',[])+review.get('transition_events',[]):
        layer=record['layer'];record_id=record.get('id')
        if record.get('silver_label_valid') is not True or record.get('annotation_source')!='assistant_visual_review':
            continue
        if (layer,record['type']) not in vocab:
            dropped['outside_frozen_vocab']+=1;continue
        if layer=='transition':
            lo,hi=record['boundary_frame_range']
            before,after=record.get('before_evidence_frame'),record.get('after_evidence_frame')
            existence=(record.get('occurrence_confirmed') is True and record.get('before_after_observed') is True
                and record.get('censored') is False
                and type(before) is int and type(after) is int and before<lo<=after<hi
                and before in record['evidence_frames'] and after in record['evidence_frames'])
            if not existence or not anchor<lo or hi-1>target:
                if hi>anchor+1 and lo<=target:dropped['transition_range_not_confirmed_or_not_fully_in_horizon']+=1
                continue
            value=1;temporal='occurrence_in_prediction_horizon'
        else:
            if not record['start_frame']<=target<record['end_frame'] or target not in record['evidence_frames']:
                continue
            value=record['value'];temporal='state_at_target_frame'
        subject=record.get('subject');identity=record.get('object_id');secondary=record.get('secondary_object_id')
        if subject=='object' and secondary:
            src,dst=identity,secondary
        elif subject in HAND_IDS and identity and secondary is None:
            src,dst=subject,identity
        else:
            dropped['unsupported_or_ambiguous_participants']+=1;continue
        if src not in slot or dst not in slot:
            dropped['entity_absent_from_observed_history']+=1;continue
        at=(slot[src],slot[dst],vocab[(layer,record['type'])])
        if mask[at] and values[at]!=value:raise ValueError('Conflicting known graph labels')
        values[at]=value;mask[at]=True
        source_records.append({'record_id':record_id,'source':src,'target':dst,'layer':layer,'type':record['type'],
            'value':value,'temporal_scope':temporal,'cell':list(at)})
    return values,mask,source_records,dict(dropped)


class EgoDemoGraphV5Dataset(Dataset):
    def __init__(self,root,split='train',*,diagnostic=False):
        self.root=Path(root);self.index=read(self.root/'training_index.json');self.split=split
        self.diagnostic=diagnostic
        if not diagnostic:
            release=read(self.root/self.index['admission_path'])
            if (release.get('interaction_training_allowed') is not True
                    or release.get('training_index_sha256')!=sha(self.root/'training_index.json')
                    or release.get('training_scope')!=self.index['training_scope']):
                raise ValueError('This index has not passed interaction-subset admission')
            evidence=release.get('evidence_sha256')
            if not isinstance(evidence,dict) or not REQUIRED_ADMISSION_EVIDENCE.issubset(evidence):
                raise ValueError('Training admission lacks required evidence bindings')
            for name,expected in evidence.items():
                if sha(self.root/name)!=expected:raise ValueError('Training admission evidence changed')
        info=self.index['splits'][split];path=self.root/info['path']
        if sha(path)!=info['sha256']:raise ValueError('Window file changed after freeze')
        self.windows=[json.loads(line) for line in path.read_text().splitlines() if line]
        self.sequences={v['sequence_id']:v for v in self.index['sequences']}
        self.event_types=self.index['event_types'];self.cache=None

    def _load_sequence(self,uid):
        if self.cache is not None and self.cache[0]==uid:return self.cache[1]
        row=self.sequences[uid]
        if row['split']!=self.split:raise ValueError('Acquisition crossed fixed split')
        for key,pathkey in [('review_sha256','review_path'),('observations_sha256','observations_path'),('tracking_review_sha256','tracking_review_path')]:
            if sha(self.root/row[pathkey])!=row[key]:raise ValueError('Bound annotation/observation/quality changed')
        sequence_path=self.root/row['sequence_path']
        if sha(sequence_path)!=row['sequence_sha256']:raise ValueError('Exported hand observations changed')
        with np.load(sequence_path,allow_pickle=False) as a:
            # Offline object geometry is deliberately not consumed as input.
            seq={k:a[k].copy() for k in ['timestamps','hands_world','hand_valid','camera_to_world']}
        with np.load(self.root/row['observations_path'],allow_pickle=False) as a:obs={k:a[k].copy() for k in a.files}
        if not np.array_equal(seq['timestamps'],obs['timestamps']):raise ValueError('Observation native time differs')
        review=read(self.root/row['review_path']);quality=read(self.root/row['tracking_review_path'])
        if quality['observations_sha256']!=row['observations_sha256']:raise ValueError('Quality describes other observations')
        valid=approved_observation_mask(obs,quality)
        data=(seq,obs,valid,review,row)
        self.cache=(uid,data)
        return data

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        window=self.windows[index];uid=window['sequence_id']
        seq,obs,valid,review,row=self._load_sequence(uid)
        start,end,target=window['start_frame'],window['end_frame'],window['target_frame']
        config=self.index['window_config']
        if not (0<=start<end<=target<len(seq['timestamps']) and end-start==config['window_frames']
                and seq['timestamps'][target]-seq['timestamps'][end-1]>=config['horizon_seconds']):
            raise ValueError('Window lacks the required strictly historical horizon')
        if not covered_range(review,start,target+1):raise ValueError('Historical-to-target interval has an unreviewed gap')
        slots=select_objects(obs,valid,start,end,config['max_objects'],config['minimum_observations'])
        object_ids=[str(obs['object_ids'][j]) for j in slots]+['']*(config['max_objects']-len(slots))
        ids=list(HAND_IDS)+object_ids
        features=build_features(seq,obs,valid,start,end,slots,row['native_image_size'],config['max_objects'])
        if self.index.get('feature_version')=='v6_pose':
            from src.mamba.pose_features_v6 import PoseTable
            seed_doc=read(self.root/'visual_observations'/uid/'seeds.json')
            seed_names={x['entity_id']:x.get('name_zh','') for x in seed_doc['seeds']}
            table=PoseTable(self.index.get('pose_refine_root',''),uid,[str(v) for v in obs['object_ids']],seed_names,len(seq['timestamps']))
            features['entity_features']=attach_pose_features(features['entity_features'],
                table.world_series(len(seq['timestamps'])) if table.ok else None,start,end,slots)
        values,mask,records,dropped=target_edges(review,ids,self.event_types,end-1,target)
        pair=features['entity_valid'].any(0)
        mask &= pair[:,None,None] & pair[None,:,None]
        if not mask.any():raise ValueError('Published window has no usable graph supervision')
        names={e.get('entity_id',e.get('id')):e.get('name_zh') for e in review['entities']}
        metadata={'sequence_id':uid,'split':self.split,**window,'entity_ids':ids,
            'entity_names_zh':['左手','右手','双手']+[names.get(v,'填充槽位') for v in object_ids],
            'source_records':records,'dropped_targets':dropped,
            'observation_mode':'causal_replay_with_annotated_initialization',
            'whole_object_6d_certified':False,'full_annotation_complete':False}
        return {**{k:torch.from_numpy(v) for k,v in features.items()},
                'graph_values':torch.from_numpy(values),'graph_label_valid':torch.from_numpy(mask),'metadata':metadata}


def collate_graph_v5(samples):
    if not samples:raise ValueError('Empty graph batch')
    return {k:[s[k] for s in samples] if k=='metadata' else torch.stack([s[k] for s in samples]) for k in samples[0]}
