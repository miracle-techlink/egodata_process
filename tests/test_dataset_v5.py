import copy
import numpy as np
import pytest
from src.mamba.dataset_v5 import (select_objects,build_features,target_edges,covered_range,
    approved_observation_mask)


def fixture():
    T,N=90,3
    seq={'timestamps':np.arange(T)/30,'hands_world':np.ones((T,2,21,3),np.float32),
         'hand_valid':np.ones((T,2),bool),'camera_to_world':np.tile(np.eye(4),(T,1,1))}
    obs={'object_ids':np.array(['visual_a','visual_b','visual_future']),
         'boxes_xyxy_native':np.tile(np.array([20,30,80,100],np.float32),(T,N,1)),
         'rgb_stats':np.ones((T,N,6),np.float32)*.3,'source_code':np.full((T,N),2,np.uint8),
         'hand_uv_native':np.full((T,2,21,2),50,np.float32),'hand_projection_valid':np.ones((T,2),bool)}
    valid=np.ones((T,N),bool);valid[:60,2]=False;obs['object_valid']=valid
    return seq,obs,valid


def record(**kwargs):
    return dict({'id':'r','layer':'relation','type':'contact','subject':'left_hand','object_id':'visual_a',
        'secondary_object_id':None,'start_frame':50,'end_frame':80,'evidence_frames':list(range(50,80)),
        'annotation_source':'assistant_visual_review','silver_label_valid':True,'value':1},**kwargs)


def test_future_objects_and_numeric_suffix_do_not_change_inputs():
    seq,obs,valid=fixture();slots=select_objects(obs,valid,10,42)
    assert slots==[0,1]
    before=build_features(seq,obs,valid,10,42,slots,[200,150])
    seq['hands_world'][42:]=1e7;seq['camera_to_world'][42:]=9e8
    obs['boxes_xyxy_native'][42:]=-5e8;obs['rgb_stats'][42:]=np.nan
    valid[42:]=True
    assert select_objects(obs,valid,10,42)==slots
    after=build_features(seq,obs,valid,10,42,slots,[200,150])
    for k in before:np.testing.assert_equal(before[k],after[k])
    assert not before['entity_valid'][:,5:].any()
    assert np.isfinite(before['entity_features']).all()


def test_unknown_and_different_unseen_object_are_not_negatives():
    types=[{'layer':'relation','code':'contact'}]
    review={'state_segments':[record(value=0),record(id='other',object_id='visual_future')],'transition_events':[]}
    values,mask,_,dropped=target_edges(review,['left_hand','right_hand','both_hands','visual_a'],types,31,65)
    assert mask.sum()==1 and values[mask].tolist()==[0]
    assert not mask[1].any() and dropped['entity_absent_from_observed_history']==1


def test_directed_object_edge_keeps_both_instances():
    types=[{'layer':'relation','code':'supported_by'}]
    r=record(type='supported_by',subject='object',secondary_object_id='visual_b')
    v,m,records,_=target_edges({'state_segments':[r]},['left_hand','visual_a','visual_b'],types,31,65)
    assert m[1,2,0] and not m[2,1,0] and v[m].tolist()==[1]
    assert records[0]['source']=='visual_a' and records[0]['target']=='visual_b'


def test_bounded_occurrence_needs_whole_horizon_and_two_witnesses():
    types=[{'layer':'transition','code':'release'}]
    r=record(layer='transition',type='release',boundary_frame_range=[45,52],before_evidence_frame=44,
             after_evidence_frame=51,occurrence_confirmed=True,before_after_observed=True,
             censored=False,evidence_frames=list(range(44,53)))
    review={'transition_events':[r]};ids=['left_hand','visual_a']
    _,mask,records,_=target_edges(review,ids,types,31,65)
    assert mask.sum()==1 and records[0]['temporal_scope']=='occurrence_in_prediction_horizon'
    for anchor,target in [(46,65),(31,50),(45,65)]:
        assert not target_edges(review,ids,types,anchor,target)[1].any()
    r['censored']=True
    assert not target_edges(review,ids,types,31,65)[1].any()


def test_unsupported_visual_signal_not_fabricated_as_scan():
    r=record(layer='visual_signal',type='illumination_onset',boundary_frame_range=[50,51])
    del r['start_frame'];del r['end_frame']
    _,mask,_,drop=target_edges({'transition_events':[r]},['left_hand','visual_a'],[{'layer':'operation','code':'scan'}],31,65)
    assert not mask.any() and drop['outside_frozen_vocab']==1


def test_conflicting_known_state_rejected():
    with pytest.raises(ValueError,match='Conflicting'):
        target_edges({'state_segments':[record(value=0),record(value=1)]},['left_hand','visual_a'],
                      [{'layer':'relation','code':'contact'}],31,65)


def test_history_target_cannot_cross_review_gap():
    review={'reviewed_ranges':[[0,50],[60,90]]}
    assert covered_range(review,0,50)
    assert not covered_range(review,0,61)
    assert covered_range(review,60,90)


def test_quality_rejection_and_missing_observations_win():
    _,obs,valid=fixture()
    q={'accepted_ranges':[{'entity_id':'visual_future','start_frame':0,'end_frame':80},
        {'entity_id':'visual_a','start_frame':10,'end_frame':20}],
       'rejected_ranges':[{'entity_id':'visual_a','start_frame':15,'end_frame':18}]}
    v=approved_observation_mask(obs,q)
    assert not v[:60,2].any() and v[60:80,2].all() and not v[80:,2].any()
    assert v[10:15,0].all() and not v[15:18,0].any() and not v[:,1].any()
    with pytest.raises(ValueError):approved_observation_mask(obs,None)
