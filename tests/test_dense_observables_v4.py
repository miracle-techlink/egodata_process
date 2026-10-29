import numpy as np

from src.annotation.build_dense_observables_v4 import (
    native_surface_distance, proximity_boundaries, sparse_frame_records, velocity,
)


def test_native_distance_respects_hand_missing_identity_exclusion_and_point_count():
    hands=np.zeros((3,2,21,3),np.float32);hands[:,1,:,0]=1
    valid=np.ones((3,2),bool);valid[1,0]=False
    frames=np.repeat([0,1,2],8);objects=np.zeros(24,np.int32)
    xyz=np.tile([.02,0,0],(24,1)).astype(np.float32)
    d,k,c=native_surface_distance(hands,valid,frames,objects,xyz,2)
    assert np.isclose(d[0,0,0],.02) and np.isclose(d[0,0,1],.98)
    assert not k[1,0,0] and np.isnan(d[1,0,0])
    assert not k[:,1].any() and c[:,0].tolist()==[8,8,8]
    d,k,c=native_surface_distance(hands,valid,frames,objects,xyz,2,excluded=[0])
    assert not k.any() and np.isnan(d).all() and not c.any()


def test_velocity_does_not_bridge_missing_or_time_gap():
    x=np.arange(5,dtype=np.float32)[:,None,None]*np.ones((1,1,3))
    t=np.array([0.,.1,.2,2.,2.1]);v=np.ones((5,1),bool);v[1]=False
    result=velocity(x,v,t)
    assert np.isnan(result[:4]).all()
    assert np.allclose(result[4],10)


def test_proximity_boundary_is_only_unverified_candidate_and_requires_adjacent_observations():
    d=np.full((5,1,2),.1,np.float32);d[1:]=.01
    k=np.ones_like(d,bool);k[0,0,1]=False
    result=proximity_boundaries(d,k,np.arange(5)*.1,['a'])
    assert len(result)==1 and result[0]['candidate_boundary_frame']==1
    assert result[0]['candidate_type']=='proximity_enter'
    assert not result[0]['silver_label_valid'] and not result[0]['human_label_valid']


def test_sparse_records_never_fill_interval_interior():
    ptr,idx=sparse_frame_records([{'evidence_frames':[1,4,4],'start_frame':1,'end_frame':5}],6)
    assert np.diff(ptr).tolist()==[0,1,0,0,1,0]
    assert idx.tolist()==[0,0]
