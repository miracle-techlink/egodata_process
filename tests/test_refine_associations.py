import numpy as np

from src.annotation.refine_associations import overlap_link,make_proposals,native_runs,co_motion_mask


def synthetic_track(points,poses,frame_count=31):
    return {'frame_index':np.arange(frame_count),'points_camera':np.tile(points,(frame_count,1,1)),
            'point_valid':np.ones((frame_count,len(points)),bool),
            'object_pose_valid':np.ones(frame_count,bool),'object_pose':poses}


def test_same_surface_links_with_correct_transform_direction():
    rng=np.random.default_rng(5);points=rng.normal(size=(40,3))*.025+[0,0,.5]
    a=np.tile(np.eye(4),(31,1,1));a[:,:3,3]=[0,0,.5]
    b_to_a=np.eye(4);b_to_a[:3,3]=[.02,-.01,.03]
    link=overlap_link(synthetic_track(points,a),synthetic_track(points,a@b_to_a))
    assert link is not None
    np.testing.assert_allclose(link['b_to_a'],b_to_a,atol=1e-8)
    recovered=(a@b_to_a)@np.linalg.inv(link['b_to_a'])
    np.testing.assert_allclose(recovered,a,atol=1e-8)


def test_distinct_stationary_objects_do_not_merge():
    points=np.random.default_rng(3).normal(size=(30,3))*.01+[0,0,.5]
    a=np.tile(np.eye(4),(31,1,1));b=a.copy();b[:,0,3]=.15
    assert overlap_link(synthetic_track(points,a),synthetic_track(points+[.15,0,0],b)) is None


def test_reference_drift_prevents_merge():
    points=np.random.default_rng(4).normal(size=(40,3))*.03+[0,0,.5]
    a=np.tile(np.eye(4),(31,1,1));b=a.copy();b[:,0,3]=np.linspace(0,.1,31)
    assert overlap_link(synthetic_track(points,a),synthetic_track(points,b)) is None


def test_missing_frames_do_not_create_false_continuous_contact():
    frame=np.array([0,1,2,20,21,22])
    assert list(native_runs(frame,np.ones(6,bool)))==[(0,3),(3,6)]
    z={'frame_index':np.arange(8),'point_valid':np.ones((8,10),bool),
       'nearest_hand_surface_m':np.c_[[.06,.01,.01,.01,.01,.01,.06,.08],np.full(8,np.nan)]}
    rows=make_proposals('obj_001',z,8)
    assert [r['type'] for r in rows]==['contact','contact_onset']
    assert all(not r['label_valid'] and not r['evidence']['direct_contact_observed'] for r in rows)
    z['nearest_hand_surface_m'][0]=np.nan
    rows=make_proposals('obj_001',z,8)
    assert [r['type'] for r in rows]==['contact']


def test_shared_motion_is_distinct_from_static_nearby_background():
    hand=np.c_[np.linspace(0,.3,60),np.zeros(60),np.full(60,.5)]
    obj=hand+[0,.04,0]
    near=np.full(60,.04);valid=np.ones(60,bool)
    assert co_motion_mask(obj,hand,near,valid).sum()>40
    stationary=np.tile([0,.04,.5],(60,1))
    assert not co_motion_mask(stationary,hand,near,valid).any()
    valid[30]=False
    result=co_motion_mask(obj,hand,near,valid)
    assert not result[30:42].any()
