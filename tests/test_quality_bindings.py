from src.annotation.quality_bindings import bind_quality_to_tracks


def test_recomputed_track_index_does_not_move_rejection_to_another_object():
    decision={'tracklet_id':'tracklet_0000','source_observation_id':'bad_mask','status':'rejected'}
    tracks=[{'tracklet_id':'tracklet_0000','source_observation_id':'new_good_mask'},
            {'tracklet_id':'tracklet_0007','source_observation_id':'bad_mask'}]
    assert bind_quality_to_tracks([decision],tracks)=={'tracklet_0007':decision}


def test_missing_source_does_not_fall_back_to_reused_track_index():
    decision={'tracklet_id':'tracklet_0000','source_observation_id':'dropped_mask','status':'rejected'}
    assert bind_quality_to_tracks([decision],[{'tracklet_id':'tracklet_0000','source_observation_id':'other'}])=={}
