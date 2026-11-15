import copy
import pytest
from src.annotation.egodemo_review import validate_draft


def draft():
    return {'sequence_id': 'seq', 'source_selection_sha256': 'sha', 'revision': 0,
            'entities': [{'id': 'obj_001', 'name_zh': '电缆', 'rigidity': 'deformable',
                          'role': 'workpiece', 'tracklets': ['tracklet_0001']}],
            'interactions': [{'layer': 'relation', 'type': 'contact', 'subject': 'left_hand',
                              'object_id': 'obj_001', 'start_frame': 2, 'end_frame': 3,
                              'review_status': 'accepted', 'label_valid': True}],
            'training_allowed': True}


def validate(d):
    return validate_draft(d, 'seq', 'sha', 10, {'tracklet_0001'})


def test_client_cannot_self_approve_draft():
    result = validate(draft())
    assert not result['training_allowed']
    assert not result['interactions'][0]['label_valid']
    assert result['interactions'][0]['review_status'] == 'pending'


def test_duplicate_track_identity_is_rejected():
    d = draft()
    d['entities'].append({**copy.deepcopy(d['entities'][0]), 'id': 'obj_002'})
    with pytest.raises(ValueError):
        validate(d)


@pytest.mark.parametrize('start,end', [(0, 0), (-1, 2), (9, 11), (1.5, 3)])
def test_invalid_native_boundary_rejected(start, end):
    d = draft()
    d['interactions'][0].update(start_frame=start, end_frame=end)
    with pytest.raises(ValueError):
        validate(d)


def test_unknown_source_or_object_rejected():
    d = draft()
    d['source_selection_sha256'] = 'changed'
    with pytest.raises(ValueError):
        validate(d)
    d = draft()
    d['interactions'][0]['object_id'] = 'obj_missing'
    with pytest.raises(ValueError):
        validate(d)

@pytest.mark.parametrize('points', [[], [[1,1],[2,2],[3,3]], [[-1,2],[20,2],[20,20]], [[1,1],[101,1],[20,20]]])
def test_invalid_surface_seed_is_rejected(points):
    d=draft();d['entities'][0]['seed_polygons']=[{'frame_index':2,'points_native':points}]
    with pytest.raises(ValueError):
        validate_draft(d,'seq','sha',10,{'tracklet_0001'},image_size=[100,100])


def test_surface_seed_keeps_native_coordinates_and_cannot_self_approve():
    d=draft();points=[[1,1],[30,1],[30,40]]
    d['entities'][0]['seed_polygons']=[{'frame_index':2,'points_native':points,'label_valid':True,'review_status':'accepted'}]
    result=validate_draft(d,'seq','sha',10,{'tracklet_0001'},image_size=[100,100])
    seed=result['entities'][0]['seed_polygons'][0]
    assert seed['points_native']==points
    assert not seed['label_valid'] and seed['review_status']=='pending'
