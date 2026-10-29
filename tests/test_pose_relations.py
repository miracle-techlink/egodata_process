"""Annotation gates and temporal graph semantics, independent of model inference."""

import copy

import pytest

from src.annotation.pose_relations import SCHEMA, frame_graph, validate_annotations


@pytest.fixture
def annotation():
    common = dict(start_frame=2, end_frame=8, evidence_frames=[2, 5, 7],
                  label_source='assistant_visual_review', review_status='pending')
    return dict(
        schema_version=SCHEMA, sequence_id='sample', frame_count=10,
        annotation_mode='offline_noncausal', allowed_as_model_input=False,
        hand_review_frames=[6],
        entities=[dict(entity_id='left_hand', kind='hand', side='left', display_name='左手'),
                  dict(entity_id='right_hand', kind='hand', side='right', display_name='右手'),
                  dict(entity_id='obj_1', kind='object', display_name='物体'),
                  dict(entity_id='table', kind='context', display_name='桌面')],
        relations=[dict(relation_id='grip', source='left_hand', target='obj_1',
                        predicate='grasp_control', **common)],
        transitions=[dict(event_id='start', source='left_hand', target='obj_1',
                          event_type='grasp_onset', **common)],
        cooperations=[dict(cooperation_id='single', mode='single_hand_manipulation',
                           participants=[dict(entity_id='left_hand', role='manipulator'),
                                         dict(entity_id='obj_1', role='patient')],
                           relation_ids=['grip'], **common)],
        task_semantics=[dict(task_id='task', description='拿起物体', **common)],
    )


def test_parallel_relations_and_missing_observations_are_preserved(annotation):
    annotation['relations'].append(dict(annotation['relations'][0], relation_id='motion', predicate='co_motion'))
    validate_annotations(annotation, ['obj_1'], 10)
    original = copy.deepcopy(annotation)
    g = frame_graph(annotation, 6, {'left_hand': False, 'obj_1': True})
    assert len(g['relations']) == 2
    assert g['relations'][0]['source_observed'] is False
    assert g['handedness_review_required'] is True
    assert g['training_allowed'] is False
    assert g['allowed_as_model_input'] is False
    assert annotation == original


@pytest.mark.parametrize('frame,expected', [(1, 0), (2, 1), (7, 1), (8, 0)])
def test_half_open_native_frame_intervals(annotation, frame, expected):
    assert len(frame_graph(annotation, frame, {})['relations']) == expected


@pytest.mark.parametrize('start,end,evidence', [(True, 8, [2]), (-1, 8, [2]),
    (2, 11, [5]), (2, 2, [2]), (2, 8, [8]), (2, 8, []), (2, 8, [2.5])])
def test_invalid_timing_and_evidence_rejected(annotation, start, end, evidence):
    annotation['relations'][0].update(start_frame=start, end_frame=end, evidence_frames=evidence)
    with pytest.raises(ValueError):
        validate_annotations(annotation, ['obj_1'], 10)


def test_unlinked_object_name_cannot_become_a_track(annotation):
    annotation['relations'][0]['target'] = 'looks_like_a_bottle'
    with pytest.raises(ValueError, match='registered entities'):
        validate_annotations(annotation, ['obj_1'], 10)


def test_missing_pose_entity_rejected(annotation):
    with pytest.raises(ValueError, match='original pose IDs'):
        validate_annotations(annotation, ['obj_1', 'obj_2'], 10)


def test_hand_identity_cannot_be_inferred_from_screen_side(annotation):
    annotation['entities'][0]['side'] = 'right'
    with pytest.raises(ValueError, match='anatomical'):
        validate_annotations(annotation, ['obj_1'], 10)


def test_assistant_visual_review_cannot_be_gold(annotation):
    annotation['relations'][0].update(review_status='accepted', reviewer='assistant', reviewed_at='today')
    with pytest.raises(ValueError, match='human review'):
        validate_annotations(annotation, ['obj_1'], 10)


@pytest.mark.parametrize('mode', ['bimanual_joint', 'bimanual_complementary', 'handover'])
def test_single_hand_cannot_claim_bimanual_cooperation(annotation, mode):
    annotation['cooperations'][0]['mode'] = mode
    with pytest.raises(ValueError, match='both hands'):
        validate_annotations(annotation, ['obj_1'], 10)


def test_support_direction_is_object_to_support(annotation):
    annotation['relations'][0]['predicate'] = 'supported_by'
    with pytest.raises(ValueError, match='entity kinds'):
        validate_annotations(annotation, ['obj_1'], 10)
    annotation['relations'][0].update(source='obj_1', target='table')
    annotation['cooperations'] = []
    validate_annotations(annotation, ['obj_1'], 10)


def test_bimanual_graph_preserves_both_hands(annotation):
    annotation['relations'].append(dict(annotation['relations'][0], relation_id='right_grip', source='right_hand'))
    c = annotation['cooperations'][0]
    c['mode'] = 'bimanual_joint'
    c['participants'].append(dict(entity_id='right_hand', role='co_manipulator'))
    c['relation_ids'].append('right_grip')
    validate_annotations(annotation, ['obj_1'], 10)
    g = frame_graph(annotation, 5, {'left_hand': True, 'right_hand': True, 'obj_1': True})
    assert {r['source'] for r in g['relations']} == {'left_hand', 'right_hand'}


def test_cooperation_cannot_refer_to_unrelated_interval(annotation):
    annotation['cooperations'][0].update(start_frame=8, end_frame=10, evidence_frames=[8])
    with pytest.raises(ValueError, match='do not overlap'):
        validate_annotations(annotation, ['obj_1'], 10)


def test_rejected_relations_do_not_appear_in_frame_graph(annotation):
    annotation['relations'][0]['review_status'] = 'rejected'
    assert frame_graph(annotation, 4, {})['relations'] == []
    assert frame_graph(annotation, 4, {})['cooperations'] == []


def test_offline_features_cannot_be_silently_used_as_input(annotation):
    annotation['allowed_as_model_input'] = True
    with pytest.raises(ValueError, match='cannot be model input'):
        validate_annotations(annotation, ['obj_1'], 10)
