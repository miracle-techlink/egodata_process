import numpy as np
import pytest

from src.dataset.complete_egodemo import operation_candidates, source_intervals


def test_multiple_verbs_preserve_uncertain_boundaries_and_identity():
    events = {'subtask_annotations': [{'subtask_id': 'a', 'start_time': .1, 'end_time': .4,
                                      'description': 'Insert the roll and close the cover.'}]}
    intervals, candidates = source_intervals(events, np.arange(20) / 30)
    assert intervals[0]['start_frame'] == 3
    assert intervals[0]['end_frame'] == 12
    assert {c['operation_type'] for c in candidates} == {'insert', 'close'}
    assert all(c['subject'] is None and c['object_id'] is None for c in candidates)
    assert all(not c['label_valid'] and c['temporal_precision'] == 'source_interval_only' for c in candidates)
    assert all('start_frame' not in c for c in candidates)


def test_word_boundaries_do_not_turn_unwind_into_wind():
    assert {x['operation_type'] for x in operation_candidates('Unwind the cable and unfold the paper.')} == {'unwind', 'unfold'}


def test_unrecognized_source_semantics_are_not_none():
    events = {'subtask_annotations': [{'subtask_id': 'a', 'start_time': 0, 'end_time': .2,
                                      'description': 'Finish preparation.'}]}
    intervals, candidates = source_intervals(events, np.arange(10) / 30)
    assert len(intervals) == 1 and candidates == []
    assert intervals[0]['review_status'] == 'pending'


@pytest.mark.parametrize('start,end', [(-1, .1), (.2, .1), (0, 10)])
def test_invalid_time_rejected(start, end):
    with pytest.raises(ValueError):
        source_intervals({'subtask_annotations': [{'start_time': start, 'end_time': end}]}, np.arange(10) / 30)


def test_negated_text_still_cannot_become_a_valid_label():
    _, candidates = source_intervals({'subtask_annotations': [{'subtask_id': 'x', 'start_time': 0,
        'end_time': .2, 'description': 'Do not open the lid.'}]}, np.arange(10) / 30)
    assert candidates[0]['label_valid'] is False
