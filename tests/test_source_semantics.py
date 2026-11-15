import json
import pytest
from src.annotation.source_semantics import parse_draft


def test_model_draft_preserves_unknown_identity_and_pending_review():
    rows = [{'source_interval_id':'a', 'description_original':'Screw the orange lid onto the jar.'}]
    response = [{'id':'a','zh':'将橙色盖子旋到罐上。','operations':[{'type':'tighten','evidence':'Screw'}],
                 'objects':[{'source_phrase':'orange lid','name_zh':'橙色盖子','rigidity':'rigid'}]}]
    parsed = parse_draft(json.dumps(response), rows)[0]
    assert not parsed['label_valid'] and parsed['object_id'] is None
    assert parsed['object_mentions'][0]['stable_object_id'] is None


def test_hallucinated_evidence_is_rejected():
    rows = [{'source_interval_id':'a', 'description_original':'Hold the lid.'}]
    response = [{'id':'a','zh':'拿住盖子。','operations':[{'type':'tighten','evidence':'Screw'}]}]
    parsed = parse_draft(json.dumps(response), rows)[0]
    assert [op['type'] for op in parsed['operations']] == ['hold']


def test_model_material_guess_cannot_make_paper_rigid():
    from src.annotation.source_semantics import rigidity_candidate
    assert rigidity_candidate('paper', 'Bend the paper.') == 'deformable'
    assert rigidity_candidate('box', 'Fold the box.') == 'deformable'
    assert rigidity_candidate('box', 'Place the box.') == 'unknown'
    assert rigidity_candidate('paper clips', 'Sort the paper clips.') == 'rigid'


def test_missing_source_event_is_rejected():
    with pytest.raises(ValueError):
        parse_draft('[]', [{'source_interval_id':'a','description_original':'Hold the lid.'}])


def test_repair_drops_hallucinated_noun_and_preserves_pending():
    from src.annotation.repair_source_semantics import repair_response
    raw = json.dumps([{'id': 'a', 'zh': '拿住盖子', 'objects': [
        {'source_phrase': '盖子', 'name_zh': '盖子'}]}]) + '\nextra prose'
    rows, audit = repair_response(raw, [{'source_interval_id': 'a', 'description_original': 'Hold the lid.'}],
                                  {'lid': '盖子', 'laser': '激光器'})
    assert rows[0]['object_mentions'][0]['source_phrase'] == 'lid'
    assert len(rows[0]['object_mentions']) == 1
    assert audit['trailing_text_removed'] and len(audit['dropped_mentions']) == 1
    assert not rows[0]['label_valid']


def test_repair_cannot_hide_missing_source_rows():
    from src.annotation.repair_source_semantics import repair_response
    with pytest.raises(ValueError):
        repair_response('[] ignored', [{'source_interval_id': 'a', 'description_original': 'Hold.'}], {})
