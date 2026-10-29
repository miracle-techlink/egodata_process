import json

import pytest

from src.dataset.assemble_human_dataset import file_asset, merge_candidates


def fixture_source(tmp_path):
    payload = tmp_path / 'source.bin'
    payload.write_bytes(b'actual test payload')
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'records': [{'sequence_id': 'fixture', 'dataset': 'EgoDemo',
        'assets': [{'role': 'data', 'path': payload.name, 'sha256': 'not_a_real_digest'}],
        'quality': {'visual_review': 'pending'}, 'capabilities': {'hand_observations': True},
        'technical_long_pass': False, 'human_verified': False}]}))
    return source


def test_no_reference_promoted_to_observations(tmp_path):
    source = fixture_source(tmp_path)
    result = merge_candidates([source], tmp_path / 'output.json')
    record = result['records'][0]
    assert record['capabilities']['hand_observations'] is False
    assert record['capabilities']['hand_metric_gt'] is False
    assert record['human_verified'] is False
    assert any(asset['role'] == 'hand_world_reference' for asset in record['assets'])
    assert record['training_allowed'] is False


def test_preview_cannot_sign_off_full_quality(tmp_path):
    source = fixture_source(tmp_path)
    review = tmp_path / 'review.json'
    review.write_text(json.dumps({'records': [{'sequence_id': 'fixture',
        'human_hand_visible': True, 'hand_view_suitable': True}]}))
    result = merge_candidates([source], tmp_path / 'output.json', review)
    record = result['records'][0]
    assert record['human_verified'] is True
    assert record['quality']['visual_review'] == 'pending'
    assert record['training_allowed'] is False


def test_duplicate_source_or_output_rejected(tmp_path):
    source = fixture_source(tmp_path)
    with pytest.raises(ValueError, match='Duplicate sequence'):
        merge_candidates([source, source], tmp_path / 'duplicate.json')
    assert not (tmp_path / 'duplicate.json').exists()
    merge_candidates([source], tmp_path / 'output.json')
    with pytest.raises(FileExistsError):
        merge_candidates([source], tmp_path / 'output.json')


def test_unknown_review_target_rejected(tmp_path):
    source = fixture_source(tmp_path)
    review = tmp_path / 'review.json'
    review.write_text(json.dumps({'records': [{'sequence_id': 'wrong'}]}))
    with pytest.raises(ValueError, match='without source records'):
        merge_candidates([source], tmp_path / 'output.json', review)


def test_history_registry_marks_old_acquisition_seen(tmp_path):
    source = fixture_source(tmp_path)
    data = json.loads(source.read_text())
    data['records'][0]['acquisition_id'] = 'previous_test'
    source.write_text(json.dumps(data))
    registry = tmp_path / 'history.json'
    registry.write_text(json.dumps({'records': [{'dataset': 'EgoDemo', 'acquisition_id': 'previous_test'}]}))
    result = merge_candidates([source], tmp_path / 'output.json', exposure_registry=registry)
    assert result['records'][0]['previously_evaluated'] is True


def test_source_test_history_survives_candidate_merge(tmp_path):
    source = fixture_source(tmp_path)
    data = json.loads(source.read_text())
    data['records'][0]['split'] = 'test'
    source.write_text(json.dumps(data))
    result = merge_candidates([source], tmp_path / 'output.json')
    assert result['records'][0]['split'] is None
    assert result['records'][0]['previous_split'] == 'test'
    assert result['records'][0]['previously_evaluated'] is True


@pytest.mark.parametrize('tampered', [False, True])
def test_structural_audit_attestation_does_not_approve_coordinates(tmp_path, tampered):
    camera = tmp_path / 'capture' / 'camera'
    camera.mkdir(parents=True)
    rgb = camera / 'frame.jpg'
    rgb.write_bytes(b'fixture frame; this test does not certify RGB decoding')
    receipt = camera.parent / 'receipt.json'
    receipt.write_text('{"fixture": true}')
    record = {'sequence_id': 'fixture', 'dataset': 'HO-Cap', 'acquisition_id': 'capture',
              'subject_id': 'subject', 'camera_id': 'camera', 'frame_count': 2,
              'assets': [file_asset(rgb, 'rgb')], 'quality': {'coordinates': 'pending'}}
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'records': [record]}))
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'sequences': [{'subject_id': 'subject', 'acquisition_id': 'capture',
        'camera_id': 'camera', 'receipt_sha256': file_asset(receipt, 'receipt')['sha256'],
        'frame_count': 2, 'structural_pass': True}]}))
    if tampered:
        receipt.write_text('{"fixture": "changed"}')
        with pytest.raises(ValueError, match='receipt changed'):
            merge_candidates([source], tmp_path / 'output.json', reference_audit=audit)
    else:
        result = merge_candidates([source], tmp_path / 'output.json', reference_audit=audit)
        assert result['records'][0]['quality']['coordinates'] == 'pending'
        assert result['records'][0]['quality']['decode'] == 'pass'
        assert result['training_allowed'] is False
