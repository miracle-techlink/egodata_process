"""Merge acquisition receipts with explicit preview decisions; never authorize training."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def file_asset(path, role, shared=False):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'role': role, 'path': str(path), 'sha256': digest.hexdigest(),
            'size_bytes': path.stat().st_size, 'identity_scope': 'shared' if shared else 'per_sequence'}


def merge_candidates(sources, output, preview_review=None, exposure_registry=None, reference_audit=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite ledger: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    review = json.loads(Path(preview_review).read_text()) if preview_review else {'records': []}
    decisions = {record['sequence_id']: record for record in review['records']}
    if len(decisions) != len(review['records']):
        raise ValueError('Duplicate preview decisions')
    review_asset = file_asset(preview_review, 'preview_review', True) if preview_review else None
    exposures = json.loads(Path(exposure_registry).read_text()) if exposure_registry else {'records': []}
    seen_before = {(record['dataset'], record['acquisition_id']) for record in exposures['records']}
    exposure_asset = file_asset(exposure_registry, 'evaluation_history', True) if exposure_registry else None
    audit = json.loads(Path(reference_audit).read_text()) if reference_audit else {'sequences': []}
    audited = {(item['subject_id'], item['acquisition_id'], item['camera_id']): item
               for item in audit['sequences']}
    audit_asset = file_asset(reference_audit, 'structural_audit', True) if reference_audit else None
    records, provenance, seen = [], [], set()
    for source in sources:
        path = Path(source).resolve()
        payload = json.loads(path.read_text())
        provenance.append(file_asset(path, 'source_ledger', True))
        for original in payload['records']:
            record = copy.deepcopy(original)
            identity = record['sequence_id']
            if identity in seen:
                raise ValueError(f'Duplicate sequence ID: {identity}')
            seen.add(identity)
            record['assets'] = copy.deepcopy(record.get('assets', []))
            for asset in record['assets']:
                asset['path'] = str((path.parent / asset['path']).resolve())
            record['assets'].append(file_asset(path, 'source_ledger', True))
            record.setdefault('evidence', {})
            if record.get('split') in ('train', 'val', 'test'):
                record['previous_split'] = record['split']
                if record['split'] == 'test':
                    record['previously_evaluated'] = True
            record['split'] = None
            record['training_allowed'] = False
            record['source_candidate_path'] = str(path)
            if exposure_asset:
                record['assets'].append(copy.deepcopy(exposure_asset))
                if (record.get('dataset'), record.get('acquisition_id')) in seen_before:
                    record['previously_evaluated'] = True
                    record['evaluation_history'] = 'Previously used in development, selection or evaluation; not unseen test.'
            if record.get('dataset') == 'EgoDemo':
                for asset in list(record['assets']):
                    if asset['role'] == 'data':
                        reference = copy.deepcopy(asset)
                        reference['role'] = 'hand_world_reference'
                        record['assets'].append(reference)
                record['pool'] = 'auxiliary'
                record['camera_motion'] = 'moving_uncompensated'
                record['capabilities']['hand_observations'] = False
                record['capabilities']['hand_metric_gt'] = False
                record['duration_definition'] = 'frame_count_over_fps'
                if record.get('technical_long_pass') is True:
                    record['continuity'] = {'continuous': True, 'concatenated': False,
                                            'repeated_frames': False, 'interpolated': False,
                                            'fps_modified': False}
                    for key in ('continuity', 'timestamp_source', 'decode', 'annotation_alignment'):
                        record['evidence'][key] = {'asset_role': 'source_ledger',
                            'description': 'Per-sequence full video decode and paired source timestamp audit; no frame fabrication or concatenation.'}
                record['evidence']['scene_id'] = {'asset_role': 'source_ledger',
                                                'description': 'Official source scene identifier retained.'}
            elif record.get('dataset') == 'HO-Cap':
                record['camera_motion'] = 'fixed'
                for key in ('subject_id', 'action_labels'):
                    record['evidence'][key] = {'asset_role': 'source_ledger',
                        'description': 'Source meta.yaml subject_id/task_id preserved in acquisition receipt.'}
                checked = audited.get((record['subject_id'], record['acquisition_id'], record['camera_id']))
                if checked:
                    rgb_path = next(Path(asset['path']) for asset in record['assets'] if asset['role'] == 'rgb')
                    receipt_asset = file_asset(rgb_path.parent.parent / 'receipt.json', 'capture_receipt')
                    if receipt_asset['sha256'] != checked['receipt_sha256']:
                        raise ValueError('Capture receipt changed after structural audit')
                    if checked['frame_count'] != record['frame_count']:
                        raise ValueError('Structural audit and source frame counts disagree')
                    record['assets'].extend([copy.deepcopy(audit_asset), receipt_asset])
                    record['quality']['decode'] = 'pass' if checked['structural_pass'] else 'fail'
                    record['structural_reference_audit'] = copy.deepcopy(checked)
                    record['evidence']['decode'] = {'asset_role': 'structural_audit',
                        'description': 'Every RGB/depth/reference frame decoded and hashed; internal K/SO3/3D-to-2D checks only, not independent annotation accuracy.'}
                    record['continuity'] = {'continuous': checked['structural_pass'], 'concatenated': False,
                                            'repeated_frames': False, 'interpolated': False, 'fps_modified': False}
                    record['evidence']['continuity'] = {'asset_role': 'capture_receipt',
                        'description': 'Original consecutive frame indices with all modalities; physical timebase remains unverified.'}
            decision = decisions.get(identity)
            if decision:
                record['assets'].append(copy.deepcopy(review_asset))
                record['human_verified'] = decision.get('human_hand_visible') is True
                record['preview_decision'] = copy.deepcopy(decision)
                record['evidence']['human_verified'] = {'asset_role': 'preview_review',
                    'description': 'Assistant visually inspected listed RGB previews; not a user approval or full-sequence annotation review.'}
                record['quality']['visual_review'] = ('fail' if decision.get('hand_view_suitable') is False else 'pending')
            records.append(record)
    unused = sorted(set(decisions) - seen)
    if unused:
        raise ValueError(f'Preview decisions without source records: {unused}')
    ledger = {'schema_version': 'human_pose_dataset_v1', 'records': records,
              'source_ledgers': provenance, 'training_allowed': False,
              'status': 'candidate_only', 'scope': 'Data acquisition and review; no training or model evaluation.'}
    with output.open('x') as stream:
        stream.write(json.dumps(ledger, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--preview-review')
    parser.add_argument('--exposure-registry')
    parser.add_argument('--reference-audit')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ledger = merge_candidates(args.source, args.output, args.preview_review, args.exposure_registry,
                              args.reference_audit)
    print(json.dumps({'records': len(ledger['records']), 'training_allowed': False}))


if __name__ == '__main__':
    main()
