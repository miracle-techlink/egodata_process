"""Read-only structural audit of downloaded HO-Cap camera/reference pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_label(label, image_shape):
    """Check internal geometry, not agreement with independent physical truth."""
    height, width = image_shape[:2]
    intrinsics = np.asarray(label['cam_K'])
    poses = np.asarray(label['obj_poses'])
    joints = np.asarray(label['hand_joints_3d'])
    pixels = np.asarray(label['hand_joints_2d'])
    segmentation = np.asarray(label['seg_mask'])
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or not np.allclose(intrinsics[2], [0, 0, 1])
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        raise ValueError('invalid intrinsics')
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError('invalid object pose shape')
    valid_objects = np.isfinite(poses).all(axis=(1, 2))
    missing_objects = np.isnan(poses).all(axis=(1, 2)) | (poses == -1).all(axis=(1, 2))
    if not (valid_objects | missing_objects).all():
        raise ValueError('partial nonfinite object pose')
    valid_objects &= ~missing_objects
    present = poses[valid_objects]
    if len(present):
        rotations = present[:, :3, :3]
        if (not np.allclose(present[:, 3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(rotations @ rotations.transpose(0, 2, 1),
                                   np.eye(3), atol=2e-3)
                or not np.allclose(np.linalg.det(rotations), 1, atol=2e-3)
                or np.any(present[:, 2, 3] <= 0)):
            raise ValueError('object pose is not a positive-depth rigid transform')
    if joints.shape != (2, 21, 3) or pixels.shape != (2, 21, 2):
        raise ValueError('unexpected hand joint layout')
    missing = (joints == -1).all(axis=-1) | np.isnan(joints).all(axis=-1)
    finite = np.isfinite(joints).all(axis=-1)
    if not (finite | missing).all():
        raise ValueError('partial nonfinite hand joint')
    valid = finite & ~missing
    if np.any(joints[..., 2][valid] <= 0):
        raise ValueError('hand reference is not in a positive-depth camera frame')
    if segmentation.shape != (height, width) or not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError('segmentation and RGB dimensions/type disagree')
    in_frame = (valid & np.isfinite(pixels).all(axis=-1)
               & (pixels[..., 0] >= 0) & (pixels[..., 0] < width)
               & (pixels[..., 1] >= 0) & (pixels[..., 1] < height))
    projected = joints[in_frame] @ intrinsics.T
    projected = projected[:, :2] / projected[:, 2:3]
    residuals = np.linalg.norm(projected - pixels[in_frame], axis=-1)
    if residuals.size and float(residuals.max()) > 2.0:
        raise ValueError('published 3D/2D hand reference reprojection exceeds 2px')
    return {'valid_objects': int(valid_objects.sum()),
            'valid_hand_joints': int(valid.sum()),
            'in_frame_hand_joints': int(in_frame.sum()),
            'hand_slots_with_in_frame_joints': int(in_frame.any(axis=1).sum()),
            'complete_hands_in_frame': int(in_frame.all(axis=1).sum()),
            'reprojection_residuals_px': residuals.tolist()}


def audit_capture(directory, preview_path=None):
    directory = Path(directory).resolve()
    receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    camera = receipt['camera_id']
    count = receipt['frame_count']
    if receipt['status'] != 'complete' or not receipt['contiguous_all_modalities'] or count < 2:
        raise ValueError('capture receipt is incomplete')
    files = {item['path']: item for item in receipt['files']}
    if len(files) != len(receipt['files']) or len(files) != count * 3:
        raise ValueError('camera receipt has duplicate or unexpected files')
    expected = {f'{camera}/{camera}-{frame:06d}.{suffix}'
                for frame in range(count) for suffix in ('color.jpg', 'depth.png', 'label.npy')}
    if set(files) != expected:
        raise ValueError('noncontiguous or mismatched frame modalities')
    report = {'subject_id': receipt['subject_id'], 'acquisition_id': receipt['acquisition_id'],
              'camera_id': camera, 'frame_count': count, 'decoded_frames': 0,
              'receipt_sha256': digest_file(receipt_path), 'verified_assets': 0,
              'errors': [], 'frames_with_in_frame_reference_hand': 0,
              'frames_with_complete_reference_hand_in_frame': 0,
              'human_verified': False, 'quality_scope': 'structural_reference_audit_only',
              'independent_accuracy': None, 'training_allowed': False,
              'hand_slot_mapping_verified': False, 'unit_scale_verified': False,
              'reference_in_frame_is_not_occlusion_or_detection_success': True}
    preview_frames = []
    residuals = []
    for frame in range(count):
        try:
            paths = {}
            for suffix in ('color.jpg', 'depth.png', 'label.npy'):
                relative = f'{camera}/{camera}-{frame:06d}.{suffix}'
                path = directory / relative
                if not path.resolve().is_relative_to(directory):
                    raise ValueError('asset symlink escapes capture directory')
                if path.stat().st_size != files[relative]['bytes'] or digest_file(path) != files[relative]['sha256']:
                    raise ValueError(f'asset hash/size changed: {relative}')
                paths[suffix] = path
                report['verified_assets'] += 1
            image = cv2.imread(str(paths['color.jpg']), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(paths['depth.png']), cv2.IMREAD_UNCHANGED)
            if image is None or depth is None or depth.dtype != np.uint16 or depth.shape != image.shape[:2]:
                raise ValueError('RGB/depth decode, dimensions or uint16 depth failed')
            with np.load(paths['label.npy'], allow_pickle=False) as label:
                checked = check_label(label, image.shape)
            residuals.extend(checked['reprojection_residuals_px'])
            report['decoded_frames'] += 1
            report['frames_with_in_frame_reference_hand'] += int(checked['in_frame_hand_joints'] > 0)
            report['frames_with_complete_reference_hand_in_frame'] += int(checked['complete_hands_in_frame'] > 0)
            if frame in (0, count // 2, count - 1):
                thumb = cv2.resize(image, (480, 360))
                cv2.putText(thumb, f'{receipt["subject_id"]} {receipt["acquisition_id"]} f{frame}',
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
                preview_frames.append(thumb)
        except (ValueError, KeyError, OSError, EOFError) as error:
            report['errors'].append({'frame': frame, 'error': f'{type(error).__name__}: {error}'})
    report['structural_pass'] = not report['errors'] and report['decoded_frames'] == count
    report['reference_hand_in_frame_coverage'] = report['frames_with_in_frame_reference_hand'] / count
    report['complete_reference_hand_in_frame_coverage'] = report['frames_with_complete_reference_hand_in_frame'] / count
    report['reprojection_comparisons'] = len(residuals)
    report['reprojection_p95_px'] = float(np.quantile(residuals, .95)) if residuals else None
    report['reprojection_max_px'] = float(max(residuals)) if residuals else None
    if preview_path and preview_frames:
        path = Path(preview_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), np.concatenate(preview_frames, axis=1)):
            raise OSError(f'Could not save {path}')
        report['preview_path'] = str(path.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for receipt in sorted(args.captures_root.glob('*/*/receipt.json')):
        name = f'{receipt.parent.parent.name}-{receipt.parent.name}'
        report = audit_capture(receipt.parent, args.out_dir / f'{name}.jpg')
        results.append(report)
        print(name, report['structural_pass'], report['decoded_frames'], flush=True)
    payload = {'structural_pass': bool(results) and all(item['structural_pass'] for item in results),
               'sequences': results, 'total_frames': sum(item['decoded_frames'] for item in results),
               'training_allowed': False, 'inference_or_training_run': False}
    (args.out_dir / 'report.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return 0 if payload['structural_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
