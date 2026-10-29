"""Complete-capture causal MediaPipe + FoundationPose production and audit.

Source timestamps may be unknown. Playback timestamps never replace them.
Preparation reads only calibration and first-frame segmentation from reference
labels. Inference reads no reference poses or later segmentation. This is
segmentation-assisted initialization, not autonomous detection or a GT benchmark.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


def file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def checked_bytes(asset):
    contents = Path(asset['path']).read_bytes()
    if hashlib.sha256(contents).hexdigest() != asset['sha256']:
        raise ValueError('Input asset changed: ' + asset['path'])
    return contents


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Pose must be finite 4x4')
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3, rtol=0)
            or pose[2, 3] <= 0):
        raise ValueError('Pose must have SO(3), homogeneous row and positive camera Z')
    return pose.copy()


def validate_timestamps(timestamps, frame_count):
    if timestamps is None:
        return [None] * frame_count
    values = json.loads(Path(timestamps).read_text()) if isinstance(timestamps, (str, Path)) else timestamps
    if isinstance(values, dict):
        values = values['timestamps']
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (frame_count,) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(np.diff(values) <= 0)):
        raise ValueError('Source timestamps must match all frames and be finite, increasing seconds')
    return values.tolist()


def prepare_hocap(acquisition_root, subject_id, acquisition_id, camera_id, output_dir,
                  depth_scale=0.001, source_timestamps=None):
    import cv2
    import yaml

    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError('depth_scale must be finite and positive')
    root, destination = Path(acquisition_root).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError('Output directory already exists')
    for identity in (subject_id, acquisition_id, camera_id):
        if not identity or Path(identity).name != identity or identity in {'.', '..'}:
            raise ValueError('Source identifiers must be single path components')
    metadata_path = root / 'labels_selected' / f'{subject_id}-{acquisition_id}.meta.yaml'
    metadata = yaml.safe_load(metadata_path.read_text())
    count = int(metadata['num_frames'])
    object_ids = metadata['object_ids']
    if count < 2 or not object_ids or len(set(object_ids)) != len(object_ids):
        raise ValueError('Need a complete multi-frame capture and unique object IDs')
    for identity in object_ids:
        if not isinstance(identity, str) or Path(identity).name != identity or identity in {'.', '..'}:
            raise ValueError('Object IDs must be safe path components')
    timestamps = validate_timestamps(source_timestamps, count)
    capture = root / 'captures_primary' / subject_id / acquisition_id / camera_id
    for suffix in ('color.jpg', 'depth.png', 'label.npy'):
        expected = {f'{camera_id}-{index:06d}.{suffix}' for index in range(count)}
        actual = {path.name for path in capture.glob(f'*.{suffix}')}
        if actual != expected:
            raise ValueError(f'Full capture {suffix} frame IDs do not match num_frames')
    first_label = capture / f'{camera_id}-000000.label.npy'
    with np.load(first_label, allow_pickle=False) as reference:
        intrinsics = reference['cam_K'].copy()
        segmentation = reference['seg_mask'].copy()
        names = reference['obj_class_names'].tolist()
    if names[:len(object_ids)] != object_ids:
        raise ValueError('First-frame object class order differs from metadata')
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1])):
        raise ValueError('Invalid camera intrinsics')
    frames = []
    image_shape = None
    for index in range(count):
        sources = {kind: file_record(capture / f'{camera_id}-{index:06d}.{suffix}')
                   for kind, suffix in [('rgb', 'color.jpg'), ('depth', 'depth.png')]}
        rgb = cv2.imdecode(np.frombuffer(checked_bytes(sources['rgb']), np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(checked_bytes(sources['depth']), np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
            raise ValueError(f'Invalid aligned RGB/uint16-depth pair at frame {index}')
        if image_shape is None:
            image_shape = list(depth.shape)
        if list(depth.shape) != image_shape or segmentation.shape != depth.shape:
            raise ValueError('Image geometry changed within capture')
        if not np.any(depth > 0):
            raise ValueError(f'No positive depth at frame {index}')
        frames.append({'frame_id': index, **sources})
    masks, objects = [], []
    first_depth = cv2.imdecode(np.frombuffer(checked_bytes(frames[0]['depth']), np.uint8), cv2.IMREAD_UNCHANGED)
    for index, identity in enumerate(object_ids):
        mask = segmentation == index + 1
        if np.count_nonzero(mask & (first_depth > 0)) < 4:
            raise ValueError(f'First-frame object mask has insufficient valid depth: {identity}')
        mesh = root / 'models' / f'{identity}.textured_mesh.obj'
        assets = [file_record(mesh)]
        assets += [file_record(path) for path in sorted(mesh.parent.glob(f'{identity}.textured_mesh.*')) if path != mesh]
        masks.append(mask)
        objects.append({'object_id': identity, 'mesh': assets[0], 'mesh_related_assets': assets[1:],
                        'mesh_unit': 'm', 'mask_value': index + 1})
    destination.mkdir(parents=True)
    (destination / 'masks').mkdir()
    for item, mask in zip(objects, masks):
        path = destination / 'masks' / (item['object_id'] + '.png')
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
            raise RuntimeError('Cannot write first-frame segmentation')
        item['first_mask'] = file_record(path)
    manifest = {'schema_version': 'full_hocap_v1', 'dataset': 'HO-Cap', 'subject_id': subject_id,
                'acquisition_id': acquisition_id, 'camera_id': camera_id, 'source_frame_count': count,
                'frame_ids': list(range(count)), 'image_shape': image_shape, 'intrinsics': intrinsics.tolist(),
                'depth_scale': float(depth_scale), 'depth_unit': 'm', 'objects': objects, 'frames': frames,
                'source_timestamp_s': timestamps, 'source_timestamp_valid': source_timestamps is not None,
                'source_timestamp_evidence': file_record(source_timestamps) if isinstance(source_timestamps, (str, Path)) else None,
                'source_metadata': file_record(metadata_path), 'first_label_provenance_only': file_record(first_label),
                'initialization': 'Published FIRST-FRAME segmentation only; no reference poses or later masks read',
                'coordinates': 'OpenCV right-handed camera: x right, y down, z forward; original mesh to camera',
                'timestamp_status': 'provided_source_timestamps' if source_timestamps is not None else 'unknown',
                'training_allowed': False, 'reference_poses_used_for_inference': False}
    write_json(destination / 'scene_manifest.json', manifest)
    return manifest


def depth_surface_points(landmarks, depth_m, intrinsics, radius=2, max_spread_m=0.04,
                         min_depth_m=0.1, max_depth_m=3.0):
    landmarks = np.asarray(landmarks, dtype=np.float32)
    depth_m = np.asarray(depth_m)
    intrinsics = np.asarray(intrinsics)
    if landmarks.shape != (2, 21, 3) or depth_m.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError('Expected landmarks (2,21,3), depth HxW and K 3x3')
    if radius < 0 or min_depth_m <= 0 or max_depth_m <= min_depth_m or max_spread_m < 0:
        raise ValueError('Invalid depth projection bounds')
    points = np.full((2, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((2, 21), dtype=bool)
    height, width = depth_m.shape
    for hand_index in range(2):
        for joint_index in range(21):
            landmark = landmarks[hand_index, joint_index]
            if not np.isfinite(landmark).all():
                continue
            pixel = landmark[:2] * [width, height]
            if not (0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1):
                continue
            column, row = np.rint(pixel).astype(int)
            patch = depth_m[max(0, row-radius):row+radius+1, max(0, column-radius):column+radius+1]
            values = patch[np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)]
            if len(values) < 3 or np.percentile(values, 90) - np.percentile(values, 10) > max_spread_m:
                continue
            depth = np.median(values)
            ray = np.linalg.solve(intrinsics, [pixel[0], pixel[1], 1.0])
            points[hand_index, joint_index] = ray * depth
            valid[hand_index, joint_index] = True
    return points, valid


def create_estimators(repository, meshes, output_dir):
    import torch
    import nvdiffrast.torch as rasterizer

    repository = Path(repository).resolve(strict=True)
    sys.path.insert(0, str(repository))
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose requires CUDA; no mock fallback')
    module = importlib.import_module('estimater')
    if Path(module.__file__).resolve() != repository / 'estimater.py':
        raise RuntimeError('Wrong FoundationPose repository imported')
    module.set_seed(0)
    scorer, refiner = module.ScorePredictor(), module.PoseRefinePredictor()
    context = rasterizer.RasterizeCudaContext()
    estimators = [module.FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                  mesh=mesh, scorer=scorer, refiner=refiner, glctx=context, debug=0,
                  debug_dir=str(Path(output_dir) / f'object_{index}')) for index, mesh in enumerate(meshes)]
    weights = {name: {'checkpoint': file_record(predictor.cfg['ckpt_dir']),
                     'config': file_record(Path(predictor.cfg['ckpt_dir']).parent / 'config.yml')}
               for name, predictor in [('scorer', scorer), ('refiner', refiner)]}
    metadata = {'backend': 'FoundationPose', 'real_model': True, 'test_only': False, 'weights': weights,
                'git_commit': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
                'git_dirty': bool(subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()),
                'source': [file_record(repository / name) for name in ['estimater.py', 'Utils.py']],
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}
    return estimators, metadata, torch.cuda.synchronize


def draw_objects(image, poses, meshes, intrinsics, object_ids):
    import cv2

    image = image.copy()
    height, width = image.shape[:2]
    visible_objects = []
    colors = [(0, 220, 255), (255, 160, 0), (60, 220, 60), (220, 80, 220)]
    for index, (pose, mesh, identity) in enumerate(zip(poses, meshes, object_ids)):
        corners = np.array(list(itertools.product(*mesh.bounds.T)))
        length = float(min(mesh.extents) * 0.6)
        points = np.vstack([corners, np.zeros((1, 3)), np.eye(3) * length])
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        projected = camera @ intrinsics.T
        pixels = projected[:, :2] / projected[:, 2:]
        usable = np.isfinite(pixels).all() and np.all(camera[:, 2] > 0) and np.max(np.abs(pixels)) < 1e7
        inside = bool(usable and np.any((pixels[:8, 0] >= 0) & (pixels[:8, 0] < width)
                                      & (pixels[:8, 1] >= 0) & (pixels[:8, 1] < height)))
        visible_objects.append(inside)
        if usable:
            pixels = np.rint(pixels).astype(np.int32)
            segments = [(start, start ^ bit, colors[index % len(colors)]) for start in range(8)
                        for bit in (1, 2, 4) if start < (start ^ bit)]
            segments += [(8, 9 + axis, color) for axis, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)])]
            for start, end, color in segments:
                visible, first, last = cv2.clipLine((0, 0, width, height), tuple(pixels[start]), tuple(pixels[end]))
                if visible:
                    cv2.line(image, first, last, color, 2, cv2.LINE_AA)
        cv2.putText(image, identity, (8, 20 + index * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    colors[index % len(colors)], 1, cv2.LINE_AA)
    return image, visible_objects


def run_scene(manifest_path, output_dir, foundationpose_repo, config_path, *, hand_backend=None, factory=None):
    import cv2
    import trimesh
    from src.hand_pose.online_hands import OnlineHands, draw_hands

    started = time.perf_counter()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(manifest_path).read_text())
    config = json.loads(Path(config_path).read_text())
    report = {'schema_version': 'full_hocap_v1', 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'scene_manifest': file_record(manifest_path), 'config_file': file_record(config_path), 'config': config,
              'frames_requested': manifest['source_frame_count'], 'frames_completed': 0, 'training_allowed': False,
              'test_only': hand_backend is not None or factory is not None, 'live_camera_validated': False,
              'source_timestamp_known': manifest['source_timestamp_valid'], 'accuracy': None,
              'initialization': manifest['initialization'], 'reference_poses_used_for_inference': False,
              'hand_camera_surface_meaning': 'Depth surface sampled at predicted 2D landmarks, NOT anatomical joint GT',
              'object_valid_meaning': 'Finite rigid positive-Z estimate, NOT independently verified tracking success',
              'code': [file_record(__file__), file_record(inspect.getfile(OnlineHands))]}
    write_json(destination / 'report.json', report)
    writer, backend, timing_stream = None, None, None
    rows, pose_rows, hand_rows, confidence_rows, surface_rows, surface_masks, projection_rows = [], [], [], [], [], [], []
    previous_threads = cv2.getNumThreads()
    try:
        if manifest['schema_version'] != 'full_hocap_v1' or config['schema_version'] != 'full_hocap_v1':
            raise ValueError('Wrong schema version')
        count = manifest['source_frame_count']
        if (manifest['frame_ids'] != list(range(count)) or len(manifest['frames']) != count
                or [record['frame_id'] for record in manifest['frames']] != list(range(count))):
            raise ValueError('Full sequence frame IDs required')
        if not np.isfinite(config['display_fps']) or config['display_fps'] <= 0:
            raise ValueError('Invalid display FPS')
        for key in ['est_refine_iter', 'track_refine_iter', 'torch_threads', 'opencv_threads']:
            if not isinstance(config[key], int) or config[key] < 1:
                raise ValueError('Invalid positive integer config: ' + key)
        if manifest['source_timestamp_valid']:
            validate_timestamps(manifest['source_timestamp_s'], count)
        elif any(value is not None for value in manifest['source_timestamp_s']):
            raise ValueError('Unknown source timing must remain null')
        intrinsics = np.asarray(manifest['intrinsics'], dtype=np.float64)
        cv2.setNumThreads(config['opencv_threads'])
        if factory is None:
            import torch
            torch.set_num_threads(config['torch_threads'])
            torch.cuda.reset_peak_memory_stats()
        meshes, masks = [], []
        for item in manifest['objects']:
            for asset in [item['mesh'], *item['mesh_related_assets']]:
                checked_bytes(asset)
            mesh = trimesh.load(item['mesh']['path'], process=False)
            if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
                raise ValueError('Need finite nondegenerate original Trimesh')
            meshes.append(mesh)
            mask = cv2.imdecode(np.frombuffer(checked_bytes(item['first_mask']), np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or list(mask.shape) != manifest['image_shape'] or np.count_nonzero(mask) < 4:
                raise ValueError('Invalid first mask')
            masks.append(mask > 0)
        estimators, metadata, synchronize = (factory or create_estimators)(foundationpose_repo, meshes, destination / 'debug')
        if len(estimators) != len(meshes):
            raise ValueError('Estimator count differs from object count')
        report['object_backend'] = metadata
        backend = hand_backend or OnlineHands(input_mirrored=False, model_complexity=config['hand_model_complexity'],
                  min_detection_confidence=config['min_detection_confidence'], min_tracking_confidence=config['min_tracking_confidence'],
                  min_handedness_score=config['min_handedness_score'])
        height, width = manifest['image_shape']
        writer = cv2.VideoWriter(str(destination / 'overlay_mp4v.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                 config['display_fps'], (width, height))
        if not writer.isOpened():
            raise RuntimeError('Cannot open video encoder')
        (destination / 'previews').mkdir()
        preview_indices = set(np.linspace(0, count-1, max(3, int(np.ceil(count * 0.05)))).astype(int).tolist())
        timing_stream = (destination / 'timing.csv').open('w', newline='')
        fields = ['frame_id', 'source_timestamp_s', 'source_timestamp_valid', 'display_timestamp_s',
                  'processing_elapsed_s', 'read_ms', 'hands_ms', 'objects_ms', 'render_write_ms', 'total_ms',
                  'left_hand_valid', 'right_hand_valid', 'objects_numeric_valid', 'objects_in_image']
        timing_writer = csv.DictWriter(timing_stream, fieldnames=fields)
        timing_writer.writeheader()
        report['initialization_seconds'] = time.perf_counter() - started
        loop_started = time.perf_counter()
        for index, source in enumerate(manifest['frames']):
            frame_started = time.perf_counter()
            bgr = cv2.imdecode(np.frombuffer(checked_bytes(source['rgb']), np.uint8), cv2.IMREAD_COLOR)
            raw_depth = cv2.imdecode(np.frombuffer(checked_bytes(source['depth']), np.uint8), cv2.IMREAD_UNCHANGED)
            if bgr is None or raw_depth is None or raw_depth.dtype != np.uint16 or raw_depth.shape != (height, width) or bgr.shape != (height, width, 3):
                raise ValueError('Input decoding/geometry mismatch')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = raw_depth.astype(np.float32) * manifest['depth_scale']
            read_done = time.perf_counter()
            timestamp = manifest['source_timestamp_s'][index]
            stream_clock_ms = timestamp * 1000 if timestamp is not None else float(index)
            hand = backend.process(rgb, stream_clock_ms)
            landmarks = np.stack([hand['left_hand'], hand['right_hand']])
            if landmarks.shape != (2, 21, 3) or np.asarray(hand['conf']).shape != (2,):
                raise ValueError('Invalid hand output shape')
            hand_valid = np.isfinite(landmarks).all(axis=(1, 2))
            if not np.all(hand_valid | np.isnan(landmarks).all(axis=(1, 2))):
                raise ValueError('Missing hands must be entirely NaN')
            surfaces, surface_valid = depth_surface_points(landmarks, depth, intrinsics,
                radius=config['depth_radius'], max_spread_m=config['depth_max_spread_m'],
                min_depth_m=config['depth_min_m'], max_depth_m=config['depth_max_m'])
            hands_done = time.perf_counter()
            synchronize()
            current_poses = []
            for object_index, estimator in enumerate(estimators):
                if index == 0:
                    pose = estimator.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=masks[object_index], iteration=config['est_refine_iter'])
                else:
                    pose = estimator.track_one(rgb=rgb, depth=depth, K=intrinsics, iteration=config['track_refine_iter'])
                current_poses.append(validate_pose(pose))
            synchronize()
            objects_done = time.perf_counter()
            overlay, in_image = draw_objects(draw_hands(bgr, hand), current_poses, meshes, intrinsics,
                                             [item['object_id'] for item in manifest['objects']])
            text = f'frame {index:06d}/{count-1:06d} | source time ' + (f'{timestamp:.3f}s' if timestamp is not None else 'UNKNOWN')
            cv2.rectangle(overlay, (0, height-28), (width-1, height-1), (0, 0, 0), -1)
            cv2.putText(overlay, text, (8, height-9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(overlay)
            if index in preview_indices or not hand_valid.any() or not all(in_image):
                if not cv2.imwrite(str(destination / 'previews' / f'{index:06d}.jpg'), overlay):
                    raise RuntimeError('Cannot write review image')
            finished = time.perf_counter()
            row = {'frame_id': index, 'source_timestamp_s': timestamp, 'source_timestamp_valid': timestamp is not None,
                   'display_timestamp_s': index / config['display_fps'], 'processing_elapsed_s': finished-loop_started,
                   'read_ms': (read_done-frame_started)*1000, 'hands_ms': (hands_done-read_done)*1000,
                   'objects_ms': (objects_done-hands_done)*1000, 'render_write_ms': (finished-objects_done)*1000,
                   'total_ms': (finished-frame_started)*1000, 'left_hand_valid': bool(hand_valid[0]),
                   'right_hand_valid': bool(hand_valid[1]), 'objects_numeric_valid': len(current_poses), 'objects_in_image': sum(in_image)}
            timing_writer.writerow(row)
            timing_stream.flush()
            rows.append(row); pose_rows.append(current_poses); hand_rows.append(landmarks)
            confidence_rows.append(hand['conf']); surface_rows.append(surfaces); surface_masks.append(surface_valid); projection_rows.append(in_image)
            report['frames_completed'] = len(rows)
            if (index+1) % 50 == 0:
                write_json(destination / 'report.json', report)
                print(json.dumps({'frames_completed': len(rows), 'frames_requested': count}), flush=True)
        report['loop_seconds'] = time.perf_counter() - loop_started
        writer.release(); writer = None
        timing_stream.close(); timing_stream = None
        report['hand_backend'] = backend.metadata()
        backend.close(); backend = None
        hand_values = np.asarray(hand_rows, dtype=np.float32)
        object_values = np.asarray(pose_rows, dtype=np.float32).transpose(1, 0, 2, 3)
        np.savez_compressed(destination / 'sequence.npz', frame_ids=np.arange(count, dtype=np.int64),
            object_ids=np.asarray([item['object_id'] for item in manifest['objects']]), object_poses=object_values,
            object_valid=np.isfinite(object_values).all(axis=(2, 3)), object_projection_visible=np.asarray(projection_rows).T,
            left_hand=hand_values[:, 0], right_hand=hand_values[:, 1], hand_conf=np.asarray(confidence_rows, dtype=np.float32),
            hand_valid=np.isfinite(hand_values).all(axis=(2, 3)), hand_camera_surface_m=np.asarray(surface_rows),
            hand_camera_surface_valid=np.asarray(surface_masks),
            source_timestamp_s=np.asarray([np.nan if value is None else value for value in manifest['source_timestamp_s']], dtype=np.float64),
            source_timestamp_valid=np.full(count, manifest['source_timestamp_valid'], dtype=bool),
            display_timestamp_s=np.arange(count, dtype=np.float64)/config['display_fps'],
            processing_elapsed_s=np.asarray([row['processing_elapsed_s'] for row in rows], dtype=np.float64))
        conversion_started = time.perf_counter()
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(destination/'overlay_mp4v.mp4'),
                        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(destination/'overlay.mp4')], check=True)
        report['video_conversion_seconds'] = time.perf_counter()-conversion_started
        durations = np.asarray([row['total_ms'] for row in rows])
        report['timing'] = {'all_frame_work_fps': float(1000/durations.mean()), 'frame_work_p50_ms': float(np.median(durations)),
                            'frame_work_p95_ms': float(np.percentile(durations, 95)), 'registration_frame_ms': float(durations[0]),
                            'steady_work_fps': float(1000/durations[1:].mean()), 'steady_work_p95_ms': float(np.percentile(durations[1:],95)),
                            'live_capture_latency_measured': False, 'keeps_up_with_source': None}
        report['environment'] = {'python': sys.version, 'platform': platform.platform(), 'opencv_module': cv2.__version__,
            'packages': {name: distribution_version(name) for name in ['numpy', 'opencv-python', 'opencv-contrib-python', 'trimesh', 'PyYAML', 'mediapipe']}}
        if not report['test_only']:
            import torch
            report['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        report['outputs'] = [file_record(destination/name) for name in ['sequence.npz', 'timing.csv', 'overlay.mp4']]
        report['status'] = 'complete'
        report['total_wall_seconds'] = time.perf_counter()-started
        write_json(destination / 'report.json', report)
        return report
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        report['total_wall_seconds'] = time.perf_counter()-started
        write_json(destination / 'report.json', report)
        raise
    finally:
        if writer is not None:
            writer.release()
        if timing_stream is not None:
            timing_stream.close()
        if backend is not None:
            backend.close()
        cv2.setNumThreads(previous_threads)


def audit_output(output_dir):
    import cv2

    root = Path(output_dir)
    report = json.loads((root / 'report.json').read_text())
    manifest = json.loads(checked_bytes(report['scene_manifest']))
    count, object_count = manifest['source_frame_count'], len(manifest['objects'])
    checks = {'complete_status': report['status'] == 'complete', 'full_processed_count': report['frames_completed'] == count,
              'real_backends': not report['test_only'] and report['object_backend']['real_model'] and not report['hand_backend']['test_only']}
    for asset in report['outputs']:
        checked_bytes(asset)
    with np.load(root/'sequence.npz', allow_pickle=False) as packed:
        checks['frame_ids'] = np.array_equal(packed['frame_ids'], np.arange(count))
        checks['object_shape'] = packed['object_poses'].shape == (object_count, count, 4, 4)
        checks['hand_shape'] = packed['left_hand'].shape == (count,21,3) and packed['right_hand'].shape == (count,21,3)
        checks['object_valid_mask'] = np.array_equal(packed['object_valid'], np.isfinite(packed['object_poses']).all(axis=(2,3)))
        valid_poses = [validate_pose(pose) for pose in packed['object_poses'].reshape(-1,4,4)]
        checks['rigid_positive_z_poses'] = len(valid_poses) == object_count*count
        hand_values = np.stack([packed['left_hand'], packed['right_hand']],axis=1)
        hand_valid = np.isfinite(hand_values).all(axis=(2,3))
        checks['hand_valid_mask'] = np.array_equal(packed['hand_valid'],hand_valid)
        checks['hand_nan_policy'] = bool(np.all(hand_valid | np.isnan(hand_values).all(axis=(2,3))))
        checks['surface_valid_mask'] = np.array_equal(packed['hand_camera_surface_valid'],np.isfinite(packed['hand_camera_surface_m']).all(axis=-1))
        checks['unknown_source_time_not_fabricated'] = bool(
            np.isfinite(packed['source_timestamp_s']).all() if manifest['source_timestamp_valid']
            else np.isnan(packed['source_timestamp_s']).all())
        checks['source_time_flags'] = bool(np.all(packed['source_timestamp_valid']==manifest['source_timestamp_valid']))
        checks['processing_time_increases'] = bool(np.all(np.diff(packed['processing_elapsed_s'])>0))
        coverage = {'hand_any_frames': int(hand_valid.any(axis=1).sum()), 'left_hand_frames': int(hand_valid[:,0].sum()),
                    'right_hand_frames': int(hand_valid[:,1].sum()), 'object_numeric_valid_frames': packed['object_valid'].sum(axis=1).tolist(),
                    'object_projection_in_image_frames': packed['object_projection_visible'].sum(axis=1).tolist(),
                    'depth_surface_valid_joints': int(packed['hand_camera_surface_valid'].sum()),
                    'joint_hand_any_all_objects_numeric_frames': int((hand_valid.any(axis=1)&packed['object_valid'].all(axis=0)).sum()),
                    'denominator_frames': count, 'not_accuracy': True}
        checks['some_human_hand_observations'] = coverage['hand_any_frames'] > 0
    with (root/'timing.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    checks['timing_all_frames'] = len(rows)==count and [int(row['frame_id']) for row in rows]==list(range(count))
    capture = cv2.VideoCapture(str(root/'overlay.mp4'))
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if list(image.shape[:2]) != manifest['image_shape']:
            raise ValueError('Video dimensions changed')
        decoded += 1
    capture.release()
    checks['video_all_frames_decodable'] = decoded==count
    result = {'schema_version':'full_hocap_v1','engineering_pass':all(checks.values()),'checks':checks,'coverage':coverage,
              'accuracy':None,'independent_accuracy_pass':None,'formal_training_allowed':False,
              'source_timestamp_known':manifest['source_timestamp_valid'],'decoded_video_frames':decoded,
              'report_sha256':file_record(root/'report.json')['sha256']}
    destination = root/'acceptance.json'
    if destination.exists():
        raise FileExistsError('Acceptance output exists; do not overwrite an audit')
    write_json(destination,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    for name in ['acquisition-root','subject-id','acquisition-id','camera-id','output-dir']:
        prepare.add_argument('--'+name,required=True)
    prepare.add_argument('--depth-scale',type=float,required=True)
    prepare.add_argument('--source-timestamps')
    run = commands.add_parser('run')
    for name in ['manifest','output-dir','foundationpose-repo','config']:
        run.add_argument('--'+name,required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--output-dir',required=True)
    args = parser.parse_args()
    if args.command=='prepare':
        result=prepare_hocap(args.acquisition_root,args.subject_id,args.acquisition_id,args.camera_id,args.output_dir,args.depth_scale,args.source_timestamps)
        print(json.dumps({'frames':result['source_frame_count'],'objects':len(result['objects']),'source_timestamp_valid':result['source_timestamp_valid']}))
    elif args.command=='run':
        result=run_scene(args.manifest,args.output_dir,args.foundationpose_repo,args.config)
        print(json.dumps({'status':result['status'],'frames_completed':result['frames_completed'],'timing':result['timing']}))
    else:
        result=audit_output(args.output_dir)
        print(json.dumps(result,indent=2))
        if not result['engineering_pass']:
            raise SystemExit(1)


if __name__=='__main__':
    main()
