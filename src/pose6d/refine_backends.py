"""Private, metric stereo and video masks for EgoDemo refinement.

These are estimated observations, never ground truth. Original camera
calibration is used before stereo inference; rectified coordinates are mapped
back to the original left camera. No missing sample is interpolated.
"""
from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np


class Rectification:
    def __init__(self, left_K, right_K, right_from_left, native_size, width=768):
        w, h = native_size
        self.size = (width, round(h * width / w))
        self.scale = np.array([self.size[0] / w, self.size[1] / h])
        self.K = [np.array(k, dtype=float).copy() for k in (left_K, right_K)]
        for k in self.K:
            k[:2] *= self.scale[:, None]
        rel = np.asarray(right_from_left)
        if not .01 < np.linalg.norm(rel[:3, 3]) < .3:
            raise ValueError('Unexpected metric stereo baseline')
        r1, r2, p1, p2, self.Q, _, _ = cv2.stereoRectify(
            *[self.K[0], None, self.K[1], None, self.size],
            rel[:3, :3].copy(), rel[:3, 3:4].copy(), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if abs(p2[1, 3]) > abs(p2[0, 3]) or p2[0, 3] >= 0:
            raise ValueError('Backend requires positive left-minus-right horizontal disparity')
        self.rotation = r1
        self.projection = p1
        self.maps = [cv2.initUndistortRectifyMap(k, None, r, p, self.size, cv2.CV_32FC1)
                     for k, r, p in zip(self.K, (r1, r2), (p1, p2))]
        self.baseline_m = float(np.linalg.norm(rel[:3, 3]))

    def images(self, left, right):
        return [cv2.remap(cv2.resize(im, self.size), *maps, cv2.INTER_LINEAR)
                for im, maps in zip((left, right), self.maps)]

    def mask(self, native_mask):
        small = cv2.resize(native_mask.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST)
        return cv2.remap(small, *self.maps[0], cv2.INTER_NEAREST).astype(bool)

    def xyz(self, disparity):
        xyz_rect = cv2.reprojectImageTo3D(disparity.astype(np.float32), self.Q)
        return xyz_rect @ self.rotation

    def original_uv(self, rectified_uv):
        xy = np.rint(rectified_uv).astype(int)
        return np.stack([m[xy[:, 1], xy[:, 0]] for m in self.maps[0]], axis=1) / self.scale

    def sample_xyz(self, original_uv, xyz_map, valid_map):
        uv = np.asarray(original_uv, np.float32).reshape(-1, 1, 2) * self.scale
        rect = cv2.undistortPoints(uv, self.K[0], None, R=self.rotation, P=self.projection)[:, 0]
        xy = np.rint(rect).astype(int)
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < self.size[0]) &
                 (xy[:, 1] >= 0) & (xy[:, 1] < self.size[1]))
        clipped = np.clip(xy, [0, 0], np.array(self.size) - 1)
        valid &= valid_map[clipped[:, 1], clipped[:, 0]]
        xyz = xyz_map[clipped[:, 1], clipped[:, 0]].copy()
        xyz[~valid] = np.nan
        return xyz, valid


class MetricStereo:
    def __init__(self, models):
        import torch
        sys.path.insert(0, str(Path(models) / 'Fast-FoundationStereo-vendor'))
        from core.foundation_stereo import FastFoundationStereo  # registers checkpoint classes
        from core.utils.utils import InputPadder
        self.torch, self.padder_class = torch, InputPadder
        self.model = torch.load(Path(models) / 'cfast_model_best_bp2_serialize.pth',
                                map_location='cpu', weights_only=False).cuda().eval()
        self.model.args.max_disp = 192
        # The official C checkpoint predates this configurable switch. The
        # published cost-volume implementation defaults to cosine normalization.
        if 'normalize' not in self.model.args:
            self.model.args.normalize = True

    def disparity(self, left, right):
        torch = self.torch
        inputs = [torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().cuda()
                  for im in (left, right)]
        padder = self.padder_class(inputs[0].shape[-2:], divis_by=32, force_square=False)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            d = self.model(*padder.pad(*inputs), iters=8, test_mode=True, optimize_build_volume='pytorch1')
        return padder.unpad(d.float())[0, 0].cpu().numpy()

    def estimate(self, rect, left, right, consistency=True):
        images = rect.images(left, right)
        disparity = self.disparity(*images)
        xyz = rect.xyz(disparity)
        valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > .1) & (xyz[..., 2] < 3.) & (disparity > .5)
        error = np.full(disparity.shape, np.nan, np.float32)
        if consistency:
            # Flip and exchange views to obtain right disparity with the same sign.
            right_disp = self.disparity(images[1][:, ::-1].copy(), images[0][:, ::-1].copy())[:, ::-1]
            y, x = np.indices(disparity.shape, dtype=np.float32)
            right_x = x - disparity
            sampled = cv2.remap(right_disp, right_x, y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
            error = np.abs(disparity - sampled)
            valid &= (right_x >= 0) & (right_x < disparity.shape[1]-1) & (error <= 1.5)
        xyz[~valid] = np.nan
        return {'xyz_camera': xyz, 'valid': valid, 'disparity': disparity,
                'left_right_error_px': error, 'rectified_left': images[0]}


class VideoMasks:
    def __init__(self, models):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        self.torch = torch
        self.model = Sam2VideoModel.from_pretrained(str(Path(models)/'sam2-hf'), local_files_only=True).cuda().eval()
        self.processor = Sam2VideoProcessor.from_pretrained(str(Path(models)/'sam2-hf'), local_files_only=True)
        self.session = None

    def start(self, image_size):
        self.session = self.processor.init_video_session(
            inference_device='cuda', inference_state_device='cuda', video_storage_device='cpu',
            processing_device='cpu', dtype=self.torch.float32)
        self.session.video_width, self.session.video_height = image_size

    def frame(self, bgr, seeds=()):
        torch = self.torch
        idx = self.session.num_frames or 0
        added = list(self.session.obj_with_new_inputs)
        for identity, mask in seeds:
            self.processor.add_inputs_to_inference_session(self.session, frame_idx=idx,
                obj_ids=identity, input_masks=np.asarray(mask, dtype=bool))
            if identity not in added:
                added.append(identity)
        self.session.obj_with_new_inputs = added
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors='pt')
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = self.model(inference_session=self.session, frame=inputs.pixel_values[0])
            masks = torch.nn.functional.interpolate(output.pred_masks.float(), size=bgr.shape[:2],
                                                     mode='bilinear', align_corners=False)[:, 0] > 0
        result = dict(zip(self.session.obj_ids, masks.cpu().numpy()))
        for identity, seed_mask in seeds:
            if np.asarray(seed_mask).sum() >= 100 and not result[identity].any():
                raise RuntimeError('Nonempty seed disappeared in the conditioning frame')
        # Streaming inference only uses the current image; retain recent output
        # memory (32 > SAM2's seven memory slots and sixteen pointer slots).
        if idx:
            self.session.processed_frames[idx-1] = None
        for outputs in self.session.output_dict_per_obj.values():
            for old in list(outputs['non_cond_frame_outputs']):
                if old < idx-32:
                    del outputs['non_cond_frame_outputs'][old]
        return result
