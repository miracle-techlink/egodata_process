"""Experimental calibrated stereo reference points + robust RGB PnP tracking.

This is an estimated, local reference-frame pose, not a CAD/canonical pose or
ground truth. Failed observations stay NaN. A reviewed rigid surface polygon
must be supplied, and independently measured accuracy remains unknown.
"""
from __future__ import annotations

import cv2
import numpy as np


def relative_camera(left_c2w, right_c2w):
    return np.linalg.inv(right_c2w) @ left_c2w


def triangulate(left_uv, right_uv, left_K, right_K, right_from_left, pixel_limit=2.):
    """Reject behind-camera, wrong-epipolar and high-reprojection matches."""
    a = np.asarray(left_uv, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(right_uv, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(right_from_left, dtype=float)
    p1, p2 = left_K @ np.eye(4)[:3], right_K @ p[:3]
    x = cv2.triangulatePoints(p1, p2, a.T, b.T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        xyz = x[:, :3] / x[:, 3:]
        right_xyz = xyz @ p[:3, :3].T + p[:3, 3]
        pa = xyz @ left_K.T
        pb = right_xyz @ right_K.T
        errors = np.maximum(np.linalg.norm(pa[:, :2] / pa[:, 2:] - a, axis=1),
                            np.linalg.norm(pb[:, :2] / pb[:, 2:] - b, axis=1))
    valid = (np.isfinite(xyz).all(axis=1) & np.isfinite(errors) & (errors < pixel_limit)
             & (xyz[:, 2] > .1) & (xyz[:, 2] < 3.) & (right_xyz[:, 2] > .1))
    return xyz, valid, errors


def unique_matches(a, b, ratio=.7):
    if a is None or b is None or min(len(a), len(b)) < 2:
        return []
    pairs = cv2.BFMatcher().knnMatch(a, b, k=2)
    chosen = sorted([x for pair in pairs if len(pair) == 2 for x, y in [pair]
                     if x.distance < ratio * y.distance], key=lambda x: x.distance)
    seen, result = set(), []
    for match in chosen:
        if match.trainIdx not in seen:
            result.append(match)
            seen.add(match.trainIdx)
    return result


def stereo_features(left_bgr, right_bgr):
    detector = cv2.SIFT_create(nfeatures=10000, contrastThreshold=.02)
    return (detector.detectAndCompute(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), None),
            detector.detectAndCompute(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY), None))


def make_reference(left_bgr, right_bgr, polygon, left_K, right_K, right_from_left, pixel_mask=None, features=None):
    if pixel_mask is None:
        mask = np.zeros(left_bgr.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    else:
        if np.asarray(pixel_mask).shape != left_bgr.shape[:2]:
            raise ValueError('Mask must match the native image shape')
        mask = (np.asarray(pixel_mask, dtype=bool).astype(np.uint8) * 255)
    # Exclude polygon edges to reduce hand/background contamination.
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    if features is None:
        features = stereo_features(left_bgr, right_bgr)
    (ka_all, da_all), (kb, db) = features
    selected = [i for i,k in enumerate(ka_all) if mask[min(mask.shape[0]-1,round(k.pt[1])),min(mask.shape[1]-1,round(k.pt[0]))]]
    ka = [ka_all[i] for i in selected]
    da = da_all[selected] if da_all is not None else None
    matches = unique_matches(da, db)
    if len(matches) < 12:
        raise ValueError('Too few unambiguous stereo surface matches')
    a = np.array([ka[m.queryIdx].pt for m in matches])
    b = np.array([kb[m.trainIdx].pt for m in matches])
    points, valid, errors = triangulate(a, b, left_K, right_K, right_from_left)
    if valid.sum() < 12:
        raise ValueError('Too few calibrated stereo points')
    # Initial camera axes define a local orientation gauge. The origin is the
    # median visible reference surface, not the object's centre of mass.
    origin = np.median(points[valid], axis=0)
    valid &= np.linalg.norm(points-origin, axis=1) < .30
    if valid.sum() < 12:
        raise ValueError('Reference points lack a coherent surface')
    return {'points_object': (points[valid]-origin).astype(np.float32),
            'descriptors': np.asarray([da[m.queryIdx] for m in matches])[valid],
            'left_uv': a[valid], 'right_uv': b[valid], 'origin_camera': origin,
            'stereo_reprojection_error_px': errors[valid], 'mask': mask,
            'reference_rotation': np.eye(3), 'object_frame': 'initial camera axes, median observed surface origin'}


def estimate(reference, frame, K, min_inliers=12):
    detector = cv2.SIFT_create(nfeatures=10000, contrastThreshold=.02)
    keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    matches = unique_matches(reference['descriptors'], descriptors)
    diagnostics = {'matches': len(matches), 'inliers': 0, 'median_reprojection_px': None}
    if len(matches) < min_inliers:
        return None, diagnostics
    xyz = np.asarray([reference['points_object'][m.queryIdx] for m in matches], np.float64)
    uv = np.asarray([keypoints[m.trainIdx].pt for m in matches], np.float64)
    ok, r, t, inliers = cv2.solvePnPRansac(xyz, uv, K, None, iterationsCount=150,
                                         reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, diagnostics
    ids = inliers.ravel()
    ok, r, t = cv2.solvePnP(xyz[ids], uv[ids], K, None, r, t, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, diagnostics
    projected, _ = cv2.projectPoints(xyz[ids], r, t, K, None)
    errors = np.linalg.norm(projected[:, 0]-uv[ids], axis=1)
    diagnostics.update(inliers=len(ids), median_reprojection_px=float(np.median(errors)))
    rotation, _ = cv2.Rodrigues(r)
    observed = xyz[ids] @ rotation.T + t.ravel()
    if np.median(errors) > 2. or np.min(observed[:, 2]) < .1 or np.max(observed[:, 2]) > 3.:
        return None, diagnostics
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, t.ravel()
    return pose, diagnostics
