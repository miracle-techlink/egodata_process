"""NumPy geometry for explicitly calibrated, time-aligned hand/camera data.

Lightwheel documents LeRobot positions in meters and quaternion order wxyz:
https://docs.lightwheel.net/egocentric_data/lerobot_data/format_specification
Camera poses are C2W; OpenCV axes are x right, y down, z forward:
https://docs.lightwheel.net/egocentric_data/mcap_data/sensor/rgb_camera/head
Use the annotated episode's own calibration and poses, never Raw frame IDs
or Raw world poses as substitutes. This module does not load or align data.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from src.dataset.check_camera_rig import quaternion_matrices


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _real_array(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real numbers")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numbers") from error


def _transform_inputs(points, camera_position, camera_quaternion_wxyz):
    values = _real_array(points, "points")
    positions = _real_array(camera_position, "camera_position")
    quaternions = _real_array(camera_quaternion_wxyz, "camera_quaternion_wxyz")
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("points must have shape (T, ..., 3)")
    frame_count = values.shape[0]
    if positions.shape != (frame_count, 3):
        raise ValueError("camera_position must have shape (T, 3), without broadcasting")
    if quaternions.shape != (frame_count, 4):
        raise ValueError("camera_quaternion_wxyz must have shape (T, 4), without broadcasting")
    if not np.isfinite(positions).all():
        raise ValueError("camera_position must be finite")
    if not np.isfinite(quaternions).all():
        raise ValueError("camera_quaternion_wxyz must be finite and nonzero")
    scales = np.max(np.abs(quaternions), axis=-1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("camera_quaternion_wxyz must be finite and nonzero")
    rotations = quaternion_matrices(quaternions / scales)
    positions = positions.reshape((frame_count,) + (1,) * (values.ndim - 2) + (3,))
    values = np.where(np.isfinite(values).all(axis=-1, keepdims=True), values, np.nan)
    return values, positions, rotations


def world_to_camera(points_world, camera_position, camera_quaternion_wxyz):
    """Return float64 camera points with the same (T, ..., 3) shape.

    The quaternion is camera-to-world in wxyz order; compute R.T @ (P - t).
    Positions must be (T, 3), quaternions (T, 4), with exactly matching T.
    Broadcasting is only over the intervening point/hand axes. Finite nonzero
    quaternions are normalized; q and -q give the same rotation. Nonfinite
    camera poses raise ValueError. Any nonfinite point becomes an all-NaN
    point, never a zero-filled observation. Inputs are not modified.

    Callers must establish pose convention, units and temporal alignment;
    this function does not align Raw and annotated frames or certify poses.
    """
    points, positions, rotations = _transform_inputs(
        points_world, camera_position, camera_quaternion_wxyz
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.einsum("tji,t...j->t...i", rotations, points - positions)
    return np.where(np.isfinite(result).all(axis=-1, keepdims=True), result, np.nan)


def camera_to_world(points_camera, camera_position, camera_quaternion_wxyz):
    """Return float64 world points (T, ..., 3), computing R @ P + t.

    This is the inverse of world_to_camera, using the same camera-to-world
    wxyz quaternion, exact sequence shapes and invalid-point policy. Inputs
    are not modified, and nonfinite camera poses raise ValueError.
    """
    points, positions, rotations = _transform_inputs(
        points_camera, camera_position, camera_quaternion_wxyz
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.einsum("tij,t...j->t...i", rotations, points) + positions
    return np.where(np.isfinite(result).all(axis=-1, keepdims=True), result, np.nan)


def project_pinhole(points_camera, intrinsics, width, height):
    """Project (..., 3) camera points with supplied undistorted calibration.

    Return a dict with points_uv: float64 (..., 2), positive_depth: bool
    (...,), and inside_image: bool (...,). positive_depth requires all three
    coordinates finite and z > 0. points_uv is all NaN for other points or
    nonfinite projected pixels; finite out-of-image pixels remain unchanged.
    inside_image additionally requires 0 <= u < width and 0 <= v < height.
    A single (3,) point and empty batches are supported. No integer pixel
    conversion, depth epsilon, clipping or guessed intrinsics is applied.

    intrinsics must provide finite real scalar fx/fy/cx/cy, with fx/fy > 0;
    principal points may lie outside the image. width/height must be positive
    integers. Invalid calibration raises ValueError. Callers must supply the
    true calibration for this undistorted image and matching optical axes;
    numeric validity alone does not establish calibration provenance.
    """
    points = _real_array(points_camera, "points_camera")
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points_camera must have shape (..., 3)")
    if not isinstance(intrinsics, Mapping):
        raise ValueError("intrinsics must provide fx, fy, cx, cy")
    calibration = {}
    for name in ("fx", "fy", "cx", "cy"):
        value = intrinsics.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"intrinsics {name} must be a finite real scalar")
        try:
            value = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"intrinsics {name} must be a finite real scalar") from error
        if not np.isfinite(value) or (name in ("fx", "fy") and value <= 0):
            raise ValueError(f"intrinsics {name} must be finite; fx and fy must be positive")
        calibration[name] = value
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    positive_depth = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    points_uv = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(
            points[..., :2], points[..., 2, None],
            out=points_uv, where=positive_depth[..., None],
        )
        points_uv *= (calibration["fx"], calibration["fy"])
        points_uv += (calibration["cx"], calibration["cy"])
    finite_uv = np.isfinite(points_uv).all(axis=-1)
    points_uv = np.where(finite_uv[..., None], points_uv, np.nan)
    inside_image = (
        positive_depth & finite_uv
        & (points_uv[..., 0] >= 0) & (points_uv[..., 0] < width)
        & (points_uv[..., 1] >= 0) & (points_uv[..., 1] < height)
    )
    return {
        "points_uv": points_uv,
        "positive_depth": positive_depth,
        "inside_image": inside_image,
    }
