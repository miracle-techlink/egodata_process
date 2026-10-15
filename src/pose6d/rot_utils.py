"""Rotation / rigid-transform utilities (pure numpy).

Conventions
-----------
- Rotation matrix R: (..., 3, 3), right-handed.
- 6D rotation representation: first two columns of R flattened as
  (a1, a2) where a1/a2 are 3-vectors (Zhou et al., "On the Continuity of
  Rotation Representations in Neural Networks", CVPR 2019). This is the
  layout task 4 consumes in `obj_seq`.
- Quaternion: (w, x, y, z), unit norm.
- Transform T: (..., 4, 4) homogeneous; project convention is
  object-to-camera (`p_cam = T[:3, :3] @ p_obj + T[:3, 3]`).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation -> (..., 6) first-two-columns 6D rep."""
    R = np.asarray(R)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {R.shape}")
    return np.swapaxes(R[..., :, :2], -1, -2).reshape(R.shape[:-2] + (6,))


def rot6d_to_matrix(x6d: np.ndarray) -> np.ndarray:
    """(..., 6) 6D rep -> (..., 3, 3) rotation via Gram-Schmidt."""
    x = np.asarray(x6d)
    if x.shape[-1] != 6:
        raise ValueError(f"expected (..., 6), got {x.shape}")
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), _EPS)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation -> (..., 4) quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {R.shape}")
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=np.float64)
    for i, m in enumerate(flat):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            q = np.array([0.25 * s,
                          (m[2, 1] - m[1, 2]) / s,
                          (m[0, 2] - m[2, 0]) / s,
                          (m[1, 0] - m[0, 1]) / s])
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s,
                          0.25 * s,
                          (m[0, 1] + m[1, 0]) / s,
                          (m[0, 2] + m[2, 0]) / s])
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s,
                          (m[0, 1] + m[1, 0]) / s,
                          0.25 * s,
                          (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s,
                          (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s,
                          0.25 * s])
        q /= np.linalg.norm(q)
        if q[0] < 0.0:
            q = -q
        out[i] = q
    return out.reshape(R.shape[:-2] + (4,))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternion (w, x, y, z) -> (..., 3, 3) rotation."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"expected (..., 4), got {q.shape}")
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _EPS)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)
    return R


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R (..., 3, 3), t (..., 3) -> homogeneous T (..., 4, 4)."""
    R = np.asarray(R)
    t = np.asarray(t)
    T = np.broadcast_to(np.eye(4), R.shape[:-2] + (4, 4)).copy()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (..., 4, 4)."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"expected (..., 4, 4), got {T.shape}")
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Ri = np.swapaxes(R, -1, -2)
    Ti = np.broadcast_to(np.eye(4), T.shape).copy()
    Ti[..., :3, :3] = Ri
    Ti[..., :3, 3] = -(Ri @ t[..., None])[..., 0]
    return Ti


def geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Rotation angle between R1 and R2 in degrees; (...) broadcast."""
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    dR = np.swapaxes(R1, -1, -2) @ R2
    cos = (np.trace(dR, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def is_valid_rotation(R: np.ndarray, atol: float = 1e-6) -> bool:
    """Check orthonormality and det == +1."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        return False
    eye = np.eye(3)
    flat = R.reshape(-1, 3, 3)
    for m in flat:
        if not np.allclose(m.T @ m, eye, atol=atol):
            return False
        if not np.isclose(np.linalg.det(m), 1.0, atol=atol):
            return False
    return True


def nan_poses(n: int, t: int) -> np.ndarray:
    """(n, t, 4, 4) float32 all-NaN pose block (contract fill value)."""
    return np.full((n, t, 4, 4), np.nan, dtype=np.float32)
