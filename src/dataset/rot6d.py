"""6D rotation representation (Zhou et al., CVPR 2019): the first two columns of R."""

import numpy as np


def matrix_to_rot6d(rot):
    """(..., 3, 3) rotation matrices -> (..., 6), NaN propagates."""
    rot = np.asarray(rot)
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {rot.shape}")
    return np.concatenate((rot[..., :, 0], rot[..., :, 1]), axis=-1)


def rot6d_to_matrix(d6):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt orthonormalization."""
    d6 = np.asarray(d6)
    if d6.shape[-1] != 6:
        raise ValueError(f"expected (..., 6), got {d6.shape}")
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(64, 3, 3)))
    q *= np.sign(np.linalg.det(q))[..., None, None]
    err = np.abs(rot6d_to_matrix(matrix_to_rot6d(q)) - q).max()
    print(f"roundtrip max abs err: {err:.2e}")
    assert err < 1e-12
