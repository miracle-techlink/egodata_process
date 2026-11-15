import numpy as np
import pytest

from src.pose6d.rot_utils import (
    geodesic_angle_deg,
    invert_transform,
    is_valid_rotation,
    make_transform,
    matrix_to_quat,
    matrix_to_rot6d,
    nan_poses,
    quat_to_matrix,
    rot6d_to_matrix,
)


def random_rotations(n, rng):
    q, _ = np.linalg.qr(rng.standard_normal((n, 3, 3)))
    # force proper rotations (det == +1)
    det = np.linalg.det(q)
    q[det < 0, :, 0] *= -1
    return q


def test_rot6d_roundtrip_random():
    rng = np.random.default_rng(0)
    R = random_rotations(64, rng)
    x6 = matrix_to_rot6d(R)
    assert x6.shape == (64, 6)
    R2 = rot6d_to_matrix(x6)
    assert np.allclose(R2, R, atol=1e-10)


def test_rot6d_known_rz90():
    Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    x6 = matrix_to_rot6d(Rz90)
    assert np.allclose(x6, [0, 1, 0, -1, 0, 0])
    assert np.allclose(rot6d_to_matrix(x6), Rz90)


def test_rot6d_single_and_batch_shapes():
    assert matrix_to_rot6d(np.eye(3)).shape == (6,)
    assert rot6d_to_matrix(np.zeros(6)).shape == (3, 3)
    assert rot6d_to_matrix(np.zeros((2, 3, 6))).shape == (2, 3, 3, 3)
    with pytest.raises(ValueError):
        matrix_to_rot6d(np.eye(4))
    with pytest.raises(ValueError):
        rot6d_to_matrix(np.zeros(5))


def test_rot6d_noisy_input_is_orthonormal():
    x6 = np.array([3.0, 0.1, 0.0, 0.05, 2.0, 0.1])
    R = rot6d_to_matrix(x6)
    assert is_valid_rotation(R, atol=1e-9)


def test_quat_roundtrip_random():
    rng = np.random.default_rng(1)
    R = random_rotations(64, rng)
    q = matrix_to_quat(R)
    assert q.shape == (64, 4)
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)
    assert np.all(q[:, 0] >= 0)  # canonical hemisphere
    R2 = quat_to_matrix(q)
    assert np.allclose(R2, R, atol=1e-10)


def test_quat_known():
    q = quat_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(q, np.eye(3))
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q2 = matrix_to_quat(Rz)
    assert np.allclose(q2, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-10)


def test_make_and_invert_transform():
    rng = np.random.default_rng(2)
    R = random_rotations(8, rng)
    t = rng.standard_normal((8, 3))
    T = make_transform(R, t)
    assert T.shape == (8, 4, 4)
    Ti = invert_transform(T)
    eye = T @ Ti
    assert np.allclose(eye, np.eye(4), atol=1e-10)


def test_geodesic_angle():
    I = np.eye(3)
    assert geodesic_angle_deg(I, I) == pytest.approx(0.0)
    Rx180 = np.diag([1.0, -1.0, -1.0])
    assert geodesic_angle_deg(I, Rx180) == pytest.approx(180.0)
    Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert geodesic_angle_deg(I, Rz90) == pytest.approx(90.0)


def test_is_valid_rotation():
    assert is_valid_rotation(np.eye(3))
    assert not is_valid_rotation(np.diag([1.0, 1.0, -1.0]))  # reflection
    assert not is_valid_rotation(np.eye(3) * 2.0)


def test_nan_poses():
    p = nan_poses(3, 7)
    assert p.shape == (3, 7, 4, 4) and p.dtype == np.float32
    assert np.isnan(p).all()
