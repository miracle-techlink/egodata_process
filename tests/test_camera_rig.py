import numpy as np
import pytest

from src.dataset.check_camera_rig import compare_conventions, quaternion_matrices


def moving_rig(inverse=False):
    angles = np.linspace(0, 0.8, 40)
    quaternions = np.column_stack((np.cos(angles / 2), np.zeros((40, 2)), np.sin(angles / 2)))
    rotations = quaternion_matrices(quaternions)
    positions = np.column_stack((angles, angles ** 2, np.sin(angles)))
    displacement = np.einsum("nij,j->ni", rotations, [0.064, 0, 0])
    if inverse:
        quaternions[:, 1:] *= -1
    return positions, positions + displacement, quaternions


@pytest.mark.parametrize("inverse,expected", [(False, "camera_to_world"), (True, "world_to_camera")])
def test_recovers_convention_from_moving_rigid_pair(inverse, expected):
    result = compare_conventions(*moving_rig(inverse))
    assert result["favored_convention"] == expected
    assert result[f"{expected}_hypothesis"]["rms_from_mean"] < 1e-14
    assert result["projection_calibrated"] is False
    assert result["automatic_coordinate_conversion_authorized"] is False


def test_stationary_camera_is_inconclusive():
    positions = np.zeros((6, 3))
    result = compare_conventions(positions, positions + [0.064, 0, 0], np.tile([1, 0, 0, 0], (6, 1)))
    assert result["favored_convention"] is None


def test_quaternion_sign_and_scale_do_not_change_rotation():
    _, _, quaternions = moving_rig()
    np.testing.assert_allclose(quaternion_matrices(quaternions), quaternion_matrices(-quaternions * 2))


@pytest.mark.parametrize("case", ["nan", "zero_quaternion", "zero_baseline", "count", "shape"])
def test_malformed_inputs_rejected(case):
    left, right, quaternions = moving_rig()
    if case == "nan":
        left[0, 0] = np.nan
    elif case == "zero_quaternion":
        quaternions[0] = 0
    elif case == "zero_baseline":
        right[0] = left[0]
    elif case == "count":
        quaternions = quaternions[:-1]
    elif case == "shape":
        left = left[:, :2]
    with pytest.raises(ValueError):
        compare_conventions(left, right, quaternions)


def test_nonrigid_pair_does_not_assert_convention():
    left, right, quaternions = moving_rig()
    right += np.random.default_rng(42).normal(size=right.shape) * 0.03
    assert compare_conventions(left, right, quaternions)["favored_convention"] is None
