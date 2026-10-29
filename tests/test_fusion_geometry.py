import warnings

import numpy as np
import pytest

from src.dataset.fusion_geometry import (
    HAND_EDGES,
    camera_to_world,
    project_pinhole,
    world_to_camera,
)


INTRINSICS = {"fx": 100.0, "fy": 200.0, "cx": 320.0, "cy": 240.0}
TRANSFORMS = (world_to_camera, camera_to_world)


def camera_sequence(frame_count):
    return np.zeros((frame_count, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (frame_count, 1))


def test_translation_and_exact_frame_correspondence():
    points = np.array([[[3.0, 5.0, 8.0]], [[3.0, 5.0, 8.0]]])
    positions = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    _, quaternions = camera_sequence(2)
    expected = np.array([[[2.0, 3.0, 5.0]], [[4.0, 7.0, 11.0]]])
    np.testing.assert_allclose(world_to_camera(points, positions, quaternions), expected)
    np.testing.assert_allclose(camera_to_world(expected, positions, quaternions), points)


def test_wxyz_rotation_uses_inverse_in_world_to_camera():
    positions = np.array([[1.0, 2.0, 3.0]])
    quaternions = np.array([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]])
    world = np.array([[[1.0, 3.0, 3.0], [0.0, 2.0, 3.0], [1.0, 2.0, 4.0]]])
    np.testing.assert_allclose(world_to_camera(world, positions, quaternions), np.eye(3)[None], atol=1e-14)
    np.testing.assert_allclose(camera_to_world(np.eye(3)[None], positions, quaternions), world, atol=1e-14)


@pytest.mark.parametrize("quaternion,expected", [
    ([1, 0, 0, 0], [1, 2, 3]),
    ([0, 1, 0, 0], [1, -2, -3]),
    ([0, 0, 1, 0], [-1, 2, -3]),
    ([0, 0, 0, 1], [-1, -2, 3]),
])
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_explicit_wxyz_order(transform, quaternion, expected):
    np.testing.assert_allclose(transform([[1, 2, 3]], [[0, 0, 0]], [quaternion]), [expected])


@pytest.mark.parametrize("shape", [(5, 3), (5, 21, 3), (5, 2, 21, 3), (5, 2, 3, 4, 3)])
def test_random_roundtrip_broadcasts_only_point_axes(shape):
    generator = np.random.default_rng(37)
    points = generator.normal(size=shape)
    positions = generator.normal(size=(5, 3))
    quaternions = generator.normal(size=(5, 4))
    camera = world_to_camera(points, positions, quaternions)
    recovered = camera_to_world(camera, positions, quaternions)
    assert camera.shape == points.shape
    assert camera.dtype == np.float64
    np.testing.assert_allclose(recovered, points, atol=1e-13)
    for frame in range(5):
        independent = world_to_camera(points[frame:frame + 1], positions[frame:frame + 1], quaternions[frame:frame + 1])
        np.testing.assert_allclose(camera[frame:frame + 1], independent)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("scale", [-1.0, 2.0, -1e300, 1e-300])
def test_quaternion_sign_and_scale_invariant(transform, scale):
    points = np.arange(36).reshape(2, 2, 3, 3)
    positions = np.ones((2, 3))
    quaternions = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, -0.5, 0.5]])
    expected = transform(points, positions, quaternions)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        np.testing.assert_allclose(transform(points, positions, quaternions * scale), expected)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_missing_hand_and_nonfinite_points_remain_nan_without_mutation(transform):
    points = np.ones((2, 2, 21, 3))
    points[0, 0] = np.nan
    points[1, 0, 2, 0] = np.inf
    points[1, 1, 0, 1] = np.nan
    positions, quaternions = camera_sequence(2)
    copies = [array.copy() for array in (points, positions, quaternions)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = transform(points, positions, quaternions)
    assert np.isnan(result[0, 0]).all()
    assert np.isnan(result[1, 0, 2]).all()
    assert np.isnan(result[1, 1, 0]).all()
    np.testing.assert_allclose(result[0, 1], 1)
    for original, expected in zip((points, positions, quaternions), copies):
        np.testing.assert_array_equal(original, expected)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("case", [
    "missing_time", "wrong_coordinate_count", "position_broadcast", "quaternion_broadcast",
    "position_extra_axis", "quaternion_extra_axis", "position_count", "quaternion_count",
    "position_nan", "position_inf", "quaternion_nan", "quaternion_inf", "quaternion_zero",
    "complex_points", "complex_positions", "complex_quaternions",
])
def test_invalid_transform_shapes_and_poses_rejected(transform, case):
    points = np.ones((2, 21, 3))
    positions, quaternions = camera_sequence(2)
    if case == "missing_time":
        points = np.ones(3)
    elif case == "wrong_coordinate_count":
        points = np.ones((2, 21, 4))
    elif case == "position_broadcast":
        positions = positions[0]
    elif case == "quaternion_broadcast":
        quaternions = quaternions[:1]
    elif case == "position_extra_axis":
        positions = positions[:, None]
    elif case == "quaternion_extra_axis":
        quaternions = quaternions[:, None]
    elif case == "position_count":
        positions = positions[:1]
    elif case == "quaternion_count":
        quaternions = np.tile(quaternions, (2, 1))
    elif case == "position_nan":
        positions[0, 0] = np.nan
    elif case == "position_inf":
        positions[0, 0] = np.inf
    elif case == "quaternion_nan":
        quaternions[0, 1] = np.nan
    elif case == "quaternion_inf":
        quaternions[0, 1] = np.inf
    elif case == "quaternion_zero":
        quaternions[0] = 0
    elif case == "complex_points":
        points = points.astype(complex)
    elif case == "complex_positions":
        positions = positions.astype(complex)
    elif case == "complex_quaternions":
        quaternions = quaternions.astype(complex)
    with pytest.raises(ValueError):
        transform(points, positions, quaternions)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("shape", [(0, 2, 21, 3), (2, 0, 3)])
def test_empty_transform_sequences(transform, shape):
    positions, quaternions = camera_sequence(shape[0])
    assert transform(np.empty(shape), positions, quaternions).shape == shape


def test_pinhole_formula_and_half_open_image_boundaries():
    pixels = np.array([[0, 0], [639, 479], [640, 200], [100, 480], [-1, 100], [320, -1]])
    points = np.column_stack(((pixels[:, 0] - 320) / 100, (pixels[:, 1] - 240) / 200, np.ones(6)))
    result = project_pinhole(points, INTRINSICS, 640, 480)
    assert set(result) == {"points_uv", "positive_depth", "inside_image"}
    assert result["points_uv"].dtype == np.float64
    assert result["positive_depth"].dtype == np.bool_
    assert result["inside_image"].dtype == np.bool_
    np.testing.assert_allclose(result["points_uv"], pixels, atol=1e-12)
    np.testing.assert_array_equal(result["positive_depth"], True)
    np.testing.assert_array_equal(result["inside_image"], [True, True, False, False, False, False])


def test_projection_depth_mask_nan_and_overflow_are_safe():
    points = np.array([
        [1, 1, 0], [1e308, 1e308, -1e-308], [np.nan, 0, 1], [0, np.inf, 1],
        [0, 0, np.inf], [0, 0, np.nan], [1e308, 0, 1e-308], [0, 0, 1e-300],
        [0, 0, -1], [2, 3, 2],
    ])
    original = points.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = project_pinhole(points, INTRINSICS, 640, 480)
    assert np.isnan(result["points_uv"][:7]).all()
    assert np.isnan(result["points_uv"][8]).all()
    np.testing.assert_array_equal(result["positive_depth"], [False] * 6 + [True, True, False, True])
    np.testing.assert_array_equal(result["inside_image"], [False] * 7 + [True, False, False])
    np.testing.assert_allclose(result["points_uv"][[7, 9]], [[320, 240], [420, 540]])
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 21, 3), (0, 3)])
def test_projection_preserves_arbitrary_leading_dimensions(shape):
    points = np.ones(shape)
    result = project_pinhole(points, INTRINSICS, np.int64(640), 480)
    assert result["points_uv"].shape == shape[:-1] + (2,)
    assert result["positive_depth"].shape == shape[:-1]
    assert result["inside_image"].shape == shape[:-1]


@pytest.mark.parametrize("name", ["fx", "fy", "cx", "cy"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf, True, "100", [100], 1 + 2j])
def test_invalid_intrinsic_values_rejected(name, value):
    intrinsics = dict(INTRINSICS, **{name: value})
    with pytest.raises(ValueError):
        project_pinhole([[0, 0, 1]], intrinsics, 640, 480)


@pytest.mark.parametrize("name", ["fx", "fy", "cx", "cy"])
def test_missing_intrinsics_never_default(name):
    intrinsics = dict(INTRINSICS)
    del intrinsics[name]
    with pytest.raises(ValueError):
        project_pinhole([[0, 0, 1]], intrinsics, 640, 480)


@pytest.mark.parametrize("name", ["fx", "fy"])
@pytest.mark.parametrize("value", [0, -1])
def test_focal_lengths_must_be_positive(name, value):
    with pytest.raises(ValueError):
        project_pinhole([[0, 0, 1]], dict(INTRINSICS, **{name: value}), 640, 480)


@pytest.mark.parametrize("intrinsics", [None, [], np.eye(3)])
def test_intrinsics_must_be_mapping(intrinsics):
    with pytest.raises(ValueError):
        project_pinhole([[0, 0, 1]], intrinsics, 640, 480)


@pytest.mark.parametrize("dimension", ["width", "height"])
@pytest.mark.parametrize("value", [0, -1, 1.5, 640.0, True, None, "640", np.nan, np.inf])
def test_image_dimensions_must_be_positive_integers(dimension, value):
    dimensions = {"width": 640, "height": 480, dimension: value}
    with pytest.raises(ValueError):
        project_pinhole([[0, 0, 1]], INTRINSICS, **dimensions)


def test_off_image_principal_points_are_not_clamped_or_guessed():
    result = project_pinhole([[0, 0, 1]], dict(INTRINSICS, cx=-20, cy=900), 640, 480)
    np.testing.assert_array_equal(result["points_uv"], [[-20, 900]])
    assert not result["inside_image"].any()


@pytest.mark.parametrize("points", [1, [], [[1, 2]], [[1, 2, 3, 4]], [[1 + 0j, 2, 3]]])
def test_invalid_projection_points_rejected(points):
    with pytest.raises(ValueError):
        project_pinhole(points, INTRINSICS, 640, 480)


def test_hand_edges_use_official_mediapipe_finger_chains():
    edges = set(HAND_EDGES)
    assert len(edges) == len(HAND_EDGES) == 21
    assert {joint for edge in edges for joint in edge} == set(range(21))
    for base in (1, 5, 9, 13, 17):
        for joint in range(base, base + 3):
            assert (joint, joint + 1) in edges
    assert {(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)} <= edges
