import numpy as np
from src.pose6d.stereo_reference import relative_camera, triangulate


def test_metric_triangulation_with_documented_baseline():
    K = np.array([[800., 0, 960], [0, 800, 728], [0, 0, 1]])
    left = np.eye(4)
    right = np.eye(4)
    right[0, 3] = .064
    relative = relative_camera(left, right)
    points = np.array([[.1, .2, .5], [-.1, -.2, .8], [.2, -.1, 1.2]])
    a = points @ K.T
    b = (points + relative[:3, 3]) @ K.T
    recovered, valid, errors = triangulate(a[:, :2]/a[:, 2:], b[:, :2]/b[:, 2:], K, K, relative)
    assert valid.all()
    np.testing.assert_allclose(recovered, points, atol=1e-12)
    assert max(errors) < 1e-10


def test_wrong_epipolar_correspondence_is_missing_not_pose():
    K = np.array([[800., 0, 960], [0, 800, 728], [0, 0, 1]])
    transform = np.eye(4)
    transform[0, 3] = -.064
    _, valid, _ = triangulate([[960, 728]], [[860, 770]], K, K, transform)
    assert not valid.any()


def test_reversed_disparity_is_behind_camera():
    K = np.array([[800., 0, 960], [0, 800, 728], [0, 0, 1]])
    transform = np.eye(4)
    transform[0, 3] = -.064
    _, valid, _ = triangulate([[900, 728]], [[950, 728]], K, K, transform)
    assert not valid.any()
