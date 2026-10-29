import numpy as np
import pytest

from src.dataset.audit_hocap_reference import check_label


def valid_label():
    intrinsics = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1.]])
    joints = np.zeros((2, 21, 3), dtype=np.float32)
    joints[..., 2] = 1
    poses = np.eye(4)[None]
    poses[:, 2, 3] = 1
    return {'cam_K': intrinsics, 'obj_poses': poses, 'hand_joints_3d': joints,
            'hand_joints_2d': np.broadcast_to([320, 240], (2, 21, 2)).copy(),
            'seg_mask': np.zeros((480, 640), dtype=np.uint8)}


def test_reference_geometry_is_not_accuracy():
    result = check_label(valid_label(), (480, 640, 3))
    assert result['in_frame_hand_joints'] == 42
    assert result['complete_hands_in_frame'] == 2
    assert max(result['reprojection_residuals_px']) == 0
    assert 'accuracy' not in result


def test_source_negative_one_hand_sentinel_not_counted():
    label = valid_label()
    label['hand_joints_3d'][0] = -1
    label['hand_joints_2d'][0] = -1
    result = check_label(label, (480, 640, 3))
    assert result['valid_hand_joints'] == 21


def test_out_of_view_2d_does_not_remove_valid_3d_reference():
    label = valid_label()
    label['hand_joints_2d'][:] = -1
    result = check_label(label, (480, 640, 3))
    assert result['in_frame_hand_joints'] == 0
    assert result['complete_hands_in_frame'] == 0
    assert result['valid_hand_joints'] == 42


@pytest.mark.parametrize('field', ['hand_joints_3d', 'obj_poses'])
def test_partial_nan_rejected(field):
    label = valid_label()
    label[field].flat[0] = np.nan
    with pytest.raises(ValueError, match='partial nonfinite'):
        check_label(label, (480, 640, 3))


def test_wrong_projection_and_wrong_rotation_rejected():
    label = valid_label()
    label['hand_joints_2d'] += 5
    with pytest.raises(ValueError, match='reprojection'):
        check_label(label, (480, 640, 3))
    label = valid_label()
    label['obj_poses'][0, 0, 0] = -1
    with pytest.raises(ValueError, match='rigid transform'):
        check_label(label, (480, 640, 3))


def test_wrong_segmentation_shape_rejected():
    label = valid_label()
    label['seg_mask'] = np.zeros((48, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match='segmentation'):
        check_label(label, (480, 640, 3))


def test_all_missing_object_sentinel_is_not_rotation():
    label = valid_label()
    label['obj_poses'][:] = -1
    result = check_label(label, (480, 640, 3))
    assert result['valid_objects'] == 0
