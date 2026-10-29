import copy

import numpy as np
import pytest

from src.dataset.audit_egodemo_mamba_v2 import audit_windows, expected_selection, masked_geometry, se3


CONFIG = dict(window_frames=4, stride_frames=3, horizon_seconds=.2, max_horizon_overshoot_median_dt=1.5)


def windows(times):
    rows = []
    for start in range(0, len(times) - 3, 3):
        anchor = start + 3
        target = int(np.searchsorted(times, times[anchor] + .2))
        if target < len(times) and times[target] - times[anchor] - .2 <= .15 + 1e-9:
            rows.append(dict(start_frame=start, end_frame=start + 4, anchor_frame=anchor,
                             target_frame=target, target_timestamp=float(times[target]),
                             actual_horizon_s=float(times[target] - times[anchor])))
    return rows


def test_native_time_completeness_and_future_horizon():
    times = np.arange(20, dtype=np.float64) / 10
    rows = windows(times)
    audit_windows(rows, times, CONFIG)
    for mutate in [lambda r: r.pop(), lambda r: r.append(r[0]),
                   lambda r: r[0].update(target_frame=4),
                   lambda r: r[0].update(actual_horizon_s=1.0)]:
        altered = copy.deepcopy(rows)
        mutate(altered)
        with pytest.raises(ValueError):
            audit_windows(altered, times, CONFIG)


def test_horizon_gap_is_rejected_not_interpolated():
    times = np.r_[np.arange(5) / 10, 3 + np.arange(10) / 10]
    rows = windows(times)
    audit_windows(rows, times, CONFIG)
    assert all(row["start_frame"] != 0 for row in rows)
    rows.append(dict(start_frame=0, end_frame=4, anchor_frame=3, target_frame=5,
                     target_timestamp=3., actual_horizon_s=2.7))
    with pytest.raises(ValueError, match="overshoot"):
        audit_windows(rows, times, CONFIG)


def test_nan_mask_and_improper_rotation_rejected():
    points = np.ones((2, 3)); valid = np.asarray([True, False])
    with pytest.raises(ValueError, match="NaN"):
        masked_geometry(points, valid, "point")
    points[1] = np.nan
    masked_geometry(points, valid, "point")
    poses = np.eye(4)[None].copy()
    poses[0, 0, 0] = -1
    with pytest.raises(ValueError, match="determinant"):
        se3(poses, np.asarray([True]), "pose")


def test_future_object_and_point_cannot_change_historical_selection():
    arrays = dict(object_observed=np.asarray([[True, False], [True, False], [True, True]]),
                  object_ids=np.asarray(["z", "a"]),
                  surface_frame_ptr=np.asarray([0, 1, 2, 4]),
                  surface_object_index=np.asarray([0, 0, 0, 1]),
                  surface_point_index=np.asarray([0, 1, 2, 3]),
                  surface_frame_index=np.asarray([0, 1, 2, 2]),
                  point_keys=np.asarray(["t/z", "t/a", "t/0", "u/a"]))
    assert expected_selection(arrays, 0, 2) == ([0], [[0, 1]])
    # Earlier lexical IDs and many future observations still cannot change slots.
    arrays["object_observed"][2] = True
    arrays["point_keys"][2:] = ["t/!", "u/!"]
    assert expected_selection(arrays, 0, 2) == ([0], [[0, 1]])
