import numpy as np

from src.pose6d.tracker import (
    IoUTracker,
    assign_object_ids,
    box_area_xyxy,
    iou_matrix,
    sorted_tracks,
)


def test_iou_matrix_known():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 5, 15, 15], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    m = iou_matrix(a, b)
    assert m.shape == (1, 3)
    assert m[0, 0] == np.float64(25 / 175)
    assert m[0, 1] == np.float64(1.0)
    assert m[0, 2] == np.float64(0.0)


def test_box_area():
    assert box_area_xyxy([0, 0, 4, 5]) == 20.0
    assert box_area_xyxy([5, 5, 1, 1]) == 0.0


def test_ids_stable_for_moving_objects():
    rng = np.random.default_rng(0)
    tr = IoUTracker(iou_threshold=0.3, max_lost=10)
    ids_a, ids_b = [], []
    for f in range(60):
        box_a = np.array([10 + 0.5 * f, 10, 40 + 0.5 * f, 40])
        box_b = np.array([100 - 0.3 * f, 80, 150 - 0.3 * f, 130])
        boxes = np.stack([box_a, box_b]) + rng.normal(0, 0.2, (2, 4))
        m = tr.update(f, boxes)
        ids_a.append(m[0])
        ids_b.append(m[1])
    assert len(set(ids_a)) == 1 and len(set(ids_b)) == 1
    assert ids_a[0] != ids_b[0]


def test_reappearance_within_max_lost_keeps_id():
    tr = IoUTracker(iou_threshold=0.3, max_lost=5)
    box = np.array([[10, 10, 40, 40]], dtype=float)
    id0 = tr.update(0, box)[0]
    for f in range(1, 5):  # 5 frames missing (1..5 would exceed with reappear at 6)
        assert tr.update(f, np.zeros((0, 4))) == {}
    assert tr.update(5, np.zeros((0, 4))) == {}
    id1 = tr.update(6, box)[0]  # gap = 6-0-1 = 5 <= max_lost
    assert id1 == id0


def test_reappearance_beyond_max_lost_new_id():
    tr = IoUTracker(iou_threshold=0.3, max_lost=2)
    box = np.array([[10, 10, 40, 40]], dtype=float)
    id0 = tr.update(0, box)[0]
    for f in (1, 2, 3, 4):
        tr.update(f, np.zeros((0, 4)))
    id1 = tr.update(5, box)[0]  # gap = 4 > max_lost=2 -> retired
    assert id1 != id0


def test_new_object_mid_clip_gets_new_id():
    tr = IoUTracker(iou_threshold=0.3, max_lost=10)
    id0 = tr.update(0, np.array([[0, 0, 10, 10]], dtype=float))[0]
    m = tr.update(5, np.array([[0, 0, 10, 10], [100, 100, 140, 140]], dtype=float))
    assert m[0] == id0
    assert m[1] != id0


def test_empty_detections_no_crash():
    tr = IoUTracker()
    for f in range(3):
        assert tr.update(f, np.zeros((0, 4))) == {}
    assert tr.all_tracks() == []


def test_ordering_by_first_frame_area_desc():
    tr = IoUTracker(iou_threshold=0.3, max_lost=10)
    small = np.array([0, 0, 10, 10], dtype=float)      # area 100
    large = np.array([50, 50, 110, 110], dtype=float)  # area 3600
    med = np.array([200, 200, 240, 240], dtype=float)  # area 1600
    m = tr.update(0, np.stack([small, large]))
    tid_small, tid_large = m[0], m[1]
    m2 = tr.update(3, np.stack([small, large, med]))
    tid_med = m2[2]
    id_map = assign_object_ids(tr.all_tracks())
    assert id_map[tid_large] == "obj_000"
    assert id_map[tid_small] == "obj_001"
    assert id_map[tid_med] == "obj_002"  # later first frame sorts after frame-0 tracks
    ordered = sorted_tracks(tr.all_tracks())
    assert [t.track_id for t in ordered] == [tid_large, tid_small, tid_med]
