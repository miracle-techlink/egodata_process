import numpy as np
import pytest

from src.annotation.ground_dense_entities_v4 import inside_polygon, old_anchors, propagate_anchor, ranges


def track(frames=None, points=6):
    frames = np.arange(10) if frames is None else np.asarray(frames)
    positions = np.zeros((len(frames), 2, points, 2), dtype=float)
    positions[..., 0] = 40 + np.arange(points)
    positions[..., 1] = 50
    return {"frame_index": frames, "timestamps": frames * .04,
            "points_uv": positions, "point_valid": np.ones((len(frames), points), dtype=bool),
            "point_ids": np.arange(points) + 20}


def test_lost_points_do_not_reappear_and_chain_does_not_restart():
    source = track()
    source["point_valid"][5, :3] = False
    observations, stops = propagate_anchor(source, 3, np.arange(10) * .04, [200, 100])
    assert [x["frame_index"] for x in observations] == [0, 1, 2, 3, 4]
    assert stops["after"] == "insufficient_uninterrupted_anchor_points"
    # Later visibility cannot heal the lost identity connection.
    assert source["point_valid"][6:].all()


def test_surviving_points_are_intersection_of_every_intervening_frame():
    source = track(points=7)
    source["point_valid"][5, 0] = False
    source["point_valid"][6, 1] = False
    source["point_valid"][7, 2] = False
    source["point_valid"][8, 3] = False
    observations, _ = propagate_anchor(source, 3, np.arange(10) * .04, [200, 100])
    last = observations[-1]
    assert last["frame_index"] == 7
    assert last["point_ids"] == [23, 24, 25, 26]
    assert last["point_count"] == 4


def test_frame_gap_breaks_identity_even_when_all_points_are_valid():
    source = track([0, 1, 2, 4, 5, 6])
    observations, stops = propagate_anchor(source, 1, np.arange(10) * .04, [200, 100])
    assert [x["frame_index"] for x in observations] == [0, 1, 2]
    assert stops["after"] == "native_frame_gap"


def test_timestamp_discontinuity_breaks_native_index_chain():
    source = track()
    times = np.arange(10) * .04
    times[5:] += .5
    source["timestamps"] = times.copy()
    observations, stops = propagate_anchor(source, 3, times, [200, 100])
    assert observations[-1]["frame_index"] == 4
    assert stops["after"] == "native_timestamp_gap"


def test_large_pixel_jump_cannot_propagate_identity():
    source = track()
    source["points_uv"][5:, :, :, 0] += 70
    observations, stops = propagate_anchor(source, 3, np.arange(10) * .04, [200, 100])
    assert observations[-1]["frame_index"] == 4
    assert stops["after"] == "insufficient_uninterrupted_anchor_points"


def test_anchor_uses_only_explicit_pixel_selected_point_ids():
    source = track(points=8)
    observations, _ = propagate_anchor(source, 3, np.arange(10) * .04, [200, 100], [21, 23, 25, 27])
    assert all(x["point_ids"] == [21, 23, 25, 27] for x in observations)
    with pytest.raises(ValueError, match="unknown point"):
        propagate_anchor(source, 3, np.arange(10) * .04, [200, 100], [99])
    assert propagate_anchor(source, 3, np.arange(10) * .04, [200, 100], [21, 23])[1] == "insufficient_anchor_points"


def test_native_time_binding_and_point_identity_are_checked():
    source = track()
    source["timestamps"][2] += .001
    with pytest.raises(ValueError, match="timestamps differ"):
        propagate_anchor(source, 3, np.arange(10) * .04, [200, 100])
    source = track()
    source["point_ids"][1] = source["point_ids"][0]
    with pytest.raises(ValueError, match="identity/frame"):
        propagate_anchor(source, 3, np.arange(10) * .04, [200, 100])


def test_pixel_polygon_and_noncontiguous_ranges():
    assert inside_polygon([[5, 5], [11, 5], [3, 4]], [[1, 1], [9, 1], [9, 9], [1, 9]]).tolist() == [True, False, True]
    assert ranges([2, 3, 6, 6, 10]) == [{"start_frame": 2, "end_frame": 4},
                                       {"start_frame": 6, "end_frame": 7}, {"start_frame": 10, "end_frame": 11}]


def test_roles_and_uncertain_names_do_not_become_identity_anchors():
    assessment = {"status": "supported", "silver_label_valid": True,
        "annotation_source": "assistant_visual_review", "review_status": "assistant_reviewed",
        "scope": "visual_identity", "evidence_frames": [7], "evidence": [],
        "reviewer_agent": "test", "reviewed_at": "2026-10-24"}
    doc = {"entities": [{"id": "obj_001", "visual_assessments": [assessment,
        {**assessment, "status": "uncertain"}, {**assessment, "scope": "role"}]},
        {"id": "visual_same_name", "visual_assessments": [assessment]}]}
    result = old_anchors(doc)
    assert len(result) == 1
    assert result[0]["entity_id"] == "obj_001" and result[0]["frame_index"] == 7
