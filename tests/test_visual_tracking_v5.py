"""Real OpenCV synthetic-pixel tests for causal observation tracking."""
import cv2
import numpy as np
import pytest

from src.annotation.track_visual_entities_v5 import PrefixTracker, project_hands, validate_seeds


def textured_frame(dx=0, dy=0):
    rng = np.random.default_rng(21)
    image = np.zeros((128, 160, 3), np.uint8)
    image[30:94, 40:104] = rng.integers(30, 250, (64, 64, 3), dtype=np.uint8)
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (160, 128))


def seed(frame=0, identity="visual_box", box=(40, 30, 104, 94)):
    return {"frame_index": frame, "entity_id": identity, "bbox_xyxy_native": list(box)}


NO_HANDS = np.empty((0, 2), np.float32)


def test_projection_inverts_real_camera_rotation_and_translation():
    camera = np.eye(4)[None].repeat(2, axis=0)
    camera[:, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    camera[:, :3, 3] = [10, 20, 30]
    local = np.ones((2, 2, 21, 3)) * [1, 2, 4]
    local[1, 1] = [0, 0, -1]
    world = np.einsum("fij,fhkj->fhki", camera[:, :3, :3], local) + camera[:, None, None, :3, 3]
    K = np.array([[100, 0, 50], [0, 150, 40], [0, 0, 1]])
    uv, valid = project_hands(world, camera, K, np.ones((2, 2), dtype=bool))
    np.testing.assert_allclose(uv[valid], np.broadcast_to([75, 115], uv[valid].shape))
    assert valid.tolist() == [[True, True], [True, False]]
    assert np.isnan(uv[1, 1]).all()


def test_invalid_hand_and_zero_depth_do_not_produce_image_coordinates():
    world = np.ones((1, 2, 21, 3))
    world[:, 0, :, 2] = 0
    uv, valid = project_hands(world, np.eye(4)[None], np.eye(3), np.array([[True, False]]))
    assert not valid.any() and np.isnan(uv).all()


def test_observed_translation_tracks_native_pixels_with_real_optical_flow():
    tracker = PrefixTracker(scale=1, max_seed_age=20, minimum_points=6)
    counts = []
    for f in range(5):
        result = tracker.step(f, textured_frame(2*f, f), [seed()] if f == 0 else [], NO_HANDS)
        assert set(result) == {"visual_box"}
        box, seed_at, points, source = result["visual_box"]
        np.testing.assert_allclose(box, np.array([40, 30, 104, 94]) + [2*f, f, 2*f, f], atol=0.6)
        assert seed_at == 0 and source == (1 if f == 0 else 2)
        counts.append(points)
    assert all(b <= a for a, b in zip(counts, counts[1:]))


def test_no_seed_never_creates_an_entity_from_visible_texture():
    tracker = PrefixTracker(scale=1)
    for f in range(5):
        assert tracker.step(f, textured_frame(f, 0), [], NO_HANDS) == {}


def test_future_pixels_and_future_seed_do_not_change_identical_prefix():
    sequences = []
    for variant in range(2):
        tracker = PrefixTracker(scale=1)
        result = []
        for f in range(8):
            image = textured_frame(f, 0) if f < 4 else np.full((128, 160, 3), variant * 255, np.uint8)
            seeds = [seed()] if f == 0 else ([seed(6, "visual_future")] if f == 6 and variant else [])
            result.append(tracker.step(f, image, seeds, NO_HANDS))
        sequences.append(result)
    for a, b in zip(sequences[0][:4], sequences[1][:4]):
        assert a.keys() == b.keys() == {"visual_box"}
        np.testing.assert_array_equal(a["visual_box"][0], b["visual_box"][0])
        assert a["visual_box"][1:] == b["visual_box"][1:]
    assert "visual_future" not in sequences[1][5] and "visual_future" in sequences[1][6]


def test_fixed_seed_expiry_and_explicit_reinitialization():
    tracker = PrefixTracker(scale=1, max_seed_age=3)
    for f in range(6):
        result = tracker.step(f, textured_frame(), [seed(f)] if f in (0, 5) else [], NO_HANDS)
        assert bool(result) == (f in (0, 1, 2, 5))
        if f == 5:
            assert result["visual_box"][1] == 5


def test_lost_texture_stops_and_does_not_reappear_without_new_seed():
    tracker = PrefixTracker(scale=1)
    assert tracker.step(0, textured_frame(), [seed()], NO_HANDS)
    assert tracker.step(1, np.zeros((128, 160, 3), np.uint8), [], NO_HANDS) == {}
    assert tracker.step(2, textured_frame(), [], NO_HANDS) == {}
    assert tracker.step(3, textured_frame(), [seed(3)], NO_HANDS)


def test_untrackable_current_visual_seed_is_valid_only_at_its_actual_frame():
    tracker = PrefixTracker(scale=1)
    flat = np.zeros((128, 160, 3), np.uint8)
    current = tracker.step(0, flat, [seed()], NO_HANDS)
    assert current["visual_box"][2] == 0 and current["visual_box"][3] == 1
    assert tracker.step(1, flat, [], NO_HANDS) == {}


def test_wrong_step_and_future_seed_rejection_do_not_poison_valid_prefix():
    tracker = PrefixTracker(scale=1)
    with pytest.raises(ValueError, match="consecutive"):
        tracker.step(1, textured_frame(), [], NO_HANDS)
    with pytest.raises(ValueError, match="seed"):
        tracker.step(0, textured_frame(), [seed(3)], NO_HANDS)
    assert tracker.last_frame == -1 and tracker.previous is None and tracker.states == {}
    assert tracker.step(0, textured_frame(), [seed()], NO_HANDS)
    with pytest.raises(ValueError, match="Duplicate"):
        tracker.step(1, textured_frame(), [seed(1), seed(1)], NO_HANDS)
    assert tracker.last_frame == 0
    assert tracker.step(1, textured_frame(), [], NO_HANDS)


@pytest.mark.parametrize("options", [{"scale": 0}, {"scale": 1.2}, {"max_seed_age": 0}, {"minimum_points": 2}])
def test_invalid_tracking_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        PrefixTracker(**options)


def test_seed_bounds_and_native_identity_validation():
    doc = {"schema_version": "assistant_annotation_v5_visual_seeds", "native_image_size": [160, 128],
           "seeds": [seed(box=(-1, 30, 104, 94))]}
    with pytest.raises(ValueError, match="outside native"):
        validate_seeds(doc, 10)
    doc["seeds"] = [seed(identity=None)]
    with pytest.raises(ValueError, match="Invalid native seed"):
        validate_seeds(doc, 10)
