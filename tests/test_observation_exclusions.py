import numpy as np
from src.pose6d.observation_exclusions import capture_exclusion_mask


def test_lateral_wearable_removed_but_distal_tool_retained():
    image = np.zeros((500, 500, 3), np.uint8)
    hands = np.full((2, 21, 2), np.nan)
    hands[0, 0] = [250, 250]
    hands[0, 9] = [250, 200]
    mask = capture_exclusion_mask(image, hands)
    assert mask[300, 340]  # controller protruding sideways from the wrist
    assert not mask[110, 250]  # tool tip distal to the fingers


def test_missing_hands_do_not_mask_entire_scene():
    mask = capture_exclusion_mask(np.zeros((120, 120, 3), np.uint8), np.full((2, 21, 2), np.nan))
    assert not mask.any()
