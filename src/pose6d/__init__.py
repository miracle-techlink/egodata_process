"""pose6d: object 6D pose extraction (task 3).

Light package init: rot_utils/tracker are pure numpy and always
importable. extract_pose6d lazily imports cv2/torch/FoundationPose/
DepthAnything, so this package imports fine without GPU deps.
"""
from . import rot_utils, tracker

__all__ = ["rot_utils", "tracker"]
