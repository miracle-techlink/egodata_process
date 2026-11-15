"""Sparse stereo surface-point tracks; deliberately has no rigid pose output."""
import cv2
import numpy as np
from src.pose6d.stereo_reference import triangulate


def follow_points(previous, current, points, valid, fb_limit=1.5):
    output = np.full_like(points, np.nan)
    alive = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(valid & np.isfinite(points).all(axis=1))
    if len(ids) == 0:
        return output, alive
    a = points[ids].astype(np.float32).reshape(-1, 1, 2)
    b, forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, a, None, winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,.01))
    recovered, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, b, None, winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,.01))
    error = np.linalg.norm(recovered[:,0]-a[:,0],axis=1)
    q=b[:,0]
    good=(forward[:,0].astype(bool)&backward[:,0].astype(bool)&(error<fb_limit)&np.isfinite(q).all(axis=1)
          &(q[:,0]>=0)&(q[:,1]>=0)&(q[:,0]<current.shape[1])&(q[:,1]<current.shape[0]))
    output[ids[good]]=q[good];alive[ids[good]]=True
    return output, alive


def stereo_points(left_uv,right_uv,left_K,right_K,right_from_left):
    valid=np.isfinite(left_uv).all(axis=1)&np.isfinite(right_uv).all(axis=1)
    xyz=np.full((len(left_uv),3),np.nan);errors=np.full(len(left_uv),np.nan)
    ids=np.flatnonzero(valid)
    if len(ids):
        points,geometric,err=triangulate(left_uv[ids],right_uv[ids],left_K,right_K,right_from_left)
        xyz[ids[geometric]]=points[geometric];errors[ids]=err
        valid[ids]=geometric
    return xyz,valid,errors
