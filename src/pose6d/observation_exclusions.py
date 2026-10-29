"""Conservative exclusion of hands and wearable capture markers at seed time."""
import cv2
import numpy as np


def capture_exclusion_mask(image_bgr, hands_uv):
    mask=np.zeros(image_bgr.shape[:2],np.uint8)
    wrists=[]
    for uv in np.asarray(hands_uv):
        finite=uv[np.isfinite(uv).all(axis=1)]
        if len(finite)>=3:cv2.fillConvexPoly(mask,cv2.convexHull(finite.astype(np.int32)),255)
        if np.isfinite(uv[[0,9]]).all():
            vector=uv[9]-uv[0]
            radius=int(np.clip(np.linalg.norm(vector)*1.05,30,220))
            wrists.append((uv[0],radius))
            # Source wrists can be centred on the wearable itself. Cover both
            # the annotated wrist and proximal cuff, not just the forearm.
            for centre in [uv[0],uv[0]-.65*vector]:
                cv2.circle(mask,tuple(centre.astype(int)),radius,255,-1)
            # Capture controllers protrude sideways beyond the cuff. Exclude
            # the proximal wrist strip, while retaining the distal tool tip.
            if np.linalg.norm(vector)>5:
                lateral=np.array([-vector[1],vector[0]])
                cuff=np.array([uv[0]+along*vector+side*lateral
                               for along,side in [(.25,-2.2),(.25,2.2),(-3,2.2),(-3,-2.2)]])
                cv2.fillConvexPoly(mask,cuff.astype(np.int32),255)
    if hasattr(cv2,'aruco'):
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        detector=cv2.aruco.ArucoDetector(dictionary,cv2.aruco.DetectorParameters())
        corners,_,_=detector.detectMarkers(cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY))
        for corners_one in corners:
            points=corners_one[0];centre=points.mean(axis=0)
            if wrists and min(np.linalg.norm(centre-w) / r for w,r in wrists)<3:
                expanded=(points-centre)*2.5+centre
                cv2.fillConvexPoly(mask,expanded.astype(np.int32),255)
    return cv2.dilate(mask,np.ones((9,9),np.uint8))
