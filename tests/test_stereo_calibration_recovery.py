import numpy as np
from scipy.spatial.transform import Rotation
from src.pose6d.refine_stereo_calibration import fundamental,residual


def test_epipolar_residual_uses_right_from_left_with_unequal_intrinsics():
    rng=np.random.default_rng(53)
    points=rng.uniform([-.3,-.2,.4],[.3,.2,1.5],(80,3))
    left=np.array([[790.,0,960],[0,780,728],[0,0,1]])
    right=np.array([[800.,0,950],[0,785,731],[0,0,1]])
    rel=np.eye(4);rel[:3,:3]=Rotation.from_euler('xyz',[.02,-.01,.005]).as_matrix();rel[:3,3]=[-.064,.0005,.0002]
    a=points@left.T;b=(points@rel[:3,:3].T+rel[:3,3])@right.T
    a=a[:,:2]/a[:,2:];b=b[:,:2]/b[:,2:]
    assert np.max(np.abs(residual(a,b,fundamental([left,right],rel))))<1e-9
    wrong=rel.copy();wrong[:3,:3]=np.eye(3)
    assert np.median(np.abs(residual(a,b,fundamental([left,right],wrong))))>5
