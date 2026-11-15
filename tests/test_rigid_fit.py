import numpy as np
from src.pose6d.rigid_fit import robust_rigid_fit


def test_recovers_real_rotation_translation_with_outliers():
    rng=np.random.default_rng(42)
    a=rng.uniform(-.1,.1,(64,3))
    angle=.4;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    t=np.array([.1,.2,.8]);b=a@r.T+t;b[:8]+=.2
    pose,stats=robust_rigid_fit(a,b)
    np.testing.assert_allclose(pose[:3,:3],r,atol=1e-8)
    np.testing.assert_allclose(pose[:3,3],t,atol=1e-8)
    assert stats['inliers']==56


def test_line_cannot_claim_full_rotation():
    a=np.zeros((30,3));a[:,0]=np.linspace(-.1,.1,30)
    pose,stats=robust_rigid_fit(a,a+[0,0,.6])
    assert pose is None and stats['status']=='rotation_unobservable'


def test_deformation_rejected_as_rigid():
    rng=np.random.default_rng(12);a=rng.uniform(-.1,.1,(64,3));b=a*[2,1,.4]+[0,0,.5]
    pose,stats=robust_rigid_fit(a,b)
    assert pose is None and stats['status']=='nonrigid_or_outliers'
