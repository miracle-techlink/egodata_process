import cv2
import numpy as np
import pytest

from src.pose6d.refine_backends import Rectification
from src.pose6d.rigid_fit import robust_rigid_fit,robust_rigid_fit_batched


def test_metric_scale_and_camera_axes_survive_rectification():
    k = np.array([[800.,0,960],[0,800,728],[0,0,1]])
    rel = np.eye(4)
    rel[:3,:3] = cv2.Rodrigues(np.array([.01,-.015,.005]))[0]
    rel[:3,3] = [-.064,.001,.002]
    rect = Rectification(k,k,rel,(1920,1456),width=768)
    # Known scene points, independently project into both original cameras.
    scene = np.array([[-.1,-.1,.5],[.15,.07,.7],[.02,.05,.4]])
    def project(xyz):
        q=xyz@k.T
        return q[:,:2]/q[:,2:]
    uv1 = project(scene)*rect.scale
    uv2 = project(scene@rel[:3,:3].T+rel[:3,3])*rect.scale
    r1,r2,p1,p2,q,_,_=cv2.stereoRectify(rect.K[0],None,rect.K[1],None,rect.size,
        rel[:3,:3].copy(),rel[:3,3:4].copy(),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    a=cv2.undistortPoints(uv1[:,None],rect.K[0],None,R=r1,P=p1)[:,0]
    b=cv2.undistortPoints(uv2[:,None],rect.K[1],None,R=r2,P=p2)[:,0]
    np.testing.assert_allclose(a[:,1],b[:,1],atol=1e-8)
    homogeneous=np.c_[a,a[:,0]-b[:,0],np.ones(len(a))]@rect.Q.T
    recovered=homogeneous[:,:3]/homogeneous[:,3:]@rect.rotation
    np.testing.assert_allclose(recovered,scene,atol=1e-8)


def test_disparity_depth_scale_and_invalid_uv():
    k=np.array([[800.,0,960],[0,800,728],[0,0,1]])
    rel=np.eye(4);rel[0,3]=-.064
    rect=Rectification(k,k,rel,(1920,1456),768)
    d=np.full(rect.size[::-1],rect.projection[0,0]*.064/.5,np.float32)
    xyz=rect.xyz(d)
    np.testing.assert_allclose(xyz[...,2],.5,atol=1e-6)
    sampled,valid=rect.sample_xyz(np.array([[960,728],[-1000,0]]),xyz,np.ones(d.shape,bool))
    assert valid.tolist()==[True,False]
    assert np.isnan(sampled[1]).all()
    np.testing.assert_allclose(sampled[0,2],.5,atol=1e-6)


def test_wrong_stereo_order_is_rejected():
    k=np.array([[800.,0,960],[0,800,728],[0,0,1]])
    rel=np.eye(4);rel[0,3]=.064
    with pytest.raises(ValueError,match='positive'):
        Rectification(k,k,rel,(1920,1456))


def test_vectorized_ransac_preserves_original_samples_and_quality_gates():
    rng=np.random.default_rng(17)
    for i in range(40):
        points=rng.normal(size=(64,3))*.04
        target=points@cv2.Rodrigues(np.array([.1,-.2,.3]))[0].T+[.1,.2,.6]
        target+=rng.normal(size=target.shape)*.001
        target[:i]+=rng.normal(size=(i,3))*.1
        if i%7==0:target[0]=np.nan
        old,a=robust_rigid_fit(points,target,trials=20)
        new,b=robust_rigid_fit_batched(points,target,trials=20)
        assert a['status']==b['status'] and a['inliers']==b['inliers']
        assert (old is None)==(new is None)
        if old is not None:np.testing.assert_allclose(old,new,atol=1e-9)
