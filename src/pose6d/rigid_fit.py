"""Robust metric surface alignment with explicit observability checks."""
import numpy as np


def align(source, target):
    a, b = np.asarray(source, float), np.asarray(target, float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = cb-r@ca
    pose = np.eye(4);pose[:3,:3]=r;pose[:3,3]=t
    return pose


def robust_rigid_fit(source, target, threshold=.012, min_inliers=8, trials=40):
    a, b = np.asarray(source,float), np.asarray(target,float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Corresponding points must have identical (N,3) shapes')
    valid=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    a,b=a[valid],b[valid]
    stats={'correspondences':len(a),'inliers':0,'rms_m':None,'status':'insufficient_points'}
    if len(a)<min_inliers:return None,stats
    singular=np.linalg.svd(a-a.mean(axis=0),compute_uv=False)
    # A point or a near-line cannot observe all three rotational degrees.
    if singular[0]<.015 or singular[1]/singular[0]<.02:
        stats['status']='rotation_unobservable';return None,stats
    rng=np.random.default_rng(0);best=np.zeros(len(a),bool)
    for _ in range(trials):
        ids=rng.choice(len(a),3,replace=False)
        if np.linalg.norm(np.cross(a[ids[1]]-a[ids[0]],a[ids[2]]-a[ids[0]]))<1e-6:continue
        pose=align(a[ids],b[ids])
        errors=np.linalg.norm(a@pose[:3,:3].T+pose[:3,3]-b,axis=1)
        good=errors<threshold
        if good.sum()>best.sum():best=good
    stats['inliers']=int(best.sum())
    if best.sum()<min_inliers or best.mean()<.65:
        stats['status']='nonrigid_or_outliers';return None,stats
    pose=align(a[best],b[best])
    errors=np.linalg.norm(a[best]@pose[:3,:3].T+pose[:3,3]-b[best],axis=1)
    rms=float(np.sqrt(np.mean(errors**2)))
    stats.update(rms_m=rms,status='estimated' if rms<threshold*.7 else 'high_residual')
    return (pose if stats['status']=='estimated' else None),stats


def robust_rigid_fit_batched(source,target,threshold=.012,min_inliers=8,trials=40):
    """Same RANSAC samples and gates, vectorized small-matrix hypotheses."""
    a,b=np.asarray(source,float),np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:
        raise ValueError('Corresponding points must have identical (N,3) shapes')
    valid=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1);a,b=a[valid],b[valid]
    stats={'correspondences':len(a),'inliers':0,'rms_m':None,'status':'insufficient_points'}
    if len(a)<min_inliers:return None,stats
    singular=np.linalg.svd(a-a.mean(axis=0),compute_uv=False)
    if singular[0]<.015 or singular[1]/singular[0]<.02:
        stats['status']='rotation_unobservable';return None,stats
    rng=np.random.default_rng(0)
    choices=np.array([rng.choice(len(a),3,replace=False) for _ in range(trials)])
    sa,sb=a[choices],b[choices]
    usable=np.linalg.norm(np.cross(sa[:,1]-sa[:,0],sa[:,2]-sa[:,0]),axis=1)>=1e-6
    sa,sb=sa[usable],sb[usable]
    best=np.zeros(len(a),bool)
    if len(sa):
        ca,cb=sa.mean(axis=1),sb.mean(axis=1)
        covariance=(sa-ca[:,None]).transpose(0,2,1)@(sb-cb[:,None])
        u,_,vt=np.linalg.svd(covariance);rotation=vt.transpose(0,2,1)@u.transpose(0,2,1)
        negative=np.linalg.det(rotation)<0
        vt[negative,-1]*=-1
        rotation=vt.transpose(0,2,1)@u.transpose(0,2,1)
        translation=cb-np.einsum('tij,tj->ti',rotation,ca)
        errors=np.linalg.norm(np.einsum('nj,tij->tni',a,rotation)+translation[:,None]-b,axis=2)
        good=errors<threshold;best=good[np.argmax(good.sum(axis=1))]
    stats['inliers']=int(best.sum())
    if best.sum()<min_inliers or best.mean()<.65:
        stats['status']='nonrigid_or_outliers';return None,stats
    pose=align(a[best],b[best]);errors=np.linalg.norm(a[best]@pose[:3,:3].T+pose[:3,3]-b[best],axis=1)
    rms=float(np.sqrt(np.mean(errors**2)))
    stats.update(rms_m=rms,status='estimated' if rms<threshold*.7 else 'high_residual')
    return (pose if stats['status']=='estimated' else None),stats
