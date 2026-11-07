"""Offline residual calibration of SDK-rectified stereo images, with held-out views."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .board import Board
from .storage import atomic_json


def matrix(intr):
    return np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1.]],float)


def solve(data, *, fixed_k_radial2=False):
    samples=data['samples']
    if len(samples)<25:raise ValueError('至少需要 25 个不同的停稳姿态，含 5 个留出验证姿态')
    ids=[s['id'] for s in samples]
    if len(set(ids))!=len(ids):raise ValueError('样本 ID 重复')
    if len({json.dumps(s['left']) for s in samples})!=len(samples):raise ValueError('不能重复照片凑数')
    board=Board(data['board']);size=(data['left']['width'],data['left']['height'])
    if size != (data['right']['width'],data['right']['height']):raise ValueError('左右分辨率不一致')
    for sample in samples:
        indexes=np.asarray(sample['ids'])
        if indexes.ndim!=1 or len(indexes)<6 or indexes.dtype.kind not in 'iu' or len(np.unique(indexes))!=len(indexes) or np.any(indexes<0) or np.any(indexes>=len(board.object_points)):
            raise ValueError('共同角点 ID 无效')
        for side in ('left','right'):
            points=np.asarray(sample[side],float)
            if points.shape!=(len(indexes),2) or not np.isfinite(points).all():raise ValueError('角点数组无效')
    held=[i for i in range(len(samples)) if i%5==4];train=[i for i in range(len(samples)) if i not in held]
    obj=[board.object_points[np.array(s['ids'],int)].astype(np.float32) for s in samples]
    xy=[[np.array(s[side],np.float32).reshape(-1,1,2) for s in samples] for side in ('left','right')]
    candidates=[]
    modes=['fixed_K_radial2'] if fixed_k_radial2 else ['factory_intrinsics_refit_extrinsics','fit_intrinsics_radial2']
    for mode in modes:
        ks=[matrix(data[side]) for side in ('left','right')];ds=[np.array(data[side]['coeffs'],float) for side in ('left','right')]
        if mode!='factory_intrinsics_refit_extrinsics':
            flags=cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K3
            if mode=='fixed_K_radial2':flags|=cv2.CALIB_FIX_FOCAL_LENGTH|cv2.CALIB_FIX_PRINCIPAL_POINT
            for j in range(2):
                _,ks[j],ds[j],_,_=cv2.calibrateCamera([obj[i] for i in train],[xy[j][i] for i in train],size,ks[j],ds[j],flags=flags)
        err,k1,d1,k2,d2,r,t,_,_=cv2.stereoCalibrate([obj[i] for i in train],[xy[0][i] for i in train],[xy[1][i] for i in train],ks[0],ds[0],ks[1],ds[1],size,flags=cv2.CALIB_FIX_INTRINSIC,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,1e-7))
        r1,r2,p1,p2,q,_,_=cv2.stereoRectify(k1,d1,k2,d2,size,r,t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
        errors=[];epi=[];normals=[]
        for i in held:
            ok,rv,tv=cv2.solvePnP(obj[i],xy[0][i],k1,d1)
            if not ok:raise ValueError('留出姿态 PnP 失败')
            rot=cv2.Rodrigues(rv)[0];normals.append(rot[:,2])
            pred=cv2.projectPoints(obj[i],cv2.Rodrigues(r@rot)[0],r@tv+t,k2,d2)[0]
            errors.extend(np.linalg.norm(pred.reshape(-1,2)-xy[1][i].reshape(-1,2),axis=1))
            a=cv2.undistortPoints(xy[0][i],k1,d1,R=r1,P=p1).reshape(-1,2)
            b=cv2.undistortPoints(xy[1][i],k2,d2,R=r2,P=p2).reshape(-1,2)
            epi.extend(np.abs(a[:,1]-b[:,1]))
        candidates.append(dict(name=mode,fit_rms_px=float(err),heldout_right_rms_px=float(np.sqrt(np.mean(np.square(errors)))),heldout_epipolar_median_px=float(np.median(epi)),heldout_epipolar_p95_px=float(np.percentile(epi,95)),baseline_m=float(np.linalg.norm(t)),K_left=k1.tolist(),D_left=d1.ravel().tolist(),K_right=k2.tolist(),D_right=d2.ravel().tolist(),R=r.tolist(),T=t.ravel().tolist(),R1=r1.tolist(),R2=r2.tolist(),P1=p1.tolist(),P2=p2.tolist(),Q=q.tolist()))
    chosen=min(candidates,key=lambda c:c['heldout_right_rms_px'])
    return dict(schema_version=1,input_space=data.get('input_space'),serial=data['serial'],board=data['board'],image_size=list(size),training_ids=[ids[i] for i in train],heldout_ids=[ids[i] for i in held],candidates=candidates,selected=chosen['name'],candidate=chosen,installed=False,independent_height_validation_pending=True,notes=['Held-out views never participate in fitting.','Selection compares cross-view prediction; it does not certify absolute depth accuracy.','Candidate applies to this SDK-rectified stream/profile only; keep firmware calibration unchanged.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('--fixed-k-radial2',action='store_true',help='Fix factory focal lengths/principal points; fit only k1/k2 before stereo extrinsics');a=p.parse_args();root=Path(a.session)
    try:
        result=solve(json.loads((root/'observations.json').read_text()),fixed_k_radial2=a.fixed_k_radial2);atomic_json(root/'stereo-result.json',result)
        atomic_json(root/'solve-status.json',dict(state='solved',hint='候选已拟合；还需独立平放板高度验证'))
    except Exception as exc:
        atomic_json(root/'solve-status.json',dict(state='error',hint=str(exc)));raise
