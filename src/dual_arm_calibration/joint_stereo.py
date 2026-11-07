"""Joint three-camera bundle adjustment with fixed K and fixed RGB distortion."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .board import Board
from .stereo import matrix
from .color_extrinsics import evaluate
from .storage import atomic_json


def transform(rv,t):
    c=np.eye(4);c[:3,:3]=cv2.Rodrigues(np.asarray(rv))[0];c[:3,3]=t
    return c


def optimize(obs, paired, initial, color_initial, rgb_model, training_ids, heldout_ids):
    board=Board(obs['board']);kl=matrix(obs['left']);kr=matrix(obs['right']);kc=matrix(rgb_model);dc=np.array(rgb_model['coeffs'])
    for side,k in [('left',kl),('right',kr)]:
        if not np.array_equal(k,np.array(initial['K_'+side])):raise ValueError('Initial model must have factory K fixed')
    rgb_by_id={s['id']:s for s in paired};by_id={s['id']:s for s in obs['samples']}
    if set(training_ids)&set(heldout_ids):raise ValueError('Training and holdout IDs overlap')
    if any(i not in rgb_by_id or i not in by_id for i in training_ids+heldout_ids):raise ValueError('Missing paired observations; split must not silently change')
    c0=np.array(color_initial);rl0=np.array(initial['R']);tl0=np.array(initial['T'])
    values=[*initial['D_left'][:2],*initial['D_right'][:2],*cv2.Rodrigues(rl0)[0].ravel(),*tl0,*cv2.Rodrigues(c0[:3,:3])[0].ravel(),*c0[:3,3]]
    groups=[]
    for i in training_ids:
        s=by_id[i];p=rgb_by_id[i];obj=board.object_points[np.array(s['ids'],int)].astype(float);co=board.object_points[np.array(p['ids'],int)].astype(float)
        ok,rv,tv=cv2.solvePnP(obj,np.array(s['left']),kl,np.array(initial['D_left']))
        if not ok:raise ValueError('Initial board pose failed')
        values.extend([*rv.ravel(),*tv.ravel()]);groups.append((obj,co,np.array(s['left']),np.array(s['right']),np.array(p['rgb'])))
    x0=np.array(values)
    def decode(x):
        return np.r_[x[:2],0.,0.,0.],np.r_[x[2:4],0.,0.,0.],transform(x[4:7],x[7:10]),transform(x[10:13],x[13:16])
    def residual(x):
        dl,dr,lr,lc=decode(x);out=[]
        for j,(obj,co,uvl,uvr,uvc) in enumerate(groups):
            pose=x[16+6*j:22+6*j];rot=cv2.Rodrigues(pose[:3])[0];t=pose[3:]
            out.append((cv2.projectPoints(obj,pose[:3],t,kl,dl)[0].reshape(-1,2)-uvl).ravel())
            for points,uv,c,k,d in [(obj,uvr,lr,kr,dr),(co,uvc,lc,kc,dc)]:
                pred=cv2.projectPoints(points,cv2.Rodrigues(c[:3,:3]@rot)[0],c[:3,:3]@t+c[:3,3],k,d)[0].reshape(-1,2)
                out.append((pred-uv).ravel())
        return np.concatenate(out)
    pattern=lil_matrix((sum(4*len(g[0])+2*len(g[1]) for g in groups),len(x0)),dtype=int);row=0
    for j,g in enumerate(groups):
        n=4*len(g[0])+2*len(g[1]);pattern[row:row+n,:16]=1;pattern[row:row+n,16+6*j:22+6*j]=1;row+=n
    def rms(e):return float(np.sqrt(np.mean(np.sum(e.reshape(-1,2)**2,axis=1))))
    before=rms(residual(x0));print('Joint initial training pixel RMS',before,flush=True)
    sol=least_squares(residual,x0,jac_sparsity=pattern.tocsr(),x_scale='jac',method='trf',loss='linear',max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-8)
    dl,dr,lr,lc=decode(sol.x)
    if not sol.success:raise ValueError(f'Joint optimization did not converge: {sol.message}')
    for k,d in [(kl,dl),(kr,dr)]:
        xx,yy=np.meshgrid(np.linspace(0,obs['left']['width']-1,30),np.linspace(0,obs['left']['height']-1,20));r2=((xx-k[0,2])/k[0,0])**2+((yy-k[1,2])/k[1,1])**2
        if np.min(1+3*d[0]*r2+5*d[1]*r2*r2)<=0:raise ValueError('Fitted radial model folds within image')
    size=(obs['left']['width'],obs['left']['height']);r1,r2,p1,p2,q,_,_=cv2.stereoRectify(kl,dl,kr,dr,size,lr[:3,:3],lr[:3,3],flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    held=[dict(by_id[i],rgb=by_id[i]['right']) for i in heldout_ids]
    stereo_evaluate=evaluate(held,board,kl,dl,kr,dr,lr)
    color_evaluate=evaluate([rgb_by_id[i] for i in heldout_ids],board,kl,dl,kc,dc,lc)
    epi=[]
    for s in held:
        a=cv2.undistortPoints(np.array(s['left']).reshape(-1,1,2),kl,dl,R=r1,P=p1).reshape(-1,2)
        b=cv2.undistortPoints(np.array(s['right']).reshape(-1,1,2),kr,dr,R=r2,P=p2).reshape(-1,2);epi.extend(abs(a[:,1]-b[:,1]))
    candidate=dict(name='joint_fixed_K_radial2',fit_rms_px=rms(sol.fun),heldout_right_rms_px=stereo_evaluate['rgb_prediction_rms_px'],heldout_epipolar_median_px=float(np.median(epi)),heldout_epipolar_p95_px=float(np.percentile(epi,95)),baseline_m=float(np.linalg.norm(lr[:3,3])),K_left=kl.tolist(),K_right=kr.tolist(),D_left=dl.tolist(),D_right=dr.tolist(),R=lr[:3,:3].tolist(),T=lr[:3,3].tolist(),R1=r1.tolist(),R2=r2.tolist(),P1=p1.tolist(),P2=p2.tolist(),Q=q.tolist())
    return dict(candidate=candidate,color_from_left=lc.tolist(),training_ids=training_ids,heldout_ids=heldout_ids,heldout_stereo=stereo_evaluate,heldout_color=color_evaluate,optimization=dict(success=bool(sol.success),message=sol.message,nfev=sol.nfev,optimality=float(sol.optimality),initial_three_view_rms_px=before,final_three_view_rms_px=rms(sol.fun)),fixed=dict(K_left=kl.tolist(),K_right=kr.tolist(),RGB=rgb_model),installed=False)


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    obs=json.loads((source/'observations.json').read_text());st=json.loads((source/'stereo-result.json').read_text());color=json.loads((source/'color-extrinsics-fit/result.json').read_text());paired=json.loads((source/'color-extrinsics-fit/paired-observations.json').read_text())['samples']
    initial_color=next(v for v in color['variants'] if v['name']=='stereo_radial2')['color_from_left']
    result=optimize(obs,paired,st['candidate'],initial_color,color['RGB_intrinsics_fixed'],st['training_ids'],st['heldout_ids'])
    atomic_json(output/'joint-result.json',result);atomic_json(output/'observations.json',obs)
    candidate=result['candidate'];atomic_json(output/'stereo-result.json',dict(schema_version=1,input_space=st['input_space'],serial=st['serial'],board=st['board'],image_size=st['image_size'],training_ids=st['training_ids'],heldout_ids=st['heldout_ids'],selected=candidate['name'],candidate=candidate,candidates=[candidate],installed=False,independent_height_validation_pending=True))
    color['variants']=[dict(name='stereo_radial2',color_from_left=result['color_from_left'],fitted_heldout=result['heldout_color'])];color['notes'].append('IR k1/k2, both extrinsics and per-training-view board poses jointly optimized; all K, RGB D, world transform and board geometry fixed.')
    atomic_json(output/'color-extrinsics-fit/result.json',color)
    print(json.dumps({'optimizer':result['optimization'],'heldout_stereo':candidate['heldout_right_rms_px'],'heldout_color':result['heldout_color']['rgb_prediction_rms_px']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
