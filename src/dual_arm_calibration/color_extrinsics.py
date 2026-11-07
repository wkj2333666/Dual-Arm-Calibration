"""Offline IR-to-color fit with fixed camera models and unchanged holdout IDs."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .board import Board
from .stereo import matrix
from .stereo_capture import detect
from .storage import atomic_json


def evaluate(samples, board, ki, di, kc, dc, transform):
    errors=[];reverse_errors=[];views=[]
    for s in samples:
        obj=board.object_points[np.array(s['ids'],int)]
        a=np.array(s['left'],float);b=np.array(s['rgb'],float)
        ok,rv,tv=cv2.solvePnP(obj,a,ki,di)
        if not ok:raise ValueError('IR PnP failed')
        r=transform[:3,:3];t=transform[:3,3:4]
        pred=cv2.projectPoints(obj,cv2.Rodrigues(r@cv2.Rodrigues(rv)[0])[0],r@tv+t,kc,dc)[0].reshape(-1,2)
        e=np.linalg.norm(pred-b,axis=1);errors.extend(e)
        ok,rv,tv=cv2.solvePnP(obj,b,kc,dc)
        if not ok:raise ValueError('RGB PnP failed')
        pred=cv2.projectPoints(obj,cv2.Rodrigues(r.T@cv2.Rodrigues(rv)[0])[0],r.T@(tv-t),ki,di)[0].reshape(-1,2)
        re=np.linalg.norm(pred-a,axis=1);reverse_errors.extend(re)
        views.append(dict(id=s['id'],corners=len(obj),rgb_prediction_rms_px=float(np.sqrt(np.mean(e**2))),ir_prediction_rms_px=float(np.sqrt(np.mean(re**2)))))
    return dict(rgb_prediction_rms_px=float(np.sqrt(np.mean(np.square(errors)))),ir_prediction_rms_px=float(np.sqrt(np.mean(np.square(reverse_errors)))),views=views)


def fit(samples, board, ki, di, kc, dc, training_ids, heldout_ids, size, factory):
    train=[s for s in samples if s['id'] in training_ids]
    held=[s for s in samples if s['id'] in heldout_ids]
    if len(train)<20 or len(held)<5:raise ValueError('Need at least 20 training and 5 heldout paired views')
    obj=[board.object_points[np.array(s['ids'],int)].astype(np.float32) for s in train]
    a=[np.array(s['left'],np.float32).reshape(-1,1,2) for s in train]
    b=[np.array(s['rgb'],np.float32).reshape(-1,1,2) for s in train]
    err,_,_,_,_,r,t,_,_=cv2.stereoCalibrate(obj,a,b,ki.copy(),di.copy(),kc.copy(),dc.copy(),size,flags=cv2.CALIB_FIX_INTRINSIC,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,1e-8))
    c=np.eye(4);c[:3,:3]=r;c[:3,3]=t.ravel()
    return dict(color_from_left=c.tolist(),fit_rms_px=float(err),training_ids=[s['id'] for s in train],heldout_ids=[s['id'] for s in held],factory_heldout=evaluate(held,board,ki,di,kc,dc,factory),fitted_heldout=evaluate(held,board,ki,di,kc,dc,c),translation_change_mm=float(np.linalg.norm(c[:3,3]-factory[:3,3])*1000),rotation_change_deg=float(np.linalg.norm(cv2.Rodrigues(c[:3,:3]@factory[:3,:3].T)[0])*180/np.pi))


def run(session, reference, output):
    output.mkdir(parents=True,exist_ok=False)
    obs=json.loads((session/'observations.json').read_text())
    stereo=json.loads((session/'stereo-result.json').read_text())
    ref=json.loads(reference.read_text())
    if obs['serial']!=ref['serial']:raise ValueError('Camera serial mismatch')
    # Fixed RGB calibration is the same model used for the trusted world transform.
    rgb=ref['calibration']['cameras']['head']['camera_model']['intrinsics']
    for key in ('width','height','fx','fy','ppx','ppy'):
        if obs['color'][key]!=rgb[key]:raise ValueError(f'RGB profile differs: {key}')
    kc=matrix(rgb);dc=np.array(rgb['coeffs'],float);factory=np.array(ref['color_from_left'])
    board=Board(obs['board']);samples=[];excluded=[]
    for s in obs['samples']:
        path=session/f'{s["id"]}_rgb.png';image=cv2.imread(str(path))
        if image is None:excluded.append(dict(id=s['id'],reason='missing RGB'));continue
        corners,ids=detect(board,image)
        common=np.intersect1d(s['ids'],ids) if ids is not None else np.array([],int)
        if len(common)<12:excluded.append(dict(id=s['id'],reason='fewer than 12 shared corners',count=len(common)));continue
        ir_ids=np.array(s['ids']);ir_xy=np.array(s['left'])
        samples.append(dict(id=s['id'],ids=common.tolist(),left=[ir_xy[np.where(ir_ids==i)[0][0]].tolist() for i in common],rgb=[corners[np.where(ids==i)[0][0]].tolist() for i in common]))
    atomic_json(output/'paired-observations.json',dict(samples=samples,excluded=excluded))
    selected=stereo['candidate'];ki=np.array(selected['K_left']);di=np.array(selected['D_left'])
    models=[('stereo_radial2',ki,di),('same_K_zero_IR_distortion',ki,np.zeros_like(di)),('factory_IR_intrinsics',matrix(obs['left']),np.array(obs['left']['coeffs']))]
    variants=[]
    for name,k,d in models:
        fitted=fit(samples,board,k,d,kc,dc,stereo['training_ids'],stereo['heldout_ids'],(obs['left']['width'],obs['left']['height']),factory)
        variants.append(dict(name=name,K_left=k.tolist(),D_left=d.tolist(),**fitted))
        print(name,'fit',fitted['fit_rms_px'],'heldout RGB',fitted['factory_heldout']['rgb_prediction_rms_px'],'->',fitted['fitted_heldout']['rgb_prediction_rms_px'],'delta mm',fitted['translation_change_mm'],flush=True)
    result=dict(serial=obs['serial'],source_session=str(session),reference_metadata=str(reference),RGB_intrinsics_fixed=rgb,world_from_RGB_fixed=ref['calibration']['cameras']['head']['world_from_camera'],factory_color_from_left=factory.tolist(),variants=variants,excluded=excluded,installed=False,notes=['Original stereo training/holdout split preserved.','RGB and IR are not hardware synchronized. Saved RGB frame timestamps were not recorded; capture used stationary board gating.','Zero-IR-distortion variant is a sensitivity control with identical K, not an independently refitted full stereo model.','SDK Y8 IR is already rectified; fitted distortion describes residual correction only.','No table height or independent static height data participates in fitting.'])
    atomic_json(output/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.session,a.reference,a.output)
