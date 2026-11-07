import copy
import cv2
import numpy as np
from dual_arm_calibration.board import Board
from dual_arm_calibration.joint_stereo import optimize, transform
from dual_arm_calibration.stereo import matrix


def test_joint_recovers_metric_geometry_and_does_not_fit_heldout_or_K():
    spec=dict(squares_x=7,squares_y=5,square_length_m=.04,marker_length_m=.028,dictionary='DICT_5X5_50')
    board=Board(spec);obj=board.object_points
    ir=dict(width=1280,height=720,fx=650.,fy=650.,ppx=640.,ppy=360.,coeffs=[0.]*5)
    rgb=dict(ir,fx=910.,fy=908.,ppx=650.,ppy=375.,coeffs=[.09,-.2,0,0,0])
    k=matrix(ir);kc=matrix(rgb);dl=np.array([-.004,.002,0,0,0]);dr=np.array([-.007,.004,0,0,0])
    lr=transform([.0003,-.0005,.0002],[-.05,.0001,-.0002]);lc=transform([.003,-.008,.005],[.015,.0002,.002])
    rng=np.random.default_rng(24);samples=[];paired=[]
    for i in range(24):
        rv=rng.uniform(-.5,.5,3);rot=cv2.Rodrigues(rv)[0];t=np.array([rng.uniform(-.2,.05),rng.uniform(-.16,.04),rng.uniform(.35,1.25)])
        uv=cv2.projectPoints(obj,rv,t,k,dl)[0].reshape(-1,2)
        right=cv2.projectPoints(obj,cv2.Rodrigues(lr[:3,:3]@rot)[0],lr[:3,:3]@t+lr[:3,3],k,dr)[0].reshape(-1,2)
        color=cv2.projectPoints(obj,cv2.Rodrigues(lc[:3,:3]@rot)[0],lc[:3,:3]@t+lc[:3,3],kc,np.array(rgb['coeffs']))[0].reshape(-1,2)
        sample=dict(id=str(i),ids=list(range(len(obj))),left=uv.tolist(),right=right.tolist());samples.append(sample);paired.append(dict(sample,rgb=color.tolist()))
    obs=dict(board=spec,left=ir,right=ir,samples=samples)
    initial=dict(K_left=k.tolist(),K_right=k.tolist(),D_left=[0.]*5,D_right=[0.]*5,R=lr[:3,:3].tolist(),T=(lr[:3,3]+[.0003,0,.0002]).tolist())
    c0=lc.copy();c0[:3,3]+=[0,.0002,-.0003]
    train=[str(i) for i in range(24) if i%4!=3];held=[str(i) for i in range(24) if i%4==3]
    result=optimize(obs,paired,initial,c0,rgb,train,held);candidate=result['candidate']
    np.testing.assert_array_equal(candidate['K_left'],k);np.testing.assert_array_equal(candidate['K_right'],k)
    np.testing.assert_allclose(candidate['T'],lr[:3,3],atol=1e-5)
    np.testing.assert_allclose(result['color_from_left'],lc,atol=1e-5)
    assert candidate['heldout_right_rms_px']<.01
    assert result['fixed']['RGB']==rgb
    changed=copy.deepcopy(obs);cp=copy.deepcopy(paired)
    for s in changed['samples']:
        if s['id'] in held:s['right']=(np.array(s['right'])+[0,5]).tolist()
    for s in cp:
        if s['id'] in held:s['rgb']=(np.array(s['rgb'])+[8,0]).tolist()
    other=optimize(changed,cp,initial,c0,rgb,train,held)
    np.testing.assert_array_equal(other['candidate']['T'],candidate['T'])
    np.testing.assert_array_equal(other['color_from_left'],result['color_from_left'])
    assert other['candidate']['heldout_right_rms_px']>4
    assert other['heldout_color']['rgb_prediction_rms_px']>7
