import copy
import cv2
import numpy as np
from dual_arm_calibration.board import Board
from dual_arm_calibration.color_extrinsics import fit


def test_metric_transform_fixed_models_and_holdout_isolation():
    board=Board(dict(squares_x=7,squares_y=5,square_length_m=.04,marker_length_m=.028,dictionary='DICT_5X5_50'))
    ki=np.array([[650.,0,640],[0,650,360],[0,0,1.]])
    kc=np.array([[910.,0,650],[0,908,375],[0,0,1.]])
    di=np.array([.001,-.002,0,0,0]);dc=np.array([.09,-.2,0,0,0])
    c=np.eye(4);c[:3,:3]=cv2.Rodrigues(np.array([.003,-.008,.005]))[0];c[:3,3]=[.015,.0002,-.0003]
    rng=np.random.default_rng(4);samples=[]
    for i in range(30):
        rv=rng.uniform(-.4,.4,3);tv=np.array([rng.uniform(-.15,0),rng.uniform(-.12,0),rng.uniform(.4,1.2)])
        ir=cv2.projectPoints(board.object_points,rv,tv,ki,di)[0].reshape(-1,2)
        rgb=cv2.projectPoints(board.object_points,cv2.Rodrigues(c[:3,:3]@cv2.Rodrigues(rv)[0])[0],c[:3,:3]@tv+c[:3,3],kc,dc)[0].reshape(-1,2)
        samples.append(dict(id=str(i),ids=list(range(len(board.object_points))),left=ir.tolist(),rgb=rgb.tolist()))
    train=[str(i) for i in range(30) if i%5!=4];held=[str(i) for i in range(30) if i%5==4]
    originals=[a.copy() for a in (ki,di,kc,dc)]
    a=fit(samples,board,ki,di,kc,dc,train,held,(1280,720),np.eye(4))
    np.testing.assert_allclose(a['color_from_left'],c,atol=1e-5)
    assert a['fitted_heldout']['rgb_prediction_rms_px']<.01
    changed=copy.deepcopy(samples)
    for s in changed:
        if s['id'] in held:s['rgb']=(np.array(s['rgb'])+[8,0]).tolist()
    b=fit(changed,board,ki,di,kc,dc,train,held,(1280,720),np.eye(4))
    np.testing.assert_allclose(a['color_from_left'],b['color_from_left'],atol=1e-12)
    assert b['fitted_heldout']['rgb_prediction_rms_px']>7
    for value,original in zip((ki,di,kc,dc),originals):np.testing.assert_array_equal(value,original)
