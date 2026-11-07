import copy
import cv2
import numpy as np
import pytest
from dual_arm_calibration.board import Board
from dual_arm_calibration.stereo import solve


def synthetic():
    spec=dict(squares_x=7,squares_y=5,square_length_m=.04,marker_length_m=.028,dictionary='DICT_5X5_50')
    obj=Board(spec).object_points
    k=np.array([[650.,0,640],[0,650,360],[0,0,1.]])
    intr=dict(width=1280,height=720,fx=650.,fy=650.,ppx=640.,ppy=360.,coeffs=[0.]*5)
    rng=np.random.default_rng(12);samples=[]
    for i in range(30):
        rv=rng.uniform(-.45,.45,3);t=np.array([rng.uniform(-.2,.05),rng.uniform(-.15,.03),rng.uniform(.45,.95)])
        a=cv2.projectPoints(obj,rv,t,k,np.zeros(5))[0].reshape(-1,2)
        b=cv2.projectPoints(obj,rv,t+[-.05,0,0],k,np.zeros(5))[0].reshape(-1,2)
        samples.append(dict(id=str(i),ids=list(range(len(obj))),left=a.tolist(),right=b.tolist()))
    return dict(serial='synthetic',board=spec,left=intr,right=intr,samples=samples,input_space='synthetic rectified')


def test_recovers_metric_baseline_and_keeps_holdout_out_of_fit():
    data=synthetic();a=solve(data)
    assert abs(a['candidate']['baseline_m']-.05)<1e-4
    assert a['candidate']['heldout_right_rms_px']<.01
    changed=copy.deepcopy(data)
    for i in range(4,30,5):
        changed['samples'][i]['right']=(np.array(changed['samples'][i]['right'])+[0,5]).tolist()
    b=solve(changed)
    for x,y in zip(a['candidates'],b['candidates']):
        np.testing.assert_allclose(x['T'],y['T'],atol=1e-9)
        assert y['heldout_right_rms_px']>4
    assert set(a['training_ids']).isdisjoint(a['heldout_ids'])
    assert not a['installed'] and a['independent_height_validation_pending']


def test_rejects_insufficient_or_duplicated_capture():
    data=synthetic()
    with pytest.raises(ValueError):solve(dict(data,samples=data['samples'][:20]))
    data['samples'][1]['left']=data['samples'][0]['left']
    with pytest.raises(ValueError,match='重复照片'):solve(data)
