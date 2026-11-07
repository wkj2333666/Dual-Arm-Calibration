import numpy as np
import json
import pytest
from dual_arm_calibration.stereo_capture import StereoStability
from dual_arm_calibration.stereo_capture import load_seed


def test_accepts_whole_board_jitter_without_worst_corner_reset():
    ids=np.arange(24);points=np.arange(48,dtype=float).reshape(-1,2)
    gate=StereoStability()
    assert not gate.update(0.,1,ids,[points,points])[0]
    noisy=points.copy();noisy[0,0]+=2.
    ready,_,motion=gate.update(.3,2,ids,[noisy,noisy])
    assert ready and motion<1.5


def test_checks_right_view_and_rejects_slow_accumulated_motion():
    ids=np.arange(24);p=np.zeros((24,2));gate=StereoStability()
    gate.update(0.,1,ids,[p,p])
    assert not gate.update(.1,2,ids,[p,p+[1.,0]])[0]
    assert not gate.update(.3,3,ids,[p,p+[2.,0]])[0]
    assert gate.update(.6,4,ids,[p,p+[2.,0]])[0]


def test_dropped_detection_and_duplicate_frames_do_not_count_as_stability():
    ids=np.arange(24);p=np.zeros((24,2));gate=StereoStability()
    gate.update(0.,1,ids,[p,p])
    assert not gate.update(.3,1,ids,[p,p])[0]
    gate.update(.4,2,np.array([],int),[])
    assert not gate.update(.8,3,ids,[p,p])[0]
    assert gate.update(1.1,4,ids,[p,p])[0]


def test_resume_preserves_original_and_rejects_mismatched_or_missing_inputs(tmp_path):
    cfg={'serial':'test-camera','board':{'squares_x':7}}
    meta=dict(cfg,samples=[{'id':'000','ids':[1,2,3]}])
    original=json.dumps(meta)
    (tmp_path/'observations.json').write_text(original)
    for side in ('left','right'):(tmp_path/f'000_{side}.png').write_bytes(b'original')
    assert load_seed(tmp_path,cfg)==meta
    assert (tmp_path/'observations.json').read_text()==original
    with pytest.raises(ValueError,match='不一致'):
        load_seed(tmp_path,dict(cfg,serial='another-camera'))
    (tmp_path/'000_right.png').unlink()
    with pytest.raises(ValueError,match='缺少'):
        load_seed(tmp_path,cfg)
