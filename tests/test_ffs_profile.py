import cv2
import numpy as np
import pytest
from dual_arm_calibration.ffs_profile import FFSCalibration


def profile():
    k=np.array([[600.,0,319.5],[0,600,239.5],[0,0,1.]])
    p1=np.column_stack([k,np.zeros(3)]);p2=p1.copy();p2[0,3]=-30;p2[0,2]-=2
    r=cv2.Rodrigues(np.array([0,.04,0]))[0]
    s=dict(K_left=k.tolist(),K_right=k.tolist(),D_left=[0]*5,D_right=[0]*5,R1=r.tolist(),R2=r.tolist(),P1=p1.tolist(),P2=p2.tolist())
    w=np.eye(4);w[1,1]=-1;w[2,2]=-1;w[2,3]=1.2
    return dict(id='synthetic',serial='test',image_size=[640,480],stereo=s,color_from_left=np.eye(4).tolist(),world_from_rgb=w.tolist(),table_z_m=.01)


def test_scale_principal_offset_inverse_rotation_and_world_height():
    p=profile();c=FFSCalibration(p)
    # At half size, 1 m rectified Z corresponds to effective disparity 15 px,
    # plus the 1 px difference of the scaled principal points.
    disp=np.full((240,320),16.,np.float32);v=c.reconstruct(disp)
    np.testing.assert_allclose(v['depth_m'],1.)
    u,vv=160,120;k=np.array(p['stereo']['P1'])[:,:3];ray=np.array([((u+.5)/.5-.5-k[0,2])/k[0,0],((vv+.5)/.5-.5-k[1,2])/k[1,1],1.])
    expected=np.array(p['stereo']['R1']).T@ray
    np.testing.assert_allclose(v['xyz_left_m'][vv,u],expected,atol=1e-7)
    np.testing.assert_allclose(v['height_above_table_m'][vv,u],1.2-expected[2]-.01,atol=1e-7)
    disp[0,0]=np.nan;disp[0,1]=0
    v=c.reconstruct(disp);assert not v['valid'][0,:2].any();assert np.isnan(v['height_above_table_m'][0,:2]).all()


def test_rejects_wrong_camera_or_image_size():
    c=FFSCalibration(profile());im=np.zeros((480,640,3),np.uint8)
    with pytest.raises(ValueError,match='serial'):c.rectify(im,im,'wrong')
    with pytest.raises(ValueError,match='resolution'):c.rectify(im[:240],im[:240],'test')
    assert c.rectify(im,im,'test')[0].shape==im.shape
