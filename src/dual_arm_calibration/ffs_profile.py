"""Apply a serial-bound calibration to SDK Y8 IR and FFS disparity (no robot access)."""
import json
from pathlib import Path
import cv2
import numpy as np


class FFSCalibration:
    def __init__(self, profile):
        self.profile=profile;self.id=profile['id'];self.serial=profile['serial']
        self.size=tuple(profile['image_size']);s=profile['stereo']
        self.r1=np.array(s['R1'],float);self.p1=np.array(s['P1'],float);self.p2=np.array(s['P2'],float)
        self.color_from_left=np.array(profile['color_from_left'],float)
        self.world_from_left=np.array(profile['world_from_rgb'],float)@self.color_from_left
        self.table_z=float(profile['table_z_m']);self.maps=[]
        for side,r,p in [('left',s['R1'],s['P1']),('right',s['R2'],s['P2'])]:
            k=np.array(s['K_'+side],float);d=np.array(s['D_'+side],float)
            self.maps.append(cv2.initUndistortRectifyMap(k,d,np.array(r),np.array(p),self.size,cv2.CV_32FC1))
        if not np.isfinite(self.world_from_left).all() or self.p1[0,0]<=0 or self.p2[0,3]>=0:
            raise ValueError('Invalid calibrated transform or stereo geometry')

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

    def rectify(self,left,right,serial):
        if serial!=self.serial:raise ValueError('Camera serial does not match active calibration')
        if left.shape!=right.shape or left.shape[:2]!=(self.size[1],self.size[0]):
            raise ValueError('Expected full SDK IR images at calibrated input resolution')
        return tuple(cv2.remap(im,*mapping,cv2.INTER_LINEAR) for im,mapping in zip((left,right),self.maps))

    def reconstruct(self,disp):
        h,w=disp.shape;sx=w/self.size[0];sy=h/self.size[1]
        corrected=disp-(self.p1[0,2]-self.p2[0,2])*sx
        valid=np.isfinite(corrected)&(corrected>0)
        z=np.divide(-self.p2[0,3]*sx,corrected,out=np.full(disp.shape,np.nan,np.float32),where=valid)
        yy,xx=np.indices(disp.shape)
        cx=(self.p1[0,2]+.5)*sx-.5;cy=(self.p1[1,2]+.5)*sy-.5
        rays=np.stack([(xx-cx)/(self.p1[0,0]*sx),(yy-cy)/(self.p1[1,1]*sy),np.ones_like(z)],-1)
        xyz=(rays*z[:,:,None])@self.r1
        height=xyz@self.world_from_left[2,:3]+self.world_from_left[2,3]-self.table_z
        return dict(disparity_px=disp,valid=valid,depth_m=z,xyz_left_m=xyz.astype(np.float32),height_above_table_m=height.astype(np.float32))

    def metadata(self):
        return dict(calibration_id=self.id,camera_serial=self.serial,input_space='SDK rectified IR, prior to residual calibration',depth_frame='new_rectified_left_optical',xyz_frame='original_left_IR_optical',height_frame='existing_world_above_table',units='meters',valid_mask_meaning='finite positive disparity only; not confidence or occlusion',table_z_m=self.table_z,height_offset_fitted=False)
