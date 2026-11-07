"""Optional exclusive RealSense IR-pair capture; never moves a robot or writes calibration."""
import argparse
import json
import signal
import shutil
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
from .board import Board
from .storage import atomic_json
from .capture import board_corner_rms


def detect(board, image):
    # Recover rejected marker candidates using known board geometry and marker codes.
    # OpenCV's board-aware refinement is particularly useful for small distant markers.
    params=board.detector.getCharucoParameters()
    if not params.tryRefineMarkers:
        params.tryRefineMarkers=True
        board.detector.setCharucoParameters(params)
    best = (None, None)
    for scale in (1, 2):
        scaled = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        corners, ids, _, _ = board.detector.detectBoard(scaled)
        if ids is not None and (best[1] is None or len(ids) > len(best[1])):
            best = ((corners.reshape(-1, 2) + .5) / scale - .5, ids.reshape(-1))
    return best


class StereoStability:
    """Compare BOTH views against one anchor so slow drift cannot accumulate."""
    def __init__(self, window_s=.25, rms_px=1.5, minimum=12):
        self.window_s=window_s;self.rms_px=rms_px;self.minimum=minimum
        self.anchor=None;self.last_frame=None

    def update(self, now, frame_number, ids, views):
        if self.last_frame is not None and frame_number<=self.last_frame:
            return False, 0., None
        self.last_frame=frame_number
        if len(ids)<self.minimum:
            self.anchor=None
            return False, 0., None
        motion=0.
        if self.anchor is not None:
            try:
                motion=max(board_corner_rms(self.anchor[1],old,ids,new,self.minimum)
                           for old,new in zip(self.anchor[2],views))
            except ValueError:
                self.anchor=None
        if self.anchor is None or motion>self.rms_px:
            self.anchor=(now,np.asarray(ids).copy(),[v.copy() for v in views])
        elapsed=now-self.anchor[0]
        return elapsed>=self.window_s, elapsed, motion


def load_seed(source, cfg):
    """Read a prior batch without changing it; only compatible images can be reused."""
    source=Path(source)
    meta=json.loads((source/'observations.json').read_text())
    if meta['serial']!=cfg['serial'] or meta['board']!=cfg['board']:
        raise ValueError('续采的相机或标定板与原批次不一致')
    for i,rec in enumerate(meta['samples']):
        if rec['id']!=f'{i:03d}':raise ValueError('原批次样本编号不连续')
        for side in ('left','right'):
            if not (source/f'{rec["id"]}_{side}.png').is_file():
                raise ValueError('原批次缺少左右红外原图')
    return meta


def capture(config, out, seed=None):
    import pyrealsense2 as rs
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(config).read_text()); board = Board(cfg['board'])
    atomic_json(out/'config.json', cfg)
    previous=load_seed(seed,cfg) if seed else None
    records = list(previous['samples']) if previous else []
    if previous:
        for rec in records:
            for suffix in ('left.png','right.png','rgb.png','depth.npz'):
                source=Path(seed)/f'{rec["id"]}_{suffix}'
                if source.exists():shutil.copy2(source,out/source.name)
        atomic_json(out/'observations.json',previous)
        atomic_json(out/'continued_from.json',dict(path=str(Path(seed).resolve()),count=len(records)))
    pipe = None; sensor = None; emitter = None; service_stopped = False
    start=None;duration=cfg.get('duration_s',300);stop_reason=None
    stop = False
    def interrupted(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    def status(state, hint, **extra):
        if start is not None:extra['remaining_s']=max(0,round(duration-(time.monotonic()-start)))
        if stop_reason is not None:extra['stop_reason']=stop_reason
        atomic_json(out/'status.json', dict(state=state, hint=hint, count=len(records), target=cfg.get('target_pairs', 30), updated=time.time(), **extra))
    status('starting', '正在准备双目采集')
    try:
        service = cfg.get('camera_service')
        if service:
            current = json.loads(Path(cfg['camera_status_file']).read_text())
            if current.get('recording'):raise ValueError('相机正在录像，请先停止录像')
            if time.time()-current['wall_time'] > 10:raise ValueError('相机状态过期')
            active = subprocess.run(['systemctl','--user','is-active',service],capture_output=True,text=True).stdout.strip()
            if active != 'active':raise ValueError('配置的相机服务当前未运行')
            # Record restoration obligation before requesting stop: even a timed-out stop may take effect.
            service_stopped = True
            subprocess.run(['systemctl','--user','stop',service],check=True,timeout=20)
        rc = rs.config(); rc.enable_device(cfg['serial'])
        for index in (1, 2):rc.enable_stream(rs.stream.infrared,index,1280,720,rs.format.y8,15)
        rc.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,6)
        rc.enable_stream(rs.stream.depth,1280,720,rs.format.z16,15)
        pipe = rs.pipeline(); profile = pipe.start(rc)
        sensor = profile.get_device().first_depth_sensor()
        emitter = sensor.get_option(rs.option.emitter_enabled); sensor.set_option(rs.option.emitter_enabled,0)
        def intr(p):
            i=p.as_video_stream_profile().intrinsics
            return dict(width=i.width,height=i.height,fx=i.fx,fy=i.fy,ppx=i.ppx,ppy=i.ppy,coeffs=list(i.coeffs))
        lp=profile.get_stream(rs.stream.infrared,1); rp=profile.get_stream(rs.stream.infrared,2)
        e=lp.get_extrinsics_to(rp)
        meta=dict(serial=cfg['serial'],board=cfg['board'],left=intr(lp),right=intr(rp),color=intr(profile.get_stream(rs.stream.color)),right_from_left=dict(rotation=np.array(e.rotation).reshape(3,3).T.tolist(),translation=list(e.translation)),input_space='SDK rectified infrared',samples=records)
        if previous:
            for key in ('left','right','color','right_from_left','input_space'):
                if meta[key]!=previous[key]:raise ValueError(f'相机参数 {key} 已变化，不能混合续采')
        atomic_json(out/'observations.json',meta)
        stability=StereoStability(cfg.get('stable_s',.25),cfg.get('stable_rms_px',1.5),cfg.get('min_corners',12))
        last_saved = None; start=time.monotonic()
        while not stop and len(records)<cfg.get('target_pairs',30) and time.monotonic()-start<duration:
            frames=pipe.wait_for_frames(5000); l=frames.get_infrared_frame(1); r=frames.get_infrared_frame(2)
            if not l or not r:continue
            if l.get_frame_number()!=r.get_frame_number() or abs(l.get_timestamp()-r.get_timestamp())>1:continue
            ims=[np.asanyarray(f.get_data()).copy() for f in (l,r)]
            detections=[detect(board,im) for im in ims]
            common = np.intersect1d(detections[0][1],detections[1][1]) if all(d[1] is not None for d in detections) else np.array([],int)
            views=[cv2.cvtColor(im,cv2.COLOR_GRAY2BGR) for im in ims]
            for view,(corners,ids) in zip(views,detections):
                if ids is not None:cv2.aruco.drawDetectedCornersCharuco(view,corners.astype(np.float32).reshape(-1,1,2),ids.reshape(-1,1))
            preview=cv2.resize(np.hstack(views),(1280,360));cv2.imwrite(str(out/'preview.tmp.jpg'),preview);(out/'preview.tmp.jpg').replace(out/'preview.jpg')
            now=time.monotonic()
            if len(common)<cfg.get('min_corners',12):
                stability.update(now,l.get_frame_number(),common,[])
                status('capturing','请让左右画面都看清完整图案',corners=int(len(common)),rejection='board_not_detected');continue
            xy=[np.array([c[np.where(ids==i)[0][0]] for i in common]) for c,ids in detections]
            left_intr=meta['left']
            k=np.array([[left_intr['fx'],0,left_intr['ppx']],[0,left_intr['fy'],left_intr['ppy']],[0,0,1.]])
            pose_ok,_,position=cv2.solvePnP(board.object_points[common],xy[0],k,np.array(left_intr['coeffs']))
            distance_m=float(position[2,0]) if pose_ok else None
            settled,stable_elapsed,motion=stability.update(now,l.get_frame_number(),common,xy)
            distinct=True
            for rec in records:
                shared=np.intersect1d(common,rec['ids'])
                if len(shared)>=cfg.get('min_corners',12):
                    old=np.array(rec['left']);ids=np.array(rec['ids'])
                    delta=np.array([xy[0][np.where(common==i)[0][0]]-old[np.where(ids==i)[0][0]] for i in shared])
                    if np.median(np.linalg.norm(delta,axis=1))<cfg.get('novelty_px',25.):distinct=False;break
            ready=settled and distinct and (last_saved is None or now-last_saved>=2.)
            status('capturing','停稳后自动保存' if distinct else '此姿态已采集，请改变位置、距离或倾角',corners=int(len(common)),board_distance_m=distance_m,stable_s=round(stable_elapsed,2),required_stable_s=stability.window_s,motion_rms_px=motion,max_motion_rms_px=stability.rms_px,rejection='waiting_stable' if not settled else 'duplicate_pose' if not distinct else None)
            if not ready:continue
            name=f'{len(records):03d}'
            for side,im in zip(('left','right'),ims):cv2.imwrite(str(out/f'{name}_{side}.png'),im)
            color=frames.get_color_frame()
            if color:cv2.imwrite(str(out/f'{name}_rgb.png'),np.asanyarray(color.get_data()))
            depth=frames.get_depth_frame()
            if depth:np.savez_compressed(out/f'{name}_depth.npz',depth_m=np.asanyarray(depth.get_data())*sensor.get_depth_scale())
            records.append(dict(id=name,ids=common.tolist(),left=xy[0].tolist(),right=xy[1].tolist(),board_distance_m=distance_m,frame_number=l.get_frame_number(),timestamp_ms=l.get_timestamp(),right_timestamp_ms=r.get_timestamp(),wall_time=time.time(),stable_s=stable_elapsed,motion_rms_px=motion))
            atomic_json(out/'observations.json',meta);last_saved=now
            status('capturing',f'已保存第 {len(records)} 组，请换姿态',corners=int(len(common)))
            print(f'Saved {name}: {len(common)} common corners',flush=True)
        stop_reason='target_reached' if len(records)>=cfg.get('target_pairs',30) else 'user_stop' if stop else 'timeout'
        status('captured',f'已保存 {len(records)} 组；相机服务正在恢复')
    except Exception as exc:
        status('error',str(exc));raise
    finally:
        errors=[]
        if sensor is not None and emitter is not None:
            try:sensor.set_option(rs.option.emitter_enabled,emitter)
            except Exception as exc:errors.append(str(exc))
        if pipe is not None:
            try:pipe.stop()
            except Exception as exc:errors.append(str(exc))
        if service_stopped:
            try:subprocess.run(['systemctl','--user','start',cfg['camera_service']],check=True,timeout=20)
            except Exception as exc:errors.append(str(exc))
        ready = None
        if service_stopped and not errors:
            for _ in range(30):
                try:
                    ready = bool(json.loads(Path(cfg['camera_status_file']).read_text()).get('ready'))
                    if ready:break
                except (OSError, ValueError):pass
                time.sleep(.5)
            if not ready:errors.append('相机服务重启后未就绪')
        atomic_json(out/'restoration.json',dict(requested_service_restart=service_stopped,service_ready=ready,errors=errors,flash_written=False))
        if errors:status('error','恢复相机失败：'+'; '.join(errors))
        elif json.loads((out/'status.json').read_text())['state'] != 'error':
            why={'target_reached':'目标组数已采齐','timeout':f'本轮 {duration:g} 秒时限已到','user_stop':'采集已停止'}.get(stop_reason,'采集已结束')
            next_step='可拟合并验证' if len(records)>=cfg.get('target_pairs',30) else '请点击继续采集补齐'
            status('captured',f'{why}；已保存 {len(records)} 组，相机已恢复；{next_step}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--seed');a=p.parse_args();capture(a.config,a.output,a.seed)
