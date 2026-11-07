"""Opt-in depth calibration process owner; site configuration is never a package default."""
import json
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from .storage import atomic_json

class DepthWorkbench:
    def __init__(self, directory, config):
        self.root=Path(directory)/'depth';self.root.mkdir(parents=True,exist_ok=True)
        self.config=Path(config).resolve();cfg=json.loads(self.config.read_text())
        if not cfg.get('serial') or not cfg.get('board'):raise ValueError('深度采集需要现场序列号和标定板规格')
        self.process=None;self.log=None;self.lock=threading.Lock();self.current=None
        existing=sorted(self.root.glob('*/config.json'),key=lambda p:p.stat().st_mtime)
        if existing:self.current=existing[-1].parent
    def state(self):
        with self.lock:
            active=self.process is not None and self.process.poll() is None
            s=dict(configured=True,active=active,id=self.current.name if self.current else None)
            if self.current:
                for name,key in [('status.json','capture'),('solve-status.json','solve'),('stereo-result.json','result'),('restoration.json','restoration')]:
                    p=self.current/name
                    if p.exists():s[key]=json.loads(p.read_text())
                if not active and s.get('capture',{}).get('state') in ('starting','capturing'):s['interrupted']=True
            p=self.root/'comparison.json'
            if p.exists():s['comparison']=json.loads(p.read_text())
            return s
    def action(self, request):
        with self.lock:
            op=request.get('op');active=self.process is not None and self.process.poll() is None
            if op=='stop':
                if active:self.process.send_signal(signal.SIGINT)
                return dict(stopping=active)
            if active:raise ValueError('已有深度采集或求解正在运行')
            if op in ('start','resume'):
                previous=self.current
                if op=='resume' and (previous is None or not (previous/'observations.json').exists()):
                    raise ValueError('没有可继续采集的批次')
                self.current=self.root/uuid.uuid4().hex
                argv=[sys.executable,'-m','dual_arm_calibration.stereo_capture','--config',str(self.config),'--output',str(self.current)]
                if op=='resume':argv+=['--seed',str(previous)]
            elif op=='solve' and self.current:
                atomic_json(self.current/'solve-status.json',dict(state='solving',hint='正在拟合并留出验证'))
                argv=[sys.executable,'-m','dual_arm_calibration.stereo',str(self.current)]
            else:raise ValueError('未知操作或尚未采集')
            if self.log:self.log.close()
            self.log=(self.root/'worker.log').open('ab')
            self.process=subprocess.Popen(argv,stdout=self.log,stderr=self.log)
            return dict(id=self.current.name)
    def preview(self):
        if self.current and (self.current/'preview.jpg').exists():return (self.current/'preview.jpg').read_bytes()
        raise ValueError('采集开始后显示双目画面')
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        if self.log:self.log.close()
