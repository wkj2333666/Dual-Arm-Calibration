"""Submit full SDK IR pairs using the FFS service's active serial-bound calibration."""
import argparse
import json
from pathlib import Path
import urllib.request
import uuid


def submit(url,left,right,output,serial,*,width=960,calibration='active',fx=None,baseline=None):
    boundary=uuid.uuid4().hex;body=bytearray();fields={'width':width}
    if calibration is not None:fields.update(calibration=calibration,camera_serial=serial)
    if fx is not None:fields['fx_px']=fx
    if baseline is not None:fields['baseline_m']=baseline
    for key,value in fields.items():body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for key,path in [('left',left),('right',right)]:
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{key}.png"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode());body.extend(Path(path).read_bytes());body.extend(b'\r\n')
    body.extend(f'--{boundary}--\r\n'.encode());req=urllib.request.Request(url.rstrip('/')+'/infer',data=body,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req,timeout=120) as response:data=response.read();seconds=response.headers.get('X-Inference-Seconds')
    Path(output).write_bytes(data)
    return dict(output=str(output),bytes=len(data),inference_s=float(seconds))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18080');p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--serial',required=True);p.add_argument('--output',required=True);p.add_argument('--width',type=int,choices=[640,960,1280],default=960);a=p.parse_args()
    print(json.dumps(submit(a.url,a.left,a.right,a.output,a.serial,width=a.width)))
