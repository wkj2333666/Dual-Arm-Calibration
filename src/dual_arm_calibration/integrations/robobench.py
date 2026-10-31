"""HTTP adapter to RoboBench's single hardware owner process."""

import io
import json
import urllib.error
import urllib.request
import zipfile

import cv2
import numpy as np


class RoboBench:
    def __init__(self, config):
        self.config = config
        self.url = config["base_url"].rstrip("/")

    def request(self, path, payload=None, timeout=12):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url + path, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            try:
                reason = json.loads(exc.read()).get("detail", str(exc))
            except ValueError:
                reason = str(exc)
            raise ValueError(str(reason)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError("无法连接 Piper 设备桥接服务：" + str(exc)) from exc

    def probe(self):
        result = json.loads(self.request("/api/dual-arm-calibration/status", timeout=5))
        expected = self.config["identities"]
        for a, serial in expected["arms"].items():
            if result["arms"][a]["serial"] != serial:
                raise ValueError(a + " 机械臂身份与接入配置不一致")
        for r, serial in expected["cameras"].items():
            if result["cameras"][r]["serial"] != serial:
                raise ValueError(r + " 相机身份与接入配置不一致")
        return result

    def preview(self, role):
        if role not in ("left", "right", "top"):
            raise ValueError("未知相机")
        return self.request(f"/api/cameras/{role}/snapshot", timeout=5)

    def capture(self, roles):
        payload = self.request("/api/dual-arm-calibration/capture", dict(roles=roles))
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            result = json.loads(z.read("metadata.json"))
            result["images"] = {
                f"{i}-{r}": cv2.imdecode(
                    np.frombuffer(z.read(f"{i}-{r}.png"), np.uint8), cv2.IMREAD_COLOR
                )
                for i in (0, 1)
                for r in roles
            }
        return result
