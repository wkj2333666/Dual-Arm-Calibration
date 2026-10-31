"""Read/capture and reviewed-motion bridge in the existing hardware owner.

Device streams reuse RoboBench's connections. Confirmed automatic trajectories
delegate to its bounded Piper executor and shared operation coordinator.
"""

import io
import json
import time
import asyncio
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np


def normalize_intrinsics(meta):
    raw = meta["intrinsics"]
    coeffs = [float(v) for v in raw.get("coeffs", [0] * 5)]
    model = raw.get("model", 0)
    if not any(coeffs):
        model = "none"
    elif model in (4, "brown_conrady", "opencv_brown_conrady"):
        model = "opencv_brown_conrady"
    else:
        raise ValueError("非零 RealSense 畸变模型需要显式转换，禁止直接套用 Brown 参数")
    return dict(
        width=int(meta.get("width", raw["width"])),
        height=int(meta.get("height", raw["height"])),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        ppx=float(raw.get("ppx", raw.get("cx"))),
        ppy=float(raw.get("ppy", raw.get("cy"))),
        model=model,
        coeffs=coeffs,
    )


def proxy_host(headers):
    """Honor browser same-origin evidence when the local gateway rewrites Host."""
    host = headers.get("host", "")
    origin = headers.get("origin")
    if origin and urlsplit(origin).netloc != host:
        if headers.get("sec-fetch-site") != "same-origin":
            raise ValueError("不接受跨站请求")
        host = urlsplit(origin).netloc
    return host


def install(app):
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import Response
    from server.calibration.hand_eye.geometry import pose_to_matrix
    from server.calibration.hand_eye.service import HandEyeService
    from server.core.config import ROOT
    from server.motion.cartesian import OrbitKinematics

    router = APIRouter(tags=["dual-arm-calibration"])
    kinematics = OrbitKinematics(ROOT)

    def service(state):
        return HandEyeService(state.store, lambda: state.devices, state.operations)

    def arm_state(svc, side):
        raw = svc._arm("wrist-" + side)
        measured = pose_to_matrix(raw["pose"])
        q = np.deg2rad(raw["joints"])
        fk = kinematics.flange.forward_kinematics([0, *q])
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip((np.trace(fk[:3, :3].T @ measured[:3, :3]) - 1) / 2, -1, 1)
                )
            )
        )
        mm = float(np.linalg.norm(fk[:3, 3] - measured[:3, 3]) * 1000)
        if mm > 3 or angle > 1.15:
            raise ValueError(
                f"{side}: 实测法兰与模型 FK 不一致 ({mm:.2f} mm / {angle:.2f}°)"
            )
        return dict(
            base_from_flange=measured.tolist(),
            joint_angles_rad=q.tolist(),
            joints_deg=raw["joints"],
            pose_mm_deg=raw["pose"],
            serial=raw.get("adapter_serial") or raw["channel"],
            channel=raw["channel"],
            fk_translation_mm=mm,
            fk_rotation_deg=angle,
        )

    def metadata(camera):
        image, meta = camera.capture_calibration()
        if not camera.source or not camera.source.startswith("realsense:"):
            raise ValueError("Piper 接入只接受实际 RealSense 图像")
        if not 0 <= time.time() - meta["timestamp"] <= 0.4:
            raise ValueError("RGB 帧已过期")
        # Native hub names RGB frame numbers explicitly.
        number = meta.get("frame_number", meta.get("rgb_frame_number"))
        if number is None:
            raise ValueError("相机缺少可追踪帧号")
        return image, dict(
            serial=camera.source.split(":", 1)[1],
            image_rotation_deg=0,
            frame_number=int(number),
            timestamp=float(meta["timestamp"]),
            intrinsics=normalize_intrinsics(meta),
        )

    from .robobench_motion import CalibrationMotion

    motion = CalibrationMotion(ROOT, kinematics, service, metadata, arm_state)

    @router.post("/api/dual-arm-calibration/motion/estimate")
    def estimate_pose(request: Request, payload: dict):
        try:
            return motion.estimate(request.app.state, payload)
        except (ValueError, KeyError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/api/dual-arm-calibration/motion/plan")
    def plan_capture(request: Request, payload: dict):
        try:
            return motion.plan(request.app.state, payload)
        except (ValueError, KeyError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/api/dual-arm-calibration/motion/{jid}")
    def capture_motion_status(jid: str):
        try:
            return motion.public(jid)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/api/dual-arm-calibration/motion/start")
    def start_capture_motion(request: Request, payload: dict):
        try:
            motion.start(
                request.app.state,
                payload["id"],
                payload["token"],
                payload.get("confirmed") is True,
            )
            return dict(ok=True)
        except (ValueError, KeyError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/api/dual-arm-calibration/motion/stop")
    def stop_capture_motion(payload: dict):
        try:
            motion.stop(payload["id"], payload["token"])
            return dict(cancel_requested=True)
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/api/dual-arm-calibration/motion/ack")
    def capture_ack(payload: dict):
        try:
            motion.acknowledge(payload["id"], payload["token"], payload["index"])
            return dict(ok=True)
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/api/dual-arm-calibration/status")
    def status(request: Request):
        state = request.app.state
        try:
            svc = service(state)
            arms = {a: arm_state(svc, a) for a in ("left", "right")}
            cameras = {
                r: metadata(state.devices.cameras[r])[1]
                for r in ("left", "right", "top")
            }
            with state.operations.lock:
                busy = bool(
                    state.operations.calibrating
                    or state.operations.reconfiguring
                    or state.operations.active()
                )
            model_file = Path(ROOT) / "public/models/piper/piper.urdf"
            import hashlib

            return dict(
                connected=True,
                busy=busy,
                arms=arms,
                cameras=cameras,
                model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
                pose_convention="measured flange; mm and fixed-axis XYZ degrees converted to metres/radians",
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/api/dual-arm-calibration/capture")
    def capture(request: Request, payload: dict):
        roles = payload.get("roles")
        if (
            not isinstance(roles, list)
            or not roles
            or len(set(roles)) != len(roles)
            or set(roles) - {"left", "right", "top"}
        ):
            raise HTTPException(422, "请选择明确的相机角色")
        svc = service(request.app.state)
        try:
            images, packets = {}, []
            with motion.capture_lease(request.app.state, payload.get("motion")):
                anchor = {a: arm_state(svc, a) for a in ("left", "right")}
                since = time.monotonic()
                for index in range(2):
                    if index:
                        while time.monotonic() - since < 0.6:
                            if request.app.state.operations.stop.wait(0.04):
                                raise ValueError("设备服务正在停止")
                            for a in anchor:
                                now = arm_state(svc, a)
                                if (
                                    np.max(
                                        np.abs(
                                            np.subtract(
                                                now["joints_deg"],
                                                anchor[a]["joints_deg"],
                                            )
                                        )
                                    )
                                    > 0.05
                                ):
                                    raise ValueError("机械臂仍在移动，请停稳后采集")
                    before = {a: arm_state(svc, a) for a in anchor}
                    frames = {}
                    for r in roles:
                        image, meta = metadata(request.app.state.devices.cameras[r])
                        images[f"{index}-{r}.png"] = image
                        frames[r] = meta
                    after = {a: arm_state(svc, a) for a in anchor}
                    for a in anchor:
                        if (
                            np.max(
                                np.abs(
                                    np.subtract(
                                        after[a]["joints_deg"], anchor[a]["joints_deg"]
                                    )
                                )
                            )
                            > 0.05
                            or np.max(
                                np.abs(
                                    np.subtract(
                                        after[a]["pose_mm_deg"],
                                        before[a]["pose_mm_deg"],
                                    )
                                )
                            )
                            > 0.1
                        ):
                            raise ValueError("取帧期间机械臂移动，样本未保存")
                    stamps = [m["timestamp"] for m in frames.values()]
                    if max(stamps) - min(stamps) > 0.05:
                        raise ValueError("相机收帧时间差超过 50 ms，请重试")
                    packets.append(dict(cameras=frames, arms=after))
                for r in roles:
                    if (
                        packets[1]["cameras"][r]["frame_number"]
                        <= packets[0]["cameras"][r]["frame_number"]
                    ):
                        raise ValueError("相机帧号未更新")
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
                z.writestr(
                    "metadata.json",
                    json.dumps(
                        dict(
                            packets=packets,
                            stable_s=time.monotonic() - since,
                            timing="software host receipt; stationary acquisition; no exposure synchronization",
                        )
                    ),
                )
                for name, image in images.items():
                    ok, png = cv2.imencode(
                        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    )
                    if not ok:
                        raise ValueError("原始图像编码失败")
                    z.writestr(name, png.tobytes())
            return Response(out.getvalue(), media_type="application/zip")
        except (ValueError, KeyError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/dual-arm-calibration")
    def workbench_redirect():
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/dual-arm-calibration/")

    @router.api_route(
        "/api/dual-arm-calibration/workbench/{path:path}", methods=["GET", "POST"]
    )
    @router.api_route("/dual-arm-calibration/{path:path}", methods=["GET", "POST"])
    async def workbench_proxy(request: Request, path: str):
        origin = request.headers.get("origin")
        try:
            host = proxy_host(request.headers)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc
        if request.method == "POST":
            if (
                request.headers.get("content-type", "").split(";")[0]
                != "application/json"
            ):
                raise HTTPException(415, "请求必须为 JSON")
            if int(request.headers.get("content-length", "0")) > 16 * 1024 * 1024:
                raise HTTPException(413, "请求过大")
        body = await request.body() if request.method == "POST" else None
        if body and len(body) > 16 * 1024 * 1024:
            raise HTTPException(413, "请求过大")
        if ".." in path.split("/"):
            raise HTTPException(404, "未知路径")

        def fetch():
            headers = {"Host": host}
            if body is not None:
                headers["Content-Type"] = "application/json"
            if origin:
                headers["Origin"] = origin
            r = urllib.request.Request(
                "http://127.0.0.1:8765/" + path,
                data=body,
                headers=headers,
                method=request.method,
            )
            try:
                response = urllib.request.urlopen(r, timeout=45)
            except urllib.error.HTTPError as exc:
                response = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                raise HTTPException(503, "标定网页服务未就绪") from exc
            with response:
                return Response(
                    response.read(),
                    status_code=response.code,
                    media_type=response.headers.get("Content-Type"),
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower()
                        in (
                            "content-disposition",
                            "cache-control",
                            "x-content-type-options",
                            "content-security-policy",
                        )
                    },
                )

        return await asyncio.to_thread(fetch)

    app.include_router(router)
    # RoboBench may already have mounted its built frontend at '/'. Keep that
    # catch-all after the added API routes, including on FastAPI's lazy routers.
    from starlette.routing import Mount

    mounts = [
        r for r in app.router.routes if isinstance(r, Mount) and r.path in ("", "/")
    ]
    app.router.routes[:] = [r for r in app.router.routes if r not in mounts] + mounts
