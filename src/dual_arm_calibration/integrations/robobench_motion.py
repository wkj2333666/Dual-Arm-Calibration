"""Piper planning and reviewed execution inside RoboBench's hardware owner."""

import copy
import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation

from ..board import Board
from ..camera_model import matrix, signature
from ..planning import candidate_views, select_views, view_features, novel_view
from ..validation import transform

CAPTURE_SPEED_DEG_S = 30.0


class CalibrationMotion:
    def __init__(self, root, kinematics, service, metadata, arm_state):
        self.root, self.kin, self.service, self.metadata, self.arm_state = (
            root,
            kinematics,
            service,
            metadata,
            arm_state,
        )
        self.lock = threading.RLock()
        self.jobs = {}

    def seed(self, state, side, request):
        current = self.arm_state(self.service(state), side)
        supplied = request.get("seed")
        if (
            supplied
            and supplied.get("source") == "workflow_hand_eye"
            and supplied.get("gate_passed") is True
        ):
            seed = copy.deepcopy(supplied)
        else:
            result = self.service(state)._load("wrist-" + side).get("result") or {}
            identity = result.get("identity", {})
            if (
                identity.get("arm_serial") != current["serial"]
                or identity.get("camera_source") != state.devices.cameras[side].source
            ):
                raise ValueError("已有腕外参与当前硬件不匹配，请先手动完成手眼初始化")
            seed = dict(
                flange_from_camera=result.get("flange_from_camera"),
                source="installed_hand_eye",
                gate_passed=result.get("quality_passed") is True,
                warning=None
                if result.get("quality_passed") is True
                else "历史手眼未通过质量检查，仅用于规划初值，执行前需核验初值和轨迹。",
            )
        x = transform(seed["flange_from_camera"])
        if np.linalg.norm(x[:3, 3]) > 0.3:
            raise ValueError("腕相机安装偏移超出规划范围")
        seed["sha256"] = hashlib.sha256(
            json.dumps(seed, sort_keys=True).encode()
        ).hexdigest()
        return seed

    def estimate(self, state, request):
        side = request.get("arm")
        if side not in ("left", "right"):
            raise ValueError("姿态估算需要选择腕相机")
        svc = self.service(state)
        with svc._hardware():
            seed = self.seed(state, side, request)
            arm = svc._stable_arm("wrist-" + side)[0]
            image, meta = self.metadata(state.devices.cameras[side])
            model = request["camera_model"]
            if meta["serial"] != model["serial"] or signature(
                meta["intrinsics"]
            ) != signature(model["factory_intrinsics"]):
                raise ValueError("相机配置改变，请重新建立标定流程")
            board = Board(request["board"])
            intr = model["intrinsics"]
            d, _ = board.detect(image, matrix(intr), intr["coeffs"])
            if not d.detected or d.corner_count < 12 or d.reprojection_error_px > 1.5:
                raise ValueError("当前腕相机需要清晰看到至少 12 个标定板角点")
            measured = self.arm_state(svc, side)
            if (
                np.max(np.abs(np.subtract(measured["joints_deg"], arm["joints"])))
                > 0.05
            ):
                raise ValueError("取图期间机械臂发生移动")
            camera = transform(measured["base_from_flange"]) @ transform(
                seed["flange_from_camera"]
            )
            target = camera @ d.camera_from_target
            return dict(
                base_from_board=target.tolist(),
                top_reference=self.top_reference(state, request, board),
                base_from_camera=camera.tolist(),
                seed=seed,
                corner_count=d.corner_count,
                reprojection_error_px=d.reprojection_error_px,
                board_position_m=target[:3, 3].tolist(),
                board_rotation_deg=Rotation.from_matrix(target[:3, :3])
                .as_euler("xyz", degrees=True)
                .tolist(),
                arm=side,
                frame=meta,
                created=time.time(),
            )

    def top_reference(self, state, request, board):
        model = request.get("top_model")
        if not model:
            return None
        try:
            image, meta = self.metadata(state.devices.cameras["top"])
            if meta["serial"] != model["serial"] or signature(
                meta["intrinsics"]
            ) != signature(model["factory_intrinsics"]):
                return None
            intr = model["intrinsics"]
            d, _ = board.detect(image, matrix(intr), intr["coeffs"])
            if d.detected and d.corner_count >= 12 and d.reprojection_error_px <= 1.5:
                return dict(
                    serial=meta["serial"],
                    camera_from_board=d.camera_from_target.tolist(),
                )
        except ValueError:
            pass
        return None

    def public(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                raise ValueError("轨迹记录不存在，请重新生成")
            return copy.deepcopy(job["public"])

    def plan(self, state, request):
        side = request.get("arm")
        dataset = request.get("dataset")
        count = request.get("count")
        if (
            side not in ("left", "right")
            or dataset not in ("intrinsics", "extrinsics")
            or not isinstance(count, int)
            or not 1 <= count <= 40
        ):
            raise ValueError("自动采集仅支持左右腕的内参或外参批次")
        jid = uuid.uuid4().hex
        token = uuid.uuid4().hex
        job = dict(
            token=token,
            cancel=threading.Event(),
            ack=threading.Event(),
            request=copy.deepcopy(request),
            public=dict(
                id=jid, state="planning", index=0, total=count, progress=0, error=None
            ),
            state=state,
        )
        with self.lock:
            self.jobs[jid] = job
        threading.Thread(target=self._plan, args=(jid, job), daemon=True).start()
        return dict(id=jid, token=token)

    def update(self, job, **values):
        with self.lock:
            job["public"].update(values)

    def camera_clearance(self, q, other_q, side, seed):
        """Supplement the legacy mesh model with a conservative mounted-camera envelope."""
        from server.motion.cartesian import base_position

        other = "right" if side == "left" else "left"
        origin = base_position(side)
        point = self.kin.flange.forward_kinematics([0, *q])[:3, 3] + origin
        radius = max(0.14, float(np.linalg.norm(transform(seed)[:3, 3])) + 0.06)
        distances = [point[2] - (-0.008) - radius]
        for center, half in (
            ([-0.17, 0, 0.56], [0.035, 0.035, 0.54]),
            ([-0.11, 0, 0.77], [0.025, 0.21, 0.13]),
        ):
            distances.append(
                np.linalg.norm(np.maximum(np.abs(point - center) - half, 0)) - radius
            )

        def segment_distance(a, b, center=point):
            delta = b - a
            t = np.clip(
                np.dot(center - a, delta) / max(np.dot(delta, delta), 1e-12), 0, 1
            )
            return np.linalg.norm(center - (a + t * delta))

        for a, b, r in self.kin.segments(other_q, other):
            distances.append(segment_distance(a, b) - radius - r)
        peer = self.kin.flange.forward_kinematics([0, *other_q])[:3, 3] + base_position(
            other
        )
        distances.append(np.linalg.norm(point - peer) - radius - 0.17)
        own = self.kin.flange.forward_kinematics([0, *q], full_kinematics=True)
        optical = (own[-1] @ transform(seed))[:3, 3] + origin
        for a, b in zip(own[:4], own[1:5]):
            distances.append(
                segment_distance(a[:3, 3] + origin, b[:3, 3] + origin, optical)
                - 0.07
                - 0.045
                + 0.025
            )
        return float(min(distances))

    def check_camera_path(self, segment, other, side, seed):
        return min(
            self.camera_clearance(
                np.deg2rad(w["joints"]), np.deg2rad(other["joints"]), side, seed
            )
            for w in segment["waypoints"]
        )

    def _plan(self, jid, job):
        from server.motion.home import HomeCollision, make_home_plan
        from server.motion.cartesian import base_position

        try:
            request = job["request"]
            state = job["state"]
            side = request["arm"]
            other_side = "right" if side == "left" else "left"
            estimate = self.estimate(state, request)
            devices = state.devices.snapshot("live")
            arms = {a["id"]: a for a in devices["arms"]}
            arm, other = copy.deepcopy(arms[side]), copy.deepcopy(arms[other_side])
            if arm.get("enabled") is not True or any(
                a.get("error_code") != 0 for a in arms.values()
            ):
                raise ValueError("双臂必须反馈正常，运动臂必须已使能")
            board = Board(request["board"])
            x = transform(estimate["seed"]["flange_from_camera"])
            proposals = candidate_views(
                estimate["base_from_board"],
                estimate["base_from_camera"],
                board.object_points,
                request["camera_model"]["intrinsics"],
                request["dataset"],
            )
            checker = HomeCollision(self.root, self.kin, arm, other)
            candidates = []
            last = np.deg2rad(arm["joints"])
            # Uniformly retain different tilts, distances and image placements.
            indices = np.unique(
                np.linspace(
                    0, max(0, len(proposals) - 1), min(240, len(proposals))
                ).astype(int)
            )
            for n, i in enumerate(indices):
                if job["cancel"].is_set():
                    raise InterruptedError("规划已停止")
                c = copy.deepcopy(proposals[i])
                if not novel_view(c["pixels"], request.get("existing_views", [])):
                    continue
                target = transform(c["base_from_camera"]) @ np.linalg.inv(x)
                try:
                    q = self.kin.flange.inverse_kinematics(
                        target_position=target[:3, 3],
                        target_orientation=target[:3, :3],
                        orientation_mode="all",
                        initial_position=[
                            0,
                            *np.clip(
                                last,
                                self.kin.limits[:, 0] + 0.03,
                                self.kin.limits[:, 1] - 0.03,
                            ),
                        ],
                    )[1:7]
                    fk = self.kin.flange.forward_kinematics([0, *q])
                    if (
                        np.linalg.norm(fk[:3, 3] - target[:3, 3]) > 0.002
                        or Rotation.from_matrix(
                            fk[:3, :3].T @ target[:3, :3]
                        ).magnitude()
                        > 0.035
                    ):
                        continue
                    if (
                        not np.isfinite(q).all()
                        or np.any(q < self.kin.limits[:, 0] + 0.02)
                        or np.any(q > self.kin.limits[:, 1] - 0.02)
                    ):
                        continue
                    internal, external = checker.check(q)
                    if internal[0] < 0 or external[0] < 0.025:
                        continue
                    if (
                        self.camera_clearance(q, np.deg2rad(other["joints"]), side, x)
                        < 0.025
                    ):
                        continue
                    c["joints_deg"] = np.rad2deg(q).tolist()
                    c["endpoint_clearance_m"] = external[0]
                    candidates.append(c)
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    continue
                finally:
                    self.update(
                        job, progress=round((n + 1) / max(1, len(indices)) * 0.65, 3)
                    )
            # Six-axis Piper cannot realize every free camera pose. Add bounded
            # joint-space proposals and judge their actual FK-projected board.
            from scipy.stats import qmc

            origin = np.deg2rad(arm["joints"])
            amplitudes = np.deg2rad([15, 18, 18, 30, 20, 30])
            for sample in qmc.Halton(6, scramble=False).random(400):
                if job["cancel"].is_set():
                    raise InterruptedError("规划已停止")
                q = np.clip(
                    origin + (sample * 2 - 1) * amplitudes,
                    self.kin.limits[:, 0] + 0.03,
                    self.kin.limits[:, 1] - 0.03,
                )
                camera = self.kin.flange.forward_kinematics([0, *q]) @ x
                feature = view_features(
                    np.linalg.inv(camera) @ transform(estimate["base_from_board"]),
                    board.object_points,
                    request["camera_model"]["intrinsics"],
                    request["dataset"],
                )
                if not feature:
                    continue
                if not novel_view(feature["pixels"], request.get("existing_views", [])):
                    continue
                internal, external = checker.check(q)
                if internal[0] < 0 or external[0] < 0.025:
                    continue
                if (
                    self.camera_clearance(q, np.deg2rad(other["joints"]), side, x)
                    < 0.025
                ):
                    continue
                candidates.append(
                    dict(
                        base_from_camera=camera.tolist(),
                        joints_deg=np.rad2deg(q).tolist(),
                        endpoint_clearance_m=external[0],
                        **feature,
                    )
                )
            selected = select_views(
                candidates, min(len(candidates), request["count"] * 3)
            )
            waypoints = []
            segments = []
            current = copy.deepcopy(arm)
            while selected and len(waypoints) < request["count"]:
                if job["cancel"].is_set():
                    raise InterruptedError("规划已停止")
                index = min(
                    range(len(selected)),
                    key=lambda i: np.linalg.norm(
                        np.subtract(selected[i]["joints_deg"], current["joints"])
                    ),
                )
                c = selected.pop(index)
                segment = make_home_plan(
                    self.root,
                    current,
                    other,
                    duration=3,
                    goal_joints=c["joints_deg"],
                    speed_deg_s=CAPTURE_SPEED_DEG_S,
                )
                if not segment["execution_allowed"]:
                    continue
                camera_gap = self.check_camera_path(segment, other, side, x)
                if camera_gap < 0.025:
                    continue
                c["duration_s"] = segment["duration_s"]
                c["clearance_m"] = min(
                    segment["checks"]["minimum_proxy_clearance_m"], camera_gap
                )
                c["links_base"] = [
                    p[:3, 3].tolist()
                    for p in self.kin.flange.forward_kinematics(
                        [0, *np.deg2rad(c["joints_deg"])], full_kinematics=True
                    )
                ]
                waypoints.append(c)
                segments.append(segment)
                current["joints"] = c["joints_deg"]
                self.update(
                    job, progress=0.65 + 0.35 * len(waypoints) / request["count"]
                )
            if len(waypoints) < request["count"]:
                raise ValueError(
                    f"仅找到 {len(waypoints)}/{request['count']} 个可达且路径检查通过的姿态，请调整板位或起始姿态后重新估算"
                )
            job.update(
                segments=segments, estimate=estimate, initial=arms, waypoints=waypoints
            )
            peer_links = [
                (p[:3, 3] + base_position(other_side) - base_position(side)).tolist()
                for p in self.kin.flange.forward_kinematics(
                    [0, *np.deg2rad(other["joints"])], full_kinematics=True
                )
            ]
            self.update(
                job,
                state="preview",
                progress=1.0,
                created=time.time(),
                arm=side,
                dataset=request["dataset"],
                estimate=estimate,
                waypoints=waypoints,
                duration_s=sum(w["duration_s"] + 0.8 for w in waypoints),
                min_clearance_mm=min(w["clearance_m"] for w in waypoints) * 1000,
                max_speed_deg_s=CAPTURE_SPEED_DEG_S,
                board=request["board"],
                initial=arms,
                execution_allowed=True,
                other_links_base=peer_links,
                table_z_base=float(-0.008 - base_position(side)[2]),
            )
        except InterruptedError as exc:
            self.update(job, state="cancelled", error=str(exc))
        except Exception as exc:
            self.update(job, state="failed", error=str(exc))

    def start(self, state, jid, token, confirmed):
        with self.lock:
            job = self.jobs.get(jid)
            if not job or token != job["token"] or not confirmed:
                raise ValueError("请先核验初值和轨迹预览，再确认执行")
            if job["cancel"].is_set():
                raise ValueError("轨迹已停止，请重新生成预览")
            if job["public"].get("max_speed_deg_s") != CAPTURE_SPEED_DEG_S:
                raise ValueError("采集速度已改变，请重新生成轨迹预览")
            if (
                job["public"]["state"] != "preview"
                or time.time() - job["public"]["created"] > 120
            ):
                raise ValueError("轨迹预览已过期，请重新生成")
        fresh = self.estimate(state, job["request"])
        before = transform(job["estimate"]["base_from_board"])
        after = transform(fresh["base_from_board"])
        tracker = job["estimate"].get("top_reference")
        if tracker:
            if not fresh.get("top_reference"):
                raise ValueError("固定顶相机未看到板，请恢复视野后重新预览")
            before = transform(tracker["camera_from_board"])
            after = transform(fresh["top_reference"]["camera_from_board"])
        elif not job["estimate"]["seed"].get("gate_passed"):
            raise ValueError(
                "规划初值未验证且固定顶相机未看到板；请让顶相机同时看到板，或先手动完成腕外参"
            )
        if np.linalg.norm(before[:3, 3] - after[:3, 3]) > (
            0.005 if tracker else 0.01
        ) or Rotation.from_matrix(
            before[:3, :3].T @ after[:3, :3]
        ).magnitude() > np.deg2rad(1 if tracker else 3):
            raise ValueError("板位姿已偏离预览，请重新估算并生成轨迹")
        svc = self.service(state)
        with state.operations.lock:
            if (
                state.operations.active()
                or state.operations.calibrating
                or state.operations.reconfiguring
                or state.store.query("SELECT id FROM runs WHERE ended IS NULL")
            ):
                raise ValueError("已有设备操作正在运行")
            for a in ("left", "right"):
                actual = svc._arm("wrist-" + a)
                initial = job["initial"][a]
                if (actual.get("adapter_serial") or actual["channel"]) != (
                    initial.get("adapter_serial") or initial["channel"]
                ) or np.max(
                    np.abs(np.subtract(actual["joints"], initial["joints"]))
                ) > 1:
                    raise ValueError("双臂身份或起点改变，请重新预览")
            operation = dict(
                kind="motion",
                task_id=None,
                cancel=job["cancel"],
                done=threading.Event(),
                motion=job["segments"][0],
                cancel_path=f"/arms/motions/{jid}/cancel",
            )
            state.operations.jobs[jid] = operation
            job["operation"] = operation
            self.update(job, state="running", phase="moving", index=0, error=None)
            operation["thread"] = threading.Thread(
                target=self._execute, args=(jid, job), daemon=True
            )
            try:
                operation["thread"].start()
            except BaseException:
                operation["done"].set()
                raise

    def _execute(self, jid, job):
        from server.motion.home import make_home_plan
        from server.motion.orbit import execute_plan

        state = job["state"]
        side = job["request"]["arm"]
        other = "right" if side == "left" else "left"
        try:
            for index, point in enumerate(job["waypoints"]):
                if job["cancel"].is_set():
                    raise InterruptedError("自动采集已停止并请求位置保持")
                arms = {
                    a: self.service(state)._arm("wrist-" + a) for a in ("left", "right")
                }
                expected = job["segments"][index]
                if (
                    np.max(
                        np.abs(
                            np.subtract(arms[side]["joints"], expected["start_joints"])
                        )
                    )
                    > 1
                    or np.max(
                        np.abs(
                            np.subtract(
                                arms[other]["joints"], expected["other_start_joints"]
                            )
                        )
                    )
                    > 1
                ):
                    raise ValueError("实际起点偏离预览，已停止")
                segment = make_home_plan(
                    self.root,
                    arms[side],
                    arms[other],
                    duration=3,
                    goal_joints=point["joints_deg"],
                    speed_deg_s=CAPTURE_SPEED_DEG_S,
                )
                if not segment["execution_allowed"]:
                    raise ValueError(
                        "路径复检未通过：" + "；".join(segment["blocking_reasons"])
                    )
                if (
                    self.check_camera_path(
                        segment,
                        arms[other],
                        side,
                        job["estimate"]["seed"]["flange_from_camera"],
                    )
                    < 0.025
                ):
                    raise ValueError("腕相机安装包络路径复检未通过")
                job["operation"]["motion"] = segment
                self.update(
                    job,
                    phase="moving",
                    index=index + 1,
                    progress=index / len(job["waypoints"]),
                )
                execute_plan(
                    segment,
                    state.devices,
                    job["cancel"],
                    lambda at, q: job["operation"].update(
                        motion_progress=at / max(segment["duration_s"], 0.001)
                    ),
                )
                job["ack"].clear()
                self.update(job, phase="awaiting_capture")
                deadline = time.monotonic() + 60
                while not job["ack"].wait(0.05):
                    if job["cancel"].is_set() or state.operations.stop.is_set():
                        raise InterruptedError("自动采集已停止")
                    if time.monotonic() > deadline:
                        raise ValueError("采集端未响应，已停止后续运动")
                if job["cancel"].is_set() or state.operations.stop.is_set():
                    raise InterruptedError("自动采集已停止")
            self.update(job, state="completed", phase="complete", progress=1.0)
        except InterruptedError as exc:
            self.update(job, state="cancelled", error=str(exc))
        except Exception as exc:
            self.update(job, state="failed", error=str(exc))
        finally:
            job["operation"]["done"].set()

    def stop(self, jid, token):
        with self.lock:
            job = self.jobs.get(jid)
            if not job or job["token"] != token:
                raise ValueError("自动采集记录无效")
            job["cancel"].set()
            job["ack"].set()

    def acknowledge(self, jid, token, index):
        with self.lock:
            job = self.jobs.get(jid)
            if (
                not job
                or job["token"] != token
                or job["public"].get("phase") != "awaiting_capture"
                or index != job["public"]["index"]
            ):
                raise ValueError("采集确认与当前点位不匹配")
            if job.get("capture_active"):
                raise ValueError("原始图像采集尚未完成")
            job["ack"].set()

    @contextmanager
    def capture_lease(self, state, claim):
        if not claim:
            with self.service(state)._hardware():
                yield
            return
        jid = claim.get("id")
        with self.lock:
            job = self.jobs.get(jid)
            if (
                not job
                or job["token"] != claim.get("token")
                or job["cancel"].is_set()
                or job["public"].get("phase") != "awaiting_capture"
            ):
                raise ValueError("自动采集不处于已停稳等待取帧的阶段")
            if job.get("capture_active"):
                raise ValueError("该点位正在取帧")
            job["capture_active"] = True
        try:
            with state.operations.lock:
                if (
                    state.operations.calibrating
                    or state.operations.reconfiguring
                    or any(
                        k != jid and not v["done"].is_set()
                        for k, v in state.operations.jobs.items()
                    )
                ):
                    raise ValueError("存在其他设备占用")
                state.operations.calibrating = True
            try:
                yield
            finally:
                with state.operations.lock:
                    state.operations.calibrating = False
        finally:
            with self.lock:
                job["capture_active"] = False
