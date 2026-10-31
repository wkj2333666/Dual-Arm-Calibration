"""Persistent manual hardware calibration using a configured capture provider."""

import copy
import json
import threading
import time
import uuid

import cv2
import numpy as np

from .board import Board
from .bridge import solve_bridge
from .camera_model import matrix, signature
from .intrinsics import coverage, solve as fit_lens
from .storage import atomic_json
from .validation import observations
from .workflow import ARMS, CAMERAS, check_rig, cross_validate, digest, table_frame
from .wrist import solve_wrist


class LiveCalibration:
    def __init__(self, directory, config):
        if config.get("provider") != "robobench":
            raise ValueError("Unsupported hardware provider")
        from .integrations.robobench import RoboBench

        self.provider = RoboBench(config)
        self.config = config
        self.directory = directory / "hardware"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "current.json"
        self.lock = threading.RLock()
        self.session = json.loads(self.path.read_text()) if self.path.exists() else None
        self.last_probe = None
        self.probed_at = 0

    def save(self):
        atomic_json(self.path, self.session)

    def probe(self, force=False):
        if force or time.monotonic() - self.probed_at > 2:
            self.last_probe = self.provider.probe()
            self.probed_at = time.monotonic()
        return copy.deepcopy(self.last_probe)

    def status(self):
        with self.lock:
            out = dict(
                configured=True,
                board=self.config["board"],
                motion_supported=False,
                board_thickness_mm=self.config.get("board_thickness_mm"),
                session=None,
            )
            try:
                out.update(connected=True, devices=self.probe(), error=None)
            except ValueError as exc:
                out.update(connected=False, error=str(exc))
            if self.session:
                s = self.session
                counts = {
                    "intrinsics-" + r: len(s["data"]["intrinsics"].get(r, []))
                    for r in CAMERAS
                }
                counts.update(
                    {"wrists-" + a: len(s["data"]["wrists"][a]) for a in ARMS}
                )
                counts.update(
                    {"bridges-" + a: len(s["data"]["bridges"][a]) for a in ARMS}
                )
                counts.update({k: len(s["data"][k]) for k in ("validation", "table")})
                out["session"] = dict(
                    id=s["id"],
                    counts=counts,
                    frozen=list(s["results"]),
                    table_enabled=s["rig"]["table"]["enabled"],
                    modes={
                        r: s["rig"]["cameras"][r]["intrinsics_mode"] for r in CAMERAS
                    },
                    sealed=s.get("sealed", False),
                    job_id=s.get("job_id"),
                    validation=s["results"].get("validation"),
                )
                out["session"]["coverage"] = {
                    r: coverage(
                        rows, s["rig"]["cameras"][r]["model"]["factory_intrinsics"]
                    )
                    for r, rows in s["data"]["intrinsics"].items()
                    if rows
                }
            return out

    def new(self, payload):
        if payload.get("board_confirmed") is not True:
            raise ValueError("请核对并确认实际标定板规格")
        devices = self.probe(force=True)
        modes = payload.get("modes", {})
        if set(modes) != set(CAMERAS) or any(
            v not in ("reuse", "fit") for v in modes.values()
        ):
            raise ValueError("请分别选择三台相机的参数来源")
        table = payload.get("table", False)
        if not isinstance(table, bool):
            raise ValueError("桌面选项无效")
        if self.session:
            atomic_json(
                self.directory / self.session["id"] / "session.json", self.session
            )
        models = {
            r: dict(
                serial=c["serial"],
                source="factory",
                image_rotation_deg=0,
                factory_intrinsics=c["intrinsics"],
                intrinsics=c["intrinsics"],
            )
            for r, c in devices["cameras"].items()
        }
        rig = dict(
            schema_version=1,
            board=self.config["board"],
            arms={
                a: dict(
                    model_id="piper:"
                    + devices["arms"][a]["serial"]
                    + ":"
                    + devices["model_sha256"]
                )
                for a in ARMS
            },
            cameras={
                r: dict(
                    model=models[r],
                    mount="fixed" if r == "top" else r + "_flange",
                    family="realsense_d435i",
                    stream="rgb",
                    intrinsics_mode=modes[r],
                )
                for r in CAMERAS
            },
            table=dict(
                enabled=table,
                thickness_mm=self.config.get("board_thickness_mm"),
                board_z_direction="into_table",
            ),
        )
        check_rig(rig)
        data = dict(
            schema_version=1,
            rig_sha256=digest(rig),
            intrinsics={r: [] for r in CAMERAS},
            wrists={a: [] for a in ARMS},
            bridges={a: [] for a in ARMS},
            validation=[],
            table=[],
        )
        self.session = dict(
            id=uuid.uuid4().hex,
            rig=rig,
            data=data,
            models=copy.deepcopy(models),
            results={},
            sealed=False,
        )
        self.save()

    def batch(self, key):
        s = self.session
        if key in ("validation", "table"):
            return s["data"][key], None
        parts = key.split("-")
        if len(parts) != 2:
            raise ValueError("未知采集阶段")
        kind, role = parts
        if kind not in ("intrinsics", "wrists", "bridges") or role not in (
            CAMERAS if kind == "intrinsics" else ARMS
        ):
            raise ValueError("未知采集阶段")
        return s["data"][kind][role], role

    def ready(self, key):
        s = self.session
        if not s or s.get("sealed"):
            raise ValueError("请新建硬件标定流程；已导出的流程不可追加样本")
        self.batch(key)
        if key in s["results"]:
            raise ValueError("该阶段已通过并冻结")
        if key.startswith("intrinsics-"):
            r = key.split("-")[1]
            if s["rig"]["cameras"][r]["intrinsics_mode"] != "fit":
                raise ValueError("此相机已选择复用参数，无需采集内参")
        if key.startswith(("wrists-", "bridges-")):
            role = key.split("-")[1]
            if (
                s["rig"]["cameras"][role]["intrinsics_mode"] == "fit"
                and "intrinsics-" + role not in s["results"]
            ):
                raise ValueError("请先完成该腕相机独立内参")
        if key.startswith("bridges-"):
            role = key.split("-")[1]
            if "wrists-" + role not in s["results"]:
                raise ValueError("请先完成该腕相机外参")
            if (
                s["rig"]["cameras"]["top"]["intrinsics_mode"] == "fit"
                and "intrinsics-top" not in s["results"]
            ):
                raise ValueError("请先完成顶相机内参")
        if key in ("validation", "table"):
            if not all("bridges-" + a in s["results"] for a in ARMS):
                raise ValueError("请先完成左右手眼与顶相机桥接")
            if key == "table" and not s["rig"]["table"]["enabled"]:
                raise ValueError("本流程已跳过桌面步骤")

    def capture(self, key, payload):
        self.ready(key)
        s = self.session
        samples, role = self.batch(key)
        if len(samples) >= 120:
            raise ValueError("每阶段最多 120 组，请检查覆盖或新建流程")
        if payload.get("stationary_confirmed") is not True:
            raise ValueError("请确认板和机械臂已停稳；腕外参批次内板必须固定")
        roles = (
            list(CAMERAS)
            if key in ("validation", "table")
            else ["top", role]
            if key.startswith("bridges-")
            else [role]
        )
        device = self.probe(force=True)
        for a in ARMS:
            identity = (
                "piper:" + device["arms"][a]["serial"] + ":" + device["model_sha256"]
            )
            if s["rig"]["arms"][a]["model_id"] != identity:
                raise ValueError("机器人模型或安装身份变化，请新建流程")
        raw = self.provider.capture(roles)
        board = Board(s["rig"]["board"])
        obs = {}
        sid = uuid.uuid4().hex
        for r in roles:
            model = s["models"][r]
            detections = []
            for i in (0, 1):
                meta = raw["packets"][i]["cameras"][r]
                if meta["serial"] != model["serial"] or signature(
                    meta["intrinsics"]
                ) != signature(model["factory_intrinsics"]):
                    raise ValueError("相机身份或原始图像配置变化，请新建流程")
                image = raw["images"][f"{i}-{r}"]
                intr = model["intrinsics"]
                if image is None or image.shape[:2] != (intr["height"], intr["width"]):
                    raise ValueError("原始图像尺寸不匹配")
                d, _ = board.detect(image, matrix(intr), intr["coeffs"])
                if (
                    not d.detected
                    or d.corner_count < 12
                    or d.reprojection_error_px > 1.5
                ):
                    raise ValueError(
                        r + " 相机至少需要清晰看到 12 个板角点，请调整后重试"
                    )
                detections.append(d)
            before, after = detections
            common = sorted(
                set(before.charuco_ids.tolist()) & set(after.charuco_ids.tolist())
            )
            if len(common) < 12:
                raise ValueError("板在稳定窗口中丢失")
            diff = [
                before.charuco_corners[before.charuco_ids.tolist().index(i)]
                - after.charuco_corners[after.charuco_ids.tolist().index(i)]
                for i in common
            ]
            if np.sqrt(np.mean(np.sum(np.square(diff), axis=1))) > 1.5:
                raise ValueError("标定板仍在移动，请停稳后采集")
            meta = raw["packets"][1]["cameras"][r]
            obs[r] = dict(
                id=sid,
                camera_frame={
                    k: meta[k]
                    for k in (
                        "serial",
                        "image_rotation_deg",
                        "frame_number",
                        "timestamp",
                    )
                },
                intrinsics=meta["intrinsics"],
                charuco_ids=after.charuco_ids.tolist(),
                charuco_corners=after.charuco_corners.tolist(),
                camera_from_target=after.camera_from_target.tolist(),
                reprojection_error_px=after.reprojection_error_px,
            )
            observations([obs[r]], board.object_points, model)
        primary = "top" if "top" in roles else role
        for old in samples:
            prior = old["cameras"][primary] if "cameras" in old else old
            common = sorted(
                set(prior["charuco_ids"]) & set(obs[primary]["charuco_ids"])
            )
            if len(common) >= 12:
                diff = [
                    np.subtract(
                        prior["charuco_corners"][prior["charuco_ids"].index(i)],
                        obs[primary]["charuco_corners"][
                            obs[primary]["charuco_ids"].index(i)
                        ],
                    )
                    for i in common
                ]
                if (
                    key not in ("table", "validation")
                    and np.sqrt(np.mean(np.sum(np.square(diff), axis=1))) < 8
                ):
                    raise ValueError("与已有观测太接近，请改变距离或倾角")
        arms = raw["packets"][1]["arms"]
        if key in ("table", "validation"):
            sample = dict(
                id=sid,
                cameras=obs,
                base_from_flange={a: arms[a]["base_from_flange"] for a in ARMS},
            )
        else:
            sample = obs[primary]
            if key.startswith(("wrists-", "bridges-")):
                sample.update(
                    base_from_flange=arms[role]["base_from_flange"],
                    joint_angles_rad=arms[role]["joint_angles_rad"],
                )
            if key.startswith("bridges-"):
                sample["wrist"] = obs[role]
        sample["measurement"] = dict(
            stable_s=raw["stable_s"], timing=raw["timing"], arms=arms
        )
        images = self.directory / s["id"] / "images"
        images.mkdir(parents=True, exist_ok=True)
        for r in roles:
            if not cv2.imwrite(str(images / f"{sid}-{r}.png"), raw["images"][f"1-{r}"]):
                raise ValueError("原始图像保存失败，样本未提交")
        sample["image_files"] = {r: f"images/{sid}-{r}.png" for r in roles}
        samples.append(sample)
        self.save()

    def fit(self, key):
        self.ready(key)
        s = self.session
        rows, role = self.batch(key)
        points = Board(s["rig"]["board"]).object_points
        if key.startswith("intrinsics-"):
            result = fit_lens(
                rows, points, camera_family="realsense_d435i", stream="rgb"
            )
        elif key.startswith("wrists-"):
            result = solve_wrist(rows, points, camera_model=s["models"][role])
        elif key.startswith("bridges-"):
            result = solve_bridge(
                rows, points, s["results"]["wrists-" + role], s["models"]["top"]
            )
        else:
            result, chains = cross_validate(
                rows,
                points,
                s["models"],
                {a: s["results"]["wrists-" + a] for a in ARMS},
                {a: s["results"]["bridges-" + a] for a in ARMS},
            )
            if key == "table":
                table_frame(chains, s["rig"]["table"])
        if not result["gate_passed"]:
            raise ValueError(
                "阶段未通过质量检查，请补采："
                + json.dumps(
                    result.get("issues", result.get("validation", {})),
                    ensure_ascii=False,
                )
            )
        s["results"][key] = result
        if key.startswith("intrinsics-"):
            s["models"][role] = result
        self.save()

    def action(self, payload, workbench):
        with self.lock:
            op = payload.get("op")
            if op == "new":
                self.new(payload)
            elif op == "capture":
                self.capture(payload.get("batch", ""), payload)
            elif op == "fit":
                self.fit(payload.get("batch", ""))
            elif op == "reset":
                key = payload.get("batch", "")
                self.ready(key)
                rows, _ = self.batch(key)
                atomic_json(
                    self.directory
                    / self.session["id"]
                    / ("previous-" + key + "-" + uuid.uuid4().hex + ".json"),
                    rows,
                )
                rows.clear()
                self.save()
            elif op == "finish":
                s = self.session
                if not s or s.get("sealed") or "validation" not in s["results"]:
                    raise ValueError("请先完成独立验证")
                if s["rig"]["table"]["enabled"] and "table" not in s["results"]:
                    raise ValueError("请先完成桌面采集与验证")
                jid = workbench.start(
                    dict(kind="hardware", rig=s["rig"], data=s["data"])
                )["id"]
                s.update(sealed=True, job_id=jid)
                self.save()
                return dict(job_id=jid)
            elif op == "resume":
                s = self.session
                if not s or not s.get("sealed"):
                    raise ValueError("没有已提交的流程")
                jobs = {j["id"]: j for j in workbench.state()["jobs"]}
                if jobs.get(s.get("job_id"), {}).get("status") != "failed":
                    raise ValueError("只有最终求解失败的流程可继续补采")
                s.update(sealed=False)
                s["results"].pop("validation", None)
                s["results"].pop("table", None)
                self.save()
            else:
                raise ValueError("未知硬件标定操作")
            return dict(ok=True)
