"""Estimate → review → execute → settle/capture handshake orchestration."""

import copy
import threading

from .workflow import digest

ACTIVE = ("planning", "starting", "running", "stopping")


class AutomaticCapture:
    def __init__(self, live):
        self.live = live
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.value = copy.deepcopy(
            (live.session or {}).get("automatic", dict(state="idle"))
        )
        if self.value.get("state") in (*ACTIVE, "preview"):
            self.value.update(
                state="interrupted",
                error="服务重启中断了自动采集，请重新预览；不会自动恢复运动",
            )
            claim = self.value.get("_claim")
            if claim:

                def stop_previous():
                    try:
                        live.provider.motion_command("stop", claim)
                    except ValueError:
                        pass

                threading.Thread(target=stop_previous, daemon=True).start()

    def status(self):
        with self.lock:
            if self.value.get("session_id") != (self.live.session or {}).get("id"):
                return dict(state="idle")
            return copy.deepcopy(
                {k: v for k, v in self.value.items() if not k.startswith("_")}
            )

    def active(self):
        return self.status().get("state") in ACTIVE

    def update(self, **changes):
        with self.lock:
            self.value.update(changes)
            value = copy.deepcopy(self.value)
        with self.live.lock:
            if self.live.session and self.live.session["id"] == value.get("session_id"):
                self.live.session["automatic"] = value
                self.live.save()

    def request(self, key):
        self.live.ready(key)
        if not key.startswith(("intrinsics-", "wrists-")):
            raise ValueError(
                "自动运动采集用于左右腕内参或外参；顶相机录像和桥接使用对应步骤"
            )
        role = key.split("-")[1]
        if role not in ("left", "right"):
            raise ValueError("顶相机固定，请移动板采集内参")
        s = self.live.session
        rows, _ = self.live.batch(key)
        dataset = "intrinsics" if key.startswith("intrinsics-") else "extrinsics"
        request = dict(
            arm=role,
            dataset=dataset,
            count=max(1, (28 if dataset == "intrinsics" else 20) - len(rows)),
            board=s["rig"]["board"],
            camera_model=s["models"][role],
            top_model=s["models"].get("top"),
            existing_views=[
                dict(charuco_ids=r["charuco_ids"], charuco_corners=r["charuco_corners"])
                for r in rows
            ],
        )
        if (28 if dataset == "intrinsics" else 20) <= len(rows):
            request["count"] = 8
        wrist = s["results"].get("wrists-" + role)
        if wrist:
            request["seed"] = dict(
                source="workflow_hand_eye",
                gate_passed=True,
                flange_from_camera=wrist["flange_from_camera"],
            )
        binding = digest(
            dict(
                session=s["id"],
                batch=key,
                model=s["models"][role],
                rig=s["data"]["rig_sha256"],
                ids=[r["id"] for r in rows],
            )
        )
        return request, binding

    def estimate(self, key):
        request, binding = self.request(key)
        result = self.live.provider.estimate(request)
        self.cancel = threading.Event()
        with self.lock:
            self.value = dict(
                state="estimated",
                session_id=self.live.session["id"],
                batch=key,
                binding=binding,
                estimate=result,
            )
        self.update()

    def plan(self, key):
        request, binding = self.request(key)
        value = self.status()
        if (
            value.get("state") not in ("estimated", "preview", "failed", "cancelled")
            or value.get("binding") != binding
            or value.get("batch") != key
            or not value.get("estimate")
        ):
            raise ValueError("请先用当前批次的单张图估算板位姿")
        self.cancel = threading.Event()
        claim = self.live.provider.plan(request)
        self.update(
            state="planning",
            _claim=claim,
            preview=None,
            error=None,
            index=0,
            total=request["count"],
            saved=0,
            skips=[],
        )
        threading.Thread(
            target=self._wait_plan, args=(claim, self.cancel), daemon=True
        ).start()

    def _wait_plan(self, claim, cancel):
        try:
            last = None
            while not cancel.is_set():
                status = self.live.provider.motion_status(claim["id"])
                if status["state"] == "preview":
                    self.update(
                        state="preview",
                        preview=status,
                        estimate=status["estimate"],
                        progress=1,
                    )
                    return
                if status["state"] in ("failed", "cancelled"):
                    self.update(state=status["state"], error=status.get("error"))
                    return
                progress = status.get("progress", 0)
                if last != progress:
                    self.update(progress=progress)
                    last = progress
                cancel.wait(0.25)
            self.live.provider.motion_command("stop", claim)
            self.update(state="cancelled")
        except Exception as exc:
            self.update(state="failed", error=str(exc))

    def start(self, key, confirmed):
        _, binding = self.request(key)
        with self.lock:
            value = copy.deepcopy(self.value)
        if (
            not confirmed
            or value.get("state") != "preview"
            or value.get("binding") != binding
        ):
            raise ValueError("请先核验当前批次的初值和轨迹预览，并勾选执行确认")
        claim = value["_claim"]
        self.cancel = threading.Event()
        self.update(state="starting")
        try:
            self.live.provider.motion_command("start", claim, confirmed=True)
            if self.cancel.is_set():
                self.live.provider.motion_command("stop", claim)
                self.update(state="cancelled")
                return
            self.update(state="running")
            threading.Thread(
                target=self._capture, args=(claim, key, self.cancel), daemon=True
            ).start()
        except Exception as exc:
            self.update(state="failed", error=str(exc))
            raise

    def _capture(self, claim, key, cancel):
        handled = set()
        skips = []
        saved = 0
        consecutive = 0
        try:
            while not cancel.is_set():
                status = self.live.provider.motion_status(claim["id"])
                if status["state"] in ("completed", "failed", "cancelled"):
                    self.update(
                        state=status["state"],
                        error=status.get("error"),
                        saved=saved,
                        skips=skips,
                        phase=status.get("phase"),
                    )
                    return
                index = status.get("index", 0)
                if status.get("phase") == "awaiting_capture" and index not in handled:
                    self.update(index=index, phase="capturing")
                    try:
                        with self.live.lock:
                            self.live.capture(
                                key,
                                dict(
                                    stationary_confirmed=True,
                                    motion=claim,
                                    board_reference=self.value["estimate"][
                                        "base_from_board"
                                    ],
                                    planning_seed=self.value["estimate"]["seed"][
                                        "flange_from_camera"
                                    ],
                                    top_reference=self.value["estimate"].get(
                                        "top_reference"
                                    ),
                                ),
                            )
                        saved += 1
                        consecutive = 0
                    except ValueError as exc:
                        if not any(text in str(exc) for text in ("角点", "太接近")):
                            raise
                        skips.append(dict(index=index, error=str(exc)))
                        consecutive += 1
                        if consecutive >= 3:
                            raise ValueError(
                                "连续三点未采到有效板，已停止，请核对板位置与规划初值"
                            ) from exc
                    handled.add(index)
                    if cancel.is_set():
                        break
                    self.live.provider.motion_command("ack", claim, index=index)
                self.update(
                    index=index,
                    total=status["total"],
                    phase=status.get("phase"),
                    saved=saved,
                    skips=skips,
                )
                cancel.wait(0.25)
            self.live.provider.motion_command("stop", claim)
            self.update(state="cancelled", saved=saved, skips=skips)
        except Exception as exc:
            try:
                self.live.provider.motion_command("stop", claim)
            except ValueError:
                pass
            self.update(state="failed", error=str(exc), saved=saved, skips=skips)

    def stop(self):
        self.cancel.set()
        with self.lock:
            claim = copy.deepcopy(self.value.get("_claim"))
            active = self.value.get("state") in ACTIVE
        if claim:
            self.live.provider.motion_command("stop", claim)
        # Do not acquire the live capture lock: the stop request stays immediate.
        with self.lock:
            self.value.update(state="stopping" if active else "cancelled")
        return dict(cancel_requested=True)
