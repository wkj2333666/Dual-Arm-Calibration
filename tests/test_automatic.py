import threading
import time
import copy
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_calibration.automatic import AutomaticCapture
from dual_arm_calibration.board import Board
from dual_arm_calibration.demo import generate
from dual_arm_calibration.integrations.robobench_motion import CalibrationMotion
from dual_arm_calibration.planning import candidate_views, select_views, novel_view


def wait_for(auto, states):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if auto.status()["state"] in states:
            return auto.status()
        time.sleep(0.01)
    raise AssertionError(auto.status())


class FakeProvider:
    def __init__(self, live):
        self.live = live
        self.commands = []
        self.state = "preview"
        self.index = 1

    def estimate(self, request):
        return dict(
            base_from_board=np.eye(4).tolist(),
            seed=dict(flange_from_camera=np.eye(4).tolist()),
            created=time.time(),
        )

    def plan(self, request):
        self.state = "preview"
        return dict(id="job", token="private")

    def motion_status(self, jid):
        return dict(
            id=jid,
            state=self.state,
            index=self.index,
            total=2,
            phase="awaiting_capture",
            estimate=self.estimate({}),
            waypoints=[],
        )

    def motion_command(self, command, claim, **fields):
        assert claim == dict(id="job", token="private")
        self.commands.append(command)
        if command == "start":
            self.state = "running"
        if command == "stop":
            self.state = "cancelled"
        if command == "ack":
            assert (
                len(self.live.rows) == self.index
            )  # never move on before data is saved
            self.index += 1
            if self.index > 2:
                self.state = "completed"


class FakeLive:
    def __init__(self):
        self.lock = threading.RLock()
        self.rows = []
        self.saved = 0
        self.session = dict(
            id="session",
            rig=dict(board={}),
            models=dict(left={}),
            data=dict(rig_sha256="rig"),
            results={},
        )
        self.provider = FakeProvider(self)

    def ready(self, key):
        assert key == "intrinsics-left"

    def batch(self, key):
        return self.rows, "left"

    def save(self):
        self.saved += 1

    def capture(self, key, payload):
        assert payload["motion"]["token"] == "private"
        self.rows.append(
            dict(id=str(len(self.rows)), charuco_ids=[], charuco_corners=[])
        )


def test_estimate_preview_confirmation_and_save_before_next_motion():
    live = FakeLive()
    auto = AutomaticCapture(live)
    with pytest.raises(ValueError, match="估算"):
        auto.plan("intrinsics-left")
    auto.estimate("intrinsics-left")
    auto.plan("intrinsics-left")
    wait_for(auto, ("preview",))
    with pytest.raises(ValueError, match="核验"):
        auto.start("intrinsics-left", False)
    assert "start" not in live.provider.commands
    auto.start("intrinsics-left", True)
    result = wait_for(auto, ("completed", "failed"))
    assert result["state"] == "completed", result
    assert len(live.rows) == 2
    assert live.provider.commands == ["start", "ack", "ack"]
    assert "_claim" not in auto.status()


def test_new_manual_sample_invalidates_reviewed_plan():
    live = FakeLive()
    auto = AutomaticCapture(live)
    auto.estimate("intrinsics-left")
    auto.plan("intrinsics-left")
    wait_for(auto, ("preview",))
    live.rows.append(dict(id="manual", charuco_ids=[], charuco_corners=[]))
    with pytest.raises(ValueError, match="核验"):
        auto.start("intrinsics-left", True)
    assert "start" not in live.provider.commands


def test_stop_does_not_wait_for_live_capture_lock():
    live = FakeLive()
    auto = AutomaticCapture(live)
    auto.estimate("intrinsics-left")
    auto.plan("intrinsics-left")
    wait_for(auto, ("preview",))
    done = threading.Event()
    with live.lock:
        thread = threading.Thread(target=lambda: (auto.stop(), done.set()))
        thread.start()
        assert done.wait(0.5)
    thread.join()
    assert live.provider.commands == ["stop"]


def test_camera_proposals_keep_extrinsics_board_visible():
    rig, data, _ = generate()
    points = Board(rig["board"]).object_points
    camera = np.linalg.inv(np.array(data["wrists"]["left"][0]["camera_from_target"]))
    intr = rig["cameras"]["left"]["model"]["intrinsics"]
    for kind in ("intrinsics", "extrinsics"):
        rows = candidate_views(np.eye(4), camera, points, intr, kind)
        assert len(rows) > 28
        assert all(
            r["visible_corners"] >= (12 if kind == "intrinsics" else len(points))
            for r in rows
        )
        for i, r in enumerate(rows):
            r["joints_deg"] = [i * 0.7, 0, 0, i * 0.2, 0, 0]
        selected = select_views(rows, 28)
        assert len(selected) == 28


def test_existing_corner_placement_is_not_planned_again():
    pixels = np.arange(48, dtype=float).reshape(24, 2)
    previous = [dict(charuco_ids=list(range(24)), charuco_corners=pixels.tolist())]
    assert not novel_view(pixels + 1, previous)
    assert novel_view(pixels + 20, previous)


def motion_fixture():
    @contextmanager
    def hardware():
        yield

    service = lambda state: SimpleNamespace(_hardware=hardware)
    motion = CalibrationMotion(None, None, service, None, None)
    operation = dict(done=threading.Event())
    state = SimpleNamespace(
        operations=SimpleNamespace(
            lock=threading.RLock(),
            calibrating=False,
            reconfiguring=False,
            jobs={"job": operation},
        )
    )
    job = dict(
        token="token",
        cancel=threading.Event(),
        ack=threading.Event(),
        public=dict(state="running", phase="awaiting_capture", index=1),
    )
    motion.jobs["job"] = job
    return motion, state, job


def test_capture_lease_excludes_other_operations_and_releases_on_error():
    motion, state, job = motion_fixture()
    with pytest.raises(RuntimeError):
        with motion.capture_lease(state, dict(id="job", token="token")):
            assert state.operations.calibrating and job["capture_active"]
            with pytest.raises(ValueError, match="尚未完成"):
                motion.acknowledge("job", "token", 1)
            raise RuntimeError("capture failed")
    assert not state.operations.calibrating and not job["capture_active"]
    state.operations.jobs["other"] = dict(done=threading.Event())
    with pytest.raises(ValueError, match="占用"):
        with motion.capture_lease(state, dict(id="job", token="token")):
            pass
    assert not state.operations.calibrating


def test_stale_wrong_or_unconfirmed_requests_never_reach_motion():
    motion, state, job = motion_fixture()
    for confirmed in (False, True):
        with pytest.raises(ValueError):
            motion.start(state, "job", "wrong", confirmed)
    job["public"].update(
        state="preview", created=time.time() - 121, max_speed_deg_s=30.0
    )
    with pytest.raises(ValueError, match="过期"):
        motion.start(state, "job", "token", True)
    with pytest.raises(ValueError, match="匹配"):
        motion.acknowledge("job", "token", 2)
    motion.stop("job", "token")
    assert job["cancel"].is_set()
    with pytest.raises(ValueError):
        with motion.capture_lease(state, dict(id="job", token="token")):
            pass


def test_owner_waits_for_saved_capture_and_cancels_last_point(monkeypatch):
    motion, state, job = motion_fixture()
    state.operations.stop = threading.Event()
    motion.check_camera_path = lambda *args: 1.0
    state.devices = object()
    arms = {a: dict(id=a, joints=[0.0] * 6) for a in ("left", "right")}
    motion.service = lambda state: SimpleNamespace(
        _arm=lambda task: copy.deepcopy(arms[task.split("-")[1]])
    )
    calls = []

    def plan(root, arm, other, **kwargs):
        assert kwargs["speed_deg_s"] == 30.0
        return dict(
            execution_allowed=True,
            duration_s=1,
            arm=arm["id"],
            goal_joints=kwargs["goal_joints"],
        )

    def execute(segment, devices, cancel, progress):
        assert not cancel.is_set()
        calls.append(segment["goal_joints"])
        arms[segment["arm"]]["joints"] = segment["goal_joints"]

    for name in ("server", "server.motion"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    home = types.ModuleType("server.motion.home")
    home.make_home_plan = plan
    orbit = types.ModuleType("server.motion.orbit")
    orbit.execute_plan = execute
    monkeypatch.setitem(sys.modules, "server.motion.home", home)
    monkeypatch.setitem(sys.modules, "server.motion.orbit", orbit)
    job.update(
        state=state,
        estimate=dict(seed=dict(flange_from_camera=np.eye(4).tolist())),
        request=dict(arm="left"),
        waypoints=[dict(joints_deg=[1.0] * 6), dict(joints_deg=[2.0] * 6)],
        segments=[
            dict(start_joints=[0.0] * 6, other_start_joints=[0.0] * 6),
            dict(start_joints=[1.0] * 6, other_start_joints=[0.0] * 6),
        ],
        operation=dict(done=threading.Event()),
    )
    thread = threading.Thread(target=motion._execute, args=("job", job))
    thread.start()

    def awaiting(index):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            s = motion.public("job")
            if s.get("phase") == "awaiting_capture" and s["index"] == index:
                return
            time.sleep(0.01)
        raise AssertionError(motion.public("job"))

    awaiting(1)
    assert len(calls) == 1 and not job["operation"]["done"].is_set()
    motion.acknowledge("job", "token", 1)
    awaiting(2)
    motion.stop("job", "token")
    thread.join(2)
    assert job["operation"]["done"].is_set()
    assert motion.public("job")["state"] == "cancelled"
    assert len(calls) == 2
