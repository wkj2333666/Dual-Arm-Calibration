import copy
import time

import cv2
import numpy as np
import pytest

from dual_arm_calibration.board import Board
from dual_arm_calibration.demo import generate
from dual_arm_calibration.integrations.robobench_bridge import (
    normalize_intrinsics,
    proxy_host,
)
from dual_arm_calibration.web import Workbench
from dual_arm_calibration.workflow import digest


class Provider:
    def __init__(self, rig):
        self.rig = rig
        self.status = dict(
            model_sha256="model",
            connected=True,
            busy=False,
            arms={
                a: dict(
                    serial=a,
                    base_from_flange=np.eye(4).tolist(),
                    joint_angles_rad=[0.0] * 6,
                )
                for a in ("left", "right")
            },
            cameras={
                r: dict(
                    serial=c["model"]["serial"],
                    intrinsics=c["model"]["factory_intrinsics"],
                )
                for r, c in rig["cameras"].items()
            },
        )
        self.image = np.full((720, 1280, 3), 255, np.uint8)
        board = Board(rig["board"]).board.generateImage((560, 400))
        self.image[160:560, 360:920] = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        self.moving = False

    def probe(self):
        return copy.deepcopy(self.status)

    def capture(self, roles, motion=None):
        packets, images = [], {}
        for i in (0, 1):
            packets.append(
                dict(
                    arms=self.status["arms"],
                    cameras={
                        r: dict(
                            self.status["cameras"][r],
                            image_rotation_deg=0,
                            frame_number=i + 1,
                            timestamp=time.time(),
                        )
                        for r in roles
                    },
                )
            )
            for r in roles:
                images[f"{i}-{r}"] = cv2.warpAffine(
                    self.image,
                    np.float32([[1, 0, 30 if i and self.moving else 0], [0, 1, 0]]),
                    (1280, 720),
                    borderValue=(255, 255, 255),
                )
        return dict(packets=packets, images=images, stable_s=0.6, timing="mock capture")


@pytest.fixture
def app(tmp_path):
    rig, _, _ = generate()
    config = dict(
        provider="robobench",
        base_url="http://127.0.0.1:1",
        board=rig["board"],
        board_thickness_mm=8,
    )
    app = Workbench(tmp_path, config)
    app.live.provider = Provider(rig)
    try:
        yield app
    finally:
        app.close()


def new(app):
    app.live.action(
        dict(
            op="new",
            board_confirmed=True,
            modes={r: "reuse" for r in ("left", "right", "top")},
            table=False,
        ),
        app,
    )


def test_hardware_requires_explicit_board_confirmation(app):
    with pytest.raises(ValueError, match="确认"):
        app.live.action(dict(op="new"), app)
    new(app)
    state = app.state()
    assert state["hardware_connected"]
    assert state["hardware"]["session"]["counts"]["wrists-left"] == 0
    assert state["draft"]["board"]["squares_x"] is None


def test_capture_saves_raw_pixels_and_rejects_duplicates(app):
    new(app)
    payload = dict(op="capture", batch="wrists-left", stationary_confirmed=True)
    app.live.action(payload, app)
    rows = app.live.session["data"]["wrists"]["left"]
    assert len(rows) == 1
    images = list(app.live.directory.glob("*/images/*.png"))
    assert len(images) == 1
    assert cv2.imread(str(images[0])).shape == (720, 1280, 3)
    with pytest.raises(ValueError, match="太接近"):
        app.live.action(payload, app)
    assert len(rows) == 1
    app.live.action(dict(op="reset", batch="wrists-left"), app)
    assert not rows
    assert list(app.live.directory.glob("*/previous-wrists-left-*.json"))
    assert images[0].exists()


def test_moving_board_missing_board_and_profile_change_are_rejected(app):
    new(app)
    app.live.provider.moving = True
    payload = dict(op="capture", batch="wrists-left", stationary_confirmed=True)
    with pytest.raises(ValueError, match="移动"):
        app.live.action(payload, app)
    app.live.provider.moving = False
    app.live.provider.image[:] = 255
    with pytest.raises(ValueError, match="角点"):
        app.live.action(payload, app)
    assert not app.live.session["data"]["wrists"]["left"]


@pytest.mark.parametrize("moved", [False, True])
def test_fixed_top_tracking_is_independent_of_wrist_seed_error(app, moved):
    from dual_arm_calibration.camera_model import matrix

    new(app)
    s = app.live.session
    intr = s["models"]["top"]["intrinsics"]
    board = Board(s["rig"]["board"])
    d, _ = board.detect(app.live.provider.image, matrix(intr), intr["coeffs"])
    reference = d.camera_from_target.copy()
    if moved:
        reference[0, 3] += 0.02
    wrong = np.eye(4)
    wrong[0, 3] = 0.2
    payload = dict(
        op="capture",
        batch="wrists-left",
        stationary_confirmed=True,
        motion=dict(id="mock", token="mock"),
        board_reference=np.eye(4).tolist(),
        planning_seed=wrong.tolist(),
        top_reference=dict(
            serial=s["models"]["top"]["serial"], camera_from_board=reference.tolist()
        ),
    )
    if moved:
        with pytest.raises(ValueError, match="顶相机检测到板移动"):
            app.live.action(payload, app)
        assert not s["data"]["wrists"]["left"]
    else:
        app.live.action(payload, app)
        row = s["data"]["wrists"]["left"][0]
        assert row["camera_frame"]["serial"] == s["models"]["left"]["serial"]
        assert (
            row["measurement"]["fixed_top_observation"]["camera_frame"]["serial"]
            == s["models"]["top"]["serial"]
        )


def test_stage_dependencies_and_frozen_results(app):
    new(app)
    with pytest.raises(ValueError, match="外参"):
        app.live.ready("bridges-left")
    with pytest.raises(ValueError, match="复用"):
        app.live.ready("intrinsics-left")
    _, data, _ = generate()
    data["intrinsics"] = {r: [] for r in ("left", "right", "top")}
    s = app.live.session
    data["rig_sha256"] = digest(s["rig"])
    s["data"] = data
    for stage in (
        "wrists-left",
        "wrists-right",
        "bridges-left",
        "bridges-right",
        "validation",
    ):
        app.live.action(dict(op="fit", batch=stage), app)
    with pytest.raises(ValueError, match="冻结"):
        app.live.ready("wrists-left")
    jid = app.live.action(dict(op="finish"), app)["job_id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = next(j for j in app.state()["jobs"] if j["id"] == jid)
        if job["status"] in ("failed", "succeeded"):
            break
        time.sleep(0.02)
    assert job["status"] == "succeeded", job
    assert job["kind"] == "hardware"
    assert s["sealed"]


def test_realsense_projection_models_are_explicit():
    intr = dict(
        width=1280,
        height=720,
        fx=850,
        fy=850,
        ppx=640,
        ppy=360,
        coeffs=[0] * 5,
        model=2,
    )
    meta = dict(intrinsics=intr, width=1280, height=720)
    assert normalize_intrinsics(meta)["model"] == "none"
    intr["coeffs"] = [0.1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="显式转换"):
        normalize_intrinsics(meta)
    intr["model"] = 4
    assert normalize_intrinsics(meta)["model"] == "opencv_brown_conrady"


def test_rewritten_host_accepts_only_browser_same_origin_evidence():
    headers = {
        "host": "127.0.0.1:8000",
        "origin": "http://127.0.0.1:5173",
        "sec-fetch-site": "same-origin",
    }
    assert proxy_host(headers) == "127.0.0.1:5173"
    headers["sec-fetch-site"] = "cross-site"
    with pytest.raises(ValueError, match="跨站"):
        proxy_host(headers)
