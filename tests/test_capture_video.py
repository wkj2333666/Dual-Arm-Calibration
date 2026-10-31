import time
import cv2
import numpy as np
import pytest
from dual_arm_calibration.adapters import ArmState, CameraFrame, load_adapter
from dual_arm_calibration.board import Board
from dual_arm_calibration.capture import capture_stationary
from dual_arm_calibration.demo import generate
from dual_arm_calibration.video import select_frames


def setup_image():
    rig, _, _ = generate()
    image = np.full((720, 1280, 3), 255, np.uint8)
    board = Board(rig["board"]).board.generateImage((560, 400))
    image[160:560, 360:920] = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    return rig, image


class Reader:
    def __init__(self, stationary=True):
        self.stationary = stationary

    def read_state(self):
        return ArmState(np.eye(4), (0.0,) * 6, time.monotonic(), self.stationary)


class Camera:
    def __init__(self, image, model, stale=False, frozen=False):
        self.image, self.model, self.stale, self.frozen = image, model, stale, frozen
        self.frame = 0

    def capture(self):
        if not self.frozen:
            self.frame += 1
        return CameraFrame(
            self.image,
            self.model["serial"],
            self.frame,
            time.monotonic() - (2 if self.stale else 0),
            self.model["factory_intrinsics"],
        )


def test_passive_capture_records_six_joint_robot():
    rig, image = setup_image()
    model = rig["cameras"]["left"]["model"]
    sample, images = capture_stationary(
        {"left": Reader()},
        {"left": Camera(image, model)},
        {"left": model},
        rig["board"],
    )
    assert len(sample["joint_angles_rad"]["left"]) == 6
    assert sample["pairing"]["stable_s"] >= 0.25
    assert len(sample["cameras"]["left"]["charuco_ids"]) >= 12
    assert images["left"].shape == image.shape


@pytest.mark.parametrize(
    "stationary,stale,frozen,match",
    [
        (False, False, False, "moving"),
        (True, True, False, "Stale"),
        (True, False, True, "advance"),
    ],
)
def test_capture_rejects_moving_stale_or_frozen(stationary, stale, frozen, match):
    rig, image = setup_image()
    model = rig["cameras"]["left"]["model"]
    with pytest.raises(ValueError, match=match):
        capture_stationary(
            {"left": Reader(stationary)},
            {"left": Camera(image, model, stale, frozen)},
            {"left": model},
            rig["board"],
        )


def test_video_decoding_and_duplicate_rejection(tmp_path):
    rig, image = setup_image()
    path = tmp_path / "board.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720)
    )
    assert writer.isOpened()
    for offset in (0, 0, 0, 40, 40, 40):
        frame = cv2.warpAffine(
            image,
            np.float32([[1, 0, offset], [0, 1, 0]]),
            (1280, 720),
            borderValue=(255, 255, 255),
        )
        writer.write(frame)
    writer.release()
    result = select_frames(path, rig["board"], rig["cameras"]["top"]["model"], stride=1)
    assert result["decoded_frames"] == 6
    assert len(result["samples"]) == 2
    assert result["coverage"][
        "issues"
    ]  # two distinct frames are not sufficient for lens fitting


def test_missing_adapter_is_explicit():
    with pytest.raises(ValueError, match="Install"):
        load_adapter("nonexistent-test-adapter")
