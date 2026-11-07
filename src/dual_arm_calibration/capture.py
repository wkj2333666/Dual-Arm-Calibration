"""Passive stationary capture for any adapters. Never commands robot motion."""

import time
import uuid
import numpy as np
from .board import Board
from .camera_model import matrix, packet_intrinsics
from .geometry import transform_error
from .validation import transform


def detect(frame, board, model):
    intr = packet_intrinsics(
        dict(
            serial=frame.serial,
            image_rotation_deg=frame.image_rotation_deg,
            intrinsics=frame.intrinsics,
        ),
        model,
    )
    if frame.bgr.shape[:2] != (intr["height"], intr["width"]):
        raise ValueError("Image dimensions differ from camera profile")
    d, _ = board.detect(frame.bgr, matrix(intr), intr["coeffs"])
    if not d.detected or d.corner_count < 12 or d.reprojection_error_px > 1.5:
        raise ValueError("Board needs at least 12 clear corners with error <= 1.5 px")
    return dict(
        camera_frame=dict(
            serial=frame.serial,
            image_rotation_deg=frame.image_rotation_deg,
            frame_number=frame.frame_number,
            host_monotonic_s=frame.host_monotonic_s,
        ),
        intrinsics=frame.intrinsics,
        charuco_ids=d.charuco_ids.tolist(),
        charuco_corners=d.charuco_corners.tolist(),
        camera_from_target=d.camera_from_target.tolist(),
        reprojection_error_px=d.reprojection_error_px,
    )


def board_corner_rms(ids_a, corners_a, ids_b, corners_b, minimum=12):
    """Whole-board image displacement in pixels; shared by stationary captures."""
    common = sorted(set(ids_a) & set(ids_b))
    if len(common) < minimum:
        raise ValueError("Too few common corners for stability check")
    a = {int(i): point for i, point in zip(ids_a, corners_a)}
    b = {int(i): point for i, point in zip(ids_b, corners_b)}
    diff = np.asarray([np.subtract(a[int(i)], b[int(i)]) for i in common])
    return float(np.sqrt(np.mean(np.sum(np.square(diff), axis=1))))


def capture_stationary(
    arms, cameras, models, board_spec, *, stable_s=0.25, timeout_s=3.0
):
    """Software pairing on one host clock; adapters must use time.monotonic().

    Returns generic arms/cameras observations plus raw BGR images for archival.
    A fixed-board wrist batch must keep the physical board fixed between calls.
    """
    if (
        not np.isfinite([stable_s, timeout_s]).all()
        or stable_s < 0.25
        or timeout_s <= stable_s
    ):
        raise ValueError("Invalid stationary capture window")
    if not cameras or set(cameras) != set(models):
        raise ValueError("Supply a model for every selected camera")
    board, start, anchor, previous = Board(board_spec), time.monotonic(), None, None
    while time.monotonic() - start < timeout_s:
        states = {r: a.read_state() for r, a in arms.items()}
        frames = {r: c.capture() for r, c in cameras.items()}
        after = {r: a.read_state() for r, a in arms.items()}
        now = time.monotonic()
        for item in [*states.values(), *frames.values(), *after.values()]:
            if (
                not np.isfinite(item.host_monotonic_s)
                or not 0 <= now - item.host_monotonic_s <= 0.5
            ):
                raise ValueError("Stale frame/state or mismatched monotonic clock")
        if (
            max(f.host_monotonic_s for f in frames.values())
            - min(f.host_monotonic_s for f in frames.values())
            > 0.05
        ):
            raise ValueError("Camera receipt skew exceeds 50 ms")
        obs = {r: detect(f, board, models[r]) for r, f in frames.items()}
        for r, state in states.items():
            if not state.stationary or not after[r].stationary:
                raise ValueError("Arm is moving")
            transform(state.base_from_flange)
            transform(after[r].base_from_flange)
            mm, deg = transform_error(state.base_from_flange, after[r].base_from_flange)
            if mm > 1 or deg > 0.2:
                raise ValueError("Arm moved during capture")
        if anchor is None:
            anchor = (now, states, obs)
        else:
            for r, o in obs.items():
                if frames[r].frame_number <= previous[r].frame_number:
                    raise ValueError("Camera frame did not advance")
                a = anchor[2][r]
                if board_corner_rms(a["charuco_ids"], a["charuco_corners"],
                                    o["charuco_ids"], o["charuco_corners"]) > 1.5:
                    raise ValueError("Board is moving")
            for r, state in states.items():
                mm, deg = transform_error(
                    anchor[1][r].base_from_flange, state.base_from_flange
                )
                if mm > 1 or deg > 0.2:
                    raise ValueError("Arm did not remain stationary")
            if now - anchor[0] >= stable_s:
                return dict(
                    id=uuid.uuid4().hex,
                    cameras=obs,
                    base_from_flange={
                        r: np.asarray(s.base_from_flange).tolist()
                        for r, s in states.items()
                    },
                    joint_angles_rad={
                        r: list(s.joint_angles_rad) for r, s in states.items()
                    },
                    pairing=dict(
                        stable_s=now - anchor[0],
                        timing="host receipt; not exposure synchronization",
                    ),
                ), {r: f.bgr for r, f in frames.items()}
        previous = frames
        time.sleep(0.04)
    raise ValueError("Stationary capture timed out")
