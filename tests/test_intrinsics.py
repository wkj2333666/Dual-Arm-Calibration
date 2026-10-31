import copy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from dual_arm_calibration.board import Board
from dual_arm_calibration.camera_model import matrix
from dual_arm_calibration.demo import generate
from dual_arm_calibration.intrinsics import solve, coverage
from dual_arm_calibration.workflow import solve as solve_workflow, digest


def lens_views(rig, distortion):
    rng = np.random.default_rng(2026)
    points = Board(rig["board"]).object_points
    model = rig["cameras"]["left"]["model"]
    intr = model["factory_intrinsics"]
    samples = []
    for index in range(500):
        r = rng.uniform(-0.7, 0.7, 3)
        z = rng.uniform(0.35, 0.75)
        center = [rng.uniform(100, 1180), rng.uniform(60, 660)]
        t = np.array(
            [
                (center[0] - intr["ppx"]) / intr["fx"] * z - 0.14,
                (center[1] - intr["ppy"]) / intr["fy"] * z - 0.1,
                z,
            ]
        )
        xy = cv2.projectPoints(points, r, t, matrix(intr), np.array(distortion))[
            0
        ].reshape(-1, 2)
        ids = np.flatnonzero(np.all((xy >= 0) & (xy < [1280, 720]), axis=1))
        if len(ids) < 12:
            continue
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = Rotation.from_rotvec(r).as_matrix(), t
        samples.append(
            dict(
                id=f"lens-{index}",
                camera_frame=dict(serial=model["serial"], image_rotation_deg=0),
                intrinsics=copy.deepcopy(intr),
                charuco_ids=ids.tolist(),
                charuco_corners=xy[ids].tolist(),
                camera_from_target=pose.tolist(),
            )
        )
        if len(samples) >= 28 and not coverage(samples, intr)["issues"]:
            return samples
    raise AssertionError("Synthetic view generator failed coverage")


def test_radial2_recovers_distortion_without_changing_factory_K():
    rig, _, _ = generate()
    samples = lens_views(rig, [0.08, -0.12, 0.0, 0.0, 0.0])
    result = solve(
        samples,
        Board(rig["board"]).object_points,
        camera_family="realsense_d435i",
        stream="rgb",
    )
    assert result["gate_passed"]
    fitted, factory = result["intrinsics"], result["factory_intrinsics"]
    for key in ("fx", "fy", "ppx", "ppy"):
        assert fitted[key] == factory[key]
    np.testing.assert_allclose(
        fitted["coeffs"], [0.08, -0.12, 0.0, 0.0, 0.0], atol=1e-4
    )


def test_workflow_can_fit_one_camera_and_reuse_the_other_two():
    rig, data, _ = generate()
    rig["cameras"]["left"].update(
        intrinsics_mode="fit", family="realsense_d435i", stream="rgb"
    )
    data["intrinsics"] = {"left": lens_views(rig, [0.0, 0.0, 0.0, 0.0, 0.0])}
    data["rig_sha256"] = digest(rig)
    result = solve_workflow(rig, data)
    assert result["quality_passed"]
    assert result["cameras"]["left"]["model"]["source"] == "independent_intrinsics"
    assert result["cameras"]["right"]["model"]["source"] == "factory"
    assert not result["table"]["calibrated"]
