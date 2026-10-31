"""Deterministic synthetic observations; no robot or camera SDK required."""

import copy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .board import Board
from .camera_model import matrix
from .geometry import make_transform
from .workflow import digest


def generate(*, table=False, seed=17):
    rng = np.random.default_rng(seed)
    spec = dict(
        squares_x=7,
        squares_y=5,
        square_length_m=0.04,
        marker_length_m=0.028,
        dictionary="DICT_5X5_50",
    )
    points = Board(spec).object_points
    intr = dict(
        width=1280,
        height=720,
        fx=850.0,
        fy=850.0,
        ppx=640.0,
        ppy=360.0,
        model="opencv_brown_conrady",
        coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    models = {
        r: dict(
            serial="synthetic-" + r,
            source="factory",
            image_rotation_deg=0,
            intrinsics=copy.deepcopy(intr),
            factory_intrinsics=copy.deepcopy(intr),
        )
        for r in ("left", "right", "top")
    }
    rig = dict(
        schema_version=1,
        board=spec,
        arms={
            a: dict(model_id="synthetic-" + str(n) + "-dof")
            for a, n in (("left", 6), ("right", 7))
        },
        cameras={
            r: dict(
                model=m,
                mount="fixed" if r == "top" else r + "_flange",
                intrinsics_mode="reuse",
            )
            for r, m in models.items()
        },
        table=dict(enabled=table, thickness_mm=8.0, board_z_direction="into_table"),
    )
    xs = {
        a: make_transform(
            Rotation.from_rotvec([0.12, -0.08, 0.06]).as_matrix(), [0.04, 0, 0.08]
        )
        for a in ("left", "right")
    }
    hs = {
        a: make_transform(np.eye(3), [x, 0, -0.9])
        for a, x in (("left", 0.4), ("right", -0.4))
    }

    def camera_pose():
        return make_transform(
            Rotation.from_rotvec(rng.uniform(-0.45, 0.45, 3)).as_matrix(),
            [
                -0.14 + rng.uniform(-0.05, 0.05),
                -0.10 + rng.uniform(-0.04, 0.04),
                rng.uniform(0.7, 1.0),
            ],
        )

    def observe(role, pose, sid):
        xy = cv2.projectPoints(
            points,
            cv2.Rodrigues(pose[:3, :3])[0],
            pose[:3, 3],
            matrix(intr),
            np.zeros(5),
        )[0].reshape(-1, 2)
        return dict(
            id=sid,
            camera_frame=dict(serial=models[role]["serial"], image_rotation_deg=0),
            intrinsics=copy.deepcopy(intr),
            charuco_ids=list(range(len(points))),
            charuco_corners=xy.tolist(),
            camera_from_target=pose.tolist(),
            reprojection_error_px=0.0,
        )

    data = dict(
        schema_version=1,
        rig_sha256=digest(rig),
        wrists={},
        bridges={},
        validation=[],
        table=[],
    )
    for a in ("left", "right"):
        b = hs[a] @ make_transform(np.eye(3), [-0.14, -0.1, 0.85])
        data["wrists"][a] = []
        for i in range(20):
            c = camera_pose()
            f = b @ np.linalg.inv(c) @ np.linalg.inv(xs[a])
            data["wrists"][a].append(
                dict(
                    observe(a, c, f"wrist-{a}-{i}"),
                    base_from_flange=f.tolist(),
                    joint_angles_rad=[0.0] * (6 if a == "left" else 7),
                )
            )
        data["bridges"][a] = []
        for i in range(12):
            h, w = camera_pose(), camera_pose()
            f = hs[a] @ h @ np.linalg.inv(w) @ np.linalg.inv(xs[a])
            data["bridges"][a].append(
                dict(
                    observe("top", h, f"bridge-{a}-{i}"),
                    wrist=observe(a, w, f"reference-{a}-{i}"),
                    base_from_flange=f.tolist(),
                )
            )
    for kind in ("validation", "table"):
        for i in range(3):
            h = (
                camera_pose()
                if kind == "validation"
                else make_transform(np.eye(3), [-0.14, -0.1, 0.85])
            )
            sample = dict(
                id=f"{kind}-{i}",
                cameras={"top": observe("top", h, f"{kind}-top-{i}")},
                base_from_flange={},
            )
            for a in ("left", "right"):
                w = camera_pose()
                sample["cameras"][a] = observe(a, w, f"{kind}-{a}-{i}")
                sample["base_from_flange"][a] = (
                    hs[a] @ h @ np.linalg.inv(w) @ np.linalg.inv(xs[a])
                ).tolist()
            data[kind].append(sample)
    if not table:
        data["table"] = []
    return rig, data, dict(wrists=xs, bridges=hs)
