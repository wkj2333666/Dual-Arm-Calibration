"""Fixed-camera pixel fit extracted from the validated paired-observation workflow."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import mean_rotation
from .camera_model import matrix


def fit(rows, points, intr):
    # Fit only the fixed head pose. Lens and measured wrist reference stay fixed.
    seed = np.eye(4)
    seed[:3, :3] = mean_rotation([r["estimate"][:3, :3] for r in rows])
    seed[:3, 3] = np.mean([r["estimate"][:3, 3] for r in rows], axis=0)
    seed = np.linalg.inv(seed)
    initial = np.r_[Rotation.from_matrix(seed[:3, :3]).as_rotvec(), seed[:3, 3]]
    worlds = [
        (
            points[r["ids"]] @ r["world_from_board"][:3, :3].T
            + r["world_from_board"][:3, 3]
        )
        for r in rows
    ]

    def residual(x):
        return np.concatenate(
            [
                (
                    cv2.projectPoints(
                        w, x[:3], x[3:], matrix(intr), np.asarray(intr["coeffs"])
                    )[0].reshape(-1, 2)
                    - r["corners"]
                ).ravel()
                for r, w in zip(rows, worlds)
            ]
        )

    result = least_squares(residual, initial, loss="soft_l1", f_scale=1.0, max_nfev=200)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("顶上相机外参求解未收敛，请检查成对观察")
    C = np.eye(4)
    C[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    C[:3, 3] = result.x[3:]
    if any(np.min((w @ C[:3, :3].T + C[:3, 3])[:, 2]) <= 0 for w in worlds):
        raise ValueError("求解出的标定板在顶上相机后方")
    return np.linalg.inv(C)


def errors(rows, H, points, intr):
    values = []
    for r in rows:
        C = np.linalg.inv(H) @ r["world_from_board"]
        pred = cv2.projectPoints(
            points[r["ids"]],
            cv2.Rodrigues(C[:3, :3])[0],
            C[:3, 3],
            matrix(intr),
            np.asarray(intr["coeffs"]),
        )[0].reshape(-1, 2)
        values.append(
            dict(
                id=r["id"],
                chain_reprojection_px=float(
                    np.sqrt(np.mean(np.sum((pred - r["corners"]) ** 2, axis=1)))
                ),
            )
        )
    return values
