"""Camera-view proposals independent of robot brand and joint count."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_model import matrix
from .intrinsics import coverage_features
from .validation import transform


def view_features(camera_from_board, points, intrinsics, dataset):
    pose = transform(camera_from_board)
    xyz = points @ pose[:3, :3].T + pose[:3, 3]
    if np.min(xyz[:, 2]) < 0.15:
        return None
    xy = cv2.projectPoints(
        points,
        cv2.Rodrigues(pose[:3, :3])[0],
        pose[:3, 3],
        matrix(intrinsics),
        np.asarray(intrinsics["coeffs"]),
    )[0].reshape(-1, 2)
    margin = 12 if dataset == "intrinsics" else 28
    visible = np.all(
        (xy >= margin)
        & (xy < [intrinsics["width"] - margin, intrinsics["height"] - margin]),
        axis=1,
    )
    if visible.sum() < (12 if dataset == "intrinsics" else len(points)):
        return None
    return dict(
        visible_corners=int(visible.sum()),
        pixels=xy.tolist(),
        coverage=coverage_features(xy[visible], intrinsics).tolist(),
    )


def candidate_views(base_from_board, base_from_camera, points, intrinsics, dataset):
    """Orbit the observed board, vary distance/tilt/roll and image placement."""
    board, camera = transform(base_from_board), transform(base_from_camera)
    center = board[:3, :3] @ np.mean(points, axis=0) + board[:3, 3]
    vector = camera[:3, 3] - center
    distance = np.linalg.norm(vector)
    if not 0.18 <= distance <= 1.0:
        raise ValueError("板距离不适合自动采集，请先调整到 0.18–1 m")
    placements = (
        [(0.5, 0.5)]
        if dataset == "extrinsics"
        else [
            (x, y)
            for x, y in [
                (0.5, 0.5),
                (0.2, 0.2),
                (0.8, 0.8),
                (0.2, 0.8),
                (0.8, 0.2),
                (0.2, 0.5),
                (0.8, 0.5),
                (0.5, 0.2),
                (0.5, 0.8),
            ]
        ]
    )
    proposals = []
    for angle in (0, 12, -12, 24, -24):
        for tilt in (0, 12, -12, 24, -24):
            orbit = Rotation.from_rotvec(
                np.deg2rad(angle) * camera[:3, 0] + np.deg2rad(tilt) * camera[:3, 1]
            ).as_matrix()
            for scale in (1.0, 0.85, 1.15):
                position = center + orbit @ vector * scale
                z = center - position
                z /= np.linalg.norm(z)
                x = np.cross(camera[:3, 1], z)
                if np.linalg.norm(x) < 0.1:
                    continue
                x /= np.linalg.norm(x)
                rotation = np.column_stack((x, np.cross(z, x), z))
                for roll in (0, 18, -18):
                    for u, v in placements:
                        a = np.arctan(
                            (u * intrinsics["width"] - intrinsics["ppx"])
                            / intrinsics["fx"]
                        )
                        b = np.arctan(
                            (v * intrinsics["height"] - intrinsics["ppy"])
                            / intrinsics["fy"]
                        )
                        out = np.eye(4)
                        out[:3, 3] = position
                        out[:3, :3] = (
                            rotation
                            @ Rotation.from_euler("z", roll, degrees=True).as_matrix()
                            @ Rotation.from_euler("yx", [-a, b]).as_matrix()
                        )
                        feature = view_features(
                            np.linalg.inv(out) @ board, points, intrinsics, dataset
                        )
                        if feature:
                            proposals.append(
                                dict(base_from_camera=out.tolist(), **feature)
                            )
    return proposals


def select_views(candidates, count):
    """Balance image coverage, relative rotation and travel after IK screening."""
    selected, votes = [], np.zeros(13)
    remaining = list(candidates)
    while remaining and len(selected) < count:

        def score(c):
            q = np.asarray(c["joints_deg"])
            diversity = min(
                (np.linalg.norm(q - np.asarray(s["joints_deg"])) for s in selected),
                default=35,
            )
            return (
                float(np.sum(np.asarray(c["coverage"]) / (1 + votes)))
                + min(diversity, 40) / 8
            )

        best = max(range(len(remaining)), key=lambda i: score(remaining[i]))
        item = remaining.pop(best)
        if any(
            np.max(np.abs(np.subtract(item["joints_deg"], s["joints_deg"]))) < 2
            for s in selected
        ):
            continue
        selected.append(item)
        votes += item["coverage"]
    return selected


def novel_view(pixels, previous):
    """Do not spend an automatic point on an already saved image placement."""
    predicted = np.asarray(pixels, float)
    for old in previous:
        ids = np.asarray(old["charuco_ids"], int)
        if (
            len(ids) >= 12
            and np.sqrt(
                np.mean(
                    np.sum(
                        (predicted[ids] - np.asarray(old["charuco_corners"])) ** 2,
                        axis=1,
                    )
                )
            )
            < 12
        ):
            return False
    return True
