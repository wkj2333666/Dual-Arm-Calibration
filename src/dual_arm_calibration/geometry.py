"""Rigid transforms and pose-diversity metrics.

Naming: T_a_b / a_from_b maps a homogeneous point expressed in frame b into
frame a.
"""

from __future__ import annotations

import math

import numpy as np


def make_transform(rotation, translation) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64)
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def invert(matrix) -> np.ndarray:
    """Inverse of a rigid transform, without a general matrix inverse."""
    m = np.asarray(matrix, dtype=np.float64)
    rotation = m[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ m[:3, 3]
    return out


def rotation_angle_deg(rotation) -> float:
    """Magnitude of a rotation matrix, in degrees."""
    cos = (np.trace(np.asarray(rotation, dtype=np.float64)[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def rotation_error_deg(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)[:3, :3]
    right = np.asarray(right, dtype=np.float64)[:3, :3]
    return rotation_angle_deg(left.T @ right)


def transform_error(left, right) -> tuple[float, float]:
    """(translation mm, rotation deg) between two transforms."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return (
        float(np.linalg.norm(left[:3, 3] - right[:3, 3]) * 1000.0),
        rotation_error_deg(left, right),
    )


def mean_rotation(rotations) -> np.ndarray:
    """Chordal mean of rotations, projected back onto SO(3) via SVD."""
    stacked = np.mean(
        [np.asarray(r, dtype=np.float64)[:3, :3] for r in rotations], axis=0
    )
    u, _, vt = np.linalg.svd(stacked)
    result = u @ vt
    if np.linalg.det(result) < 0:  # reflection, not a rotation
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def rotation_axis(rotation) -> np.ndarray | None:
    """Unit rotation axis, or None when the rotation is too small to define one."""
    m = np.asarray(rotation, dtype=np.float64)[:3, :3]
    angle = math.radians(rotation_angle_deg(m))
    if angle < math.radians(1.0):
        return None
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # 180 deg rotation: recover the axis from the symmetric part
        values, vectors = np.linalg.eigh(m + np.eye(3))
        axis = vectors[:, int(np.argmax(values))]
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return None
    return axis / norm


def relative_rotations(transforms) -> list[np.ndarray]:
    """Every pairwise relative rotation in the set."""
    out = []
    for i in range(len(transforms)):
        a = np.asarray(transforms[i], dtype=np.float64)[:3, :3]
        for j in range(i + 1, len(transforms)):
            b = np.asarray(transforms[j], dtype=np.float64)[:3, :3]
            out.append(a.T @ b)
    return out


def axis_spread_deg(transforms) -> float:
    """Largest angle between any two relative-rotation axes in the set.

    Hand-eye calibration is only determined when the motions rotate about at
    least two non-parallel axes. A set of poses that all pivot about the same
    joint gives a small spread here and an under-determined solve, no matter
    how many samples it contains.
    """
    axes = []
    for rotation in relative_rotations(transforms):
        if rotation_angle_deg(rotation) < 5.0:
            continue
        axis = rotation_axis(rotation)
        if axis is not None:
            axes.append(axis)
    if len(axes) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            cos = abs(float(np.dot(axes[i], axes[j])))  # sign of an axis is arbitrary
            worst = max(worst, math.degrees(math.acos(min(1.0, cos))))
    return worst


def min_rotation_to_set(candidate, transforms) -> float:
    """Smallest rotation between a candidate pose and any pose already kept."""
    if not transforms:
        return 180.0
    return min(rotation_error_deg(candidate, item) for item in transforms)
