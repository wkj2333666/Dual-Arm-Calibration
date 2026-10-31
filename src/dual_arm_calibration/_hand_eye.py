"""Hand-eye solvers with an end-to-end reprojection check.

Two geometries:

  eye_in_hand   camera bolted to the flange, board fixed in the world.
                unknown X = flange_from_camera, invariant = base_from_target
  eye_to_hand   camera fixed in the world, board bolted to the flange.
                unknown X = base_from_camera,  invariant = flange_from_target

cv2.calibrateHandEye solves AX=XB and always returns "cam2gripper". Feeding it
flange_from_base instead of base_from_flange turns that same output into
base_from_camera, which is the eye_to_hand unknown.

Consistency and reprojection residuals measure different aspects of the fit.
Neither replaces measuring the physical board or checking pose observability.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import (
    invert,
    make_transform,
    mean_rotation,
    rotation_error_deg,
)

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


class SolveError(RuntimeError):
    pass


def _as_array(value):
    return np.asarray(value, dtype=np.float64)


def _invariant(sample, transform, kind):
    """The quantity that must be identical for every sample."""
    base_from_flange = _as_array(sample["base_from_flange"])
    camera_from_target = _as_array(sample["camera_from_target"])
    if kind == "eye_in_hand":  # base_from_target
        return base_from_flange @ transform @ camera_from_target
    return (
        invert(base_from_flange) @ transform @ camera_from_target
    )  # flange_from_target


def _predict_camera_from_target(sample, transform, invariant, kind):
    base_from_flange = _as_array(sample["base_from_flange"])
    if kind == "eye_in_hand":
        return invert(transform) @ invert(base_from_flange) @ invariant
    return invert(transform) @ base_from_flange @ invariant


def _mean_transform(transforms):
    translation = np.mean([t[:3, 3] for t in transforms], axis=0)
    return make_transform(mean_rotation(transforms), translation)


def _reprojection(sample, board_points, predicted) -> float:
    """RMS pixel error of the solved chain against the corners actually seen."""
    ids = sample.get("charuco_ids")
    corners = sample.get("charuco_corners")
    if ids is None or corners is None or not len(ids):
        return float("nan")
    intr = sample["intrinsics"]
    camera_matrix = np.array(
        [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]],
        dtype=np.float64,
    )
    distortion = np.asarray(intr.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    object_points = np.asarray(board_points, dtype=np.float64)[
        np.asarray(ids, dtype=int)
    ]
    rvec, _ = cv2.Rodrigues(predicted[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points, rvec, predicted[:3, 3], camera_matrix, distortion
    )
    observed = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    return float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - observed) ** 2, axis=1)))
    )


def evaluate(samples, transform, kind, board_points=None, invariant=None) -> dict:
    invariants = [_invariant(s, transform, kind) for s in samples]
    mean = _mean_transform(invariants) if invariant is None else invariant
    residuals = []
    for sample, estimate in zip(samples, invariants):
        entry = {
            "id": sample["id"],
            "translation_mm": float(
                np.linalg.norm(estimate[:3, 3] - mean[:3, 3]) * 1000.0
            ),
            "rotation_deg": rotation_error_deg(mean, estimate),
            "detect_reprojection_px": float(sample.get("reprojection_error_px", 0.0)),
        }
        if board_points is not None:
            entry["chain_reprojection_px"] = _reprojection(
                sample,
                board_points,
                _predict_camera_from_target(sample, transform, mean, kind),
            )
        residuals.append(entry)

    def rms(key):
        values = [r[key] for r in residuals if not math.isnan(r.get(key, float("nan")))]
        return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")

    return {
        "invariant": mean,
        "residuals": residuals,
        "translation_rms_mm": rms("translation_mm"),
        "rotation_rms_deg": rms("rotation_deg"),
        "reprojection_rms_px": rms("chain_reprojection_px"),
    }


def _solve_once(samples, kind, board_points=None):
    base_from_flange = [_as_array(s["base_from_flange"]) for s in samples]
    camera_from_target = [_as_array(s["camera_from_target"]) for s in samples]
    # eye_to_hand feeds the inverted arm poses; that is the whole difference.
    arm = (
        base_from_flange
        if kind == "eye_in_hand"
        else [invert(m) for m in base_from_flange]
    )

    best = None
    for name, method in METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [m[:3, :3] for m in arm],
                [m[:3, 3].reshape(3, 1) for m in arm],
                [m[:3, :3] for m in camera_from_target],
                [m[:3, 3].reshape(3, 1) for m in camera_from_target],
                method=method,
            )
        except cv2.error:
            continue
        transform = make_transform(rotation, np.asarray(translation).reshape(3))
        if not np.isfinite(transform).all():
            continue
        report = evaluate(samples, transform, kind, board_points)
        score = report["translation_rms_mm"] + 2.0 * report["rotation_rms_deg"]
        if not math.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, name, transform, report)
    if best is None:
        raise SolveError("求解失败：所有方法都不收敛，通常是姿态变化不足")
    return best[1], best[2], best[3]


def _pack(transform, invariant):
    rvec_x, _ = cv2.Rodrigues(transform[:3, :3])
    rvec_i, _ = cv2.Rodrigues(invariant[:3, :3])
    return np.concatenate(
        [rvec_x.reshape(3), transform[:3, 3], rvec_i.reshape(3), invariant[:3, 3]]
    )


def _unpack(vector):
    rot_x, _ = cv2.Rodrigues(vector[0:3].reshape(3, 1))
    rot_i, _ = cv2.Rodrigues(vector[6:9].reshape(3, 1))
    return (make_transform(rot_x, vector[3:6]), make_transform(rot_i, vector[9:12]))


def refine(
    samples, transform, invariant, kind, board_points, *, require_pose_improvement=True
):
    """Least-squares refinement of X and the invariant against pixel residuals.

    The closed-form methods minimise a rotation/translation algebraic cost, not
    reprojection error. Legacy callers also require PnP pose consistency to
    improve. Set require_pose_improvement=False when independently validating
    the pixel fit, so noisy per-frame PnP is not treated as ground truth.
    Falls back if scipy is unavailable.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return transform, invariant, False

    usable = [
        s
        for s in samples
        if s.get("charuco_ids") is not None and s.get("charuco_corners") is not None
    ]
    if len(usable) < 3:
        return transform, invariant, False

    board_points = np.asarray(board_points, dtype=np.float64)

    def residual(vector):
        x_transform, inv_transform = _unpack(vector)
        errors = []
        for sample in usable:
            predicted = _predict_camera_from_target(
                sample, x_transform, inv_transform, kind
            )
            ids = np.asarray(sample["charuco_ids"], dtype=int)
            intr = sample["intrinsics"]
            camera_matrix = np.array(
                [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]],
                dtype=np.float64,
            )
            distortion = np.asarray(
                intr.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64
            )
            rvec, _ = cv2.Rodrigues(predicted[:3, :3])
            projected, _ = cv2.projectPoints(
                board_points[ids], rvec, predicted[:3, 3], camera_matrix, distortion
            )
            observed = np.asarray(sample["charuco_corners"], dtype=np.float64).reshape(
                -1, 2
            )
            errors.append((projected.reshape(-1, 2) - observed).ravel())
        return np.concatenate(errors)

    start = _pack(transform, invariant)
    try:
        result = least_squares(residual, start, method="lm", max_nfev=400)
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return transform, invariant, False
    if not result.success or not np.isfinite(result.x).all():
        return transform, invariant, False
    if np.linalg.norm(residual(result.x)) > np.linalg.norm(residual(start)):
        return transform, invariant, False
    refined_x, refined_invariant = _unpack(result.x)
    if np.linalg.norm(refined_x[:3, 3] - transform[:3, 3]) > 0.20:
        return transform, invariant, False  # ran away; keep the closed-form answer
    before = evaluate(samples, transform, kind, board_points, invariant=invariant)
    after = evaluate(
        samples, refined_x, kind, board_points, invariant=refined_invariant
    )
    # Compare against the same samples and explicit invariant. A lower pixel
    # cost alone can hide a large deterioration in the measured 3-D chain.
    # The independently validated pixel fit does not treat per-image PnP
    # rotations as ground truth. Legacy callers keep their previous policy.
    metrics = (
        ("translation_rms_mm", "rotation_rms_deg", "reprojection_rms_px")
        if require_pose_improvement
        else ("reprojection_rms_px",)
    )
    for sample in usable:
        predicted = _predict_camera_from_target(
            sample, refined_x, refined_invariant, kind
        )
        obj = board_points[np.asarray(sample["charuco_ids"], int)]
        if np.min((obj @ predicted[:3, :3].T + predicted[:3, 3])[:, 2]) <= 0:
            return transform, invariant, False
    for metric in metrics:
        if not math.isfinite(after[metric]) or after[metric] > before[metric] + 1e-6:
            return transform, invariant, False
    return refined_x, refined_invariant, True


def solve(
    samples,
    kind,
    board_points=None,
    min_samples=10,
    do_refine=True,
    strict_outliers=False,
) -> dict:
    if kind not in ("eye_in_hand", "eye_to_hand"):
        raise ValueError("kind 必须是 eye_in_hand 或 eye_to_hand")
    usable = [
        s
        for s in samples
        if s.get("included", True)
        and s.get("base_from_flange") is not None
        and s.get("camera_from_target") is not None
    ]
    if len(usable) < min_samples:
        raise SolveError(f"至少需要 {min_samples} 个有效样本，当前 {len(usable)} 个")

    method, transform, report = _solve_once(usable, kind, board_points)

    # Reject outliers on a robust (median + 3*MAD) band, with a floor so a very
    # tight set does not start discarding perfectly good samples.
    def limit(key, floor):
        values = np.array([r[key] for r in report["residuals"]], dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(floor, median + 3.0 * mad)

    translation_limit = limit("translation_mm", 3.0)
    rotation_limit = limit("rotation_deg", 1.5)
    outliers = [
        r["id"]
        for r in report["residuals"]
        if r["translation_mm"] > translation_limit
        or r["rotation_deg"] > rotation_limit
        or r["detect_reprojection_px"] > 2.0
    ]
    kept = [s for s in usable if s["id"] not in set(outliers)]
    if strict_outliers and outliers and len(kept) < min_samples:
        raise SolveError(
            f"剔除异常样本后仅剩 {len(kept)} 个，至少需要 {min_samples} 个；请补采。异常样本：{', '.join(map(str, outliers))}"
        )
    if outliers and len(kept) >= min_samples:
        method, transform, report = _solve_once(kept, kind, board_points)
    else:
        outliers, kept = [], usable

    refined = False
    if do_refine and board_points is not None:
        transform, invariant, refined = refine(
            kept, transform, report["invariant"], kind, board_points
        )
        if refined:
            report = evaluate(kept, transform, kind, board_points, invariant=invariant)

    name = "flange_from_camera" if kind == "eye_in_hand" else "base_from_camera"
    inverse_name = "camera_from_flange" if kind == "eye_in_hand" else "camera_from_base"
    invariant_name = (
        "base_from_target" if kind == "eye_in_hand" else "flange_from_target"
    )
    return {
        "kind": kind,
        "method": method,
        "refined": refined,
        "used_samples": len(kept),
        "used_sample_ids": [s["id"] for s in kept],
        "outlier_ids": outliers,
        name: transform.tolist(),
        inverse_name: invert(transform).tolist(),
        invariant_name: report["invariant"].tolist(),
        "translation_rms_mm": report["translation_rms_mm"],
        "rotation_rms_deg": report["rotation_rms_deg"],
        "reprojection_rms_px": report["reprojection_rms_px"],
        "residuals": report["residuals"],
    }
