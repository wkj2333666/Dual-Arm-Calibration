"""Per-camera RGB lens fitting from immutable ChArUco observations.

Keep factory K fixed; fit Brown coefficients only when cross-validation improves.
No hardware writes or image undistortion: projections use the fitted forward model.
"""

import copy
import cv2
import numpy as np


def matrix(intr):
    return np.array(
        [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]], float
    )


def signature(intr):
    return {
        k: intr.get(k)
        for k in ("width", "height", "fx", "fy", "ppx", "ppy", "model", "coeffs")
    }


def valid_lens(intr):
    """Reject folded/nonfinite fitted maps over the entire image, not just corners."""
    K = matrix(intr)
    D = np.asarray(intr["coeffs"], float)
    xs = (np.linspace(0, intr["width"] - 1, 25) - intr["ppx"]) / intr["fx"]
    ys = (np.linspace(0, intr["height"] - 1, 17) - intr["ppy"]) / intr["fy"]
    xy = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T

    def project(p):
        return cv2.projectPoints(
            np.c_[p, np.ones(len(p))], np.zeros(3), np.zeros(3), K, D
        )[0].reshape(-1, 2)

    p = project(xy)
    dx = (project(xy + [1e-5, 0]) - p) / 1e-5
    dy = (project(xy + [0, 1e-5]) - p) / 1e-5
    return bool(
        np.isfinite(p).all()
        and np.isfinite(dx).all()
        and np.isfinite(dy).all()
        and np.all(dx[:, 0] > 0)
        and np.all(dy[:, 1] > 0)
        and np.all(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0] > 0)
    )


def fit_distortion(samples, points, intr, *, fit_model="brown5"):
    if fit_model not in ("brown5", "radial2"):
        raise ValueError("Unknown distortion fit model")
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_FOCAL_LENGTH
        | cv2.CALIB_FIX_PRINCIPAL_POINT
    )
    initial = np.asarray(intr.get("coeffs", [0.0] * 5), float).copy()
    if fit_model == "radial2":
        flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
        initial = np.zeros(5)
    objects = [np.asarray(points, np.float32)[s["charuco_ids"]] for s in samples]
    images = [np.asarray(s["charuco_corners"], np.float32) for s in samples]
    _, K, D, _, _ = cv2.calibrateCamera(
        objects,
        images,
        (intr["width"], intr["height"]),
        matrix(intr),
        initial,
        flags=flags,
    )
    fitted = copy.deepcopy(intr)
    fitted.update(model="opencv_brown_conrady", coeffs=D.ravel().tolist())
    if not valid_lens(fitted):
        raise ValueError("Fitted camera model folds outside observed board")
    return fitted


def apply_model(samples, points, intr):
    result = copy.deepcopy(samples)
    K = matrix(intr)
    D = np.asarray(intr.get("coeffs", [0.0] * 5), float)
    for s in result:
        obj = np.asarray(points, float)[s["charuco_ids"]]
        observed = np.asarray(s["charuco_corners"], float)
        ok, r, t = cv2.solvePnP(obj, observed, K, D)
        if not ok:
            raise ValueError("PnP failed with calibrated camera model")
        R = cv2.Rodrigues(r)[0]
        if np.min((obj @ R.T + t.ravel())[:, 2]) <= 0:
            raise ValueError("Board behind camera")
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        projected = cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
        s["camera_from_target"] = T.tolist()
        s["intrinsics"] = copy.deepcopy(intr)
        s["reprojection_error_px"] = float(
            np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1)))
        )
    return result


def pixel_rms(samples):
    return float(
        np.sqrt(
            np.average(
                [s["reprojection_error_px"] ** 2 for s in samples],
                weights=[len(s["charuco_ids"]) for s in samples],
            )
        )
    )


def packet_intrinsics(packet, model):
    if (
        packet.get("serial") != model["serial"]
        or packet.get("image_rotation_deg", 0) != model["image_rotation_deg"]
        or signature(packet["intrinsics"]) != signature(model["factory_intrinsics"])
    ):
        raise ValueError(
            "Camera profile does not match calibrated lens model; recalibrate before projection"
        )
    return copy.deepcopy(model["intrinsics"])
