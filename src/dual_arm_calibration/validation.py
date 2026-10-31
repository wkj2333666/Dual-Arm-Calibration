"""Validate observations at the hardware-independent boundary."""

import numpy as np
from scipy.spatial.transform import Rotation
from .camera_model import signature


def transform(value):
    m = np.asarray(value, dtype=float)
    if (
        m.shape != (4, 4)
        or not np.isfinite(m).all()
        or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("Expected a finite SE(3) transform in metres")
    return m


def camera_model(model):
    if (
        model.get("source") == "independent_intrinsics"
        and model.get("gate_passed") is not True
    ):
        raise ValueError("Independently fitted intrinsics did not pass validation")
    if not model.get("serial") or model.get("source") not in (
        "factory",
        "reused",
        "independent_intrinsics",
    ):
        raise ValueError("Camera model requires serial and explicit parameter source")
    if model.get("image_rotation_deg", 0) not in (0, 90, 180, 270):
        raise ValueError("Invalid image rotation")
    for intr in (model["intrinsics"], model["factory_intrinsics"]):
        vals = [intr[k] for k in ("width", "height", "fx", "fy", "ppx", "ppy")]
        if not np.isfinite(vals).all() or min(vals[:4]) <= 0:
            raise ValueError("Invalid camera dimensions or intrinsic matrix")
        if any(int(intr[k]) != intr[k] for k in ("width", "height")):
            raise ValueError("Image dimensions must be integers")
        coeffs = np.asarray(intr.get("coeffs", []), float)
        if (
            coeffs.shape not in ((4,), (5,), (8,), (12,), (14,))
            or not np.isfinite(coeffs).all()
        ):
            raise ValueError("Invalid OpenCV distortion coefficients")
        zero_inverse = intr.get("model") in (
            "inverse_brown_conrady",
            "distortion.inverse_brown_conrady",
        ) and not np.any(coeffs)
        if (
            intr.get("model") not in ("opencv_brown_conrady", "brown_conrady", "none")
            and not zero_inverse
        ):
            raise ValueError(
                "Convert the camera model to OpenCV forward Brown before solving"
            )
        if intr["model"] == "none" and np.any(coeffs):
            raise ValueError("Undistorted model must have zero coefficients")
    a, b = model["intrinsics"], model["factory_intrinsics"]
    if (a["width"], a["height"]) != (b["width"], b["height"]):
        raise ValueError("Fitted and raw profiles differ in image dimensions")


def observations(samples, points, model, *, arm=False, minimum=1):
    camera_model(model)
    if len(samples) < minimum:
        raise ValueError(f"At least {minimum} observations required")
    ids = [s["id"] for s in samples]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Observation IDs must be nonempty and unique")
    for s in samples:
        f = s["camera_frame"]
        if (
            f["serial"] != model["serial"]
            or f.get("image_rotation_deg", 0) != model.get("image_rotation_deg", 0)
            or signature(s["intrinsics"]) != signature(model["factory_intrinsics"])
        ):
            raise ValueError("Observation does not match the camera/profile binding")
        ii = np.asarray(s["charuco_ids"])
        xy = np.asarray(s["charuco_corners"], float)
        if (
            ii.ndim != 1
            or len(ii) < 6
            or not np.issubdtype(ii.dtype, np.integer)
            or len(set(ii.tolist())) != len(ii)
            or ii.min() < 0
            or ii.max() >= len(points)
            or xy.shape != (len(ii), 2)
            or not np.isfinite(xy).all()
        ):
            raise ValueError("Invalid ChArUco corners or IDs")
        intr = model["factory_intrinsics"]
        if np.any(xy < 0) or np.any(xy >= [intr["width"], intr["height"]]):
            raise ValueError("Corners fall outside the bound image profile")
        transform(s["camera_from_target"])
        if arm:
            transform(s["base_from_flange"])
            if "joint_angles_rad" in s:
                q = np.asarray(s["joint_angles_rad"], float)
                if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
                    raise ValueError(
                        "Invalid measured joints; no fixed joint count is assumed"
                    )


def rotation_coverage(samples):
    rotations = [
        Rotation.from_matrix(transform(s["base_from_flange"])[:3, :3]) for s in samples
    ]
    vectors = np.array([(r * rotations[0].inv()).as_rotvec() for r in rotations])
    singular = np.linalg.svd(vectors, compute_uv=False)
    if len(singular) < 2 or singular[1] < 0.25:
        raise ValueError(
            "Insufficient rotation coverage: rotate about two different axes"
        )
    return singular.tolist()
