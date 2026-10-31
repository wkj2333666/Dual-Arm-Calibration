"""Independent RGB intrinsic calibration. No robot pose or motor interface.

For wrist cameras the board stays fixed and the arm moves; for the overhead
camera the camera stays fixed and the board moves. These camera-only samples
must never be used as fixed-table hand-eye poses. Validation thresholds are project acceptance rules,
not a claim about manufacturer's accuracy.
"""

import copy
import cv2
import numpy as np
from .camera_model import matrix, signature, fit_distortion, apply_model, pixel_rms

MIN_VIEWS = 20
CELL_MIN = 2
EDGE_MIN = 3
EDGE_FRACTION = 0.15


def coverage_features(corners, intr):
    """One vote per image per cell/edge, shared by planning and validation."""
    xy = np.asarray(corners, float).reshape(-1, 2) / [intr["width"], intr["height"]]
    cells = np.zeros(9, int)
    if len(xy):
        occupied = np.clip((xy * 3).astype(int), 0, 2)
        cells[np.unique(occupied[:, 1] * 3 + occupied[:, 0])] = 1
    return np.r_[
        cells,
        [
            np.any(xy[:, 0] < EDGE_FRACTION),
            np.any(xy[:, 0] > 1 - EDGE_FRACTION),
            np.any(xy[:, 1] < EDGE_FRACTION),
            np.any(xy[:, 1] > 1 - EDGE_FRACTION),
        ],
    ]


def coverage(samples, intr):
    cells = np.zeros((3, 3), int)
    edges = np.zeros(4, int)
    normals = []
    for s in samples:
        features = coverage_features(s["charuco_corners"], intr)
        cells += features[:9].reshape(3, 3)
        edges += features[9:]
        n = np.asarray(s["camera_from_target"])[:3, 2]
        normals.append(n if n[2] >= 0 else -n)
    spread = (
        np.linalg.svd(np.asarray(normals) - np.mean(normals, axis=0), compute_uv=False)
        if len(normals) > 2
        else np.zeros(3)
    )
    issues = []
    if len(samples) < MIN_VIEWS:
        issues.append(f"还需 {MIN_VIEWS - len(samples)} 张不同姿态的图像")
    if np.any(cells < CELL_MIN):
        issues.append(f"九宫格每区需要至少 {CELL_MIN} 张图的角点覆盖")
    for name, n in zip(("左边缘", "右边缘", "上边缘", "下边缘"), edges):
        if n < EDGE_MIN:
            issues.append(f"{name}还需 {EDGE_MIN - int(n)} 张图的角点覆盖")
    if spread[1] < 0.3:
        issues.append("倾角变化不足，请增加相机与标定板的左右、上下相对倾角")
    return dict(
        cells=cells.tolist(),
        edges=dict(zip(("left", "right", "top", "bottom"), edges.tolist())),
        normal_spread=spread.tolist(),
        issues=issues,
    )


def fit(samples, points, intr):
    # Keep factory K exactly fixed; only k1/k2 vary. Board poses remain
    # per-image nuisance variables, independent of robot/hand-eye extrinsics.
    return fit_distortion(samples, points, intr, fit_model="radial2")


def projections(intr, rays):
    return cv2.projectPoints(
        rays, np.zeros(3), np.zeros(3), matrix(intr), np.asarray(intr["coeffs"])
    )[0].reshape(-1, 2)


def failure_details(candidates):
    limits = {"heldout_rms_px": 0.5, "fit_rms_px": 0.5, "map_std_max_px": 2.0}
    usable = [
        c
        for c in candidates
        if all(np.isfinite(c.get(k, float("inf"))) for k in limits)
    ]
    if not usable:
        return [
            "内参求解失败："
            + "；".join(
                dict.fromkeys(c.get("error", "未得到有效模型") for c in candidates)
            )
        ], None
    best = min(usable, key=lambda c: max(c[k] / limit for k, limit in limits.items()))
    labels = {
        "heldout_rms_px": "留出误差",
        "fit_rms_px": "拟合误差",
        "map_std_max_px": "整幅图重采样标准差",
    }
    failed = [
        f"{labels[k]} {best[k]:.3f} px > {limit:g} px"
        for k, limit in limits.items()
        if best[k] > limit
    ]
    detail = {k: best[k] for k in ("k3_fitted", *limits)}
    note = (
        f"留出误差 {best['heldout_rms_px']:.3f} px，拟合误差 {best['fit_rms_px']:.3f} px。"
        "已有覆盖达标；需增加能约束镜头参数的不同距离、倾角和边缘视图。"
    )
    return ["内参未通过：" + "；".join(failed) + "。" + note], detail


def solve(samples, points, *, camera_family, stream):
    if camera_family != "realsense_d435i" or stream != "rgb":
        raise ValueError("radial2 profile is specific to RealSense D435i RGB")
    if not samples:
        raise ValueError("请先采集内参图像")
    intr = samples[0]["intrinsics"]
    frame = samples[0]["camera_frame"]
    for s in samples:
        if signature(s["intrinsics"]) != signature(intr) or any(
            s["camera_frame"].get(k) != frame.get(k)
            for k in ("serial", "image_rotation_deg")
        ):
            raise ValueError("不能混用不同相机或分辨率的图像")
    cov = coverage(samples, intr)
    if cov["issues"]:
        raise ValueError("；".join(cov["issues"]))
    xy = (
        np.array(
            np.meshgrid(
                np.linspace(0, intr["width"] - 1, 25),
                np.linspace(0, intr["height"] - 1, 17),
            )
        )
        .reshape(2, -1)
        .T
    )
    rays = np.c_[
        (xy[:, 0] - intr["ppx"]) / intr["fx"],
        (xy[:, 1] - intr["ppy"]) / intr["fy"],
        np.ones(len(xy)),
    ]
    candidates = []
    try:
        held = []
        maps = []
        for fold in range(4):
            training = [s for i, s in enumerate(samples) if i % 4 != fold]
            test = [s for i, s in enumerate(samples) if i % 4 == fold]
            model = fit(training, points, intr)
            held.extend(apply_model(test, points, model))
            maps.append(projections(model, rays))
        fitted = fit(samples, points, intr)
        uncertainty = np.sqrt(np.sum(np.var(maps, axis=0), axis=1))
        candidate = dict(
            fit_model="radial2",
            k3_fitted=False,
            intrinsics=fitted,
            heldout_rms_px=pixel_rms(held),
            fit_rms_px=pixel_rms(apply_model(samples, points, fitted)),
            map_std_p95_px=float(np.quantile(uncertainty, 0.95)),
            map_std_max_px=float(uncertainty.max()),
            map_std_max_pixel=xy[int(np.argmax(uncertainty))].tolist(),
        )
        candidate["passed"] = bool(
            candidate["heldout_rms_px"] <= 0.5
            and candidate["fit_rms_px"] <= 0.5
            and candidate["map_std_max_px"] <= 2.0
        )
        candidates.append(candidate)
    except (ValueError, cv2.error, np.linalg.LinAlgError) as e:
        candidates.append(
            dict(fit_model="radial2", k3_fitted=False, passed=False, error=str(e))
        )
    good = [c for c in candidates if c["passed"]]
    if not good:
        issues, diagnostic = failure_details(candidates)
        return dict(
            fit_model="radial2",
            gate_passed=False,
            coverage=cov,
            candidates=candidates,
            issues=issues,
            diagnostic_candidate=diagnostic,
        )
    chosen = good[0]
    return dict(
        version=3,
        source="independent_intrinsics",
        profile="realsense_d435i_rgb_radial2",
        camera_family=camera_family,
        stream=stream,
        fit_model="radial2",
        gate_passed=True,
        serial=frame["serial"],
        image_rotation_deg=frame.get("image_rotation_deg", 0),
        factory_intrinsics=copy.deepcopy(intr),
        intrinsics=chosen["intrinsics"],
        coverage=cov,
        validation={k: v for k, v in chosen.items() if k != "intrinsics"},
        candidates=candidates,
        sample_ids=[s["id"] for s in samples],
        folds=4,
    )
