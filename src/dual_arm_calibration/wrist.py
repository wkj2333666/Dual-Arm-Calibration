"""Hand-eye fitting with camera-model and unseen-pose validation."""

import copy
import numpy as np
from ._hand_eye import _solve_once, refine, evaluate, SolveError
from .camera_model import apply_model
from .validation import observations, rotation_coverage


def fit(samples, points):
    method, X, report = _solve_once(samples, "eye_in_hand", points)
    X, B, refined = refine(
        samples,
        X,
        report["invariant"],
        "eye_in_hand",
        points,
        require_pose_improvement=False,
    )
    report = evaluate(samples, X, "eye_in_hand", points, invariant=B)
    return method, X, B, refined, report


def solve_wrist(samples, points, min_samples=20, *, camera_model):
    observations(samples, points, camera_model, arm=True, minimum=min_samples)
    rotation_coverage(samples)
    if len(samples) < min_samples:
        raise SolveError(f"至少需要 {min_samples} 个有效姿态")
    model = copy.deepcopy(camera_model)
    corrected = apply_model(samples, points, model["intrinsics"])
    method, X, B, refined, report = fit(corrected, points)
    # Judge outliers against the fitted pixel chain, not the noisy PnP poses
    # that initialized it. Never restore rejected frames to satisfy the count.
    errors = np.array([r["chain_reprojection_px"] for r in report["residuals"]])
    median = np.median(errors)
    limit = max(2.0, float(median + 3 * np.median(abs(errors - median))))
    rejected = [s["id"] for s, e in zip(corrected, errors) if e > limit]
    kept = [s for s in corrected if s["id"] not in rejected]
    if len(kept) < min_samples:
        raise SolveError(
            f"剔除异常样本后仅剩 {len(kept)} 个，至少需要 {min_samples} 个；请补采。异常样本："
            + ", ".join(rejected)
        )
    if rejected:
        method, X, B, refined, report = fit(kept, points)
    # Hold the independently chosen lens fixed; refit hand-eye on each training fold.
    # Include all original observations here, even final-fit rejected frames.
    residuals = []
    folds = []
    for fold in range(4):
        training = [s for i, s in enumerate(samples) if i % 4 != fold]
        held = [s for i, s in enumerate(samples) if i % 4 == fold]
        rotation_coverage(training)
        train, train_model = apply_model(training, points, model["intrinsics"]), model
        _, tx, tb, _, _ = fit(train, points)
        test = apply_model(held, points, train_model["intrinsics"])
        r = evaluate(test, tx, "eye_in_hand", points, invariant=tb)
        residuals.extend(r["residuals"])
        folds.append(
            dict(
                fold=fold,
                train_ids=[s["id"] for s in training],
                heldout_ids=[s["id"] for s in held],
                camera_model_source=train_model["source"],
                reprojection_rms_px=r["reprojection_rms_px"],
            )
        )
    heldout = float(
        np.sqrt(np.mean([r["chain_reprojection_px"] ** 2 for r in residuals]))
    )
    gate = bool(
        report["translation_rms_mm"] <= 5
        and report["rotation_rms_deg"] <= 1
        and report["reprojection_rms_px"] <= 2
        and heldout <= 2
    )
    return dict(
        kind="eye_in_hand",
        method=method,
        refined=refined,
        solver_version=3 if camera_model else 2,
        flange_from_camera=X.tolist(),
        camera_from_flange=np.linalg.inv(X).tolist(),
        base_from_target=B.tolist(),
        used_samples=len(kept),
        used_sample_ids=[s["id"] for s in kept],
        outlier_ids=rejected,
        **{
            k: report[k]
            for k in (
                "translation_rms_mm",
                "rotation_rms_deg",
                "reprojection_rms_px",
                "residuals",
            )
        },
        camera_model=model,
        validation=dict(
            folds=4,
            intrinsics_fixed=camera_model is not None,
            heldout_reprojection_rms_px=heldout,
            details=folds,
            residuals=residuals,
            includes_rejected_samples=True,
        ),
        gate_passed=gate,
    )
