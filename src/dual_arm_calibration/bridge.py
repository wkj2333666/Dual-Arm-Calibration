"""Connect a fixed camera to either base using paired board observations."""

import copy
import numpy as np
from .camera_model import apply_model
from .validation import observations, transform
from ._bridge_fit import fit, errors


def prepare(samples, points, wrist, top_model):
    observations(samples, points, top_model, arm=True, minimum=12)
    observations(
        [dict(s["wrist"], id=s["id"]) for s in samples], points, wrist["camera_model"]
    )
    rows = []
    for s in samples:
        h = apply_model([s], points, top_model["intrinsics"])[0]
        w = apply_model([s["wrist"]], points, wrist["camera_model"]["intrinsics"])[0]
        base_from_board = (
            transform(s["base_from_flange"])
            @ transform(wrist["flange_from_camera"])
            @ transform(w["camera_from_target"])
        )
        estimate = base_from_board @ np.linalg.inv(transform(h["camera_from_target"]))
        rows.append(
            dict(
                id=s["id"],
                world_from_board=base_from_board,
                estimate=estimate,
                corners=np.asarray(h["charuco_corners"]),
                ids=h["charuco_ids"],
            )
        )
    return rows


def solve_bridge(samples, points, wrist, top_model):
    if not wrist.get("gate_passed"):
        raise ValueError("Wrist calibration must pass before bridging")
    rows = prepare(samples, points, wrist, top_model)
    intr = top_model["intrinsics"]
    pose = fit(rows, points, intr)
    residuals = errors(rows, pose, points, intr)
    held = []
    for fold in range(4):
        training = [r for i, r in enumerate(rows) if i % 4 != fold]
        test = [r for i, r in enumerate(rows) if i % 4 == fold]
        held.extend(errors(test, fit(training, points, intr), points, intr))
    rms = lambda rr: float(
        np.sqrt(np.mean([r["chain_reprojection_px"] ** 2 for r in rr]))
    )
    fit_rms, held_rms = rms(residuals), rms(held)
    return dict(
        base_from_top=pose.tolist(),
        camera_model=copy.deepcopy(top_model),
        used_sample_ids=[s["id"] for s in samples],
        reprojection_rms_px=fit_rms,
        validation=dict(folds=4, heldout_reprojection_rms_px=held_rms, residuals=held),
        gate_passed=bool(fit_rms <= 2 and held_rms <= 2),
    )
