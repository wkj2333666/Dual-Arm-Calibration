"""Deterministic offline workflow. Solving never opens hardware or moves arms."""

import copy
import hashlib
import json
import numpy as np
from . import __version__
from .board import Board
from .camera_model import apply_model
from .geometry import mean_rotation, transform_error
from .validation import camera_model, observations, transform
from .intrinsics import solve as fit_intrinsics
from .wrist import solve_wrist
from .bridge import solve_bridge

ARMS = ("left", "right")
CAMERAS = (*ARMS, "top")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def mean_pose(poses):
    m = np.eye(4)
    m[:3, :3] = mean_rotation(poses)
    m[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return m


def check_rig(rig):
    if rig.get("schema_version") != 1:
        raise ValueError("Unsupported rig schema")
    if set(rig["arms"]) != set(ARMS) or set(rig["cameras"]) != set(CAMERAS):
        raise ValueError("Expected left/right arms and left/right/top cameras")
    for a in ARMS:
        if not rig["arms"][a].get("model_id"):
            raise ValueError("Each arm needs a model/installation identity")
    for r in CAMERAS:
        c = rig["cameras"][r]
        if c.get("mount") != ("fixed" if r == "top" else r + "_flange"):
            raise ValueError("Invalid camera mounting role")
        camera_model(c["model"])
        if c.get("intrinsics_mode", "reuse") not in ("reuse", "fit"):
            raise ValueError("Choose reuse or fit for each camera")
    if len({rig["cameras"][r]["model"]["serial"] for r in CAMERAS}) != 3:
        raise ValueError("Camera serials must be distinct")
    Board(rig["board"])
    table = rig.get("table", {"enabled": False})
    if not isinstance(table.get("enabled", False), bool):
        raise ValueError("table.enabled must be a boolean")
    if table.get("enabled"):
        thickness = float(table["thickness_mm"])
        if not np.isfinite(thickness) or not 0 <= thickness <= 100:
            raise ValueError("Board thickness must be between 0 and 100 mm")
        if table.get("board_z_direction") not in ("into_table", "out_of_table"):
            raise ValueError("Declare the board normal direction explicitly")


def board_chains(samples, points, models, wrists, bridges):
    """All three cameras observe the same stationary board in each sample."""
    estimates = []
    for s in samples:
        poses = {}
        for r in CAMERAS:
            obs = dict(s["cameras"][r], id=s["id"])
            observations([obs], points, models[r])
            pose = transform(
                apply_model([obs], points, models[r]["intrinsics"])[0][
                    "camera_from_target"
                ]
            )
            if r != "top":
                pose = (
                    np.linalg.inv(transform(bridges[r]["base_from_top"]))
                    @ transform(s["base_from_flange"][r])
                    @ transform(wrists[r]["flange_from_camera"])
                    @ pose
                )
            poses[r] = pose
        estimates.append(poses)
    return estimates


def cross_validate(samples, points, models, wrists, bridges):
    if len(samples) < 3:
        raise ValueError(
            "At least three independent three-camera validation observations required"
        )
    chains = board_chains(samples, points, models, wrists, bridges)
    residuals = []
    for s, poses in zip(samples, chains):
        for a in ARMS:
            mm, deg = transform_error(poses["top"], poses[a])
            residuals.append(
                dict(id=s["id"], arm=a, translation_mm=mm, rotation_deg=deg)
            )
    mm = max(r["translation_mm"] for r in residuals)
    deg = max(r["rotation_deg"] for r in residuals)
    return dict(
        gate_passed=mm <= 5 and deg <= 1,
        max_translation_mm=mm,
        max_rotation_deg=deg,
        residuals=residuals,
    ), chains


def table_frame(chains, config):
    poses = [p for group in chains for p in group.values()]
    avg = mean_pose(poses)
    errors = [transform_error(avg, p) for p in poses]
    if max(e[0] for e in errors) > 5 or max(e[1] for e in errors) > 1:
        raise ValueError("Table board moved or camera chains disagree (5 mm / 1 deg)")
    world_from_board = np.eye(4)
    if config["board_z_direction"] == "into_table":
        world_from_board[1, 1] = world_from_board[2, 2] = -1
    world_from_board[2, 3] = config["thickness_mm"] / 1000
    return world_from_board @ np.linalg.inv(avg), world_from_board


def solve(rig, data):
    check_rig(rig)
    if data.get("schema_version") != 1 or data.get("rig_sha256") != digest(rig):
        raise ValueError("Dataset is not bound to this exact rig configuration")
    # IDs are globally unique across independently collected batches.
    batches = list(data.get("intrinsics", {}).values()) + list(data["wrists"].values())
    batches += list(data["bridges"].values()) + [
        data["validation"],
        data.get("table", []),
    ]
    ids = [s["id"] for batch in batches for s in batch]
    if len(ids) != len(set(ids)):
        raise ValueError("Batches overlap: observation IDs must be globally unique")
    points = Board(rig["board"]).object_points
    models, wrists, bridges = {}, {}, {}
    for r in CAMERAS:
        cfg = rig["cameras"][r]
        models[r] = copy.deepcopy(cfg["model"])
        if cfg.get("intrinsics_mode", "reuse") == "fit":
            samples = data["intrinsics"][r]
            observations(samples, points, models[r], minimum=20)
            fitted = fit_intrinsics(
                samples, points, camera_family=cfg["family"], stream=cfg["stream"]
            )
            if not fitted["gate_passed"]:
                raise ValueError(
                    f"{r}: intrinsic validation failed: {fitted.get('issues')}"
                )
            models[r] = fitted
    for a in ARMS:
        wrists[a] = solve_wrist(data["wrists"][a], points, camera_model=models[a])
        if not wrists[a]["gate_passed"]:
            raise ValueError(f"{a}: wrist validation failed")
        bridges[a] = solve_bridge(data["bridges"][a], points, wrists[a], models["top"])
        if not bridges[a]["gate_passed"]:
            raise ValueError(f"{a}: fixed-camera validation failed")
    validation, _ = cross_validate(data["validation"], points, models, wrists, bridges)
    if not validation["gate_passed"]:
        raise ValueError("Independent cross-camera validation failed")
    world_from_top = np.eye(4)
    table = dict(calibrated=False, table_z_m=None)
    if rig.get("table", {}).get("enabled", False):
        report, chains = cross_validate(
            data.get("table", []), points, models, wrists, bridges
        )
        if not report["gate_passed"]:
            raise ValueError("Table camera agreement failed")
        world_from_top, world_from_board = table_frame(chains, rig["table"])
        table = dict(
            calibrated=True,
            table_z_m=0.0,
            world_from_board=world_from_board.tolist(),
            thickness_mm=rig["table"]["thickness_mm"],
            validation=report,
        )
    return dict(
        schema_version=1,
        package_version=__version__,
        rig_sha256=digest(rig),
        dataset_sha256=digest(data),
        units="metres",
        transform_convention="a_from_b",
        quality_passed=True,
        world_frame="table" if table["calibrated"] else "top_camera",
        world_from_top=world_from_top.tolist(),
        world_from_base={
            a: (
                world_from_top @ np.linalg.inv(transform(bridges[a]["base_from_top"]))
            ).tolist()
            for a in ARMS
        },
        cameras={
            r: dict(
                model=models[r],
                mount=rig["cameras"][r]["mount"],
                **(
                    {"flange_from_camera": wrists[r]["flange_from_camera"]}
                    if r in ARMS
                    else {}
                ),
            )
            for r in CAMERAS
        },
        wrists=wrists,
        bridges=bridges,
        table=table,
        validation=validation,
        tcp_verified=False,
    )
