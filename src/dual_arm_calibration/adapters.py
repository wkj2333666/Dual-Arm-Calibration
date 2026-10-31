"""Optional external hardware contracts; no SDK is imported by the core."""

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class ArmState:
    base_from_flange: np.ndarray
    joint_angles_rad: tuple[float, ...]
    host_monotonic_s: float
    stationary: bool


@dataclass(frozen=True)
class CameraFrame:
    bgr: np.ndarray
    serial: str
    frame_number: int
    host_monotonic_s: float
    intrinsics: dict
    image_rotation_deg: int = 0


class ArmReader(Protocol):
    def read_state(self) -> ArmState: ...


class CameraReader(Protocol):
    def capture(self) -> CameraFrame: ...


class MotionAdapter(ArmReader, Protocol):
    """An optional capability. Planning/collision limits remain adapter-owned."""

    def plan(self, targets: list, constraints: dict) -> object: ...
    def execute(self, reviewed_plan: object) -> None: ...
    def stop(self) -> None: ...


def load_adapter(name, **config):
    matches = list(entry_points(group="dual_arm_calibration.adapters", name=name))
    if len(matches) != 1:
        raise ValueError(f"Install exactly one adapter entry point named {name!r}")
    return matches[0].load()(**config)
