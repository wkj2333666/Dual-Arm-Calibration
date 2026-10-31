"""The ChArUco board: geometry, detection, and a printable sheet."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import make_transform


def _dictionary(name: str):
    key = getattr(cv2.aruco, name, None)
    if key is None:
        raise ValueError(f"未知的 aruco 字典: {name}")
    return cv2.aruco.getPredefinedDictionary(key)


@dataclass
class Detection:
    detected: bool
    corner_count: int
    coverage: float = 0.0
    reprojection_error_px: float = 0.0
    tilt_deg: float = 0.0
    distance_m: float = 0.0
    camera_from_target: np.ndarray | None = None
    charuco_ids: np.ndarray | None = None
    charuco_corners: np.ndarray | None = None
    reason: str = ""


class Board:
    def __init__(self, spec: dict):
        self.spec = dict(spec)
        self.dictionary = _dictionary(spec["dictionary"])
        self.board = cv2.aruco.CharucoBoard(
            (int(spec["squares_x"]), int(spec["squares_y"])),
            float(spec["square_length_m"]),
            float(spec["marker_length_m"]),
            self.dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.object_points = self.board.getChessboardCorners()

    @property
    def interior_corners(self) -> int:
        return (int(self.spec["squares_x"]) - 1) * (int(self.spec["squares_y"]) - 1)

    @property
    def size_m(self) -> tuple[float, float]:
        s = float(self.spec["square_length_m"])
        return (int(self.spec["squares_x"]) * s, int(self.spec["squares_y"]) * s)

    def detect(self, image, camera_matrix, distortion) -> tuple[Detection, np.ndarray]:
        """Detect the board and solve its pose. Returns (detection, overlay)."""
        overlay = image.copy()
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(image)
        if marker_ids is not None and len(marker_ids):
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        count = 0 if ids is None else len(ids)
        if ids is not None and count:
            cv2.aruco.drawDetectedCornersCharuco(overlay, corners, ids, (80, 255, 80))
        if ids is None or count < 6:
            return Detection(False, count, reason="角点太少"), overlay
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        distortion = np.asarray(distortion, dtype=np.float64)
        object_points = self.object_points[ids.reshape(-1)].astype(np.float64)
        image_points = corners.reshape(-1, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return Detection(False, count, reason="PnP 失败"), overlay

        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        error = float(
            np.sqrt(
                np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))
            )
        )

        rotation, _ = cv2.Rodrigues(rvec)
        camera_from_target = make_transform(rotation, tvec.reshape(3))

        height, width = image.shape[:2]
        hull = cv2.convexHull(image_points.astype(np.float32))
        coverage = float(cv2.contourArea(hull) / float(width * height))

        # Board normal is its +Z in camera frame; tilt is the angle away from
        # facing the camera. Near 0 the pose is weakly constrained in depth,
        # so the capture HUD nudges you towards 20-60 deg.
        normal = rotation[:, 2]
        view = tvec.reshape(3)
        view = view / max(1e-9, float(np.linalg.norm(view)))
        tilt = float(
            np.degrees(np.arccos(np.clip(abs(float(np.dot(normal, view))), 0.0, 1.0)))
        )

        cv2.drawFrameAxes(
            overlay,
            camera_matrix,
            distortion,
            rvec,
            tvec,
            float(self.spec["square_length_m"]) * 2.0,
            2,
        )
        return Detection(
            detected=True,
            corner_count=count,
            coverage=coverage,
            reprojection_error_px=error,
            tilt_deg=tilt,
            distance_m=float(np.linalg.norm(tvec)),
            camera_from_target=camera_from_target,
            charuco_ids=ids.reshape(-1).copy(),
            charuco_corners=image_points.copy(),
        ), overlay

    def printable(self, dpi: int = 300, margin_mm: float = 10.0) -> np.ndarray:
        """Render the board at true scale for printing at 100% (no 'fit to page')."""
        px_per_m = dpi / 0.0254
        width_m, height_m = self.size_m
        size = (round(width_m * px_per_m), round(height_m * px_per_m))
        image = self.board.generateImage(size)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        margin = round(margin_mm / 1000.0 * px_per_m)
        canvas = np.full(
            (image.shape[0] + 2 * margin, image.shape[1] + 2 * margin, 3),
            255,
            dtype=np.uint8,
        )
        canvas[margin : margin + image.shape[0], margin : margin + image.shape[1]] = (
            image
        )

        square_px = round(float(self.spec["square_length_m"]) * px_per_m)
        label = (
            f"{self.spec['squares_x']}x{self.spec['squares_y']}  "
            f"square {self.spec['square_length_m'] * 1000:.0f}mm  "
            f"marker {self.spec['marker_length_m'] * 1000:.0f}mm  "
            f"{self.spec['dictionary']}  @{dpi}dpi 100%"
        )
        cv2.putText(
            canvas,
            label,
            (margin, max(18, margin - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        # A one-square ruler to check the print scale with calipers.
        base = canvas.shape[0] - margin // 2
        cv2.line(canvas, (margin, base), (margin + square_px, base), (0, 0, 0), 2)
        for x in (margin, margin + square_px):
            cv2.line(canvas, (x, base - 8), (x, base + 8), (0, 0, 0), 2)
        cv2.putText(
            canvas,
            f"= {self.spec['square_length_m'] * 1000:.0f} mm",
            (margin + square_px + 8, base + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return canvas
