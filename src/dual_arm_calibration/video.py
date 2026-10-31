"""Select diverse, sharp board views from an already recorded camera video."""

import cv2
import numpy as np
from .board import Board
from .camera_model import matrix
from .intrinsics import coverage, coverage_features
from .validation import camera_model


def select_frames(path, board_spec, model, *, stride=3, maximum=80, sharpness=40.0):
    camera_model(model)
    if stride < 1 or maximum < 20 or sharpness < 0:
        raise ValueError("Invalid frame selection settings")
    board, intr = Board(board_spec), model["factory_intrinsics"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    candidates, index = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if frame.shape[:2] != (intr["height"], intr["width"]):
                raise ValueError("Video resolution differs from the bound camera model")
            if (index - 1) % stride:
                continue
            d, _ = board.detect(frame, matrix(intr), intr["coeffs"])
            if not d.detected or d.corner_count < 12:
                continue
            x, y, w, h = cv2.boundingRect(d.charuco_corners.astype(np.float32))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
                max(0, y) : y + h, max(0, x) : x + w
            ]
            if not gray.size or cv2.Laplacian(gray, cv2.CV_64F).var() < sharpness:
                continue
            candidates.append(
                dict(
                    id=f"video-frame-{index}",
                    video_frame=index - 1,
                    camera_frame=dict(
                        serial=model["serial"],
                        image_rotation_deg=model.get("image_rotation_deg", 0),
                    ),
                    intrinsics=intr,
                    charuco_ids=d.charuco_ids.tolist(),
                    charuco_corners=d.charuco_corners.tolist(),
                    camera_from_target=d.camera_from_target.tolist(),
                    reprojection_error_px=d.reprojection_error_px,
                )
            )
    finally:
        cap.release()
    selected, counts = [], np.zeros(13)
    while candidates and len(selected) < maximum:
        best = max(
            range(len(candidates)),
            key=lambda i: float(
                np.sum(
                    coverage_features(candidates[i]["charuco_corners"], intr)
                    / (1 + counts)
                )
            ),
        )
        s = candidates.pop(best)
        duplicate = False
        for old in selected:
            common = sorted(set(old["charuco_ids"]) & set(s["charuco_ids"]))
            if len(common) >= 12:
                diff = [
                    np.subtract(
                        old["charuco_corners"][old["charuco_ids"].index(i)],
                        s["charuco_corners"][s["charuco_ids"].index(i)],
                    )
                    for i in common
                ]
                if np.sqrt(np.mean(np.sum(np.square(diff), axis=1))) < 8:
                    duplicate = True
                    break
        if not duplicate:
            selected.append(s)
            counts += coverage_features(s["charuco_corners"], intr)
    selected.sort(key=lambda s: s["video_frame"])
    return dict(
        samples=selected, decoded_frames=index, coverage=coverage(selected, intr)
    )
