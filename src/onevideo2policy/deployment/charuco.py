"""ChArUco detection and intrinsic calibration helpers for physical cameras."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def create_board(config: dict[str, Any]) -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional video dependency
        raise RuntimeError("ChArUco calibration requires the video extra") from exc
    dictionary_name = str(config["dictionary"])
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return cv2.aruco.CharucoBoard(
        (int(config["squares_x"]), int(config["squares_y"])),
        float(config["square_length_m"]),
        float(config["marker_length_m"]),
        dictionary,
    )


def detect_charuco(
    image: NDArray[np.uint8], board: Any
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
    """Return matched board-frame xyz, image pixels, and ChArUco corner IDs."""
    import cv2

    array = np.asarray(image)
    if array.ndim == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError("calibration image must be grayscale or BGR")
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or corners is None or len(ids) < 6:
        raise RuntimeError("fewer than six ChArUco corners were detected")
    flat_ids = ids.reshape(-1).astype(np.int32)
    board_points = board.getChessboardCorners()[flat_ids].astype(np.float32)
    return board_points, corners.reshape(-1, 2).astype(np.float32), flat_ids


def calibrate_intrinsics(
    object_points: list[NDArray[np.float32]],
    image_points: list[NDArray[np.float32]],
    image_size: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Estimate pinhole intrinsics and distortion from multiple board observations."""
    import cv2

    if len(object_points) < 5 or len(object_points) != len(image_points):
        raise ValueError("intrinsic calibration requires at least five matched views")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise RuntimeError("intrinsic calibration produced nonfinite values")
    return matrix, distortion.reshape(-1), float(rms)


def transform_points(
    transform: NDArray[np.floating], points: NDArray[np.floating]
) -> NDArray[np.float64]:
    matrix = np.asarray(transform, dtype=np.float64)
    xyz = np.asarray(points, dtype=np.float64)
    if matrix.shape != (4, 4) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("transform must be [4,4] and points must be [N,3]")
    return (matrix[:3, :3] @ xyz.T).T + matrix[:3, 3]
