from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def estimate_rigid_transform(
    source_points: NDArray[np.floating], target_points: NDArray[np.floating]
) -> tuple[NDArray[np.float64], float]:
    """Fit a proper rigid transform from matched source-frame to target-frame points."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or source.shape != target.shape
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or np.linalg.matrix_rank(source - source.mean(axis=0)) < 2
    ):
        raise ValueError("at least three finite noncollinear matched 3D points are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    residual = (rotation @ source.T).T + translation - target
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return transform, rmse


def estimate_camera_to_robot(
    robot_points_m: NDArray[np.floating],
    image_points_px: NDArray[np.floating],
    intrinsics: NDArray[np.floating],
    distortion: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Estimate a camera-to-robot transform from measured 3D/2D correspondences."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional video dependency
        raise RuntimeError("extrinsic calibration requires the video extra") from exc
    robot_points = np.asarray(robot_points_m, dtype=np.float64)
    image_points = np.asarray(image_points_px, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    coefficients = (
        np.zeros(5, dtype=np.float64)
        if distortion is None
        else np.asarray(distortion, dtype=np.float64)
    )
    if robot_points.ndim != 2 or robot_points.shape[1] != 3 or len(robot_points) < 6:
        raise ValueError("at least six robot-frame 3D points are required")
    if image_points.shape != (len(robot_points), 2):
        raise ValueError("one image pixel is required for every robot point")
    if matrix.shape != (3, 3) or not all(
        np.isfinite(value).all() for value in (robot_points, image_points, matrix, coefficients)
    ):
        raise ValueError("calibration inputs must be finite and correctly shaped")
    success, rotation_vector, translation = cv2.solvePnP(
        robot_points,
        image_points,
        matrix,
        coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed to estimate camera extrinsics")
    rotation_robot_to_camera, _ = cv2.Rodrigues(rotation_vector)
    robot_to_camera = np.eye(4, dtype=np.float64)
    robot_to_camera[:3, :3] = rotation_robot_to_camera
    robot_to_camera[:3, 3] = translation[:, 0]
    camera_to_robot = np.linalg.inv(robot_to_camera)
    depths = (robot_to_camera[:3, :3] @ robot_points.T + translation).T[:, 2]
    if np.any(depths <= 0):
        raise RuntimeError("estimated calibration places reference points behind the camera")
    return camera_to_robot


def reprojection_rmse(
    camera_to_robot: NDArray[np.floating],
    robot_points_m: NDArray[np.floating],
    image_points_px: NDArray[np.floating],
    intrinsics: NDArray[np.floating],
    distortion: NDArray[np.floating] | None = None,
) -> float:
    """Measure pixel RMSE for robot-frame points under a camera calibration."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional video dependency
        raise RuntimeError("reprojection measurement requires the video extra") from exc
    camera_to_robot = np.asarray(camera_to_robot, dtype=np.float64)
    robot_points = np.asarray(robot_points_m, dtype=np.float64)
    image_points = np.asarray(image_points_px, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if (
        camera_to_robot.shape != (4, 4)
        or robot_points.ndim != 2
        or robot_points.shape[1] != 3
        or len(robot_points) == 0
        or image_points.shape != (len(robot_points), 2)
        or matrix.shape != (3, 3)
        or not all(
            np.isfinite(value).all()
            for value in (camera_to_robot, robot_points, image_points, matrix)
        )
    ):
        raise ValueError("reprojection inputs must contain finite matched 3D/2D points")
    robot_to_camera = np.linalg.inv(camera_to_robot)
    rotation_vector, _ = cv2.Rodrigues(robot_to_camera[:3, :3])
    coefficients = (
        np.zeros(5, dtype=np.float64)
        if distortion is None
        else np.asarray(distortion, dtype=np.float64)
    )
    projected, _ = cv2.projectPoints(
        robot_points,
        rotation_vector,
        robot_to_camera[:3, 3],
        matrix,
        coefficients,
    )
    residual = projected[:, 0] - image_points
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def validate_calibration(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate metric camera/robot calibration and return measured diagnostics."""
    errors: list[str] = []
    task_to_robot = np.asarray(bundle.get("task_to_robot", []), dtype=np.float64)
    if _transform_error(task_to_robot) is not None:
        errors.append("task_to_robot must be a finite proper rigid transform")
    cameras = bundle.get("cameras", [])
    if len(cameras) != 2:
        errors.append("exactly two calibrated cameras are required")
    names = [camera.get("name") for camera in cameras]
    if set(names) != {"agentview", "frontview"}:
        errors.append("calibrated cameras must be named agentview and frontview")
    camera_diagnostics = []
    for camera in cameras:
        name = str(camera.get("name", "unnamed"))
        intrinsics = np.asarray(camera.get("intrinsics", []), dtype=np.float64)
        resolution = camera.get("resolution")
        transform = np.asarray(camera.get("camera_to_robot", []), dtype=np.float64)
        distortion = np.asarray(camera.get("distortion_coefficients", []), dtype=np.float64)
        raw_rmse = camera.get("reprojection_rmse_px")
        rmse = float(raw_rmse) if raw_rmse is not None else float("inf")
        camera_errors = []
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            camera_errors.append("invalid intrinsics")
        elif (
            intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1])
        ):
            camera_errors.append("intrinsics have invalid focal length or homogeneous row")
        elif resolution != [84, 84]:
            camera_errors.append("resolution must be [84, 84] for the frozen policy")
        elif not (0 <= intrinsics[0, 2] < 84 and 0 <= intrinsics[1, 2] < 84):
            camera_errors.append("principal point lies outside the 84x84 image")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            camera_errors.append("invalid camera_to_robot transform")
            rotation_error = float("inf")
            determinant = float("nan")
        else:
            rotation = transform[:3, :3]
            rotation_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
            determinant = float(np.linalg.det(rotation))
            if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
                camera_errors.append("transform has invalid homogeneous row")
            if rotation_error > 1e-4 or not np.isclose(determinant, 1.0, atol=1e-4):
                camera_errors.append("transform rotation is not proper orthonormal")
        if distortion.ndim != 1 or not np.isfinite(distortion).all():
            camera_errors.append("invalid distortion coefficients")
        elif not bundle.get("simulation_only", True) and distortion.size not in {4, 5, 8, 12, 14}:
            camera_errors.append("physical camera requires calibrated distortion coefficients")
        if not np.isfinite(rmse) or rmse > 2.0:
            camera_errors.append("reprojection RMSE exceeds 2 px")
        errors.extend(f"{name}: {message}" for message in camera_errors)
        camera_diagnostics.append(
            {
                "name": name,
                "reprojection_rmse_px": rmse,
                "rotation_orthogonality_error": rotation_error,
                "rotation_determinant": determinant,
                "errors": camera_errors,
            }
        )
    normal = np.asarray(bundle.get("table_normal_robot", []), dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        errors.append("invalid table normal")
        table_tilt_deg = float("nan")
    else:
        normal /= np.linalg.norm(normal)
        table_tilt_deg = float(
            np.degrees(np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1, 1)))
        )
        if table_tilt_deg > 5:
            errors.append("table normal differs from robot +Z by more than 5 degrees")
    bounds = np.asarray(bundle.get("workspace_bounds_m", []), dtype=np.float64)
    if (
        bounds.shape != (3, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        errors.append("workspace_bounds_m must be finite [3,2] lower/upper bounds")
    table_height = bundle.get("table_height_robot_m")
    try:
        finite_table_height = bool(np.isfinite(float(table_height)))
    except (TypeError, ValueError):
        finite_table_height = False
    if not finite_table_height:
        errors.append("table_height_robot_m must be measured and finite")
    for field in ("home_position_robot_m", "target_center_robot_m"):
        position = np.asarray(bundle.get(field, []), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            errors.append(f"{field} must be a finite robot-frame xyz position")
        elif bounds.shape == (3, 2) and np.isfinite(bounds).all() and (
            np.any(position < bounds[:, 0]) or np.any(position > bounds[:, 1])
        ):
            errors.append(f"{field} lies outside workspace bounds")
    for field in ("source_diameter_m", "target_outer_diameter_m", "target_height_m"):
        try:
            dimension = float(bundle.get(field))
        except (TypeError, ValueError):
            dimension = float("nan")
        if not np.isfinite(dimension) or dimension <= 0:
            errors.append(f"{field} must be measured, finite, and positive")
    for field in ("simulation_only", "operator_approved", "emergency_stop_verified"):
        if type(bundle.get(field)) is not bool:
            errors.append(f"{field} must be Boolean")
    return {
        "valid": not errors,
        "errors": errors,
        "camera_diagnostics": camera_diagnostics,
        "table_tilt_deg": table_tilt_deg,
        "simulation_only": bool(bundle.get("simulation_only", True)),
        "operator_approved": bool(bundle.get("operator_approved", False)),
        "emergency_stop_verified": bool(bundle.get("emergency_stop_verified", False)),
    }


def validate_camera_alignment(
    bundle: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Compare measured cameras with the frozen policy's training geometry."""
    errors: list[str] = []
    diagnostics = []
    task_to_robot = np.asarray(bundle["task_to_robot"], dtype=np.float64)
    robot_to_task = np.linalg.inv(task_to_robot)
    measured_by_name = {camera.get("name"): camera for camera in bundle.get("cameras", [])}
    for expected in reference.get("cameras", []):
        name = expected["name"]
        measured = measured_by_name.get(name)
        if measured is None:
            errors.append(f"missing measured camera {name}")
            continue
        actual_transform = robot_to_task @ np.asarray(
            measured["camera_to_robot"], dtype=np.float64
        )
        expected_transform = np.asarray(expected["camera_to_task"], dtype=np.float64)
        translation_error = float(
            np.linalg.norm(actual_transform[:3, 3] - expected_transform[:3, 3])
        )
        rotation_delta = expected_transform[:3, :3].T @ actual_transform[:3, :3]
        rotation_error = float(
            np.degrees(
                np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1.0, 1.0))
            )
        )
        actual_intrinsics = np.asarray(measured["intrinsics"], dtype=np.float64)
        expected_intrinsics = np.asarray(expected["intrinsics"], dtype=np.float64)
        focal_relative_error = float(
            np.max(
                np.abs(actual_intrinsics[[0, 1], [0, 1]] - expected_intrinsics[[0, 1], [0, 1]])
                / expected_intrinsics[[0, 1], [0, 1]]
            )
        )
        principal_error = float(
            np.linalg.norm(actual_intrinsics[:2, 2] - expected_intrinsics[:2, 2])
        )
        camera_errors = []
        if translation_error > float(reference["max_translation_error_m"]):
            camera_errors.append("translation differs from training camera")
        if rotation_error > float(reference["max_rotation_error_deg"]):
            camera_errors.append("rotation differs from training camera")
        if focal_relative_error > float(reference["max_focal_relative_error"]):
            camera_errors.append("focal length differs from training camera")
        if principal_error > float(reference["max_principal_point_error_px"]):
            camera_errors.append("principal point differs from training camera")
        errors.extend(f"{name}: {message}" for message in camera_errors)
        diagnostics.append(
            {
                "name": name,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "focal_relative_error": focal_relative_error,
                "principal_point_error_px": principal_error,
                "errors": camera_errors,
            }
        )
    target_robot = np.r_[np.asarray(bundle["target_center_robot_m"], dtype=np.float64), 1.0]
    target_task = (robot_to_task @ target_robot)[:3]
    target_error = float(
        np.linalg.norm(target_task - np.asarray(reference["target_center_task_m"]))
    )
    if target_error > float(reference["max_target_position_error_m"]):
        errors.append("target center differs from the frozen training task")
    normal_robot = np.asarray(bundle["table_normal_robot"], dtype=np.float64)
    normal_task = robot_to_task[:3, :3] @ normal_robot
    normal_task /= np.linalg.norm(normal_task)
    table_normal_error = float(
        np.degrees(np.arccos(np.clip(np.dot(normal_task, [0, 0, 1]), -1.0, 1.0)))
    )
    table_point_robot = np.array([0.0, 0.0, float(bundle["table_height_robot_m"]), 1.0])
    table_height_task = float((robot_to_task @ table_point_robot)[2])
    table_height_error = abs(table_height_task - float(reference["table_height_task_m"]))
    if table_normal_error > 5.0:
        errors.append("table normal differs from the frozen training task")
    if table_height_error > float(reference["max_table_height_error_m"]):
        errors.append("table height differs from the frozen training task")
    dimension_errors = {}
    for field in ("source_diameter_m", "target_outer_diameter_m", "target_height_m"):
        dimension_errors[field] = abs(float(bundle[field]) - float(reference[field])) / float(
            reference[field]
        )
        if dimension_errors[field] > float(reference["max_object_dimension_relative_error"]):
            errors.append(f"{field} differs from the frozen training task")
    return {
        "valid": not errors,
        "errors": errors,
        "camera_diagnostics": diagnostics,
        "target_position_error_m": target_error,
        "table_normal_error_deg": table_normal_error,
        "table_height_error_m": table_height_error,
        "object_dimension_relative_errors": dimension_errors,
    }


def hardware_ready(bundle: dict[str, Any], diagnostics: dict[str, Any]) -> bool:
    serial = str(bundle.get("robot_serial", ""))
    return bool(
        diagnostics["valid"]
        and not diagnostics["simulation_only"]
        and diagnostics["operator_approved"]
        and diagnostics["emergency_stop_verified"]
        and serial
        and not serial.startswith("REPLACE_")
        and serial != "SIMULATION-FIXTURE"
    )


def transform_point(
    transform: NDArray[np.floating], point: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Apply a homogeneous rigid transform to one xyz point."""
    matrix = np.asarray(transform, dtype=np.float64)
    xyz = np.asarray(point, dtype=np.float64)
    if _transform_error(matrix) is not None or xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("transform and point must be finite rigid-transform inputs")
    return (matrix @ np.r_[xyz, 1.0])[:3]


def _transform_error(transform: NDArray[np.float64]) -> str | None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return "invalid shape or values"
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
        return "invalid homogeneous row"
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        return "rotation is not orthonormal"
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        return "rotation determinant is not one"
    return None
