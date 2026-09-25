from __future__ import annotations

from typing import Any

import numpy as np


def validate_calibration(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate metric camera/robot calibration and return measured diagnostics."""
    errors: list[str] = []
    cameras = bundle.get("cameras", [])
    if len(cameras) != 2:
        errors.append("exactly two calibrated cameras are required")
    camera_diagnostics = []
    for camera in cameras:
        name = str(camera.get("name", "unnamed"))
        intrinsics = np.asarray(camera.get("intrinsics", []), dtype=np.float64)
        transform = np.asarray(camera.get("camera_to_robot", []), dtype=np.float64)
        raw_rmse = camera.get("reprojection_rmse_px")
        rmse = float(raw_rmse) if raw_rmse is not None else float("inf")
        camera_errors = []
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            camera_errors.append("invalid intrinsics")
        elif intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or intrinsics[2, 2] != 1:
            camera_errors.append("intrinsics have invalid focal length or homogeneous row")
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
    return {
        "valid": not errors,
        "errors": errors,
        "camera_diagnostics": camera_diagnostics,
        "table_tilt_deg": table_tilt_deg,
        "simulation_only": bool(bundle.get("simulation_only", True)),
        "operator_approved": bool(bundle.get("operator_approved", False)),
        "emergency_stop_verified": bool(bundle.get("emergency_stop_verified", False)),
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
