"""Estimate both physical camera extrinsics from robot-frame point observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from onevideo2policy.deployment.calibration import (
    estimate_camera_to_robot,
    estimate_rigid_transform,
    reprojection_rmse,
    validate_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    task_frame = observations["task_frame"]
    task_fit = task_frame["fit"]
    task_to_robot, task_frame_rmse = estimate_rigid_transform(
        np.asarray(task_fit["task_points_m"], dtype=np.float64),
        np.asarray(task_fit["robot_points_m"], dtype=np.float64),
    )
    task_validation = task_frame["validation"]
    validation_task_points = np.asarray(task_validation["task_points_m"], dtype=np.float64)
    validation_robot_points = np.asarray(task_validation["robot_points_m"], dtype=np.float64)
    if (
        validation_task_points.ndim != 2
        or validation_task_points.shape[1] != 3
        or len(validation_task_points) == 0
        or validation_robot_points.shape != validation_task_points.shape
    ):
        raise ValueError("task-frame validation requires matched 3D points")
    predicted_robot_points = (
        task_to_robot[:3, :3] @ validation_task_points.T
    ).T + task_to_robot[:3, 3]
    task_validation_rmse = float(
        np.sqrt(np.mean(np.sum((predicted_robot_points - validation_robot_points) ** 2, axis=1)))
    )
    if task_validation_rmse > 0.005:
        raise ValueError(f"task-frame validation RMSE exceeds 5 mm: {task_validation_rmse:.6f}")
    cameras = []
    diagnostics = []
    for camera in observations.get("cameras", []):
        matrix = np.asarray(camera["intrinsics"], dtype=np.float64)
        distortion = np.asarray(camera.get("distortion_coefficients", []), dtype=np.float64)
        fit = camera["fit"]
        validation = camera["validation"]
        camera_to_robot = estimate_camera_to_robot(
            np.asarray(fit["robot_points_m"], dtype=np.float64),
            np.asarray(fit["pixels"], dtype=np.float64),
            matrix,
            distortion,
        )
        fit_rmse = reprojection_rmse(
            camera_to_robot,
            np.asarray(fit["robot_points_m"], dtype=np.float64),
            np.asarray(fit["pixels"], dtype=np.float64),
            matrix,
            distortion,
        )
        validation_rmse = reprojection_rmse(
            camera_to_robot,
            np.asarray(validation["robot_points_m"], dtype=np.float64),
            np.asarray(validation["pixels"], dtype=np.float64),
            matrix,
            distortion,
        )
        cameras.append(
            {
                "name": camera["name"],
                "resolution": camera.get("resolution", [84, 84]),
                "intrinsics": matrix.tolist(),
                "distortion_coefficients": distortion.tolist(),
                "camera_to_robot": camera_to_robot.tolist(),
                "reprojection_rmse_px": validation_rmse,
            }
        )
        diagnostics.append(
            {
                "name": camera["name"],
                "fit_points": len(fit["robot_points_m"]),
                "validation_points": len(validation["robot_points_m"]),
                "fit_rmse_px": fit_rmse,
                "validation_rmse_px": validation_rmse,
            }
        )
    bundle = {
        "schema_version": 1,
        "simulation_only": False,
        "robot": observations.get("robot", "Panda"),
        "robot_serial": observations.get("robot_serial", "REPLACE_WITH_ROBOT_SERIAL"),
        "operator_approved": False,
        "emergency_stop_verified": False,
        "workspace_bounds_m": observations["workspace_bounds_m"],
        "table_normal_robot": observations["table_normal_robot"],
        "table_height_robot_m": observations["table_height_robot_m"],
        "task_to_robot": task_to_robot.tolist(),
        "home_position_robot_m": observations["home_position_robot_m"],
        "target_center_robot_m": observations["target_center_robot_m"],
        "source_diameter_m": observations["source_diameter_m"],
        "target_outer_diameter_m": observations["target_outer_diameter_m"],
        "target_height_m": observations["target_height_m"],
        "cameras": cameras,
        "estimation_diagnostics": {
            "task_frame_fit_points": len(task_fit["task_points_m"]),
            "task_frame_rmse_m": task_frame_rmse,
            "task_frame_validation_points": len(validation_task_points),
            "task_frame_validation_rmse_m": task_validation_rmse,
            "cameras": diagnostics,
            "capture_diagnostics": observations.get("charuco_diagnostics"),
        },
    }
    report = validate_calibration(bundle)
    if not report["valid"]:
        raise ValueError("Estimated calibration failed: " + "; ".join(report["errors"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_frame_rmse_m": task_frame_rmse,
                "task_frame_validation_rmse_m": task_validation_rmse,
                "cameras": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
