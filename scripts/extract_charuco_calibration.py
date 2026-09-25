"""Detect ChArUco captures and build physical calibration observations automatically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.deployment.charuco import (
    calibrate_intrinsics,
    create_board,
    detect_charuco,
    transform_points,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.captures.read_text(encoding="utf-8"))
    board = create_board(raw["board"])
    runtime_width, runtime_height = raw.get("runtime_resolution", [84, 84])
    output_cameras = []
    camera_diagnostics = []
    for camera in raw["cameras"]:
        native_width, native_height = camera["capture_resolution"]
        object_views = []
        pixel_views = []
        detections = []
        for capture in camera["captures"]:
            image_path = (args.captures.parent / capture["image"]).resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            if image.shape[:2] != (native_height, native_width):
                raise ValueError(
                    f"{image_path} has shape {image.shape[:2]}; "
                    "expected capture resolution"
                )
            board_points, pixels, ids = detect_charuco(image, board)
            object_views.append(board_points)
            pixel_views.append(pixels)
            detections.append((capture, board_points, pixels, ids))
        matrix_native, distortion, intrinsic_rmse = calibrate_intrinsics(
            object_views, pixel_views, (native_width, native_height)
        )
        scale_x = runtime_width / native_width
        scale_y = runtime_height / native_height
        matrix_runtime = matrix_native.copy()
        matrix_runtime[0] *= scale_x
        matrix_runtime[1] *= scale_y
        split_points = {"fit": ([], []), "validation": ([], [])}
        capture_reports = []
        for capture, board_points, pixels, ids in detections:
            split = capture["split"]
            if split not in split_points:
                raise ValueError("capture split must be fit or validation")
            robot_points = transform_points(
                np.asarray(capture["board_to_robot"], dtype=np.float64), board_points
            )
            runtime_pixels = pixels * np.array([scale_x, scale_y], dtype=np.float32)
            split_points[split][0].extend(robot_points.tolist())
            split_points[split][1].extend(runtime_pixels.tolist())
            capture_reports.append(
                {
                    "image": capture["image"],
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "split": split,
                    "corners": len(ids),
                }
            )
        output_cameras.append(
            {
                "name": camera["name"],
                "resolution": [runtime_width, runtime_height],
                "intrinsics": matrix_runtime.tolist(),
                "distortion_coefficients": distortion.tolist(),
                "fit": {
                    "robot_points_m": split_points["fit"][0],
                    "pixels": split_points["fit"][1],
                },
                "validation": {
                    "robot_points_m": split_points["validation"][0],
                    "pixels": split_points["validation"][1],
                },
            }
        )
        camera_diagnostics.append(
            {
                "name": camera["name"],
                "intrinsic_rms_px_native": intrinsic_rmse,
                "captures": capture_reports,
            }
        )
    output = {
        key: raw[key]
        for key in (
            "schema_version",
            "robot",
            "robot_serial",
            "workspace_bounds_m",
            "table_normal_robot",
            "table_height_robot_m",
            "home_position_robot_m",
            "target_center_robot_m",
            "source_diameter_m",
            "target_outer_diameter_m",
            "target_height_m",
            "task_frame",
        )
    }
    output["cameras"] = output_cameras
    output["charuco_diagnostics"] = camera_diagnostics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cameras": camera_diagnostics}, indent=2))


if __name__ == "__main__":
    main()
