"""Run one hardware-armed physical trial through a user-supplied robot SDK adapter."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from onevideo2policy.deployment.camera import OpenCVCamera, SynchronizedCameraPair
from onevideo2policy.deployment.package import load_deployment_package
from onevideo2policy.deployment.runtime import execute_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument(
        "--robot-factory", required=True, help="Python module:function adapter factory"
    )
    parser.add_argument("--controller", required=True, choices=("scripted", "learned"))
    parser.add_argument("--agent-camera", required=True)
    parser.add_argument("--front-camera", required=True)
    parser.add_argument("--scripted-source-m", nargs=3, type=float)
    parser.add_argument("--source-reference-m", nargs=3, type=float)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--reset-id", required=True)
    parser.add_argument("--pair-order", required=True, choices=(1, 2), type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Required to send robot commands")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise ValueError("--execute is required after reviewing the hardware-ready preflight")
    package = load_deployment_package(args.deployment)
    robot = load_robot(args.robot_factory, package.calibration, package.safety_config)
    cameras_by_name = {camera["name"]: camera for camera in package.calibration["cameras"]}
    agent = calibrated_camera(args.agent_camera, cameras_by_name["agentview"])
    try:
        front = calibrated_camera(args.front_camera, cameras_by_name["frontview"])
    except Exception:
        agent.close()
        raise
    cameras = SynchronizedCameraPair(
        agent,
        front,
        max_age_s=float(package.safety_config["abort_on_camera_timeout_s"]),
        max_skew_s=float(package.safety_config["max_camera_skew_s"]),
    )
    try:
        execution = execute_trial(
            controller=args.controller,
            robot=robot,
            cameras=cameras,
            envelope=package.envelope,
            checkpoint_path=package.checkpoint_path,
            task_to_robot=np.asarray(package.calibration["task_to_robot"]),
            home_m=np.asarray(package.calibration["home_position_robot_m"]),
            target_m=np.asarray(package.calibration["target_center_robot_m"]),
            scripted_source_m=(
                None if args.scripted_source_m is None else np.asarray(args.scripted_source_m)
            ),
            source_reference_m=(
                None if args.source_reference_m is None else np.asarray(args.source_reference_m)
            ),
        )
    finally:
        cameras.close()
    output = {
        "schema_version": 1,
        "trial_id": args.trial_id,
        "reset_id": args.reset_id,
        "pair_order": args.pair_order,
        "controller": execution.controller,
        "execution_complete": True,
        "source_position_robot_m": execution.source_position_m.tolist(),
        "commands_sent": execution.commands_sent,
        "planned_duration_s": execution.planned_duration_s,
        "max_force_n": execution.peak_force_n,
        "localization_error_m": execution.localization_error_m,
        "success": None,
        "safety_abort": False,
        "notes": "Set success after checking physical bowl containment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def load_robot(factory_spec: str, calibration: dict, safety_config: dict) -> Any:
    module_name, separator, function_name = factory_spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("robot factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name)
    robot = factory(calibration=calibration, safety_config=safety_config)
    if not all(callable(getattr(robot, name, None)) for name in ("state", "command", "stop")):
        raise TypeError("robot adapter must implement state(), command(), and stop()")
    return robot


def calibrated_camera(device: str, calibration: dict) -> OpenCVCamera:
    parsed_device: int | str = int(device) if device.isdigit() else device
    width, height = calibration["resolution"]
    return OpenCVCamera(
        parsed_device,
        width=int(width),
        height=int(height),
        intrinsics=np.asarray(calibration["intrinsics"], dtype=np.float64),
        distortion=np.asarray(calibration.get("distortion_coefficients", []), dtype=np.float64),
    )


if __name__ == "__main__":
    main()
