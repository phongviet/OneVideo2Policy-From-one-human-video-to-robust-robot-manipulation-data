"""Validate calibration, checkpoint inference, and a safety-limited dry-run plan."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from onevideo2policy.deployment.calibration import (
    hardware_ready,
    transform_point,
    validate_calibration,
    validate_camera_alignment,
)
from onevideo2policy.deployment.package import sha256
from onevideo2policy.deployment.planning import build_pick_place_commands
from onevideo2policy.deployment.safety import (
    DryRunRobot,
    RobotState,
    SafetyEnvelope,
    SafetySupervisor,
)
from onevideo2policy.deployment.waypoint_runtime import predict_source_position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--safety", required=True, type=Path)
    parser.add_argument("--camera-reference", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--smoke-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-simulation-fixture", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    safety_config = yaml.safe_load(args.safety.read_text(encoding="utf-8"))
    camera_reference = json.loads(args.camera_reference.read_text(encoding="utf-8"))
    calibration_report = validate_calibration(calibration)
    if not calibration_report["valid"]:
        raise ValueError("Calibration failed: " + "; ".join(calibration_report["errors"]))
    if calibration_report["simulation_only"] and not args.allow_simulation_fixture:
        raise ValueError("Simulation calibration requires --allow-simulation-fixture")
    alignment_report = validate_camera_alignment(calibration, camera_reference)
    if not alignment_report["valid"]:
        raise ValueError("Camera alignment failed: " + "; ".join(alignment_report["errors"]))
    envelope = SafetyEnvelope(
        workspace_bounds_m=np.asarray(safety_config["workspace_bounds_m"], dtype=np.float64),
        max_cartesian_step_m=float(safety_config["max_cartesian_step_m"]),
        max_cartesian_speed_m_s=float(safety_config["max_cartesian_speed_m_s"]),
        max_force_n=float(safety_config["max_force_n"]),
        gripper_range=tuple(safety_config["gripper_range"]),
    )
    envelope.validate()
    for field in ("abort_on_camera_timeout_s", "max_camera_skew_s"):
        if not np.isfinite(float(safety_config[field])) or float(safety_config[field]) <= 0:
            raise ValueError(f"{field} must be finite and positive")
    calibration_bounds = np.asarray(calibration["workspace_bounds_m"], dtype=np.float64)
    if np.any(envelope.workspace_bounds_m[:, 0] < calibration_bounds[:, 0]) or np.any(
        envelope.workspace_bounds_m[:, 1] > calibration_bounds[:, 1]
    ):
        raise ValueError("Safety workspace exceeds the calibrated workspace")
    with np.load(args.smoke_data, allow_pickle=False) as smoke:
        agent = smoke["images"][0]
        front = smoke["images_front"][0]
        expected = smoke["ball_positions"][0]
    predicted = predict_source_position(args.checkpoint, agent, front)
    inference_error = float(np.linalg.norm(predicted - expected))
    if not np.isfinite(predicted).all() or inference_error > 0.03:
        raise ValueError(f"Checkpoint smoke inference error is {inference_error:.4f} m")

    home = np.asarray(calibration["home_position_robot_m"], dtype=np.float64)
    source = transform_point(calibration["task_to_robot"], predicted)
    target = np.asarray(calibration["target_center_robot_m"], dtype=np.float64)
    commands = build_pick_place_commands(home, source, target, envelope)
    robot = DryRunRobot(RobotState(home, force_n=0.0, timestamp_s=0.0))
    supervisor = SafetySupervisor(robot, envelope)
    for command in commands:
        supervisor.send(command)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_target = args.output / "visual_waypoint.pt"
    shutil.copy2(args.checkpoint, checkpoint_target)
    shutil.copy2(args.calibration, args.output / "calibration.json")
    shutil.copy2(args.safety, args.output / "safety.yaml")
    shutil.copy2(args.camera_reference, args.output / "camera-reference.json")
    report = {
        "schema_version": 1,
        "status": "software_preflight_pass",
        "hardware_ready": hardware_ready(calibration, calibration_report),
        "calibration": calibration_report,
        "training_camera_alignment": alignment_report,
        "checkpoint": {
            "sha256": sha256(checkpoint_target),
            "smoke_prediction_m": predicted.tolist(),
            "smoke_expected_m": expected.tolist(),
            "smoke_error_m": inference_error,
            "smoke_prediction_robot_m": source.tolist(),
        },
        "dry_run": {
            "commands": len(commands),
            "final_position_m": robot.current.position_m.tolist(),
            "elapsed_command_time_s": robot.current.timestamp_s,
            "stopped": robot.stopped,
            "max_step_m": envelope.max_cartesian_step_m,
            "max_speed_m_s": envelope.max_cartesian_speed_m_s,
            "camera_timeout_s": float(safety_config["abort_on_camera_timeout_s"]),
            "max_camera_skew_s": float(safety_config["max_camera_skew_s"]),
        },
        "arming_blockers": arming_blockers(calibration, calibration_report),
    }
    (args.output / "preflight-report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "files": {
            name: sha256(args.output / name)
            for name in (
                "visual_waypoint.pt",
                "calibration.json",
                "safety.yaml",
                "camera-reference.json",
                "preflight-report.json",
            )
        },
        "preflight_report": "preflight-report.json",
    }
    (args.output / "deployment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def arming_blockers(calibration: dict, diagnostics: dict) -> list[str]:
    blockers = []
    if diagnostics["simulation_only"]:
        blockers.append("replace simulation fixture with measured physical calibration")
    if not diagnostics["operator_approved"]:
        blockers.append("operator approval is false")
    if not diagnostics["emergency_stop_verified"]:
        blockers.append("emergency stop is not verified")
    if (
        not calibration.get("robot_serial")
        or str(calibration.get("robot_serial")).startswith("REPLACE_")
        or calibration.get("robot_serial") == "SIMULATION-FIXTURE"
    ):
        blockers.append("physical robot serial is absent")
    return blockers


if __name__ == "__main__":
    main()
