from __future__ import annotations

import numpy as np
import pytest

from onevideo2policy.deployment.calibration import hardware_ready, validate_calibration
from onevideo2policy.deployment.results import FROZEN_CHECKPOINT_SHA256, validate_results
from onevideo2policy.deployment.safety import (
    CartesianCommand,
    DryRunRobot,
    RobotState,
    SafetyEnvelope,
    SafetySupervisor,
    interpolate_keyframes,
)


def valid_calibration() -> dict:
    camera = {
        "intrinsics": [[100, 0, 42], [0, 100, 42], [0, 0, 1]],
        "camera_to_robot": np.eye(4).tolist(),
        "reprojection_rmse_px": 0.8,
    }
    return {
        "simulation_only": False,
        "operator_approved": True,
        "emergency_stop_verified": True,
        "robot_serial": "PANDA-001",
        "workspace_bounds_m": [[-0.1, 0.2], [-0.3, 0.2], [0.8, 1.1]],
        "table_normal_robot": [0, 0, 1],
        "cameras": [{"name": "a", **camera}, {"name": "b", **camera}],
    }


def test_calibration_contract_and_hardware_interlocks() -> None:
    bundle = valid_calibration()
    report = validate_calibration(bundle)
    assert report["valid"]
    assert hardware_ready(bundle, report)
    bundle["simulation_only"] = True
    report = validate_calibration(bundle)
    assert not hardware_ready(bundle, report)
    bundle["simulation_only"] = False
    bundle["robot_serial"] = "REPLACE_WITH_ROBOT_SERIAL"
    report = validate_calibration(bundle)
    assert not hardware_ready(bundle, report)


def test_safety_supervisor_accepts_interpolated_path() -> None:
    envelope = SafetyEnvelope(
        np.array([[-0.1, 0.2], [-0.3, 0.2], [0.8, 1.1]]),
        max_cartesian_step_m=0.01,
        max_cartesian_speed_m_s=0.08,
    )
    start = np.array([0.0, 0.0, 0.9])
    commands = interpolate_keyframes(
        np.array([start, [0.03, 0.0, 0.9]]),
        np.array([-1.0, 1.0]),
        max_step_m=0.01,
        speed_m_s=0.08,
    )
    robot = DryRunRobot(RobotState(start, 0.0, 0.0))
    supervisor = SafetySupervisor(robot, envelope)
    for command in commands:
        supervisor.send(command)
    assert len(robot.commands) == 3
    assert np.allclose(robot.current.position_m, [0.03, 0.0, 0.9])


def test_safety_supervisor_stops_on_force_and_rejects_workspace() -> None:
    envelope = SafetyEnvelope(np.array([[-0.1, 0.2], [-0.3, 0.2], [0.8, 1.1]]))
    start = np.array([0.0, 0.0, 0.9])
    robot = DryRunRobot(RobotState(start, 16.0, 0.0))
    supervisor = SafetySupervisor(robot, envelope)
    with pytest.raises(RuntimeError, match="force"):
        supervisor.send(CartesianCommand(start, -1.0, 0.1))
    assert robot.stopped
    robot = DryRunRobot(RobotState(start, 0.0, 0.0))
    supervisor = SafetySupervisor(robot, envelope)
    with pytest.raises(ValueError, match="workspace"):
        supervisor.send(CartesianCommand(np.array([0.3, 0.0, 0.9]), -1.0, 5.0))
    with pytest.raises(ValueError, match="speed"):
        supervisor.send(CartesianCommand(start, -1.0, np.nan))


def test_physical_result_gate_requires_paired_safe_successes() -> None:
    trials = []
    for controller in ("scripted", "learned"):
        for index in range(5):
            trials.append(
                {
                    "trial_id": f"{controller}-{index}",
                    "controller": controller,
                    "success": index < 4,
                    "safety_abort": False,
                    "max_force_n": 8.0,
                }
            )
    raw = {
        "robot_serial": "PANDA-001",
        "calibration_sha256": "a" * 64,
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "trials": trials,
    }
    report = validate_results(raw, trials_per_controller=5, min_success_rate=0.8, max_force_n=15)
    assert report["status"] == "pass"
    raw["trials"][0]["safety_abort"] = True
    report = validate_results(raw, trials_per_controller=5, min_success_rate=0.8, max_force_n=15)
    assert report["status"] == "incomplete_or_failed"
    raw["trials"][0]["safety_abort"] = "false"
    report = validate_results(raw, trials_per_controller=5, min_success_rate=0.8, max_force_n=15)
    assert "invalid required fields" in report["errors"][-1]
