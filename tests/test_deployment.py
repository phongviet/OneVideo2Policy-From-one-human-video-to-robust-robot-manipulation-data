from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from onevideo2policy.deployment import package as deployment_package
from onevideo2policy.deployment.calibration import (
    estimate_camera_to_robot,
    estimate_rigid_transform,
    hardware_ready,
    reprojection_rmse,
    validate_calibration,
    validate_camera_alignment,
)
from onevideo2policy.deployment.camera import CameraFrame, SynchronizedCameraPair
from onevideo2policy.deployment.physical_assets import (
    BALL_DIAMETER_M,
    BOWL_HEIGHT_M,
    BOWL_OUTER_DIAMETER_M,
    audit_binary_stl,
    bowl_triangles,
    sphere_triangles,
    write_binary_stl,
)
from onevideo2policy.deployment.planning import build_pick_place_commands
from onevideo2policy.deployment.results import FROZEN_CHECKPOINT_SHA256, validate_results
from onevideo2policy.deployment.runtime import execute_trial
from onevideo2policy.deployment.safety import (
    CartesianCommand,
    DryRunRobot,
    RobotState,
    SafetyEnvelope,
    SafetySupervisor,
    interpolate_keyframes,
)


class FakeCamera:
    def __init__(self, timestamp_s: float = 1.0) -> None:
        self.timestamp_s = timestamp_s
        self.closed = False

    def read(self) -> CameraFrame:
        return CameraFrame(np.zeros((84, 84, 3), dtype=np.uint8), self.timestamp_s)

    def close(self) -> None:
        self.closed = True


class ForceSpikeRobot(DryRunRobot):
    def command(self, command: CartesianCommand) -> None:
        super().command(command)
        self.current = RobotState(self.current.position_m, 20.0, self.current.timestamp_s)


def valid_calibration() -> dict:
    camera = {
        "resolution": [84, 84],
        "intrinsics": [[100, 0, 42], [0, 100, 42], [0, 0, 1]],
        "distortion_coefficients": [0, 0, 0, 0, 0],
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
        "table_height_robot_m": 0.8,
        "task_to_robot": np.eye(4).tolist(),
        "home_position_robot_m": [0.1, -0.2, 1.0],
        "target_center_robot_m": [0.0, 0.1, 0.85],
        "source_diameter_m": 0.038,
        "target_outer_diameter_m": 0.099,
        "target_height_m": 0.057,
        "cameras": [{"name": "agentview", **camera}, {"name": "frontview", **camera}],
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


def test_camera_extrinsic_estimation_recovers_synthetic_transform() -> None:
    pytest.importorskip("cv2")
    points = np.array(
        [
            [-0.10, -0.08, 0.00],
            [0.10, -0.08, 0.02],
            [-0.10, 0.08, 0.04],
            [0.10, 0.08, 0.06],
            [0.00, -0.04, 0.10],
            [0.06, 0.00, 0.14],
            [-0.06, 0.02, 0.18],
            [0.02, 0.06, 0.22],
        ],
        dtype=np.float64,
    )
    intrinsics = np.array([[200, 0, 42], [0, 200, 42], [0, 0, 1]], dtype=np.float64)
    camera_points = points + np.array([0, 0, 1.0])
    pixels = np.column_stack(
        (
            intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsics[1, 2],
        )
    )
    transform = estimate_camera_to_robot(points, pixels, intrinsics)
    assert np.allclose(transform[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(transform[:3, 3], [0, 0, -1], atol=1e-6)
    assert reprojection_rmse(transform, points, pixels, intrinsics) < 1e-6


def test_task_to_robot_fit_recovers_rigid_transform() -> None:
    task = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.3, 0], [0.1, 0.1, 0.2]])
    angle = np.deg2rad(12)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    translation = np.array([0.5, 0.1, -0.9])
    robot = (rotation @ task.T).T + translation
    transform, rmse = estimate_rigid_transform(task, robot)
    assert np.allclose(transform[:3, :3], rotation)
    assert np.allclose(transform[:3, 3], translation)
    assert rmse < 1e-12


def test_camera_alignment_enforces_training_geometry() -> None:
    bundle = valid_calibration()
    reference = {
        "max_translation_error_m": 0.02,
        "max_rotation_error_deg": 5.0,
        "max_focal_relative_error": 0.05,
        "max_principal_point_error_px": 2.0,
        "target_center_task_m": bundle["target_center_robot_m"],
        "max_target_position_error_m": 0.02,
        "table_height_task_m": bundle["table_height_robot_m"],
        "max_table_height_error_m": 0.01,
        "source_diameter_m": bundle["source_diameter_m"],
        "target_outer_diameter_m": bundle["target_outer_diameter_m"],
        "target_height_m": bundle["target_height_m"],
        "max_object_dimension_relative_error": 0.1,
        "cameras": [
            {
                **copy.deepcopy(camera),
                "camera_to_task": copy.deepcopy(camera["camera_to_robot"]),
            }
            for camera in bundle["cameras"]
        ],
    }
    assert validate_camera_alignment(bundle, reference)["valid"]
    bundle["cameras"][0]["camera_to_robot"][0][3] = 0.03
    report = validate_camera_alignment(bundle, reference)
    assert not report["valid"]
    assert "translation" in report["errors"][0]
    bundle = valid_calibration()
    bundle["source_diameter_m"] = 0.03
    report = validate_camera_alignment(bundle, reference)
    assert "source_diameter_m differs from the frozen training task" in report["errors"]


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


def test_pick_place_plan_changes_gripper_only_while_stationary() -> None:
    envelope = SafetyEnvelope(np.array([[-0.1, 0.2], [-0.35, 0.25], [0.79, 1.08]]))
    commands = build_pick_place_commands(
        np.array([0.1, -0.2, 1.02]),
        np.array([0.07, -0.2, 0.82]),
        np.array([0.0, 0.1575, 0.857]),
        envelope,
    )
    changes = [
        index
        for index in range(1, len(commands))
        if commands[index].gripper != commands[index - 1].gripper
    ]
    assert len(changes) == 2
    for index in changes:
        assert np.allclose(commands[index].position_m, commands[index - 1].position_m)
        assert commands[index].duration_s == 0.5


def test_synchronized_camera_pair_rejects_stale_or_skewed_frames() -> None:
    pair = SynchronizedCameraPair(FakeCamera(1.0), FakeCamera(1.02), clock=lambda: 1.1)
    agent, front = pair.capture()
    assert agent.rgb.shape == front.rgb.shape == (84, 84, 3)
    pair = SynchronizedCameraPair(FakeCamera(1.0), FakeCamera(1.2), clock=lambda: 1.2)
    with pytest.raises(RuntimeError, match="skew"):
        pair.capture()
    pair = SynchronizedCameraPair(FakeCamera(1.0), FakeCamera(1.0), clock=lambda: 1.3)
    with pytest.raises(RuntimeError, match="stale"):
        pair.capture()


def test_scripted_physical_runtime_uses_camera_watchdog_and_safety() -> None:
    envelope = SafetyEnvelope(np.array([[-0.1, 0.2], [-0.35, 0.25], [0.79, 1.08]]))
    home = np.array([0.1, -0.2, 1.02])
    robot = DryRunRobot(RobotState(home, 0.0, 0.0))
    cameras = SynchronizedCameraPair(FakeCamera(), FakeCamera(), clock=lambda: 1.0)
    execution = execute_trial(
        controller="scripted",
        robot=robot,
        cameras=cameras,
        envelope=envelope,
        checkpoint_path="unused-for-scripted-controller.pt",
        task_to_robot=np.eye(4),
        home_m=home,
        target_m=np.array([0.0, 0.1575, 0.857]),
        scripted_source_m=np.array([0.07, -0.2, 0.82]),
    )
    assert execution.commands_sent == len(robot.commands)
    assert execution.commands_sent > 100
    assert execution.peak_force_n == 0
    assert np.allclose(robot.current.position_m, home)


def test_deployment_package_verifies_arming_state_and_hashes(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "visual_waypoint.pt"
    checkpoint.write_bytes(b"frozen-test-checkpoint")
    checkpoint_hash = deployment_package.sha256(checkpoint)
    monkeypatch.setattr(deployment_package, "FROZEN_CHECKPOINT_SHA256", checkpoint_hash)
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}\n", encoding="utf-8")
    safety = tmp_path / "safety.yaml"
    safety.write_text(
        "workspace_bounds_m: [[-0.1, 0.2], [-0.3, 0.2], [0.8, 1.1]]\n"
        "max_cartesian_step_m: 0.01\n"
        "max_cartesian_speed_m_s: 0.08\n"
        "max_force_n: 15.0\n"
        "gripper_range: [-1.0, 1.0]\n",
        encoding="utf-8",
    )
    reference = tmp_path / "camera-reference.json"
    reference.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "preflight-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "software_preflight_pass",
                "hardware_ready": True,
                "arming_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    files = [
        "visual_waypoint.pt",
        "calibration.json",
        "safety.yaml",
        "camera-reference.json",
        "preflight-report.json",
    ]
    manifest = {
        "files": {name: deployment_package.sha256(tmp_path / name) for name in files},
        "preflight_report": "preflight-report.json",
    }
    (tmp_path / "deployment-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    loaded = deployment_package.load_deployment_package(tmp_path)
    assert loaded.checkpoint_path == checkpoint
    safety.write_text(safety.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        deployment_package.load_deployment_package(tmp_path)


def test_printable_task_meshes_are_watertight_and_dimensionally_exact(tmp_path) -> None:
    fixtures = {
        "ball.stl": (
            sphere_triangles(BALL_DIAMETER_M / 2, 32, 16),
            np.full(3, BALL_DIAMETER_M * 1000),
        ),
        "bowl.stl": (
            bowl_triangles(BOWL_OUTER_DIAMETER_M / 2, BOWL_HEIGHT_M, 0.004, 32),
            np.array(
                [BOWL_OUTER_DIAMETER_M * 1000, BOWL_OUTER_DIAMETER_M * 1000, BOWL_HEIGHT_M * 1000]
            ),
        ),
    }
    for name, (triangles, expected_extents) in fixtures.items():
        path = tmp_path / name
        write_binary_stl(path, triangles * 1000)
        audit = audit_binary_stl(path)
        assert audit["watertight"]
        assert np.allclose(audit["extents_mm"], expected_extents, atol=1e-4)


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
    robot = ForceSpikeRobot(RobotState(start, 0.0, 0.0))
    supervisor = SafetySupervisor(robot, envelope)
    with pytest.raises(RuntimeError, match="force"):
        supervisor.send(CartesianCommand(start, -1.0, 0.1))
    assert robot.stopped


def test_physical_result_gate_requires_paired_safe_successes() -> None:
    trials = []
    for controller in ("scripted", "learned"):
        for index in range(5):
            trials.append(
                {
                    "trial_id": f"{controller}-{index}",
                    "reset_id": f"reset-{index}",
                    "pair_order": (
                        1
                        if (controller == "scripted") == (index % 2 == 0)
                        else 2
                    ),
                    "controller": controller,
                    "success": index < 4,
                    "safety_abort": False,
                    "max_force_n": 8.0,
                }
            )
    raw = {
        "robot_serial": "PANDA-001",
        "deployment_manifest_sha256": "c" * 64,
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
