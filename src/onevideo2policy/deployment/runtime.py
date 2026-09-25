"""Hardware-neutral execution loop for one physical pick-and-place trial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from onevideo2policy.deployment.calibration import transform_point
from onevideo2policy.deployment.camera import SynchronizedCameraPair
from onevideo2policy.deployment.planning import build_pick_place_commands
from onevideo2policy.deployment.safety import RobotInterface, SafetyEnvelope, SafetySupervisor
from onevideo2policy.deployment.waypoint_runtime import predict_source_position


@dataclass(frozen=True)
class TrialExecution:
    controller: str
    source_position_m: NDArray[np.float64]
    commands_sent: int
    planned_duration_s: float
    peak_force_n: float
    localization_error_m: float | None


def execute_trial(
    *,
    controller: str,
    robot: RobotInterface,
    cameras: SynchronizedCameraPair,
    envelope: SafetyEnvelope,
    checkpoint_path: str | Path,
    task_to_robot: NDArray[np.floating],
    home_m: NDArray[np.floating],
    target_m: NDArray[np.floating],
    scripted_source_m: NDArray[np.floating] | None = None,
    source_reference_m: NDArray[np.floating] | None = None,
) -> TrialExecution:
    """Capture, localize, plan, and execute one safety-supervised physical trial."""
    if controller not in {"scripted", "learned"}:
        raise ValueError("controller must be scripted or learned")
    initial_agent, initial_front = cameras.capture()
    if controller == "learned":
        source_task = predict_source_position(
            checkpoint_path, initial_agent.rgb, initial_front.rgb
        ).astype(np.float64)
        source = transform_point(task_to_robot, source_task)
    elif scripted_source_m is None:
        raise ValueError("scripted controller requires a measured source position")
    else:
        source = np.asarray(scripted_source_m, dtype=np.float64)

    commands = build_pick_place_commands(
        np.asarray(home_m, dtype=np.float64),
        source,
        np.asarray(target_m, dtype=np.float64),
        envelope,
    )
    supervisor = SafetySupervisor(robot, envelope)
    peak_force = 0.0
    sent = 0
    try:
        for command in commands:
            cameras.capture()
            state = robot.state()
            peak_force = max(peak_force, float(state.force_n))
            supervisor.send(command)
            sent += 1
            state = robot.state()
            if not np.isfinite(state.force_n) or state.force_n > envelope.max_force_n:
                robot.stop()
                raise RuntimeError("force reading invalid or limit exceeded after command")
            peak_force = max(peak_force, float(state.force_n))
        final_state = robot.state()
        peak_force = max(peak_force, float(final_state.force_n))
    except Exception:
        robot.stop()
        raise

    reference = None if source_reference_m is None else np.asarray(source_reference_m)
    localization_error = (
        None if reference is None else float(np.linalg.norm(source - reference))
    )
    return TrialExecution(
        controller=controller,
        source_position_m=source,
        commands_sent=sent,
        planned_duration_s=float(sum(command.duration_s for command in commands)),
        peak_force_n=peak_force,
        localization_error_m=localization_error,
    )
