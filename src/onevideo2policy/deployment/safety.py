from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RobotState:
    position_m: NDArray[np.float64]
    force_n: float
    timestamp_s: float
    fault: bool = False


@dataclass(frozen=True)
class CartesianCommand:
    position_m: NDArray[np.float64]
    gripper: float
    duration_s: float


@dataclass(frozen=True)
class SafetyEnvelope:
    workspace_bounds_m: NDArray[np.float64]
    max_cartesian_step_m: float = 0.01
    max_cartesian_speed_m_s: float = 0.08
    max_force_n: float = 15.0
    gripper_range: tuple[float, float] = (-1.0, 1.0)

    def validate(self) -> None:
        bounds = np.asarray(self.workspace_bounds_m, dtype=np.float64)
        if (
            bounds.shape != (3, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("workspace bounds must have ordered shape [3,2]")
        limits = np.asarray(
            [self.max_cartesian_step_m, self.max_cartesian_speed_m_s, self.max_force_n]
        )
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("safety limits must be positive")
        if (
            len(self.gripper_range) != 2
            or not np.isfinite(self.gripper_range).all()
            or self.gripper_range[0] >= self.gripper_range[1]
        ):
            raise ValueError("gripper range must contain finite ordered limits")


class RobotInterface(Protocol):
    def state(self) -> RobotState: ...

    def command(self, command: CartesianCommand) -> None: ...

    def stop(self) -> None: ...


@dataclass
class DryRunRobot:
    current: RobotState
    commands: list[CartesianCommand] = field(default_factory=list)
    stopped: bool = False

    def state(self) -> RobotState:
        return self.current

    def command(self, command: CartesianCommand) -> None:
        self.commands.append(command)
        self.current = RobotState(
            position_m=np.asarray(command.position_m, dtype=np.float64),
            force_n=self.current.force_n,
            timestamp_s=self.current.timestamp_s + command.duration_s,
        )

    def stop(self) -> None:
        self.stopped = True


class SafetySupervisor:
    def __init__(self, robot: RobotInterface, envelope: SafetyEnvelope) -> None:
        envelope.validate()
        self.robot = robot
        self.envelope = envelope

    def send(self, command: CartesianCommand) -> None:
        _, current = self._validated_state()
        target = np.asarray(command.position_m, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("command position must be finite xyz")
        bounds = self.envelope.workspace_bounds_m
        if np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
            raise ValueError("command leaves workspace bounds")
        distance = float(np.linalg.norm(target - current))
        if distance > self.envelope.max_cartesian_step_m + 1e-9:
            raise ValueError("command exceeds Cartesian step limit")
        if (
            not np.isfinite(command.duration_s)
            or command.duration_s <= 0
            or distance / command.duration_s > self.envelope.max_cartesian_speed_m_s + 1e-9
        ):
            raise ValueError("command exceeds Cartesian speed limit")
        low, high = self.envelope.gripper_range
        if not low <= command.gripper <= high:
            raise ValueError("gripper command outside configured range")
        self.robot.command(command)
        self._validated_state()

    def _validated_state(self) -> tuple[RobotState, NDArray[np.float64]]:
        state = self.robot.state()
        current = np.asarray(state.position_m, dtype=np.float64)
        if state.fault:
            self.robot.stop()
            raise RuntimeError("robot reports a fault")
        if not np.isfinite(state.force_n) or state.force_n > self.envelope.max_force_n:
            self.robot.stop()
            raise RuntimeError("force reading invalid or limit exceeded")
        if not np.isfinite(state.timestamp_s):
            self.robot.stop()
            raise RuntimeError("robot timestamp is invalid")
        if current.shape != (3,) or not np.isfinite(current).all():
            self.robot.stop()
            raise RuntimeError("robot position is invalid")
        bounds = self.envelope.workspace_bounds_m
        if np.any(current < bounds[:, 0]) or np.any(current > bounds[:, 1]):
            self.robot.stop()
            raise RuntimeError("robot position is outside workspace bounds")
        return state, current


def interpolate_keyframes(
    keyframes_m: NDArray[np.floating],
    gripper: NDArray[np.floating],
    *,
    max_step_m: float,
    speed_m_s: float,
    hold_duration_s: float = 0.5,
) -> list[CartesianCommand]:
    keyframes = np.asarray(keyframes_m, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64)
    if (
        keyframes.ndim != 2
        or keyframes.shape[1] != 3
        or gripper.shape != (len(keyframes),)
        or not np.isfinite(keyframes).all()
        or not np.isfinite(gripper).all()
    ):
        raise ValueError("keyframes must be [N,3] with one gripper value each")
    limits = np.asarray([max_step_m, speed_m_s, hold_duration_s], dtype=np.float64)
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("interpolation limits must be positive")
    commands = []
    for index in range(1, len(keyframes)):
        delta = keyframes[index] - keyframes[index - 1]
        steps = max(1, int(np.ceil(np.linalg.norm(delta) / max_step_m)))
        for step in range(1, steps + 1):
            position = keyframes[index - 1] + delta * step / steps
            distance = np.linalg.norm(delta) / steps
            duration_s = (
                hold_duration_s
                if distance < 1e-12
                else max(float(distance / speed_m_s), 0.05)
            )
            commands.append(
                CartesianCommand(
                    position_m=position,
                    gripper=float(gripper[index]),
                    duration_s=duration_s,
                )
            )
    return commands
