"""Safety-limited Cartesian planning for the measured pick-and-place task."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from onevideo2policy.deployment.safety import (
    CartesianCommand,
    SafetyEnvelope,
    interpolate_keyframes,
)


def build_pick_place_commands(
    home_m: NDArray[np.floating],
    source_m: NDArray[np.floating],
    target_m: NDArray[np.floating],
    envelope: SafetyEnvelope,
) -> list[CartesianCommand]:
    """Plan approach, stationary grasp/release, transfer, retreat, and home motion."""
    home = np.asarray(home_m, dtype=np.float64)
    source = np.asarray(source_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (home, source, target)):
        raise ValueError("home, source, and target must be finite xyz positions")
    source_contact = source.copy()
    source_contact[2] = max(source_contact[2], envelope.workspace_bounds_m[2, 0])
    target_release = target + np.array([0.0, 0.0, 0.04])
    transfer_height = max(home[2], source_contact[2] + 0.10, target_release[2] + 0.10)
    keyframes = np.vstack(
        (
            home,
            [source[0], source[1], transfer_height],
            source_contact,
            source_contact,
            [source[0], source[1], transfer_height],
            [target[0], target[1], transfer_height],
            target_release,
            target_release,
            [target[0], target[1], transfer_height],
            home,
        )
    )
    gripper = np.array([-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], dtype=np.float64)
    return interpolate_keyframes(
        keyframes,
        gripper,
        max_step_m=envelope.max_cartesian_step_m,
        speed_m_s=envelope.max_cartesian_speed_m_s,
        hold_duration_s=0.5,
    )
