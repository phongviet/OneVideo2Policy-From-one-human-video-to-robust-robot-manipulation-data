from __future__ import annotations

import numpy as np

from onevideo2policy.generation.registration import (
    opencv_camera_to_mujoco_pose,
    register_task_frames,
)


def test_task_registration_maps_target_up_and_direction() -> None:
    source_sim = np.array([0.2, -0.3, 0.8])
    target_sim = np.array([0.0, 0.1, 0.85])
    source_world = np.array([0.6, 0.2, 0.7])
    target_world = np.array([0.3, 0.3, 0.75])
    normal_world = np.array([0.0, -np.sqrt(0.5), -np.sqrt(0.5)])
    transform = register_task_frames(
        source_sim, target_sim, source_world, target_world, normal_world
    )
    mapped_target = (transform @ np.r_[target_sim, 1])[:3]
    mapped_up = transform[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(mapped_target, target_world)
    assert np.allclose(mapped_up, normal_world)
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0)


def test_opencv_camera_conversion_preserves_center_and_fov() -> None:
    world_to_camera = np.eye(4)
    intrinsics = np.array([[200.0, 0, 160], [0, 200.0, 90], [0, 0, 1]])
    position, quaternion, fovy = opencv_camera_to_mujoco_pose(
        world_to_camera, intrinsics, 180
    )
    assert np.allclose(position, 0)
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    assert np.isclose(fovy, np.degrees(2 * np.arctan(180 / 400)))
