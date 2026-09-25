from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def register_task_frames(
    source_sim: NDArray[np.floating],
    target_sim: NDArray[np.floating],
    source_world: NDArray[np.floating],
    target_world: NDArray[np.floating],
    normal_world: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Rigidly align target center, table normal, and source direction."""
    sim_basis = task_basis(source_sim, target_sim, np.array([0.0, 0.0, 1.0]))
    world_basis = task_basis(source_world, target_world, normal_world)
    rotation = world_basis @ sim_basis.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(target_world) - rotation @ np.asarray(target_sim)
    return transform


def task_basis(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    up: NDArray[np.floating],
) -> NDArray[np.float64]:
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    forward = np.asarray(source, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    forward -= up * np.dot(forward, up)
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("source and target need a nonzero tabletop separation")
    forward /= norm
    side = np.cross(up, forward)
    side /= np.linalg.norm(side)
    return np.column_stack((forward, side, up))


def opencv_camera_to_mujoco_pose(
    world_to_camera: NDArray[np.floating],
    intrinsics: NDArray[np.floating],
    image_height: int,
) -> tuple[NDArray[np.float64], NDArray[np.float32], float]:
    """Convert an OpenCV world-to-camera matrix to MuJoCo position, wxyz, and fovy."""
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if world_to_camera.shape != (4, 4) or intrinsics.shape != (3, 3) or image_height <= 0:
        raise ValueError("invalid camera matrices or image height")
    camera_to_world = np.linalg.inv(world_to_camera)
    cv_to_gl = np.diag([1.0, -1.0, -1.0])
    local_to_world_gl = camera_to_world[:3, :3] @ cv_to_gl
    quaternion = rotation_matrix_to_quaternion(local_to_world_gl)
    fovy = float(np.degrees(2 * np.arctan(image_height / (2 * intrinsics[1, 1]))))
    return camera_to_world[:3, 3], quaternion, fovy


def rotation_matrix_to_quaternion(rotation: NDArray[np.floating]) -> NDArray[np.float32]:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        following, last = (axis + 1) % 3, (axis + 2) % 3
        scale = 2 * np.sqrt(
            1 + rotation[axis, axis] - rotation[following, following] - rotation[last, last]
        )
        quaternion = np.zeros(4)
        quaternion[axis + 1] = 0.25 * scale
        quaternion[0] = (rotation[last, following] - rotation[following, last]) / scale
        quaternion[following + 1] = (
            rotation[following, axis] + rotation[axis, following]
        ) / scale
        quaternion[last + 1] = (rotation[last, axis] + rotation[axis, last]) / scale
    return (quaternion / np.linalg.norm(quaternion)).astype(np.float32)
