from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def fit_fixed_radius_sphere(
    points_m: NDArray[np.floating], radius_m: float, *, iterations: int = 30
) -> tuple[NDArray[np.float64], float]:
    """Robustly fit a known-radius sphere center to a visible depth surface."""
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 12 or radius_m <= 0:
        raise ValueError("sphere fitting needs at least 12 3D points and a positive radius")
    center = np.median(points, axis=0)
    center[2] += radius_m
    for _ in range(iterations):
        delta = center - points
        distance = np.linalg.norm(delta, axis=1)
        residual = distance - radius_m
        jacobian = delta / np.maximum(distance[:, None], 1e-8)
        centered = residual - np.median(residual)
        mad = np.median(np.abs(centered)) + 1e-6
        weights = np.minimum(1.0, 2.5 * mad / np.maximum(np.abs(centered), 1e-8))
        step = np.linalg.lstsq(jacobian * weights[:, None], -residual * weights, rcond=None)[0]
        center += step
        if np.linalg.norm(step) < 1e-7:
            break
    surface_error = np.abs(np.linalg.norm(points - center, axis=1) - radius_m)
    return center, float(np.median(surface_error))


def recover_sphere_relative_trajectory(
    depth_frames_m: Sequence[NDArray[np.floating]],
    source_masks: Sequence[NDArray[np.bool_]],
    intrinsics: NDArray[np.floating],
    camera_to_world: NDArray[np.floating],
    target_centers_camera_m: NDArray[np.floating],
    *,
    radius_m: float,
    max_surface_error_m: float = 0.006,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    list[dict[str, Any]],
]:
    """Recover sphere translation relative to the target with camera rotation removed.

    Sphere rotation is intentionally identity because it is unobservable and irrelevant
    for a rotationally symmetric collision object.
    """
    frame_count = len(depth_frames_m)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    target_centers = np.asarray(target_centers_camera_m, dtype=np.float64)
    if len(source_masks) != frame_count or frame_count < 2:
        raise ValueError("depth and source mask sequences must match")
    if camera_to_world.shape != (frame_count, 4, 4) or target_centers.shape != (
        frame_count,
        3,
    ):
        raise ValueError("camera poses and target centers must match frame count")
    relative_centers = np.full((frame_count, 3), np.nan, dtype=np.float64)
    world_centers = np.full((frame_count, 3), np.nan, dtype=np.float64)
    accepted = np.zeros(frame_count, dtype=bool)
    diagnostics = []
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    for frame_id, (depth, mask) in enumerate(zip(depth_frames_m, source_masks, strict=True)):
        depth = np.asarray(depth, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if depth.shape != mask.shape:
            raise ValueError("each depth and mask frame must have matching dimensions")
        y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0.1) & (depth < 10.0))
        record: dict[str, Any] = {"frame_id": frame_id, "depth_points": len(x)}
        if len(x) >= 12:
            z = depth[y, x]
            points = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
            center_camera, surface_error = fit_fixed_radius_sphere(points, radius_m)
            record["median_surface_error_m"] = surface_error
            if surface_error <= max_surface_error_m:
                relative_camera = center_camera - target_centers[frame_id]
                relative_centers[frame_id] = camera_to_world[frame_id, :3, :3] @ relative_camera
                world_centers[frame_id] = (camera_to_world[frame_id] @ np.r_[center_camera, 1.0])[
                    :3
                ]
                accepted[frame_id] = True
                record["accepted"] = True
                record["center_camera_m"] = center_camera.tolist()
            else:
                record["accepted"] = False
                record["rejection"] = "surface_error"
        else:
            record["accepted"] = False
            record["rejection"] = "insufficient_depth"
        diagnostics.append(record)
    valid_frames = np.flatnonzero(accepted)
    if len(valid_frames) < 2:
        raise ValueError("fewer than two accepted sphere observations")
    frame_ids = np.arange(frame_count)
    for axis in range(3):
        relative_centers[:, axis] = np.interp(
            frame_ids, valid_frames, relative_centers[valid_frames, axis]
        )
        world_centers[:, axis] = np.interp(
            frame_ids, valid_frames, world_centers[valid_frames, axis]
        )
    return relative_centers, world_centers, accepted, diagnostics
