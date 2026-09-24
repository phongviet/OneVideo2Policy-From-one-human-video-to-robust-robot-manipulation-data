from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def estimate_rgbd_step(
    first_rgb: NDArray[np.uint8],
    first_depth_m: NDArray[np.floating],
    second_rgb: NDArray[np.uint8],
    second_depth_m: NDArray[np.floating],
    intrinsics: NDArray[np.floating],
    *,
    exclusion_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], dict[str, float | int]]:
    """Estimate a metric first-camera to second-camera transform with RGB-D PnP."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional video extra
        raise RuntimeError("RGB-D odometry requires: uv sync --extra video") from exc
    first_depth_m = np.asarray(first_depth_m, dtype=np.float32)
    second_depth_m = np.asarray(second_depth_m, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if first_rgb.shape != second_rgb.shape or first_rgb.shape[:2] != first_depth_m.shape:
        raise ValueError("RGB and depth frame dimensions must match")
    if second_depth_m.shape != first_depth_m.shape or intrinsics.shape != (3, 3):
        raise ValueError("invalid depth or intrinsic dimensions")
    valid = np.isfinite(first_depth_m) & (first_depth_m > 0.1) & (first_depth_m < 10.0)
    if exclusion_mask is not None:
        exclusion_mask = np.asarray(exclusion_mask, dtype=bool)
        if exclusion_mask.shape != valid.shape:
            raise ValueError("exclusion mask shape must match depth")
        valid &= ~exclusion_mask
    feature_mask = valid.astype(np.uint8) * 255
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=8)
    first_points, first_descriptors = orb.detectAndCompute(
        cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY), feature_mask
    )
    second_points, second_descriptors = orb.detectAndCompute(
        cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY), None
    )
    if first_descriptors is None or second_descriptors is None:
        raise ValueError("Insufficient RGB features for odometry")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    first_xy = np.float32([first_points[match.queryIdx].pt for match in matches])
    second_xy = np.float32([second_points[match.trainIdx].pt for match in matches])
    width, height = first_rgb.shape[1], first_rgb.shape[0]
    x_index = np.clip(np.rint(first_xy[:, 0]).astype(int), 0, width - 1)
    y_index = np.clip(np.rint(first_xy[:, 1]).astype(int), 0, height - 1)
    depth = first_depth_m[y_index, x_index]
    depth_valid = np.isfinite(depth) & (depth > 0.1) & (depth < 10.0)
    first_xy, second_xy, depth = first_xy[depth_valid], second_xy[depth_valid], depth[depth_valid]
    if len(depth) < 20:
        raise ValueError("Fewer than 20 depth-backed feature matches")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    object_points = np.column_stack(
        (
            (first_xy[:, 0] - cx) * depth / fx,
            (first_xy[:, 1] - cy) * depth / fy,
            depth,
        )
    ).astype(np.float32)
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points,
        second_xy,
        intrinsics,
        None,
        iterationsCount=200,
        reprojectionError=2.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 20:
        raise ValueError("RGB-D PnP failed to find 20 inliers")
    inlier_indices = inliers[:, 0]
    success, rotation_vector, translation = cv2.solvePnP(
        object_points[inlier_indices],
        second_xy[inlier_indices],
        intrinsics,
        None,
        rotation_vector,
        translation,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("RGB-D PnP refinement failed")
    rotation = cv2.Rodrigues(rotation_vector)[0]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation[:, 0]
    projected = cv2.projectPoints(
        object_points[inlier_indices], rotation_vector, translation, intrinsics, None
    )[0][:, 0]
    reprojection = np.linalg.norm(projected - second_xy[inlier_indices], axis=1)
    transformed = object_points[inlier_indices] @ rotation.T + translation[:, 0]
    second_x = np.clip(np.rint(second_xy[inlier_indices, 0]).astype(int), 0, width - 1)
    second_y = np.clip(np.rint(second_xy[inlier_indices, 1]).astype(int), 0, height - 1)
    measured_second_depth = second_depth_m[second_y, second_x]
    depth_pairs = np.isfinite(measured_second_depth) & (measured_second_depth > 0.1)
    depth_residual = np.abs(transformed[depth_pairs, 2] - measured_second_depth[depth_pairs])
    diagnostics: dict[str, float | int] = {
        "descriptor_matches": len(matches),
        "depth_matches": len(object_points),
        "inliers": len(inlier_indices),
        "inlier_ratio": len(inlier_indices) / len(object_points),
        "median_reprojection_error_px": float(np.median(reprojection)),
        "translation_m": float(np.linalg.norm(translation)),
        "rotation_deg": float(np.degrees(np.linalg.norm(rotation_vector))),
        "depth_consistency_pairs": int(np.count_nonzero(depth_pairs)),
        "median_depth_residual_m": (
            float(np.median(depth_residual)) if len(depth_residual) else float("nan")
        ),
    }
    return transform, diagnostics


def accumulate_world_to_camera(
    relative_transforms: Sequence[NDArray[np.floating]],
) -> NDArray[np.float64]:
    """Accumulate transforms that map camera i coordinates into camera i+1."""
    poses = [np.eye(4, dtype=np.float64)]
    for transform in relative_transforms:
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("relative transforms must have shape 4x4")
        poses.append(transform @ poses[-1])
    return np.stack(poses)


def estimate_rgbd_trajectory(
    rgb_frames: Sequence[NDArray[np.uint8]],
    depth_frames_m: Sequence[NDArray[np.floating]],
    intrinsics: NDArray[np.floating],
    *,
    exclusion_masks: Sequence[NDArray[np.bool_]] | None = None,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    if len(rgb_frames) != len(depth_frames_m) or len(rgb_frames) < 2:
        raise ValueError("at least two matching RGB-D frames are required")
    if exclusion_masks is not None and len(exclusion_masks) != len(rgb_frames):
        raise ValueError("exclusion mask count must match frame count")
    transforms = []
    diagnostics = []
    for frame_id in range(len(rgb_frames) - 1):
        transform, step = estimate_rgbd_step(
            rgb_frames[frame_id],
            depth_frames_m[frame_id],
            rgb_frames[frame_id + 1],
            depth_frames_m[frame_id + 1],
            intrinsics,
            exclusion_mask=(None if exclusion_masks is None else exclusion_masks[frame_id]),
        )
        transforms.append(transform)
        diagnostics.append({"from_frame": frame_id, "to_frame": frame_id + 1, **step})
    return accumulate_world_to_camera(transforms), diagnostics
