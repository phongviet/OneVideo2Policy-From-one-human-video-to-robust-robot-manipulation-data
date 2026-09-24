from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianScene:
    """A metric RGB-D initialized 3D Gaussian scene.

    Coordinates use metres. Quaternions are stored as ``[w, x, y, z]`` and colors
    use linear values in ``[0, 1]``. Labels are 0=static, 1=source, 2=target.
    """

    means_m: NDArray[np.float32]
    log_scales_m: NDArray[np.float32]
    rotations_wxyz: NDArray[np.float32]
    opacities: NDArray[np.float32]
    colors_rgb: NDArray[np.float32]
    labels: NDArray[np.uint8]

    def validate(self) -> None:
        count = len(self.means_m)
        expected = {
            "means_m": (count, 3),
            "log_scales_m": (count, 3),
            "rotations_wxyz": (count, 4),
            "opacities": (count,),
            "colors_rgb": (count, 3),
            "labels": (count,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if count == 0 or not np.isfinite(self.means_m).all():
            raise ValueError("scene must contain finite Gaussian means")
        if not np.isfinite(self.log_scales_m).all() or not np.isfinite(self.colors_rgb).all():
            raise ValueError("scene parameters must be finite")
        if np.any(self.opacities <= 0) or np.any(self.opacities > 1):
            raise ValueError("opacities must be in (0, 1]")
        if np.any(self.colors_rgb < 0) or np.any(self.colors_rgb > 1):
            raise ValueError("colors must be in [0, 1]")

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(1, dtype=np.int64),
            means_m=self.means_m,
            log_scales_m=self.log_scales_m,
            rotations_wxyz=self.rotations_wxyz,
            opacities=self.opacities,
            colors_rgb=self.colors_rgb,
            labels=self.labels,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> GaussianScene:
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("Unsupported Gaussian scene schema")
            scene = cls(
                means_m=data["means_m"].astype(np.float32),
                log_scales_m=data["log_scales_m"].astype(np.float32),
                rotations_wxyz=data["rotations_wxyz"].astype(np.float32),
                opacities=data["opacities"].astype(np.float32),
                colors_rgb=data["colors_rgb"].astype(np.float32),
                labels=data["labels"].astype(np.uint8),
            )
        scene.validate()
        return scene


def initialize_gaussians_from_rgbd(
    rgb: NDArray[np.uint8],
    depth_m: NDArray[np.floating],
    intrinsics: NDArray[np.floating],
    *,
    source_mask: NDArray[np.bool_] | None = None,
    target_mask: NDArray[np.bool_] | None = None,
    stride: int = 3,
    opacity: float = 0.9,
) -> GaussianScene:
    """Unproject an RGB-D frame into anisotropic 3D Gaussian parameters."""
    rgb = np.asarray(rgb)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[:2] != depth_m.shape or rgb.shape[2] != 3:
        raise ValueError("rgb and depth must have matching HxW dimensions")
    if intrinsics.shape != (3, 3) or stride <= 0 or not 0 < opacity <= 1:
        raise ValueError("invalid intrinsics, stride, or opacity")
    height, width = depth_m.shape
    sampled_y, sampled_x = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[sampled_y, sampled_x]
    valid = np.isfinite(z) & (z > 0.1) & (z < 10.0)
    x_px = sampled_x[valid].astype(np.float32)
    y_px = sampled_y[valid].astype(np.float32)
    z = z[valid]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (x_px - cx) * z / fx
    y = (y_px - cy) * z / fy
    means = np.column_stack((x, y, z)).astype(np.float32)

    # One standard deviation covers about half of the sampled pixel footprint.
    tangent_scale = np.maximum(z * stride * 0.55 / np.sqrt(fx * fy), 1e-5)
    normal_scale = np.maximum(tangent_scale * 0.35, 1e-5)
    scales = np.column_stack((tangent_scale, tangent_scale, normal_scale)).astype(np.float32)
    rotations = np.zeros((len(means), 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    colors = rgb[sampled_y[valid], sampled_x[valid]].astype(np.float32) / 255.0
    labels = np.zeros(len(means), dtype=np.uint8)
    if source_mask is not None:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != depth_m.shape:
            raise ValueError("source mask shape must match depth")
        labels[source_mask[sampled_y[valid], sampled_x[valid]]] = 1
    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool)
        if target_mask.shape != depth_m.shape:
            raise ValueError("target mask shape must match depth")
        target_values = target_mask[sampled_y[valid], sampled_x[valid]]
        # The placed source can project inside the target mask. Visible source pixels
        # retain priority because their measured depth occludes the target surface.
        labels[target_values & (labels == 0)] = 2
    scene = GaussianScene(
        means_m=means,
        log_scales_m=np.log(scales),
        rotations_wxyz=rotations,
        opacities=np.full(len(means), opacity, dtype=np.float32),
        colors_rgb=colors,
        labels=labels,
    )
    scene.validate()
    return scene


def transform_label(
    scene: GaussianScene, label: int, transform: NDArray[np.floating]
) -> GaussianScene:
    """Apply a rigid transform to one semantic Gaussian group."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("transform must be a 4x4 rigid matrix")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("transform rotation must be orthonormal")
    selected = scene.labels == label
    means = scene.means_m.copy()
    homogeneous = np.column_stack((means[selected], np.ones(np.count_nonzero(selected))))
    means[selected] = (homogeneous @ transform.T)[:, :3]
    rotations = scene.rotations_wxyz.copy()
    transform_quaternion = _rotation_matrix_to_quaternion(rotation)
    rotations[selected] = _quaternion_multiply(transform_quaternion, rotations[selected])
    result = replace(scene, means_m=means, rotations_wxyz=rotations)
    result.validate()
    return result


def transform_scene(scene: GaussianScene, transform: NDArray[np.floating]) -> GaussianScene:
    result = scene
    for label in np.unique(scene.labels):
        result = transform_label(result, int(label), transform)
    return result


def fuse_gaussian_scenes(
    scenes: list[GaussianScene],
    camera_to_world: NDArray[np.floating],
    *,
    reference_index: int,
    voxel_size_m: float = 0.01,
    min_static_observations: int = 2,
) -> GaussianScene:
    """Fuse static Gaussians across views and retain movable groups from one view."""
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if len(scenes) < 2 or camera_to_world.shape != (len(scenes), 4, 4):
        raise ValueError("scenes and camera poses must contain at least two matching views")
    if not 0 <= reference_index < len(scenes):
        raise ValueError("reference index is outside the scene list")
    if voxel_size_m <= 0 or min_static_observations <= 0:
        raise ValueError("voxel size and minimum observations must be positive")
    transformed = [
        transform_scene(scene, camera_to_world[index]) for index, scene in enumerate(scenes)
    ]
    static_means = []
    static_scales = []
    static_rotations = []
    static_opacities = []
    static_colors = []
    frame_ids = []
    for frame_id, scene in enumerate(transformed):
        selected = scene.labels == 0
        static_means.append(scene.means_m[selected])
        static_scales.append(scene.log_scales_m[selected])
        static_rotations.append(scene.rotations_wxyz[selected])
        static_opacities.append(scene.opacities[selected])
        static_colors.append(scene.colors_rgb[selected])
        frame_ids.append(np.full(np.count_nonzero(selected), frame_id, dtype=np.int32))
    means = np.concatenate(static_means)
    log_scales = np.concatenate(static_scales)
    rotations = np.concatenate(static_rotations)
    opacities = np.concatenate(static_opacities)
    colors = np.concatenate(static_colors)
    frames = np.concatenate(frame_ids)
    voxels = np.floor(means / voxel_size_m).astype(np.int64)
    _, first_indices, inverse = np.unique(voxels, axis=0, return_inverse=True, return_index=True)
    voxel_count = int(inverse.max()) + 1
    observation_pairs = np.unique(np.column_stack((inverse, frames)), axis=0)
    support = np.bincount(observation_pairs[:, 0], minlength=voxel_count)
    keep_voxel = support >= min_static_observations
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)

    def mean_by_voxel(values: NDArray[np.floating]) -> NDArray[np.float64]:
        output = np.zeros((voxel_count,) + values.shape[1:], dtype=np.float64)
        np.add.at(output, inverse, values)
        return output / counts.reshape((-1,) + (1,) * (values.ndim - 1))

    fused_means = mean_by_voxel(means)[keep_voxel].astype(np.float32)
    fused_scales = mean_by_voxel(log_scales)[keep_voxel].astype(np.float32)
    fused_opacities = mean_by_voxel(opacities)[keep_voxel].astype(np.float32)
    fused_colors = mean_by_voxel(colors)[keep_voxel].astype(np.float32)
    fused_rotations = rotations[first_indices[keep_voxel]].astype(np.float32)

    reference = transformed[reference_index]
    movable = reference.labels != 0
    fused = GaussianScene(
        means_m=np.concatenate((fused_means, reference.means_m[movable])),
        log_scales_m=np.concatenate((fused_scales, reference.log_scales_m[movable])),
        rotations_wxyz=np.concatenate((fused_rotations, reference.rotations_wxyz[movable])),
        opacities=np.concatenate((fused_opacities, reference.opacities[movable])),
        colors_rgb=np.concatenate((fused_colors, reference.colors_rgb[movable])),
        labels=np.concatenate(
            (np.zeros(len(fused_means), dtype=np.uint8), reference.labels[movable])
        ),
    )
    fused.validate()
    return fused


def _rotation_matrix_to_quaternion(rotation: NDArray[np.floating]) -> NDArray[np.float32]:
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
        next_axis, last_axis = (axis + 1) % 3, (axis + 2) % 3
        scale = 2 * np.sqrt(
            1
            + rotation[axis, axis]
            - rotation[next_axis, next_axis]
            - rotation[last_axis, last_axis]
        )
        quaternion = np.zeros(4)
        quaternion[axis + 1] = 0.25 * scale
        quaternion[0] = (rotation[last_axis, next_axis] - rotation[next_axis, last_axis]) / scale
        quaternion[next_axis + 1] = (rotation[next_axis, axis] + rotation[axis, next_axis]) / scale
        quaternion[last_axis + 1] = (rotation[last_axis, axis] + rotation[axis, last_axis]) / scale
    return (quaternion / np.linalg.norm(quaternion)).astype(np.float32)


def _quaternion_multiply(
    left: NDArray[np.floating], right: NDArray[np.floating]
) -> NDArray[np.float32]:
    lw, lx, ly, lz = np.broadcast_to(left, right.shape).T
    rw, rx, ry, rz = np.asarray(right).T
    return np.column_stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    ).astype(np.float32)


def _quaternion_to_rotation_matrix(
    quaternion: NDArray[np.floating],
) -> NDArray[np.float64]:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)


def render_gaussians(
    scene: GaussianScene,
    intrinsics: NDArray[np.floating],
    world_to_camera: NDArray[np.floating],
    *,
    width: int,
    height: int,
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Render projected 3D Gaussians with front-to-back alpha compositing."""
    scene.validate()
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    if intrinsics.shape != (3, 3) or world_to_camera.shape != (4, 4):
        raise ValueError("invalid camera matrices")
    homogeneous = np.column_stack((scene.means_m, np.ones(len(scene.means_m))))
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    z = camera[:, 2]
    valid = z > 1e-4
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = fx * camera[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * camera[:, 1] / np.maximum(z, 1e-6) + cy
    rotations = _quaternion_to_rotation_matrix(scene.rotations_wxyz)
    scales2 = np.exp(2 * scene.log_scales_m).astype(np.float64)
    covariance_world = np.einsum(
        "nij,njk,nlk->nil", rotations, scales2[:, :, None] * np.eye(3), rotations
    )
    camera_rotation = world_to_camera[:3, :3]
    covariance_camera = np.einsum(
        "ij,njk,lk->nil", camera_rotation, covariance_world, camera_rotation
    )
    jacobian = np.zeros((len(scene.means_m), 2, 3), dtype=np.float64)
    safe_z = np.maximum(z, 1e-6)
    jacobian[:, 0, 0] = fx / safe_z
    jacobian[:, 0, 2] = -fx * camera[:, 0] / safe_z**2
    jacobian[:, 1, 1] = fy / safe_z
    jacobian[:, 1, 2] = -fy * camera[:, 1] / safe_z**2
    covariance_screen = np.einsum("nij,njk,nlk->nil", jacobian, covariance_camera, jacobian)
    covariance_screen[:, 0, 0] += 0.35**2
    covariance_screen[:, 1, 1] += 0.35**2
    max_sigma = np.sqrt(np.linalg.eigvalsh(covariance_screen)[:, 1])
    valid &= (u + 3 * max_sigma >= 0) & (u - 3 * max_sigma < width)
    valid &= (v + 3 * max_sigma >= 0) & (v - 3 * max_sigma < height)

    color = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float32)
    transmittance = np.ones((height, width), dtype=np.float32)
    for index in np.flatnonzero(valid)[np.argsort(z[valid])]:
        radius = max(1, int(np.ceil(3 * max_sigma[index])))
        x0, x1 = (
            max(0, int(np.floor(u[index])) - radius),
            min(width, int(np.floor(u[index])) + radius + 1),
        )
        y0, y1 = (
            max(0, int(np.floor(v[index])) - radius),
            min(height, int(np.floor(v[index])) + radius + 1),
        )
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        delta = np.stack((grid_x - u[index], grid_y - v[index]), axis=-1)
        inverse_covariance = np.linalg.inv(covariance_screen[index])
        mahalanobis2 = np.einsum("...i,ij,...j->...", delta, inverse_covariance, delta)
        alpha = scene.opacities[index] * np.exp(-0.5 * mahalanobis2)
        alpha = np.minimum(alpha.astype(np.float32), 0.995)
        patch_t = transmittance[y0:y1, x0:x1]
        contribution = patch_t * alpha
        color[y0:y1, x0:x1] += contribution[..., None] * scene.colors_rgb[index]
        depth[y0:y1, x0:x1] += contribution * z[index]
        transmittance[y0:y1, x0:x1] *= 1.0 - alpha
    accumulated_alpha = 1.0 - transmittance
    background = np.asarray(background_rgb, dtype=np.float32)
    color += transmittance[..., None] * background
    visible = accumulated_alpha > 1e-6
    depth[visible] /= accumulated_alpha[visible]
    return np.clip(color, 0, 1), depth, accumulated_alpha
