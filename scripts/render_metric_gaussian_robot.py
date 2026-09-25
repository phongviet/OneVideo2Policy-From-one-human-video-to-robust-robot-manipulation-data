"""Register robosuite to HOI4D metric anchors and depth-composite it with Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import robosuite as suite
import robosuite_ball_bowl_env  # noqa: F401

from onevideo2policy.generation.gaussian_scene import (
    GaussianScene,
    render_gaussians,
)
from onevideo2policy.generation.registration import (
    opencv_camera_to_mujoco_pose,
    register_task_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth-dir", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--camera-trajectory", required=True, type=Path)
    parser.add_argument("--object-trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", default=24, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.camera_trajectory, allow_pickle=False) as camera:
        intrinsics = camera["intrinsics"].astype(np.float64)
        camera_to_world = camera["camera_to_world"]
        world_to_camera = camera["world_to_camera"]
    with np.load(args.object_trajectory, allow_pickle=False) as objects:
        source_world = objects["source_world_xyz_m"]
        relative_xyz = objects["relative_xyz_m"]
    width = int(manifest["resolution"]["width"])
    height = int(manifest["resolution"]["height"])
    observed_rgb, depth_m, source_mask, target_mask = load_rgbd(
        args.frame, manifest, args.manifest, args.depth_dir, args.masks, width, height
    )
    plane_center, plane_normal, plane_diagnostics = fit_table_plane(
        depth_m,
        intrinsics,
        camera_to_world[args.frame],
        source_mask | target_mask,
        source_world[args.frame],
    )
    source_hoi = source_world[args.frame]
    target_hoi = source_hoi - relative_xyz[args.frame]

    env = suite.make(
        "BallToBowl",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=height,
        camera_widths=width,
        camera_depths=True,
        camera_segmentations="class",
        control_freq=20,
        horizon=1,
        hard_reset=False,
        reward_shaping=False,
    )
    try:
        env.reset()
        source_sim = env.ball_position
        target_sim = env.bowl_position
        sim_to_hoi = register_task_frames(
            source_sim, target_sim, source_hoi, target_hoi, plane_normal
        )
        registered_world_to_camera = world_to_camera[args.frame] @ sim_to_hoi
        configure_mujoco_camera(
            env, "agentview", registered_world_to_camera, intrinsics, height
        )
        observation = env._get_observations(force_update=True)
        # Robosuite exposes MuJoCo's bottom-up offscreen convention; HOI4D and the
        # Gaussian renderer use top-down OpenCV image coordinates.
        sim_rgb = np.flipud(observation["agentview_image"])
        sim_depth_m = np.flipud(
            metric_mujoco_depth(env, np.squeeze(observation["agentview_depth"]))
        )
        segmentation = np.flipud(np.squeeze(observation["agentview_segmentation_class"]))
        class_names = list(env.model.classes_to_ids.keys())
    finally:
        env.close()

    scene = GaussianScene.load(args.scene)
    static = select_static(scene)
    gaussian_rgb, gaussian_depth_m, gaussian_alpha = render_gaussians(
        static,
        intrinsics,
        world_to_camera[args.frame],
        width=width,
        height=height,
    )
    foreground = segmentation > 0
    depth_visible = foreground & (
        (gaussian_depth_m <= 0) | (sim_depth_m <= gaussian_depth_m + 0.005)
    )
    depth_occluded = foreground & ~depth_visible
    composite = (gaussian_rgb * 255).astype(np.uint8)
    composite[depth_visible] = sim_rgb[depth_visible]

    projected_target_hoi = project(target_hoi, world_to_camera[args.frame], intrinsics)
    projected_target_sim = project(
        transform_point(target_sim, sim_to_hoi), world_to_camera[args.frame], intrinsics
    )
    bowl_pixels = segmentation == class_names.index("FixedHollowCylinderObject") + 1
    bowl_y, bowl_x = np.where(bowl_pixels)
    bowl_centroid = (
        np.array([bowl_x.mean(), bowl_y.mean()]) if len(bowl_x) else np.array([np.nan, np.nan])
    )
    mapped_up = sim_to_hoi[:3, :3] @ np.array([0.0, 0.0, 1.0])
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "metric task-anchor registration with Gaussian/simulator depth ordering",
        "frame_id": args.frame,
        "anchors": ["placed bowl center", "table normal", "source-to-target direction"],
        "sim_to_hoi4d": sim_to_hoi.tolist(),
        "rotation_determinant": float(np.linalg.det(sim_to_hoi[:3, :3])),
        "table_normal_alignment_deg": float(
            np.degrees(np.arccos(np.clip(np.dot(mapped_up, plane_normal), -1, 1)))
        ),
        "target_anchor_3d_error_m": float(
            np.linalg.norm(transform_point(target_sim, sim_to_hoi) - target_hoi)
        ),
        "target_anchor_reprojection_error_px": float(
            np.linalg.norm(projected_target_sim - projected_target_hoi)
        ),
        "rendered_bowl_centroid_to_anchor_px": float(
            np.linalg.norm(bowl_centroid - projected_target_hoi)
        ),
        "table_plane": plane_diagnostics,
        "class_ids": {name: index + 1 for index, name in enumerate(class_names)},
        "gaussian_static_count": len(static.means_m),
        "gaussian_coverage": float(np.mean(gaussian_alpha > 0.5)),
        "simulator_foreground_pixels": int(np.count_nonzero(foreground)),
        "depth_visible_foreground_pixels": int(np.count_nonzero(depth_visible)),
        "depth_occluded_foreground_pixels": int(np.count_nonzero(depth_occluded)),
        "depth_ordered_fraction_visible": float(
            np.count_nonzero(depth_visible) / max(np.count_nonzero(foreground), 1)
        ),
        "limitations": [
            "Registration uses semantic 3D task anchors rather than image correspondences.",
            "The sparse Gaussian scene has holes and does not model contact shadows.",
            "The full registered rollout is validated by generate_metric_gaussian_demo.py.",
        ],
    }
    write_rgb(args.output / "observed.png", observed_rgb)
    write_rgb(args.output / "registered-simulator.png", sim_rgb)
    write_rgb(args.output / "static-gaussian.png", (gaussian_rgb * 255).astype(np.uint8))
    write_rgb(args.output / "metric-composite.png", composite)
    audit = np.hstack((observed_rgb, sim_rgb, composite))
    for index, label in enumerate(("observed", "registered simulator", "depth composite")):
        cv2.putText(
            audit,
            label,
            (index * width + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
    write_rgb(args.output / "audit.png", audit)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def load_rgbd(
    frame_id: int,
    manifest: dict[str, Any],
    manifest_path: Path,
    depth_dir: Path,
    masks: Path,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = manifest["frames"][frame_id]
    rgb_path = Path(frame["rgb"])
    if not rgb_path.is_absolute():
        rgb_path = manifest_path.parent / rgb_path
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    source_id = int(frame["source_frame_id"])
    raw_depth = cv2.imread(str(depth_dir / f"{source_id:05d}.png"), cv2.IMREAD_UNCHANGED)
    source = cv2.imread(str(masks / "source" / f"{frame_id:06d}.png"), 0)
    target = cv2.imread(str(masks / "target" / f"{frame_id:06d}.png"), 0)
    if bgr is None or raw_depth is None or source is None or target is None:
        raise FileNotFoundError(f"RGB-D or masks missing for frame {frame_id}")
    depth_m = cv2.resize(
        raw_depth.astype(np.float32) / 1000,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth_m, source > 0, target > 0


def fit_table_plane(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    exclusion: np.ndarray,
    source_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    yy, xx = np.mgrid[0 : depth_m.shape[0] : 4, 0 : depth_m.shape[1] : 4]
    z = depth_m[yy, xx]
    valid = (
        (z > 0.2)
        & (z < 2.0)
        & ~exclusion[yy, xx]
        & (xx > int(depth_m.shape[1] * 0.375))
        & (yy > int(depth_m.shape[0] * 0.47))
    )
    z = z[valid]
    camera_points = np.column_stack(
        (
            (xx[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (yy[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    world_points = (
        np.column_stack((camera_points, np.ones(len(camera_points)))) @ camera_to_world.T
    )[:, :3]
    generator = np.random.default_rng(7)
    best: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(3000):
        sample = world_points[generator.choice(len(world_points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-7:
            continue
        normal /= norm
        residual = np.abs((world_points - sample[0]) @ normal)
        score = int(np.count_nonzero(residual < 0.008))
        if best is None or score > best[0]:
            best = (score, normal, sample[0], residual)
    if best is None:
        raise ValueError("Table plane RANSAC failed")
    _, _, _, residual = best
    inliers = residual < 0.008
    center = world_points[inliers].mean(axis=0)
    _, _, axes = np.linalg.svd(world_points[inliers] - center, full_matrices=False)
    normal = axes[-1]
    if np.dot(normal, source_world - center) < 0:
        normal = -normal
    refined_residual = np.abs((world_points - center) @ normal)
    diagnostics = {
        "candidate_points": len(world_points),
        "inliers": int(np.count_nonzero(inliers)),
        "inlier_ratio": float(np.mean(inliers)),
        "median_inlier_residual_m": float(np.median(refined_residual[inliers])),
        "normal_world": normal.tolist(),
        "center_world_m": center.tolist(),
    }
    return center, normal, diagnostics


def configure_mujoco_camera(
    env: Any,
    camera_name: str,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
) -> None:
    position, quaternion, fovy = opencv_camera_to_mujoco_pose(
        world_to_camera, intrinsics, height
    )
    camera_id = env.sim.model.camera_name2id(camera_name)
    env.sim.model.cam_pos[camera_id] = position
    env.sim.model.cam_quat[camera_id] = quaternion
    env.sim.model.cam_fovy[camera_id] = fovy
    env.sim.forward()


def metric_mujoco_depth(env: Any, normalized: np.ndarray) -> np.ndarray:
    extent = env.sim.model.stat.extent
    far = env.sim.model.vis.map.zfar * extent
    near = env.sim.model.vis.map.znear * extent
    return near / (1 - normalized * (1 - near / far))


def select_static(scene: GaussianScene) -> GaussianScene:
    selected = scene.labels == 0
    static = GaussianScene(
        means_m=scene.means_m[selected],
        log_scales_m=scene.log_scales_m[selected],
        rotations_wxyz=scene.rotations_wxyz[selected],
        opacities=scene.opacities[selected],
        colors_rgb=scene.colors_rgb[selected],
        labels=scene.labels[selected],
    )
    static.validate()
    return static


def transform_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform @ np.r_[point, 1.0])[:3]


def project(point: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    camera = transform_point(point, world_to_camera)
    return np.array(
        [
            intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2],
        ]
    )


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
