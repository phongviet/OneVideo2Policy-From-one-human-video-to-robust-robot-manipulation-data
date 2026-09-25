"""Replay one robot demo through two metric-registered Gaussian scene cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import robosuite as suite
import robosuite_ball_bowl_env  # noqa: F401
from render_metric_gaussian_robot import (
    configure_mujoco_camera,
    fit_table_plane,
    load_rgbd,
    metric_mujoco_depth,
    select_static,
)

from onevideo2policy.generation.gaussian_scene import GaussianScene, render_gaussians
from onevideo2policy.generation.registration import register_task_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth-dir", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--camera-trajectory", required=True, type=Path)
    parser.add_argument("--object-trajectory", required=True, type=Path)
    parser.add_argument("--demonstrations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-frames", nargs=2, default=[24, 36], type=int)
    parser.add_argument("--render-width", default=320, type=int)
    parser.add_argument("--render-height", default=180, type=int)
    parser.add_argument("--output-size", default=84, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    calibration_width = int(manifest["resolution"]["width"])
    calibration_height = int(manifest["resolution"]["height"])
    with np.load(args.camera_trajectory, allow_pickle=False) as camera:
        intrinsics = camera["intrinsics"].astype(np.float64)
        camera_to_world = camera["camera_to_world"]
        world_to_camera = camera["world_to_camera"]
    with np.load(args.object_trajectory, allow_pickle=False) as objects:
        source_world = objects["source_world_xyz_m"]
        relative_xyz = objects["relative_xyz_m"]
    with np.load(args.demonstrations, allow_pickle=False) as demonstrations:
        end = int(demonstrations["episode_ends"][0])
        original = {key: demonstrations[key][:end].copy() for key in demonstrations.files}
    original["episode_ends"] = np.asarray([end], dtype=np.int64)

    anchor_frame = args.camera_frames[0]
    _, depth_m, source_mask, target_mask = load_rgbd(
        anchor_frame,
        manifest,
        args.manifest,
        args.depth_dir,
        args.masks,
        calibration_width,
        calibration_height,
    )
    _, plane_normal, plane_diagnostics = fit_table_plane(
        depth_m,
        intrinsics,
        camera_to_world[anchor_frame],
        source_mask | target_mask,
        source_world[anchor_frame],
    )
    target_hoi = source_world[anchor_frame] - relative_xyz[anchor_frame]

    env = suite.make(
        "BallToBowl",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "frontview"],
        camera_heights=args.render_height,
        camera_widths=args.render_width,
        camera_depths=True,
        camera_segmentations="class",
        control_freq=20,
        horizon=1,
        hard_reset=False,
        reward_shaping=False,
    )
    video_writer = cv2.VideoWriter(
        str(args.output / "metric-registered-demo.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (args.render_width * 2, args.render_height),
    )
    if not video_writer.isOpened():
        env.close()
        raise RuntimeError("Could not open metric demo video")
    try:
        env.reset()
        set_robot_and_ball(env, original["joint_positions"][0], original["ball_positions"][0])
        sim_to_hoi = register_task_frames(
            original["ball_positions"][0],
            env.bowl_position,
            source_world[anchor_frame],
            target_hoi,
            plane_normal,
        )
        scaled_intrinsics = intrinsics.copy()
        scaled_intrinsics[0] *= args.render_width / calibration_width
        scaled_intrinsics[1] *= args.render_height / calibration_height
        for name, frame_id in zip(
            ("agentview", "frontview"), args.camera_frames, strict=True
        ):
            configure_mujoco_camera(
                env,
                name,
                world_to_camera[frame_id] @ sim_to_hoi,
                scaled_intrinsics,
                args.render_height,
            )
        static = select_static(GaussianScene.load(args.scene))
        gaussian_backgrounds = []
        gaussian_depths = []
        for frame_id in args.camera_frames:
            rgb, depth, _ = render_gaussians(
                static,
                scaled_intrinsics,
                world_to_camera[frame_id],
                width=args.render_width,
                height=args.render_height,
            )
            gaussian_backgrounds.append((rgb * 255).astype(np.uint8))
            gaussian_depths.append(depth)
        registered_images = [[], []]
        visible_counts = np.zeros(2, dtype=np.int64)
        occluded_counts = np.zeros(2, dtype=np.int64)
        foreground_counts = np.zeros(2, dtype=np.int64)
        for sample_id in range(end):
            set_robot_and_ball(
                env,
                original["joint_positions"][sample_id],
                original["ball_positions"][sample_id],
            )
            observation = env._get_observations(force_update=True)
            pair = []
            for view_id, name in enumerate(("agentview", "frontview")):
                sim_rgb = np.flipud(observation[f"{name}_image"])
                sim_depth = np.flipud(
                    metric_mujoco_depth(env, np.squeeze(observation[f"{name}_depth"]))
                )
                segmentation = np.flipud(
                    np.squeeze(observation[f"{name}_segmentation_class"])
                )
                foreground = segmentation > 0
                visible = foreground & (
                    (gaussian_depths[view_id] <= 0)
                    | (sim_depth <= gaussian_depths[view_id] + 0.005)
                )
                composite = gaussian_backgrounds[view_id].copy()
                composite[visible] = sim_rgb[visible]
                registered_images[view_id].append(
                    cv2.resize(
                        composite,
                        (args.output_size, args.output_size),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                foreground_counts[view_id] += np.count_nonzero(foreground)
                visible_counts[view_id] += np.count_nonzero(visible)
                occluded_counts[view_id] += np.count_nonzero(foreground & ~visible)
                pair.append(composite)
            video_writer.write(
                cv2.cvtColor(np.hstack(pair), cv2.COLOR_RGB2BGR)
            )
            if (sample_id + 1) % 50 == 0:
                print(f"rendered {sample_id + 1}/{end}", flush=True)
    finally:
        video_writer.release()
        env.close()

    original["images"] = np.asarray(registered_images[0], dtype=np.uint8)
    original["images_front"] = np.asarray(registered_images[1], dtype=np.uint8)
    np.savez_compressed(args.output / "demonstrations.npz", **original)
    report = {
        "schema_version": 1,
        "status": "pass",
        "method": "dual-view metric task-anchor registration and depth compositing",
        "samples": end,
        "episodes": 1,
        "camera_frames": args.camera_frames,
        "render_resolution": [args.render_width, args.render_height],
        "dataset_resolution": [args.output_size, args.output_size],
        "sim_to_hoi4d": sim_to_hoi.tolist(),
        "table_plane": plane_diagnostics,
        "per_view": [
            {
                "name": name,
                "foreground_pixels": int(foreground_counts[index]),
                "visible_pixels": int(visible_counts[index]),
                "occluded_pixels": int(occluded_counts[index]),
                "visible_fraction": float(
                    visible_counts[index] / max(foreground_counts[index], 1)
                ),
            }
            for index, name in enumerate(("agentview", "frontview"))
        ],
        "synchronization": {
            "states_actions_copied_without_resampling": True,
            "episode_end": end,
            "image_count_matches_actions": len(original["actions"]) == end,
        },
        "limitations": [
            "Sparse Gaussian holes and missing contact shadows remain visible.",
            "This replays one successful trajectory rather than evaluating a physical robot.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def set_robot_and_ball(env: object, joints: np.ndarray, ball_position: np.ndarray) -> None:
    env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes] = joints
    ball_qpos = env.sim.data.get_joint_qpos(env.objects[0].joints[0]).copy()
    ball_qpos[:3] = ball_position
    env.sim.data.set_joint_qpos(env.objects[0].joints[0], ball_qpos)
    env.sim.forward()


if __name__ == "__main__":
    main()
