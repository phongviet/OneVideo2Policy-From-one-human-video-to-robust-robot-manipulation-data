"""Render randomized ball placements for low-cost visual localization training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import robosuite as suite
import robosuite_ball_bowl_env  # noqa: F401

from onevideo2policy.generation.compositing import composite_task_foreground, load_video_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=500, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--gaussian-background-video", type=Path)
    parser.add_argument("--camera-jitter-m", default=0.0, type=float)
    parser.add_argument("--lighting-scale", default=1.0, type=float)
    parser.add_argument(
        "--robot-poses-data",
        type=Path,
        help="Optional demonstration NPZ supplying varied Panda joint positions.",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    camera_options = {}
    if args.gaussian_background_video:
        camera_options["camera_segmentations"] = "class"
    env = suite.make(
        "BallToBowl",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "frontview"],
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=1,
        hard_reset=False,
        reward_shaping=False,
        **camera_options,
    )
    backgrounds = (
        load_video_frames(args.gaussian_background_video, 84, 84)
        if args.gaussian_background_video
        else None
    )
    robot_poses = None
    if args.robot_poses_data:
        with np.load(args.robot_poses_data, allow_pickle=False) as pose_data:
            robot_poses = pose_data["joint_positions"].copy()
    camera_ids = [env.sim.model.camera_name2id(name) for name in ("agentview", "frontview")]
    base_camera_positions = env.sim.model.cam_pos[camera_ids].copy()
    base_light_diffuse = env.sim.model.light_diffuse.copy()
    base_light_ambient = env.sim.model.light_ambient.copy()
    rng = np.random.default_rng(args.seed + 1)
    foreground_fractions = []
    records: dict[str, list[np.ndarray | int]] = {
        key: []
        for key in (
            "images",
            "images_front",
            "proprio",
            "objects",
            "actions",
            "ball_positions",
            "bowl_positions",
            "phases",
        )
    }
    started = time.perf_counter()
    try:
        for index in range(args.samples):
            env.sim.model.cam_pos[camera_ids] = base_camera_positions + rng.uniform(
                -args.camera_jitter_m, args.camera_jitter_m, size=(2, 3)
            )
            env.sim.model.light_diffuse[:] = np.clip(
                base_light_diffuse * args.lighting_scale, 0, 1
            )
            env.sim.model.light_ambient[:] = np.clip(
                base_light_ambient * args.lighting_scale, 0, 1
            )
            obs = env.reset()
            if robot_poses is not None:
                pose = robot_poses[index * len(robot_poses) // args.samples]
                env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes] = pose
                env.sim.forward()
                obs = env._get_observations(force_update=True)
            agent_image = obs["agentview_image"].copy()
            front_image = obs["frontview_image"].copy()
            if backgrounds:
                background = backgrounds[index % len(backgrounds)]
                agent_image, agent_alpha = composite_task_foreground(
                    agent_image, obs["agentview_segmentation_class"], background
                )
                front_image, front_alpha = composite_task_foreground(
                    front_image,
                    obs["frontview_segmentation_class"],
                    np.fliplr(background),
                )
                foreground_fractions.extend((agent_alpha.mean(), front_alpha.mean()))
            records["images"].append(agent_image)
            records["images_front"].append(front_image)
            records["proprio"].append(obs["robot0_proprio-state"].copy())
            records["objects"].append(obs["object-state"].copy())
            records["actions"].append(np.zeros(env.action_dim, dtype=np.float32))
            records["ball_positions"].append(env.ball_position)
            records["bowl_positions"].append(env.bowl_position)
            records["phases"].append(0)
            if (index + 1) % 100 == 0:
                print(f"rendered {index + 1}/{args.samples}", flush=True)
    finally:
        env.close()
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["episode_ends"] = np.arange(1, args.samples + 1, dtype=np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "localization.npz", **arrays)
    positions = arrays["ball_positions"]
    report = {
        "status": "pass",
        "task": "randomized measured ball localization",
        "samples": args.samples,
        "position_min_m": positions.min(axis=0).tolist(),
        "position_max_m": positions.max(axis=0).tolist(),
        "elapsed_seconds": time.perf_counter() - started,
        "appearance": "gaussian_background_composite" if backgrounds else "robosuite_rgb",
        "mean_foreground_fraction": (
            float(np.mean(foreground_fractions)) if foreground_fractions else None
        ),
        "camera_alignment": ("appearance_only_unregistered" if backgrounds else "native_robosuite"),
        "robot_pose_diversity": "demonstration_joint_poses"
        if robot_poses is not None
        else "reset_pose",
        "camera_jitter_m": args.camera_jitter_m,
        "lighting_scale": args.lighting_scale,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
