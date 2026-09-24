"""Evaluate the learned visual waypoint policy on the measured ball-to-bowl task."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import robosuite as suite
import robosuite_ball_bowl_env  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robosuite_policy_models import VisualWaypointPolicy  # noqa: E402

from onevideo2policy.generation.compositing import (  # noqa: E402
    composite_task_foreground,
    load_video_frames,
)


def image_tensor(obs: dict, device: str, background: np.ndarray | None = None) -> torch.Tensor:
    agent = obs["agentview_image"]
    front = obs["frontview_image"]
    if background is not None:
        agent, _ = composite_task_foreground(agent, obs["agentview_segmentation_class"], background)
        front, _ = composite_task_foreground(
            front, obs["frontview_segmentation_class"], np.fliplr(background)
        )
    image = np.concatenate((agent, front), axis=-1)
    tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0) / 255


def set_ball_position(env, position: np.ndarray) -> dict:
    qpos = env.sim.data.get_joint_qpos(env.objects[0].joints[0]).copy()
    qpos[:3] = position
    env.sim.data.set_joint_qpos(env.objects[0].joints[0], qpos)
    env.sim.forward()
    return env._get_observations(force_update=True)


def set_camera_observables(env, enabled: bool) -> None:
    for name, observable in env._observables.items():
        if name.startswith(("agentview_", "frontview_")):
            observable.set_enabled(enabled)


def phase_goal(
    phase: int, eef: np.ndarray, grasp_xy: np.ndarray, ball: np.ndarray, bowl: np.ndarray
) -> tuple[np.ndarray, float]:
    if phase == 0:
        return np.r_[grasp_xy, ball[2] + 0.10], -1.0
    if phase == 1:
        return np.r_[grasp_xy, ball[2] - 0.005], -1.0
    if phase == 2:
        return eef, 1.0
    if phase == 3:
        return np.r_[grasp_xy, 1.02], 1.0
    if phase == 4:
        return np.r_[bowl[:2], 1.02], 1.0
    if phase == 5:
        return np.r_[bowl[:2], bowl[2] + 0.03], 1.0
    if phase == 6:
        return eef, -1.0
    return np.r_[bowl[:2], 1.02], -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", default=5, type=int)
    parser.add_argument("--max-steps", default=650, type=int)
    parser.add_argument("--seed", default=1729, type=int)
    parser.add_argument("--initial-positions-data", type=Path)
    parser.add_argument("--gaussian-background-video", type=Path)
    parser.add_argument(
        "--oracle-ball-position",
        action="store_true",
        help="Use simulator ball XY to isolate controller performance from perception.",
    )
    parser.add_argument(
        "--continuous-localization",
        action="store_true",
        help="Re-estimate during approach instead of freezing the unobstructed first frame.",
    )
    parser.add_argument("--grasp-retries", default=4, type=int)
    parser.add_argument("--camera-jitter-m", default=0.0, type=float)
    parser.add_argument("--lighting-scale", default=1.0, type=float)
    args = parser.parse_args()
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["kind"] != "visual_waypoint":
        raise ValueError("checkpoint must be a visual_waypoint policy")
    model = VisualWaypointPolicy(6).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    positions = None
    if args.initial_positions_data:
        with np.load(args.initial_positions_data, allow_pickle=False) as source:
            ends = source["episode_ends"].astype(int)
            starts = np.r_[0, ends[:-1]]
            positions = source["ball_positions"][starts]
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
        horizon=args.max_steps,
        hard_reset=False,
        reward_shaping=False,
        **camera_options,
    )
    backgrounds = (
        load_video_frames(args.gaussian_background_video, 84, 84)
        if args.gaussian_background_video
        else None
    )
    camera_ids = [env.sim.model.camera_name2id(name) for name in ("agentview", "frontview")]
    base_camera_positions = env.sim.model.cam_pos[camera_ids].copy()
    base_light_diffuse = env.sim.model.light_diffuse.copy()
    base_light_ambient = env.sim.model.light_ambient.copy()
    rng = np.random.default_rng(args.seed + 1)
    results = []
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            set_camera_observables(env, True)
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
            if positions is not None:
                obs = set_ball_position(env, positions[episode % len(positions)])
            true_initial = env.ball_position.copy()
            history = []
            phase = hold = 0
            grasp_attempts = 1
            retry_xy_errors = []
            success = False
            initial_prediction = None
            background = backgrounds[episode % len(backgrounds)] if backgrounds else None
            for _step in range(args.max_steps):
                eef = obs["robot0_eef_pos"].copy()
                ball = env.ball_position
                bowl = env.bowl_position
                if phase in (0, 1):
                    if initial_prediction is None or args.continuous_localization:
                        if args.oracle_ball_position:
                            prediction = ball.copy()
                        else:
                            with torch.inference_mode():
                                prediction = (
                                    model(image_tensor(obs, device, background))[0].cpu().numpy()
                                    * checkpoint["can_position_std"]
                                    + checkpoint["can_position_mean"]
                                )
                        if initial_prediction is None:
                            initial_prediction = prediction.copy()
                        history.append(prediction)
                        history = history[-5:]
                        grasp_xy = np.median(history, axis=0)[:2]
                        if not args.continuous_localization:
                            set_camera_observables(env, False)
                goal, gripper = phase_goal(phase, eef, grasp_xy, ball, bowl)
                action = np.r_[np.clip((goal - eef) / 0.05, -0.25, 0.25), np.zeros(3), gripper]
                obs, _, done, _ = env.step(action)
                error = np.linalg.norm(goal - eef)
                if phase == 3 and error < 0.008:
                    grasped = env._check_grasp(
                        gripper=env.robots[0].gripper,
                        object_geoms=env.objects[0].contact_geoms,
                    )
                    if not grasped and grasp_attempts <= args.grasp_retries:
                        if args.oracle_ball_position:
                            retry_prediction = env.ball_position.copy()
                        else:
                            set_camera_observables(env, True)
                            recovery_obs = env._get_observations(force_update=True)
                            with torch.inference_mode():
                                retry_prediction = (
                                    model(image_tensor(recovery_obs, device, background))[0]
                                    .cpu()
                                    .numpy()
                                    * checkpoint["can_position_std"]
                                    + checkpoint["can_position_mean"]
                                )
                            set_camera_observables(env, False)
                        grasp_xy = retry_prediction[:2]
                        retry_xy_errors.append(
                            float(np.linalg.norm(grasp_xy - env.ball_position[:2]))
                        )
                        grasp_attempts += 1
                        phase = 0
                    else:
                        phase += 1
                    hold = 0
                elif phase in (0, 1, 4, 7) and error < 0.008:
                    phase += 1
                    hold = 0
                elif phase == 5 and np.linalg.norm(eef[:2] - bowl[:2]) < 0.008 and eef[2] < 0.925:
                    phase += 1
                    hold = 0
                elif phase in (2, 6):
                    hold += 1
                    if hold >= 25:
                        phase += 1
                        hold = 0
                success = bool(env._check_success())
                if success or done:
                    break
            results.append(
                {
                    "episode": episode,
                    "success": success,
                    "steps": _step + 1,
                    "terminal_phase": phase,
                    "grasp_attempts": grasp_attempts,
                    "retry_xy_errors_m": retry_xy_errors,
                    "initial_ball_position": true_initial.tolist(),
                    "initial_prediction": initial_prediction.tolist(),
                    "initial_localization_error_m": float(
                        np.linalg.norm(initial_prediction - true_initial)
                    ),
                    "initial_xy_localization_error_m": float(
                        np.linalg.norm(initial_prediction[:2] - true_initial[:2])
                    ),
                    "final_ball_position": env.ball_position.tolist(),
                    "final_bowl_error_m": float(
                        np.linalg.norm(env.ball_position - env.bowl_position)
                    ),
                }
            )
            print(f"episode {episode + 1}/{args.episodes}: success={success}", flush=True)
    finally:
        env.close()
    successes = sum(item["success"] for item in results)
    report = {
        "status": "closed_loop_evaluation",
        "task": "measured HOI4D ball-to-bowl geometry",
        "policy": (
            "oracle ball position with scripted controller"
            if args.oracle_ball_position
            else "dual-view initial visual waypoint with scripted controller"
        ),
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "median_initial_localization_error_m": float(
            np.median([item["initial_localization_error_m"] for item in results])
        ),
        "median_initial_xy_localization_error_m": float(
            np.median([item["initial_xy_localization_error_m"] for item in results])
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "appearance": "gaussian_background_composite" if backgrounds else "robosuite_rgb",
        "camera_alignment": ("appearance_only_unregistered" if backgrounds else "native_robosuite"),
        "camera_jitter_m": args.camera_jitter_m,
        "lighting_scale": args.lighting_scale,
        "episodes_detail": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
