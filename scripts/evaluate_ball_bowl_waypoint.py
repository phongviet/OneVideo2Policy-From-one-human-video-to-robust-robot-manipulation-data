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


def image_tensor(obs: dict, device: str) -> torch.Tensor:
    image = np.concatenate((obs["agentview_image"], obs["frontview_image"]), axis=-1)
    tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0) / 255


def set_ball_position(env, position: np.ndarray) -> dict:
    qpos = env.sim.data.get_joint_qpos(env.objects[0].joints[0]).copy()
    qpos[:3] = position
    env.sim.data.set_joint_qpos(env.objects[0].joints[0], qpos)
    env.sim.forward()
    return env._get_observations(force_update=True)


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
    parser.add_argument("--max-steps", default=450, type=int)
    parser.add_argument("--seed", default=1729, type=int)
    parser.add_argument("--initial-positions-data", type=Path)
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
    )
    results = []
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            if positions is not None:
                obs = set_ball_position(env, positions[episode % len(positions)])
            true_initial = env.ball_position.copy()
            history = []
            phase = hold = 0
            success = False
            initial_prediction = None
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
                                    model(image_tensor(obs, device))[0].cpu().numpy()
                                    * checkpoint["can_position_std"]
                                    + checkpoint["can_position_mean"]
                                )
                        if initial_prediction is None:
                            initial_prediction = prediction.copy()
                        history.append(prediction)
                        history = history[-5:]
                        grasp_xy = np.median(history, axis=0)[:2]
                goal, gripper = phase_goal(phase, eef, grasp_xy, ball, bowl)
                action = np.r_[np.clip((goal - eef) / 0.05, -0.25, 0.25), np.zeros(3), gripper]
                obs, _, done, _ = env.step(action)
                error = np.linalg.norm(goal - eef)
                if phase in (0, 1, 3, 4, 7) and error < 0.008:
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
        "episodes_detail": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
