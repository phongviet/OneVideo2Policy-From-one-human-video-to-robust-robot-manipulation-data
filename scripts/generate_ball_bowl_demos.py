"""Generate successful Panda demonstrations for the measured HOI4D ball and bowl."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import robosuite as suite
import robosuite_ball_bowl_env  # noqa: F401


def action_toward(current: np.ndarray, goal: np.ndarray, gripper: float) -> np.ndarray:
    return np.r_[np.clip((goal - current) / 0.05, -0.25, 0.25), np.zeros(3), gripper]


def collect_episode(env, max_steps: int) -> tuple[dict[str, np.ndarray], bool]:
    obs = env.reset()
    grasp_xy = env.ball_position[:2].copy()
    phase = hold = 0
    keys = (
        "images",
        "images_front",
        "proprio",
        "objects",
        "actions",
        "joint_positions",
        "joint_targets",
        "eef_positions",
        "ball_positions",
        "bowl_positions",
        "phases",
    )
    records = {key: [] for key in keys}
    success = False
    for _ in range(max_steps):
        eef = obs["robot0_eef_pos"]
        ball = env.ball_position
        bowl = env.bowl_position
        if phase == 0:
            goal, gripper = np.r_[grasp_xy, ball[2] + 0.10], -1.0
        elif phase == 1:
            goal, gripper = np.r_[grasp_xy, ball[2] - 0.005], -1.0
        elif phase == 2:
            goal, gripper = eef, 1.0
        elif phase == 3:
            goal, gripper = np.r_[grasp_xy, 1.02], 1.0
        elif phase == 4:
            goal, gripper = np.r_[bowl[:2], 1.02], 1.0
        elif phase == 5:
            goal, gripper = np.r_[bowl[:2], bowl[2] + 0.03], 1.0
        elif phase == 6:
            goal, gripper = eef, -1.0
        else:
            goal, gripper = np.r_[bowl[:2], 1.02], -1.0
        action = action_toward(eef, goal, gripper)
        records["images"].append(obs["agentview_image"].copy())
        records["images_front"].append(obs["frontview_image"].copy())
        records["proprio"].append(obs["robot0_proprio-state"].copy())
        records["objects"].append(obs["object-state"].copy())
        records["actions"].append(action.copy())
        records["joint_positions"].append(obs["robot0_joint_pos"].copy())
        records["eef_positions"].append(eef.copy())
        records["ball_positions"].append(ball.copy())
        records["bowl_positions"].append(bowl.copy())
        records["phases"].append(phase)
        next_obs, _, done, _ = env.step(action)
        records["joint_targets"].append(np.r_[next_obs["robot0_joint_pos"].copy(), gripper])
        obs = next_obs
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
    return {key: np.asarray(value) for key, value in records.items()}, success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--max-attempts", default=20, type=int)
    parser.add_argument("--max-steps", default=450, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)
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
    episodes = []
    attempts = 0
    started = time.perf_counter()
    try:
        while len(episodes) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            episode, success = collect_episode(env, args.max_steps)
            if success:
                episodes.append(episode)
                print(
                    f"successful episode {len(episodes)}/{args.episodes} after {attempts} attempts",
                    flush=True,
                )
    finally:
        env.close()
    if not episodes:
        raise RuntimeError("Measured ball-to-bowl expert produced no successful episodes")
    episode_ends = np.cumsum([len(episode["actions"]) for episode in episodes])
    concatenated = {
        key: np.concatenate([episode[key] for episode in episodes]) for key in episodes[0]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "demonstrations.npz", **concatenated, episode_ends=episode_ends
    )
    report = {
        "status": "pass" if len(episodes) == args.episodes else "partial",
        "scope": "measured HOI4D ball and bowl dimensions in robosuite",
        "environment": "BallToBowl",
        "robot": "Panda",
        "controller": "OSC_POSE scripted waypoint expert",
        "ball_radius_m": robosuite_ball_bowl_env.BallToBowl.BALL_RADIUS,
        "bowl_outer_radius_m": robosuite_ball_bowl_env.BallToBowl.BOWL_OUTER_RADIUS,
        "bowl_height_m": robosuite_ball_bowl_env.BallToBowl.BOWL_HALF_HEIGHT * 2,
        "requested_episodes": args.episodes,
        "successful_episodes": len(episodes),
        "attempts": attempts,
        "samples": int(episode_ends[-1]),
        "episode_lengths": [int(len(episode["actions"])) for episode in episodes],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
