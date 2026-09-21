"""Generate successful headless robosuite can-to-bin demonstrations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import robosuite as suite


def action_toward(
    current: np.ndarray, goal: np.ndarray, gripper: float, action_limit: float
) -> np.ndarray:
    translation = np.clip((goal - current) / 0.05, -action_limit, action_limit)
    return np.r_[translation, np.zeros(3), gripper]


def collect_episode(
    env,
    *,
    max_steps: int,
    recovery_probability: float,
    recovery_magnitude: float,
    action_limit: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], bool]:
    obs = env.reset()
    while obs["Can_pos"][1] < -0.30:
        obs = env.reset()
    target = env.target_bin_placements[env.object_id].copy()
    grasp_xy = obs["Can_pos"][:2].copy()
    phase, hold = 0, 0
    record_keys = (
        "images",
        "images_front",
        "proprio",
        "objects",
        "actions",
        "joint_positions",
        "joint_targets",
        "eef_positions",
        "can_positions",
        "target_positions",
        "phases",
        "recovery_states",
    )
    records = {key: [] for key in record_keys}
    success = False
    for _ in range(max_steps):
        recovery_state = False
        if phase in (0, 1, 3, 4, 5, 7) and rng.random() < recovery_probability:
            perturbation = np.zeros(7)
            perturbation[:3] = rng.uniform(-recovery_magnitude, recovery_magnitude, 3)
            perturbation[-1] = -1.0 if phase < 2 or phase >= 6 else 1.0
            obs, _, done, _ = env.step(perturbation)
            recovery_state = True
            if done:
                break
        eef = obs["robot0_eef_pos"]
        obj = obs["Can_pos"]
        if phase == 0:
            goal, gripper = np.r_[grasp_xy, 1.02], -1.0
        elif phase == 1:
            goal, gripper = np.r_[grasp_xy, 0.88], -1.0
        elif phase == 2:
            goal, gripper = eef, 1.0
        elif phase == 3:
            goal, gripper = np.r_[grasp_xy, 1.06], 1.0
        elif phase == 4:
            goal, gripper = np.r_[target[:2], 1.06], 1.0
        elif phase == 5:
            goal, gripper = np.r_[target[:2], 0.885], 1.0
        elif phase == 6:
            goal, gripper = eef, -1.0
        else:
            goal, gripper = np.r_[target[:2], 1.06], -1.0
        action = action_toward(eef, goal, gripper, action_limit)
        records["images"].append(obs["agentview_image"].copy())
        records["images_front"].append(obs["frontview_image"].copy())
        records["proprio"].append(obs["robot0_proprio-state"].copy())
        records["objects"].append(obs["object-state"].copy())
        records["actions"].append(action.copy())
        records["joint_positions"].append(obs["robot0_joint_pos"].copy())
        records["eef_positions"].append(eef.copy())
        records["can_positions"].append(obj.copy())
        records["target_positions"].append(target.copy())
        records["phases"].append(phase)
        records["recovery_states"].append(recovery_state)
        next_obs, _, done, _ = env.step(action)
        records["joint_targets"].append(np.r_[next_obs["robot0_joint_pos"].copy(), gripper])
        obs = next_obs
        error = np.linalg.norm(goal - eef)
        if phase in (0, 1, 3, 4, 5, 7) and error < 0.012:
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
        if not np.isfinite(obj).all():
            break
    return {key: np.asarray(value) for key, value in records.items()}, success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-probability", type=float, default=0.03)
    parser.add_argument("--recovery-magnitude", type=float, default=0.25)
    parser.add_argument("--action-limit", type=float, default=0.4)
    args = parser.parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "frontview"],
        camera_heights=84,
        camera_widths=84,
        reward_shaping=True,
        single_object_mode=2,
        object_type="can",
        control_freq=20,
        horizon=args.max_steps,
        ignore_done=False,
        hard_reset=False,
    )
    episodes, attempts = [], 0
    started = time.perf_counter()
    try:
        while len(episodes) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            episode, success = collect_episode(
                env,
                max_steps=args.max_steps,
                recovery_probability=args.recovery_probability,
                recovery_magnitude=args.recovery_magnitude,
                action_limit=args.action_limit,
                rng=rng,
            )
            if success:
                episodes.append(episode)
                print(
                    f"successful episode {len(episodes)}/{args.episodes} after {attempts} attempts",
                    flush=True,
                )
    finally:
        env.close()
    if not episodes:
        raise RuntimeError("Scripted controller produced no successful episodes")
    keys = episodes[0].keys()
    concatenated = {key: np.concatenate([episode[key] for episode in episodes]) for key in keys}
    episode_ends = np.cumsum([len(episode["actions"]) for episode in episodes])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "demonstrations.npz", **concatenated, episode_ends=episode_ends
    )
    report = {
        "status": "pass" if len(episodes) == args.episodes else "partial",
        "scope": "robosuite built-in can-to-bin task; not the measured filmed objects",
        "environment": "PickPlace",
        "robot": "Panda",
        "controller": "OSC_POSE scripted waypoint expert",
        "cameras": ["agentview 84x84 RGB", "frontview 84x84 RGB"],
        "stored_action_representations": [
            "OSC_POSE delta with gripper",
            "next-step absolute joint positions with gripper",
        ],
        "recovery_probability": args.recovery_probability,
        "recovery_magnitude": args.recovery_magnitude,
        "action_limit": args.action_limit,
        "recovery_samples": int(concatenated["recovery_states"].sum()),
        "requested_episodes": args.episodes,
        "successful_episodes": len(episodes),
        "attempts": attempts,
        "samples": int(episode_ends[-1]),
        "episode_lengths": [int(len(episode["actions"])) for episode in episodes],
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "uses robosuite's built-in can and bin geometry",
            (
                "initial can poses with y < -0.30 m are rejected as outside this "
                "controller's workspace"
            ),
            "does not validate the real-video scale, reconstructed meshes, or physical transfer",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
