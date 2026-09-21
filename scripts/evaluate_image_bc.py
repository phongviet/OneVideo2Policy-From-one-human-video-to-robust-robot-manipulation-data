"""Evaluate a compact image behavior-cloning policy in headless robosuite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import robosuite as suite
import torch
from train_image_bc import ImageBC


def predict_action(
    model: ImageBC,
    image: np.ndarray,
    proprio: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    translation_gain: float,
    discrete_gripper: bool,
) -> np.ndarray:
    rgb = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0) / 255
    state = torch.from_numpy((proprio.astype(np.float32) - mean) / std)
    state = state.to(device=device).unsqueeze(0)
    with torch.inference_mode():
        action = model(rgb, state).squeeze(0).cpu().numpy()
    action[:3] *= translation_gain
    if discrete_gripper:
        action[-1] = 1.0 if action[-1] >= 0 else -1.0
    return action.clip(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--translation-gain", type=float, default=1.0)
    parser.add_argument("--discrete-gripper", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ImageBC(checkpoint["proprio_dim"], checkpoint["action_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(checkpoint["proprio_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["proprio_std"], dtype=np.float32)
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
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
    results = []
    preview_frames = []
    started = time.perf_counter()
    try:
        for episode_id in range(args.episodes):
            obs = env.reset()
            reset_count = 1
            while obs["Can_pos"][1] < -0.30:
                obs = env.reset()
                reset_count += 1
            initial_position = obs["Can_pos"].copy()
            success = False
            for step in range(args.max_steps):
                if episode_id == 0 and step % 10 == 0:
                    preview_frames.append(obs["agentview_image"].copy())
                action = predict_action(
                    model,
                    obs["agentview_image"],
                    obs["robot0_proprio-state"],
                    mean,
                    std,
                    device,
                    args.translation_gain,
                    args.discrete_gripper,
                )
                obs, _, done, _ = env.step(action)
                success = bool(env._check_success())
                if success or done:
                    break
            target = env.target_bin_placements[env.object_id]
            results.append(
                {
                    "episode": episode_id,
                    "success": success,
                    "steps": step + 1,
                    "resets_to_workspace": reset_count,
                    "initial_can_position": initial_position.tolist(),
                    "final_can_position": obs["Can_pos"].tolist(),
                    "final_xy_error_to_bin_center": float(
                        np.linalg.norm(obs["Can_pos"][:2] - target[:2])
                    ),
                }
            )
    finally:
        env.close()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "first_rollout_preview.npz",
        images=np.asarray(preview_frames, dtype=np.uint8),
    )
    successes = sum(row["success"] for row in results)
    report = {
        "status": "closed_loop_evaluation",
        "scope": "robosuite built-in can-to-bin task",
        "device": device,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "translation_gain": args.translation_gain,
        "discrete_gripper": args.discrete_gripper,
        "median_final_xy_error_to_bin_center": float(
            np.median([row["final_xy_error_to_bin_center"] for row in results])
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "episodes_detail": results,
        "limitation": (
            "This evaluates simulator imitation only; it does not establish real-object transfer."
        ),
    }
    (args.output / "closed_loop_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
