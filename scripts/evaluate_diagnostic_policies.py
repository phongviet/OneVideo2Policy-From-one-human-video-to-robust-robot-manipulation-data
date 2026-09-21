"""Run closed-loop evaluations for local robosuite diagnostic policies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import robosuite as suite
import torch
from robosuite.controllers import (
    load_composite_controller_config,
    load_part_controller_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robosuite_policy_models import (  # noqa: E402
    ChunkImagePolicy,
    DiffusionChunkPolicy,
    PhaseImagePolicy,
    StatePolicy,
    sample_diffusion_actions,
)


def phase_goal(
    phase: int,
    eef: np.ndarray,
    grasp_xy: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    if phase == 0:
        return np.r_[grasp_xy, 1.02], -1.0
    if phase == 1:
        return np.r_[grasp_xy, 0.88], -1.0
    if phase == 2:
        return eef, 1.0
    if phase == 3:
        return np.r_[grasp_xy, 1.06], 1.0
    if phase == 4:
        return np.r_[target[:2], 1.06], 1.0
    if phase == 5:
        return np.r_[target[:2], 0.885], 1.0
    if phase == 6:
        return eef, -1.0
    return np.r_[target[:2], 1.06], -1.0


def make_model(checkpoint: dict, device: str) -> torch.nn.Module:
    kind = checkpoint["kind"]
    if kind == "state":
        model = StatePolicy(checkpoint["state_dim"], checkpoint["action_dim"])
    elif kind == "phase":
        model = PhaseImagePolicy(checkpoint["proprio_dim"], checkpoint["action_dim"], 3)
    elif kind in ("chunk", "absolute_chunk"):
        model = ChunkImagePolicy(
            checkpoint["proprio_dim"],
            checkpoint["action_dim"],
            checkpoint["horizon"],
            6 if checkpoint["dual_camera"] else 3,
            bounded=kind != "absolute_chunk",
        )
    elif kind == "diffusion":
        model = DiffusionChunkPolicy(
            checkpoint["proprio_dim"],
            checkpoint["action_dim"],
            checkpoint["horizon"],
            image_channels=6 if checkpoint["dual_camera"] else 3,
        )
    else:
        raise ValueError(f"Unsupported closed-loop policy kind: {kind}")
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def image_tensor(obs: dict, dual: bool, device: str) -> torch.Tensor:
    images = [obs["agentview_image"]]
    if dual:
        images.append(obs["frontview_image"])
    image = torch.from_numpy(np.concatenate(images, axis=-1)).to(device=device, dtype=torch.float32)
    return image.permute(2, 0, 1).unsqueeze(0) / 255


def set_can_position(env, position: np.ndarray) -> dict:
    obj = env.objects[env.object_id]
    joint = obj.joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = position
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    return env._get_observations(force_update=True)


def absolute_joint_controller() -> dict:
    """Build a Panda controller that accepts joint angles in radians."""
    config = load_composite_controller_config(robot="Panda")
    arm = load_part_controller_config(default_controller="JOINT_POSITION")
    arm["input_type"] = "absolute"
    arm["gripper"] = {"type": "GRIP"}
    config["body_parts"]["right"] = arm
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--execute-chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--initial-positions-data", type=Path)
    args = parser.parse_args()
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = make_model(checkpoint, device)
    cameras = ["agentview", "frontview"] if checkpoint["dual_camera"] else ["agentview"]
    positions = None
    if args.initial_positions_data:
        source = np.load(args.initial_positions_data)
        ends = source["episode_ends"].astype(int)
        starts = np.r_[0, ends[:-1]]
        positions = (
            source["can_positions"][starts]
            if "can_positions" in source
            else source["objects"][starts, 7:10]
        )
    env = suite.make(
        "PickPlace",
        robots="Panda",
        controller_configs=(
            absolute_joint_controller() if checkpoint["kind"] == "absolute_chunk" else None
        ),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cameras,
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
    results, trajectories, trajectory_ends = [], [], []
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            resets = 1
            while obs["Can_pos"][1] < -0.30:
                obs = env.reset()
                resets += 1
            if positions is not None:
                obs = set_can_position(env, positions[episode % len(positions)])
            initial_can = obs["Can_pos"].copy()
            grasp_xy = initial_can[:2].copy()
            target = env.target_bin_placements[env.object_id].copy()
            phase, hold, queue = 0, 0, []
            success = False
            for step in range(args.max_steps):
                eef = obs["robot0_eef_pos"].copy()
                prop = obs["robot0_proprio-state"].astype(np.float32)
                with torch.inference_mode():
                    if checkpoint["kind"] == "state":
                        state = np.r_[prop, obs["object-state"]].astype(np.float32)
                        state = (state - checkpoint["state_mean"]) / checkpoint["state_std"]
                        action = model(torch.from_numpy(state).to(device).unsqueeze(0))[0]
                        action = action.cpu().numpy()
                    elif checkpoint["kind"] == "phase":
                        normalized = (prop - checkpoint["proprio_mean"]) / checkpoint["proprio_std"]
                        action = (
                            model(
                                image_tensor(obs, False, device),
                                torch.from_numpy(normalized).to(device).unsqueeze(0),
                                torch.tensor([phase], device=device),
                            )[0]
                            .cpu()
                            .numpy()
                        )
                    elif checkpoint["kind"] in ("chunk", "absolute_chunk"):
                        if not queue:
                            normalized = (prop - checkpoint["proprio_mean"]) / checkpoint[
                                "proprio_std"
                            ]
                            predicted = (
                                model(
                                    image_tensor(obs, checkpoint["dual_camera"], device),
                                    torch.from_numpy(normalized).to(device).unsqueeze(0),
                                )[0]
                                .cpu()
                                .numpy()
                            )
                            queue = list(predicted[: args.execute_chunk])
                        action = queue.pop(0)
                    else:
                        if not queue:
                            normalized = (prop - checkpoint["proprio_mean"]) / checkpoint[
                                "proprio_std"
                            ]
                            predicted = (
                                sample_diffusion_actions(
                                    model,
                                    image_tensor(obs, checkpoint["dual_camera"], device),
                                    torch.from_numpy(normalized).to(device).unsqueeze(0),
                                )[0]
                                .cpu()
                                .numpy()
                            )
                            queue = list(predicted[: args.execute_chunk])
                        action = queue.pop(0)
                action = np.asarray(action)
                if checkpoint["kind"] == "absolute_chunk":
                    # The controller consumes radians directly; only the gripper is normalized.
                    action[:7] = np.clip(action[:7], -3.0, 3.0)
                    action[-1] = np.clip(action[-1], -1.0, 1.0)
                else:
                    action = action.clip(-1, 1)
                trajectories.append(np.r_[episode, step, phase, eef, obs["Can_pos"], action])
                obs, _, done, _ = env.step(action)
                goal, _ = phase_goal(phase, eef, grasp_xy, target)
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
            trajectory_ends.append(len(trajectories))
            results.append(
                {
                    "episode": episode,
                    "success": success,
                    "steps": step + 1,
                    "resets_to_workspace": resets,
                    "initial_can_position": initial_can.tolist(),
                    "final_can_position": obs["Can_pos"].tolist(),
                    "terminal_phase": phase,
                    "final_xy_error_to_bin_center": float(
                        np.linalg.norm(obs["Can_pos"][:2] - target[:2])
                    ),
                }
            )
            print(
                f"episode {episode + 1}/{args.episodes}: success={success}, phase={phase}",
                flush=True,
            )
    finally:
        env.close()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "trajectories.npz",
        values=np.asarray(trajectories, dtype=np.float32),
        episode_ends=np.asarray(trajectory_ends, dtype=np.int64),
    )
    successes = sum(item["success"] for item in results)
    report = {
        "status": "closed_loop_evaluation",
        "policy_kind": checkpoint["kind"],
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "execute_chunk": (
            args.execute_chunk
            if checkpoint["kind"] in ("chunk", "diffusion", "absolute_chunk")
            else 1
        ),
        "training_initial_positions": args.initial_positions_data is not None,
        "median_final_xy_error_to_bin_center": float(
            np.median([item["final_xy_error_to_bin_center"] for item in results])
        ),
        "terminal_phase_counts": {
            str(phase): sum(item["terminal_phase"] == phase for item in results)
            for phase in sorted({item["terminal_phase"] for item in results})
        },
        "elapsed_seconds": time.perf_counter() - started,
        "episodes_detail": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
