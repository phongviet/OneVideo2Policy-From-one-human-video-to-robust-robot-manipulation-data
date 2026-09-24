"""Render randomized ball placements for low-cost visual localization training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import robosuite as suite
import robosuite_ball_bowl_env  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=500, type=int)
    parser.add_argument("--seed", default=2026, type=int)
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
        horizon=1,
        hard_reset=False,
        reward_shaping=False,
    )
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
            obs = env.reset()
            records["images"].append(obs["agentview_image"].copy())
            records["images_front"].append(obs["frontview_image"].copy())
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
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
