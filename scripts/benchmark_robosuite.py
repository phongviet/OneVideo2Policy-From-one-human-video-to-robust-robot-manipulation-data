"""Measure headless Panda PickPlace startup and rendered step rate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import robosuite as suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    report = {"robosuite": suite.__version__, "task": "PickPlace", "robot": "Panda"}
    for size in (84, 128):
        start = time.perf_counter()
        env = suite.make(
            "PickPlace",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=size,
            camera_widths=size,
            control_freq=20,
            horizon=args.steps,
            ignore_done=True,
        )
        observation = env.reset()
        startup = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(args.steps):
            observation, _, _, _ = env.step(np.zeros(env.action_dim))
        seconds = time.perf_counter() - start
        report[str(size)] = {
            "startup_seconds": startup,
            "steps_per_second": args.steps / seconds,
            "camera_shape": list(observation["agentview_image"].shape),
            "action_dim": env.action_dim,
        }
        env.close()
    report["note"] = "Zero-action throughput only; no task success evaluation."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
