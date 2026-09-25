"""Render the frozen physical camera and task layout in the policy task frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("configs/physical_camera_reference.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/physical-camera-layout.png")
    )
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    base = np.asarray(reference["simulation_robot_base_in_task"], dtype=np.float64)[:3, 3]
    target = np.asarray(reference["target_center_task_m"], dtype=np.float64)
    source = np.array([0.0731500512, -0.2025296591, 0.8191606110])
    cameras = []
    for camera in reference["cameras"]:
        transform = np.asarray(camera["camera_to_task"], dtype=np.float64)
        cameras.append((camera["name"], transform[:3, 3], transform[:3, 2]))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    views = (
        (0, 1, "Top view", "task x (m)", "task y (m)"),
        (0, 2, "Side view", "task x (m)", "task z (m)"),
    )
    for axis, (horizontal, vertical, title, xlabel, ylabel) in zip(axes, views, strict=True):
        axis.scatter(
            base[horizontal],
            base[vertical],
            marker="s",
            s=90,
            color="#333333",
            label="Panda base",
        )
        axis.scatter(source[horizontal], source[vertical], s=80, color="#f28e2b", label="source")
        axis.scatter(target[horizontal], target[vertical], s=110, color="#4e79a7", label="target")
        for name, position, forward in cameras:
            axis.scatter(position[horizontal], position[vertical], marker="^", s=90, label=name)
            axis.arrow(
                position[horizontal], position[vertical], forward[horizontal] * 0.18,
                forward[vertical] * 0.18, width=0.006, head_width=0.035,
                length_includes_head=True, color="#59a14f",
            )
            if title == "Top view":
                offset = (-5, -18) if name == "frontview" else (5, 10)
            else:
                offset = (-5, 6) if name == "frontview" else (5, 6)
            alignment = "right" if name == "frontview" else "left"
            axis.annotate(
                name,
                (position[horizontal], position[vertical]),
                xytext=offset,
                textcoords="offset points",
                horizontalalignment=alignment,
            )
        if vertical == 2:
            axis.axhline(
                reference["table_height_task_m"],
                color="#9c755f",
                linestyle="--",
                label="table",
            )
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.15)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes[0].legend(
        unique.values(), unique.keys(), loc="upper left", ncol=2, frameon=False, fontsize=8
    )
    figure.suptitle("Frozen physical transfer layout (robosuite task frame)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
