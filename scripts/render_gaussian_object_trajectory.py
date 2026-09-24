"""Render a recovered object trajectory through a fused metric Gaussian scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.generation.gaussian_scene import (
    GaussianScene,
    render_gaussians,
    transform_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--camera-trajectory", required=True, type=Path)
    parser.add_argument("--object-trajectory", required=True, type=Path)
    parser.add_argument("--source-masks", required=True, type=Path)
    parser.add_argument("--reference-frame", default=1, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", default=320, type=int)
    parser.add_argument("--height", default=180, type=int)
    parser.add_argument("--calibration-width", default=640, type=int)
    parser.add_argument("--calibration-height", default=360, type=int)
    parser.add_argument("--fps", default=5.0, type=float)
    args = parser.parse_args()
    scene = GaussianScene.load(args.scene)
    with np.load(args.camera_trajectory, allow_pickle=False) as camera:
        intrinsics = camera["intrinsics"].astype(np.float64)
        world_to_camera = camera["world_to_camera"]
    with np.load(args.object_trajectory, allow_pickle=False) as objects:
        world_xyz = objects["source_world_xyz_m"]
        observed = objects["depth_observed"]
    if len(world_to_camera) != len(world_xyz):
        raise ValueError("camera and object trajectories must have matching frame counts")
    source = scene.labels == 1
    if not np.any(source):
        raise ValueError("Gaussian scene contains no source object group")
    scale_x = args.width / args.calibration_width
    scale_y = args.height / args.calibration_height
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[0] *= scale_x
    scaled_intrinsics[1] *= scale_y
    args.output.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output / "object-trajectory.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open Gaussian trajectory video writer")
    reference_translation = world_xyz[args.reference_frame]
    coverage = []
    centroid_errors = []
    for frame_id in range(len(world_xyz)):
        source_transform = np.eye(4)
        source_transform[:3, 3] = world_xyz[frame_id] - reference_translation
        animated = transform_label(scene, 1, source_transform)
        rendered, _, alpha = render_gaussians(
            animated,
            scaled_intrinsics,
            world_to_camera[frame_id],
            width=args.width,
            height=args.height,
        )
        writer.write(cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        coverage.append(float(np.mean(alpha > 0.5)))
        if observed[frame_id]:
            camera_points = (
                np.column_stack((animated.means_m[source], np.ones(np.count_nonzero(source))))
                @ world_to_camera[frame_id].T
            )[:, :3]
            projected = np.column_stack(
                (
                    scaled_intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2]
                    + scaled_intrinsics[0, 2],
                    scaled_intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2]
                    + scaled_intrinsics[1, 2],
                )
            )
            mask = cv2.imread(str(args.source_masks / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(args.source_masks / f"{frame_id:06d}.png")
            y, x = np.nonzero(mask > 0)
            if len(x):
                mask_center = np.array(
                    [x.mean() * args.width / mask.shape[1], y.mean() * args.height / mask.shape[0]]
                )
                centroid_errors.append(float(np.linalg.norm(projected.mean(axis=0) - mask_center)))
    writer.release()
    report = {
        "schema_version": 1,
        "method": "rigid source Gaussian translation along recovered metric trajectory",
        "frame_count": len(world_xyz),
        "resolution": {"width": args.width, "height": args.height},
        "source_gaussian_count": int(np.count_nonzero(source)),
        "mean_render_coverage": float(np.mean(coverage)),
        "observed_centroid_frames": len(centroid_errors),
        "median_source_centroid_error_px": float(np.median(centroid_errors)),
        "p90_source_centroid_error_px": float(np.quantile(centroid_errors, 0.9)),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
