"""Fuse metric HOI4D RGB-D keyframes into a temporally filtered Gaussian scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.generation.gaussian_scene import (
    fuse_gaussian_scenes,
    initialize_gaussians_from_rgbd,
    render_gaussians,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth-dir", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-ids", nargs="+", type=int, default=[1, 12, 24, 36, 48, 60, 71])
    parser.add_argument("--reference-frame", default=1, type=int)
    parser.add_argument("--validation-frame", default=30, type=int)
    parser.add_argument("--stride", default=6, type=int)
    parser.add_argument("--voxel-size-m", default=0.01, type=float)
    parser.add_argument("--min-static-observations", default=2, type=int)
    args = parser.parse_args()
    if args.reference_frame not in args.frame_ids:
        raise ValueError("reference frame must be included in frame IDs")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        intrinsics = trajectory["intrinsics"]
        camera_to_world = trajectory["camera_to_world"]
        world_to_camera = trajectory["world_to_camera"]
    width = int(manifest["resolution"]["width"])
    height = int(manifest["resolution"]["height"])

    def load_frame(frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame = manifest["frames"][frame_id]
        rgb_path = Path(frame["rgb"])
        if not rgb_path.is_absolute():
            rgb_path = args.manifest.parent / rgb_path
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        source_id = int(frame["source_frame_id"])
        raw_depth = cv2.imread(str(args.depth_dir / f"{source_id:05d}.png"), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or raw_depth is None:
            raise FileNotFoundError(f"RGB-D frame {frame_id}")
        depth_m = cv2.resize(
            raw_depth.astype(np.float32) / 1000.0,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        source = cv2.imread(
            str(args.masks / "source" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE
        )
        target = cv2.imread(
            str(args.masks / "target" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE
        )
        if source is None or target is None:
            raise FileNotFoundError(f"object masks for frame {frame_id}")
        return (
            cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            depth_m,
            source > 0,
            target > 0,
        )

    scenes = []
    for frame_id in args.frame_ids:
        rgb, depth, source, target = load_frame(frame_id)
        scenes.append(
            initialize_gaussians_from_rgbd(
                rgb,
                depth,
                intrinsics,
                source_mask=source,
                target_mask=target,
                stride=args.stride,
            )
        )
    reference_index = args.frame_ids.index(args.reference_frame)
    fused = fuse_gaussian_scenes(
        scenes,
        camera_to_world[args.frame_ids],
        reference_index=reference_index,
        voxel_size_m=args.voxel_size_m,
        min_static_observations=args.min_static_observations,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fused.save(args.output / "scene.npz")
    validation_rgb, _, validation_source, validation_target = load_frame(args.validation_frame)
    rendered, rendered_depth, alpha = render_gaussians(
        fused,
        intrinsics,
        world_to_camera[args.validation_frame],
        width=width,
        height=height,
    )
    cv2.imwrite(
        str(args.output / "held-out-render.png"),
        cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    evaluated = (alpha > 0.5) & ~validation_source & ~validation_target
    mse = float(np.mean((rendered[evaluated] - validation_rgb[evaluated] / 255.0) ** 2))
    depth_visual = np.zeros((height, width), dtype=np.uint8)
    valid_depth = rendered_depth > 0
    if np.any(valid_depth):
        low, high = np.percentile(rendered_depth[valid_depth], [2, 98])
        depth_visual[valid_depth] = np.clip(
            (rendered_depth[valid_depth] - low) / max(high - low, 1e-6) * 255, 0, 255
        )
    cv2.imwrite(str(args.output / "held-out-depth.png"), depth_visual)
    report = {
        "schema_version": 1,
        "method": "metric RGB-D keyframe fusion with temporal voxel support",
        "frame_ids": args.frame_ids,
        "reference_frame": args.reference_frame,
        "validation_frame": args.validation_frame,
        "voxel_size_m": args.voxel_size_m,
        "min_static_observations": args.min_static_observations,
        "gaussian_count": len(fused.means_m),
        "label_counts": {
            "static": int(np.count_nonzero(fused.labels == 0)),
            "source": int(np.count_nonzero(fused.labels == 1)),
            "target": int(np.count_nonzero(fused.labels == 2)),
        },
        "held_out_render": {
            "evaluated_coverage": float(np.mean(evaluated)),
            "mse": mse,
            "psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
