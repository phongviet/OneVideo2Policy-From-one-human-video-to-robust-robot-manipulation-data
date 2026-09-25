"""Tune the compact Gaussian footprint on one view and verify it on separate views."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from onevideo2policy.generation.gaussian_scene import GaussianScene, render_gaussians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-frame", default=30, type=int)
    parser.add_argument("--test-frames", nargs="+", default=[42, 54, 66], type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = GaussianScene.load(args.scene)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        intrinsics = trajectory["intrinsics"]
        world_to_camera = trajectory["world_to_camera"]
    width = int(manifest["resolution"]["width"])
    height = int(manifest["resolution"]["height"])
    frame_ids = [args.selection_frame, *args.test_frames]
    observations = {
        frame_id: load_observation(frame_id, manifest, args.manifest, args.masks)
        for frame_id in frame_ids
    }
    baseline_renders: dict[int, np.ndarray] = {}
    evaluation_masks: dict[int, np.ndarray] = {}
    baseline_psnr: dict[int, float] = {}
    for frame_id in frame_ids:
        rendered, _, alpha = render_gaussians(
            scene,
            intrinsics,
            world_to_camera[frame_id],
            width=width,
            height=height,
        )
        rgb, source, target = observations[frame_id]
        evaluated = (alpha > 0.5) & ~source & ~target
        baseline_renders[frame_id] = rendered
        evaluation_masks[frame_id] = evaluated
        baseline_psnr[frame_id] = psnr(rendered, rgb, evaluated)

    grid: list[dict[str, float]] = []
    best: tuple[float, float, float] | None = None
    for scale_multiplier in [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]:
        for opacity in [0.4, 0.55, 0.7, 0.85, 0.95, 1.0]:
            candidate = with_global_footprint(scene, scale_multiplier, opacity)
            rendered, _, _ = render_gaussians(
                candidate,
                intrinsics,
                world_to_camera[args.selection_frame],
                width=width,
                height=height,
            )
            score = psnr(
                rendered,
                observations[args.selection_frame][0],
                evaluation_masks[args.selection_frame],
            )
            grid.append(
                {
                    "scale_multiplier": scale_multiplier,
                    "opacity": opacity,
                    "selection_psnr_db": score,
                }
            )
            if best is None or score > best[0]:
                best = (score, scale_multiplier, opacity)
    assert best is not None
    _, selected_scale, selected_opacity = best
    optimized = with_global_footprint(scene, selected_scale, selected_opacity)
    optimized.save(args.output / "scene.npz")
    evaluations: list[dict[str, Any]] = []
    optimized_renders: dict[int, np.ndarray] = {}
    for frame_id in frame_ids:
        rendered, _, _ = render_gaussians(
            optimized,
            intrinsics,
            world_to_camera[frame_id],
            width=width,
            height=height,
        )
        optimized_renders[frame_id] = rendered
        optimized_score = psnr(
            rendered, observations[frame_id][0], evaluation_masks[frame_id]
        )
        evaluations.append(
            {
                "frame_id": frame_id,
                "role": "selection" if frame_id == args.selection_frame else "test",
                "evaluated_coverage": float(np.mean(evaluation_masks[frame_id])),
                "baseline_psnr_db": baseline_psnr[frame_id],
                "optimized_psnr_db": optimized_score,
                "improvement_db": optimized_score - baseline_psnr[frame_id],
            }
        )
    report = {
        "schema_version": 1,
        "method": "global Gaussian scale and opacity held-out sweep",
        "selection_frame": args.selection_frame,
        "test_frames": args.test_frames,
        "fixed_evaluation_support": "baseline alpha > 0.5, excluding source and target masks",
        "selected": {
            "scale_multiplier": selected_scale,
            "opacity": selected_opacity,
        },
        "evaluations": evaluations,
        "mean_test_improvement_db": float(
            np.mean([item["improvement_db"] for item in evaluations if item["role"] == "test"])
        ),
        "grid": grid,
        "limitations": [
            "The sweep tunes global footprint parameters, not independent per-Gaussian means.",
            "Camera poses and RGB-D initialized colors remain fixed.",
            "Robot camera registration requires cross-scene correspondences and is not "
            "inferred here.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    audit_frame = args.test_frames[0]
    write_audit(
        observations[audit_frame][0],
        baseline_renders[audit_frame],
        optimized_renders[audit_frame],
        args.output / "audit.png",
    )
    print(json.dumps({key: report[key] for key in ["selected", "evaluations"]}, indent=2))


def with_global_footprint(
    scene: GaussianScene, scale_multiplier: float, opacity: float
) -> GaussianScene:
    return replace(
        scene,
        log_scales_m=scene.log_scales_m + np.float32(np.log(scale_multiplier)),
        opacities=np.full_like(scene.opacities, opacity),
    )


def load_observation(
    frame_id: int, manifest: dict[str, Any], manifest_path: Path, masks: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_path = Path(manifest["frames"][frame_id]["rgb"])
    if not frame_path.is_absolute():
        frame_path = manifest_path.parent / frame_path
    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    source = cv2.imread(str(masks / "source" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(masks / "target" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if bgr is None or source is None or target is None:
        raise FileNotFoundError(f"RGB or mask missing for frame {frame_id}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return rgb, source > 0, target > 0


def psnr(rendered: np.ndarray, observed: np.ndarray, evaluated: np.ndarray) -> float:
    mse = float(np.mean((rendered[evaluated] - observed[evaluated]) ** 2))
    return float(-10 * np.log10(max(mse, 1e-12)))


def write_audit(
    observed: np.ndarray, baseline: np.ndarray, optimized: np.ndarray, path: Path
) -> None:
    panels = []
    for name, image in [("observed", observed), ("baseline", baseline), ("optimized", optimized)]:
        panel = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(panel, name, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panels.append(panel)
    cv2.imwrite(str(path), np.hstack(panels))


if __name__ == "__main__":
    main()
