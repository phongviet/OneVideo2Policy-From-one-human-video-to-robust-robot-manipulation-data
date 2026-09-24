"""Measure Depth Anything V2 Metric Small on frozen Place frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--target-masks", required=True, type=Path)
    parser.add_argument("--source-masks", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[384, 518])
    parser.add_argument("--frame-ids", nargs="+", type=int, default=[0, 14, 28, 42, 56])
    parser.add_argument("--relative", action="store_true", help="Use relative-depth weights")
    args = parser.parse_args()
    module_root = args.repo.resolve() if args.relative else (args.repo / "metric_depth").resolve()
    sys.path.insert(0, str(module_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    if not args.relative:
        model_config["max_depth"] = 20
    model = DepthAnythingV2(**model_config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for size in args.sizes:
        records = []
        for frame_id in args.frame_ids:
            path = args.frames / f"{frame_id:06d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            mask_path = args.target_masks / f"{frame_id:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f"Invalid target mask: {mask_path}")
            source_mask = None
            if args.source_masks is not None:
                source_path = args.source_masks / f"{frame_id:06d}.png"
                source_mask = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
                if source_mask is None or source_mask.shape != image.shape[:2]:
                    raise ValueError(f"Invalid source mask: {source_path}")
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                depth = model.infer_image(image, size)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not np.isfinite(depth).all():
                raise ValueError(f"Non-finite depth in frame {frame_id}")
            np.save(args.output / f"depth_{size}_{frame_id:06d}.npy", depth.astype(np.float32))
            height, width = depth.shape
            table_patch = depth[
                round(height * 0.70) : round(height * 0.93),
                round(width * 0.77) : round(width * 0.94),
            ]
            record = {
                "frame_id": frame_id,
                "seconds": elapsed,
                "peak_gpu_allocated_mb": (
                    torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
                ),
                "prediction_range": [float(depth.min()), float(depth.max())],
                "target_median_prediction": float(np.median(depth[mask > 0])),
                "table_patch_median_prediction": float(np.median(table_patch)),
            }
            if source_mask is not None:
                if not np.any(source_mask > 0):
                    raise ValueError(f"Empty source mask in frame {frame_id}")
                record["source_median_prediction"] = float(np.median(depth[source_mask > 0]))
                record["source_minus_target_prediction"] = (
                    record["source_median_prediction"] - record["target_median_prediction"]
                )
            records.append(record)
            lo, hi = np.percentile(depth, [2, 98]) if args.relative else (0.5, 3.0)
            color = cv2.applyColorMap(
                np.clip((depth - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )
            cv2.imwrite(str(args.output / f"depth_{size}_{frame_id:06d}.png"), color)
        target_values = [entry["target_median_prediction"] for entry in records]
        table_values = [entry["table_patch_median_prediction"] for entry in records]
        results[str(size)] = {
            "frames": records,
            "median_seconds": statistics.median(entry["seconds"] for entry in records),
            "max_peak_gpu_allocated_mb": max(entry["peak_gpu_allocated_mb"] for entry in records)
            if device == "cuda"
            else None,
            "target_depth_cv": statistics.pstdev(target_values) / statistics.mean(target_values),
            "table_patch_depth_cv": statistics.pstdev(table_values) / statistics.mean(table_values),
        }
    report = {
        "model": (
            "Depth Anything V2 Relative Small"
            if args.relative
            else "Depth Anything V2 Metric Hypersim Small"
        ),
        "output_units": "arbitrary relative disparity" if args.relative else "predicted metres",
        "device": device,
        "checkpoint_sha256": sha256(args.checkpoint),
        "repo": str(args.repo.resolve()),
        "measurement_note": (
            "Temporal depth variability is a diagnostic, not accuracy: camera motion, "
            "predicted masks, and no measured ground truth limit interpretation."
        ),
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
