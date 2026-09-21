"""Compare small depth models on the local UniHand RGB-D fixture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    if not args.relative:
        config["max_depth"] = 20
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    report = {"model": "relative_small" if args.relative else "metric_hypersim_small"}
    for size in (384, 518):
        rows = []
        for frame_id in (0, 8, 17, 26, 34):
            rgb = np.load(args.fixture / "rgb" / f"{frame_id:06d}.npy")
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_m = (
                np.load(args.fixture / "depth" / f"{frame_id:06d}.npy").astype(np.float32) / 1000
            )
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model.infer_image(image, size)
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            valid = (depth_m > 0.3) & (depth_m < 3.0) & np.isfinite(prediction)
            comparable = 1 / np.maximum(prediction, 1e-3) if args.relative else prediction
            scale = np.median(depth_m[valid]) / np.median(comparable[valid])
            rows.append(
                {
                    "frame": frame_id,
                    "seconds": seconds,
                    "peak_gpu_allocated_mb": (
                        torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
                    ),
                    "valid_fraction": float(valid.mean()),
                    "scale_aligned_absrel": float(
                        np.mean(np.abs(comparable[valid] * scale - depth_m[valid]) / depth_m[valid])
                    ),
                    "raw_absrel": (
                        None
                        if args.relative
                        else float(
                            np.mean(np.abs(prediction[valid] - depth_m[valid]) / depth_m[valid])
                        )
                    ),
                }
            )
        report[str(size)] = {
            "rows": rows,
            "mean_scale_aligned_absrel": float(np.mean([r["scale_aligned_absrel"] for r in rows])),
            "mean_raw_absrel": (
                None if args.relative else float(np.mean([r["raw_absrel"] for r in rows]))
            ),
        }
    report["note"] = (
        "Assumes fixture uint16 depth units are millimetres. Scale alignment uses a "
        "per-frame ground-truth median and is diagnostic only."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
