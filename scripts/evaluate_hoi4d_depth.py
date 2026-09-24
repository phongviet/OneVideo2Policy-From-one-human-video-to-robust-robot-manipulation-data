"""Evaluate saved monocular depth predictions against aligned HOI4D sensor depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def depth_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prediction) & np.isfinite(target) & (target > 0.1) & (target < 10.0)
    if not np.any(valid):
        raise ValueError("No valid sensor depth pixels")
    predicted = prediction[valid]
    measured = target[valid]
    ratio = np.maximum(predicted / measured, measured / predicted)
    return {
        "abs_rel": float(np.mean(np.abs(predicted - measured) / measured)),
        "rmse_m": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "valid_pixels": int(np.count_nonzero(valid)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--sensor-depth", required=True, type=Path)
    parser.add_argument("--input-size", default=518, type=int)
    parser.add_argument("--frame-ids", required=True, nargs="+", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for frame_id in args.frame_ids:
        source_frame_id = int(manifest["frames"][frame_id]["source_frame_id"])
        prediction_path = args.predictions / f"depth_{args.input_size}_{frame_id:06d}.npy"
        sensor_path = args.sensor_depth / f"{source_frame_id:05d}.png"
        prediction = np.load(prediction_path).astype(np.float32)
        sensor_mm = cv2.imread(str(sensor_path), cv2.IMREAD_UNCHANGED)
        if sensor_mm is None:
            raise FileNotFoundError(sensor_path)
        sensor_m = cv2.resize(
            sensor_mm.astype(np.float32) / 1000.0,
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        valid = np.isfinite(prediction) & (sensor_m > 0.1) & (sensor_m < 10.0)
        scale = float(np.median(sensor_m[valid] / prediction[valid]))
        records.append(
            {
                "frame_id": frame_id,
                "source_frame_id": source_frame_id,
                "median_scale": scale,
                "raw": depth_metrics(prediction, sensor_m),
                "scale_aligned": depth_metrics(prediction * scale, sensor_m),
            }
        )

    report = {
        "model_output": "Depth Anything V2 Metric Hypersim Small, predicted metres",
        "reference": "HOI4D aligned sensor depth, uint16 millimetres",
        "alignment": "independent median scale per frame; diagnostic only",
        "frames": records,
        "summary": {
            "median_scale": float(np.median([item["median_scale"] for item in records])),
            "raw_mean_abs_rel": float(np.mean([item["raw"]["abs_rel"] for item in records])),
            "scale_aligned_mean_abs_rel": float(
                np.mean([item["scale_aligned"]["abs_rel"] for item in records])
            ),
            "scale_aligned_mean_rmse_m": float(
                np.mean([item["scale_aligned"]["rmse_m"] for item in records])
            ),
            "scale_aligned_mean_delta_1": float(
                np.mean([item["scale_aligned"]["delta_1"] for item in records])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
