"""Audit whether real-video depth and tracks support calibrated 3D motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coefficient_of_variation(values: np.ndarray) -> float:
    return float(np.std(values) / abs(np.mean(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depth-report", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--camera-transforms", type=Path, required=True)
    parser.add_argument("--fixture-report", type=Path, required=True)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    depth_report = json.loads(args.depth_report.read_text())
    fixture_report = json.loads(args.fixture_report.read_text())
    measurements = json.loads(args.measurements.read_text()) if args.measurements else None
    frames = depth_report["results"]["518"]["frames"]
    n_frames = len(manifest["frames"])
    if len(frames) != n_frames or [row["frame_id"] for row in frames] != list(range(n_frames)):
        raise ValueError("Depth report must cover every manifest frame in order")
    proxy = np.load(args.proxy)
    transforms = np.load(args.camera_transforms)
    if transforms.shape != (n_frames, 3, 3):
        raise ValueError("Camera transforms have the wrong shape")
    relative_xy = proxy["relative_xy_px"]
    target_xy = proxy["target_xy_stabilized_px"]
    if relative_xy.shape != (n_frames, 2) or target_xy.shape != (n_frames, 2):
        raise ValueError("Proxy centroid arrays have the wrong shape")
    source_depth = np.array([row["source_median_prediction"] for row in frames])
    target_depth = np.array([row["target_median_prediction"] for row in frames])
    table_depth = np.array([row["table_patch_median_prediction"] for row in frames])
    depth_difference = source_depth - target_depth
    if not all(np.isfinite(x).all() for x in (source_depth, target_depth, table_depth)):
        raise ValueError("Non-finite depth summary")
    height, width = manifest["resolution"]["height"], manifest["resolution"]["width"]
    center = np.array([width / 2, height / 2, 1.0])
    center_motion = (transforms @ center)[:, :2] - center[:2]
    reliable = proxy["target_centroid_reliable"]
    target_reference = np.median(target_xy[reliable], axis=0)
    target_drift = np.linalg.norm(target_xy[reliable] - target_reference, axis=1)
    has_measured_diameter = bool(measurements and measurements["source"].get("diameter_m"))
    times = np.array([frame["timestamp_s"] for frame in manifest["frames"]])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "evidence_trajectory.npz",
        timestamp_s=times,
        source_relative_to_target_xy_px=relative_xy,
        source_depth_model_prediction=source_depth,
        target_depth_model_prediction=target_depth,
        source_minus_target_depth_model_prediction=depth_difference,
        camera_center_alignment_px=center_motion,
    )
    report = {
        "status": "uncalibrated_geometry_evidence",
        "scope": (
            "57-frame real-video diagnostics; planar scale anchored by measured source diameter, "
            "with no calibrated 3D poses"
            if has_measured_diameter
            else "57-frame real-video diagnostics; no recovered 6D poses or measured metric scale"
        ),
        "camera_metadata": {
            "device": "iPhone 11 Pro Max",
            "source": "QuickTime MOV metadata",
            "lens_intrinsics_present": False,
        },
        "input_sha256": {
            "manifest": sha256(args.manifest),
            "depth_report": sha256(args.depth_report),
            "proxy": sha256(args.proxy),
            "camera_transforms": sha256(args.camera_transforms),
            **({"measurements": sha256(args.measurements)} if args.measurements else {}),
        },
        "measured_geometry": measurements,
        "camera_alignment": {
            "max_image_center_shift_px": float(np.max(np.linalg.norm(center_motion, axis=1))),
            "median_static_target_centroid_drift_px": float(np.median(target_drift)),
            "max_static_target_centroid_drift_px": float(np.max(target_drift)),
        },
        "depth_diagnostics": {
            "model": depth_report["model"],
            "target_temporal_cv": coefficient_of_variation(target_depth),
            "table_patch_temporal_cv": coefficient_of_variation(table_depth),
            "source_minus_target_median_prediction": float(np.median(depth_difference)),
            "source_minus_target_range_prediction": [
                float(np.min(depth_difference)),
                float(np.max(depth_difference)),
            ],
            "source_minus_target_temporal_cv": coefficient_of_variation(depth_difference),
            "unihand_fixture_raw_absrel": fixture_report["518"]["mean_raw_absrel"],
            "interpretation": (
                "Predicted metre values have unverified scale on this clip. The separate "
                "RGB-D fixture shows scale bias, and the source-target depth difference "
                "varies strongly; this signal is not a metric 3D trajectory."
            ),
        },
        "translation_evidence": {
            "initial_relative_xy_px": relative_xy[0].tolist(),
            "final_relative_xy_px": relative_xy[-1].tolist(),
            "metres_per_pixel": float(proxy["metres_per_pixel"]),
            "scale_anchor": str(proxy["scale_anchor"]),
            "metric_scale_status": (
                "planar xy anchored by measured source diameter; calibrated 3D remains unavailable"
                if has_measured_diameter
                else "awaiting measured dimensions and initial separation"
            ),
        },
        "orientation_evidence": {
            "status": "unestimated",
            "reason": (
                "A single view of a nearly round source with occlusion does not "
                "constrain full 3D rotation."
            ),
        },
        "next_required_inputs": [
            (
                "measured source height"
                if measurements and measurements["source"].get("diameter_m")
                else "measured source diameter and height"
            ),
            "measured target length, width, and height",
            "measured initial source-target center separation",
            "camera intrinsics or calibration capture for perspective 3D",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
