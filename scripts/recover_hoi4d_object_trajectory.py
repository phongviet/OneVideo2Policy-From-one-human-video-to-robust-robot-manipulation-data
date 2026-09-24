"""Recover the HOI4D ball trajectory from masked metric depth and RGB-D odometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.pose_tracking.sphere_trajectory import recover_sphere_relative_trajectory


def longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth-dir", required=True, type=Path)
    parser.add_argument("--source-masks", required=True, type=Path)
    parser.add_argument("--object-poses", required=True, type=Path)
    parser.add_argument("--camera-trajectory", required=True, type=Path)
    parser.add_argument("--source-radius-m", required=True, type=float)
    parser.add_argument("--target-radius-m", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-surface-error-m", default=0.006, type=float)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.camera_trajectory, allow_pickle=False) as trajectory:
        intrinsics = trajectory["intrinsics"]
        camera_to_world = trajectory["camera_to_world"]
    width = int(manifest["resolution"]["width"])
    height = int(manifest["resolution"]["height"])
    depth_frames = []
    source_masks = []
    target_centers = []
    source_frame_ids = []
    for frame in manifest["frames"]:
        frame_id = int(frame["frame_id"])
        source_id = int(frame["source_frame_id"])
        raw_depth = cv2.imread(str(args.depth_dir / f"{source_id:05d}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(args.source_masks / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if raw_depth is None or mask is None:
            raise FileNotFoundError(f"depth or source mask for frame {frame_id}")
        pose = json.loads((args.object_poses / f"{source_id}.json").read_text(encoding="utf-8"))
        bowl = next(item for item in pose["dataList"] if item["label"] == "bowl")
        center = bowl["center"]
        depth_frames.append(
            cv2.resize(
                raw_depth.astype(np.float32) / 1000.0,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        source_masks.append(mask > 0)
        target_centers.append([center["x"], center["y"], center["z"]])
        source_frame_ids.append(source_id)
    relative_xyz, world_xyz, accepted, diagnostics = recover_sphere_relative_trajectory(
        depth_frames,
        source_masks,
        intrinsics,
        camera_to_world,
        np.asarray(target_centers),
        radius_m=args.source_radius_m,
        max_surface_error_m=args.max_surface_error_m,
    )
    poses = np.repeat(np.eye(4)[None], len(relative_xyz), axis=0)
    poses[:, :3, 3] = relative_xyz
    step_distance = np.linalg.norm(np.diff(relative_xyz, axis=0), axis=1)
    observed_ids = np.flatnonzero(accepted)
    final_distance = float(np.linalg.norm(relative_xyz[-1]))
    accepted_errors = [
        item["median_surface_error_m"] for item in diagnostics if item.get("accepted")
    ]
    report = {
        "schema_version": 1,
        "method": "known-radius robust sphere fit in target-relative metric coordinates",
        "rotation": "identity; ball rotation is unobservable under spherical symmetry",
        "frame_count": len(relative_xyz),
        "accepted_depth_frames": int(np.count_nonzero(accepted)),
        "interpolated_frames": int(np.count_nonzero(~accepted)),
        "first_observed_frame": int(observed_ids[0]),
        "last_observed_frame": int(observed_ids[-1]),
        "longest_interpolated_run": longest_false_run(accepted),
        "median_surface_error_m": float(np.median(accepted_errors)),
        "median_step_distance_m": float(np.median(step_distance)),
        "max_step_distance_m": float(np.max(step_distance)),
        "initial_observed_separation_m": float(np.linalg.norm(relative_xyz[observed_ids[0]])),
        "final_separation_m": final_distance,
        "target_radius_m": args.target_radius_m,
        "observed_place": final_distance <= args.target_radius_m,
        "diagnostics": diagnostics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "object-trajectory.npz",
        schema_version=np.asarray(1),
        source_frame_ids=np.asarray(source_frame_ids),
        source_to_target=poses,
        relative_xyz_m=relative_xyz,
        source_world_xyz_m=world_xyz,
        depth_observed=accepted,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in report.items() if key != "diagnostics"}, indent=2)
    )


if __name__ == "__main__":
    main()
