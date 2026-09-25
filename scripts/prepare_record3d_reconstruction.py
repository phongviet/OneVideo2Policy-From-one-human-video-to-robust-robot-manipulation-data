"""Prepare the audited EM1-0406 action camera crop and metric RGB-D reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.video.record3d import Record3DArchive

ROI_XYXY = (520, 1300, 850, 1600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = Record3DArchive(args.archive)
    rgb = archive.read_rgb(args.frame)
    depth_m = archive.read_depth_m(args.frame)
    mask = audited_action_camera_mask(rgb)
    depth_mask = cv2.resize(
        mask.astype(np.uint8),
        (depth_m.shape[1], depth_m.shape[0]),
        interpolation=cv2.INTER_AREA,
    ) > 0.5
    valid = depth_mask & np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 2.0)
    yy, xx = np.where(valid)
    z = depth_m[yy, xx]
    intrinsics = archive.intrinsics_for_depth()
    points = np.column_stack(
        (
            (xx - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    center = np.median(points, axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    coordinates = (points - center) @ axes.T
    extents_m = np.percentile(coordinates, 98, axis=0) - np.percentile(
        coordinates, 2, axis=0
    )
    extents_m = np.sort(extents_m)[::-1]
    crop_rgba = square_rgba_crop(rgb, mask)
    rgba_path = args.output / "action_camera_rgba.png"
    gray_path = args.output / "action_camera_gray.png"
    cv2.imwrite(str(rgba_path), cv2.cvtColor(crop_rgba, cv2.COLOR_RGBA2BGRA))
    gray_rgb = crop_rgba[:, :, :3].copy()
    gray_rgb[crop_rgba[:, :, 3] == 0] = 128
    cv2.imwrite(str(gray_path), cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR))
    write_ply(
        args.output / "action_camera_depth_points.ply", points, rgb, xx, yy, depth_m.shape
    )
    report = {
        "schema_version": 1,
        "source_archive": str(args.archive),
        "source_archive_sha256": sha256(args.archive),
        "frame_id": args.frame,
        "method": "aligned metric Record3D depth under audited dark-object mask",
        "roi_xyxy": ROI_XYXY,
        "gray_threshold": 105,
        "depth_samples": len(points),
        "robust_extent_percentiles": [2, 98],
        "metric_extents_m_descending": extents_m.tolist(),
        "metric_extent_ratios_descending": (extents_m / extents_m[0]).tolist(),
        "depth_m_percentiles": np.percentile(z, [0, 5, 50, 95, 100]).tolist(),
        "intrinsics_depth": intrinsics.tolist(),
        "inputs": {
            "rgba": {"path": rgba_path.name, "sha256": sha256(rgba_path)},
            "gray": {"path": gray_path.name, "sha256": sha256(gray_path)},
        },
        "limitations": [
            "Extents are a robust PCA bounding box of visible RGB-D surfaces.",
            "The depth camera has 192x256 resolution, so this is not caliper-grade metrology.",
            "The mask threshold and ROI are fixed for the audited EM1-0406 frame 0.",
        ],
    }
    (args.output / "metric_reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def audited_action_camera_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x0, y0, x1, y1 = ROI_XYXY
    roi = np.zeros(gray.shape, dtype=np.uint8)
    roi[y0:y1, x0:x1] = 1
    binary = ((gray < 105) & (roi > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    candidates = [
        component
        for component in range(1, count)
        if stats[component, cv2.CC_STAT_AREA] > 500
    ]
    if not candidates:
        raise ValueError("No action-camera component found")
    selected = max(candidates, key=lambda component: stats[component, cv2.CC_STAT_AREA])
    return labels == selected


def square_rgba_crop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    yy, xx = np.where(mask)
    padding = 30
    x0, x1 = max(0, int(xx.min()) - padding), min(rgb.shape[1], int(xx.max()) + padding + 1)
    y0, y1 = max(0, int(yy.min()) - padding), min(rgb.shape[0], int(yy.max()) + padding + 1)
    rgba = np.dstack((rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(np.uint8) * 255))
    side = max(rgba.shape[:2])
    square = np.zeros((side, side, 4), dtype=np.uint8)
    offset_y = (side - rgba.shape[0]) // 2
    offset_x = (side - rgba.shape[1]) // 2
    square[offset_y : offset_y + rgba.shape[0], offset_x : offset_x + rgba.shape[1]] = rgba
    return square


def write_ply(
    path: Path,
    points: np.ndarray,
    rgb: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    depth_shape: tuple[int, int],
) -> None:
    depth_height, depth_width = depth_shape
    rgb_x = np.clip(
        np.rint((xx + 0.5) * rgb.shape[1] / depth_width - 0.5).astype(int),
        0,
        rgb.shape[1] - 1,
    )
    rgb_y = np.clip(
        np.rint((yy + 0.5) * rgb.shape[0] / depth_height - 0.5).astype(int),
        0,
        rgb.shape[0] - 1,
    )
    colors = rgb[rgb_y, rgb_x]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point, color in zip(points, colors, strict=True):
            stream.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
