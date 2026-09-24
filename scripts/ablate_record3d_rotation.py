#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from onevideo2policy.video.record3d import Record3DArchive

ROI_REFERENCE = (520, 1300, 850, 1600)
ROI_PLACED = (500, 900, 850, 1250)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate correspondence-constrained rotation on EM1-0406"
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/sources/EM1-0406.r3d")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/evaluation/em1_rotation_ablation")
    )
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=330)
    parser.add_argument("--end-frame", type=int, default=390)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args()


def crop(image: NDArray[np.uint8], roi: tuple[int, int, int, int]) -> NDArray[np.uint8]:
    x0, y0, x1, y1 = roi
    return image[y0:y1, x0:x1]


def apply_transform(points: NDArray[np.floating], transform: NDArray[np.floating]) -> NDArray:
    return np.asarray(points) @ transform[:, :2].T + transform[:, 2]


def angle_deg(transform: NDArray[np.floating]) -> float:
    return math.degrees(math.atan2(transform[1, 0], transform[0, 0]))


def mask_pose(image: NDArray[np.uint8]) -> tuple[NDArray, float, float, NDArray]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = (gray < 105).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    candidates = [
        component
        for component in range(1, count)
        if stats[component, cv2.CC_STAT_AREA] > 500
    ]
    if not candidates:
        raise ValueError("No dark object component found in audited ROI")
    selected = max(candidates, key=lambda component: stats[component, cv2.CC_STAT_AREA])
    yy, xx = np.where(labels == selected)
    points = np.column_stack((xx, yy)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, -1]
    orientation = math.atan2(principal[1], principal[0])
    return center, float(len(points)), orientation, labels == selected


def mask_only_transform(
    reference: NDArray[np.uint8], observed: NDArray[np.uint8]
) -> NDArray[np.float64]:
    first_center, first_area, first_angle, _ = mask_pose(reference)
    second_center, second_area, second_angle, _ = mask_pose(observed)
    rotation = (second_angle - first_angle + math.pi / 2) % math.pi - math.pi / 2
    scale = math.sqrt(second_area / first_area)
    cosine, sine = math.cos(rotation), math.sin(rotation)
    linear = scale * np.array([[cosine, -sine], [sine, cosine]])
    translation = second_center - linear @ first_center
    return np.column_stack((linear, translation))


def bootstrap_median_interval(values: list[float], seed: int = 7) -> list[float]:
    generator = np.random.default_rng(seed)
    data = np.asarray(values)
    medians = np.median(generator.choice(data, (5000, len(data)), replace=True), axis=1)
    return [float(x) for x in np.percentile(medians, [2.5, 97.5])]


def make_audit_figure(
    reference: NDArray[np.uint8],
    observed: NDArray[np.uint8],
    held_out: NDArray[np.float32],
    target: NDArray[np.float32],
    mask_prediction: NDArray[np.float64],
    track_prediction: NDArray[np.float64],
    mask_median: float,
    track_median: float,
    path: Path,
) -> None:
    import cv2

    left = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(observed, cv2.COLOR_RGB2BGR)
    for source, truth, baseline, tracked in zip(
        held_out, target, mask_prediction, track_prediction, strict=True
    ):
        cv2.circle(left, tuple(np.rint(source).astype(int)), 4, (0, 255, 255), -1)
        truth_xy = tuple(np.rint(truth).astype(int))
        cv2.circle(right, truth_xy, 5, (0, 255, 0), -1)
        cv2.line(right, truth_xy, tuple(np.rint(baseline).astype(int)), (0, 0, 255), 2)
        cv2.line(right, truth_xy, tuple(np.rint(tracked).astype(int)), (255, 100, 0), 2)
    height = max(left.shape[0], right.shape[0])
    if left.shape[0] < height:
        left = cv2.copyMakeBorder(
            left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    if right.shape[0] < height:
        right = cv2.copyMakeBorder(
            right,
            0,
            height - right.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
    canvas = np.hstack((left, right))
    cv2.putText(canvas, "reference: held-out points", (12, 28), 0, 0.65, (0, 255, 255), 2)
    cv2.putText(
        canvas,
        f"mask-only {mask_median:.1f}px (red) | tracked {track_median:.1f}px (blue)",
        (reference.shape[1] + 8, 28),
        0,
        0.55,
        (20, 20, 20),
        2,
    )
    cv2.imwrite(str(path), canvas)


def main() -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install the video extra: uv sync --extra video") from exc
    args = parse_args()
    cv2.setRNGSeed(7)
    archive = Record3DArchive(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = crop(archive.read_rgb(args.reference_frame), ROI_REFERENCE)
    detector = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.015)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    matcher = cv2.BFMatcher()
    frame_results: list[dict[str, Any]] = []
    mask_errors: list[float] = []
    track_errors: list[float] = []
    audit_data = None

    for frame_id in range(args.start_frame, args.end_frame + 1, args.stride):
        observed = crop(archive.read_rgb(frame_id), ROI_PLACED)
        observed_gray = cv2.cvtColor(observed, cv2.COLOR_RGB2GRAY)
        observed_keypoints, observed_descriptors = detector.detectAndCompute(observed_gray, None)
        pairs = matcher.knnMatch(reference_descriptors, observed_descriptors, k=2)
        matches = [first for first, second in pairs if first.distance < 0.78 * second.distance]
        first_xy = np.float32([reference_keypoints[item.queryIdx].pt for item in matches])
        second_xy = np.float32([observed_keypoints[item.trainIdx].pt for item in matches])
        _, inlier_mask = cv2.estimateAffinePartial2D(
            first_xy,
            second_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=5000,
            confidence=0.999,
        )
        if inlier_mask is None:
            raise ValueError(f"RANSAC failed for frame {frame_id}")
        first_xy = first_xy[inlier_mask[:, 0] > 0]
        second_xy = second_xy[inlier_mask[:, 0] > 0]
        if len(first_xy) < 8:
            raise ValueError(f"Frame {frame_id} has only {len(first_xy)} verified matches")
        order = np.argsort(first_xy[:, 0] + 0.01 * first_xy[:, 1])
        held_out_indices = order[::3]
        fit_indices = np.setdiff1d(order, held_out_indices)
        tracked_transform, _ = cv2.estimateAffinePartial2D(
            first_xy[fit_indices],
            second_xy[fit_indices],
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=5000,
            confidence=0.999,
        )
        if tracked_transform is None:
            raise ValueError(f"Tracked fit failed for frame {frame_id}")
        baseline_transform = mask_only_transform(reference, observed)
        held_out = first_xy[held_out_indices]
        target = second_xy[held_out_indices]
        baseline_prediction = apply_transform(held_out, baseline_transform)
        tracked_prediction = apply_transform(held_out, tracked_transform)
        frame_mask_errors = np.linalg.norm(baseline_prediction - target, axis=1)
        frame_track_errors = np.linalg.norm(tracked_prediction - target, axis=1)
        mask_errors.extend(frame_mask_errors.tolist())
        track_errors.extend(frame_track_errors.tolist())
        frame_results.append(
            {
                "frame_id": frame_id,
                "verified_correspondences": len(first_xy),
                "fit_correspondences": len(fit_indices),
                "held_out_correspondences": len(held_out_indices),
                "mask_only_rotation_deg": angle_deg(baseline_transform),
                "tracked_rotation_deg": angle_deg(tracked_transform),
                "mask_only_median_held_out_error_px": float(np.median(frame_mask_errors)),
                "tracked_median_held_out_error_px": float(np.median(frame_track_errors)),
            }
        )
        if frame_id == args.start_frame:
            audit_data = (
                observed,
                held_out,
                target,
                baseline_prediction,
                tracked_prediction,
                float(np.median(frame_mask_errors)),
                float(np.median(frame_track_errors)),
            )

    mask_median = float(np.median(mask_errors))
    track_median = float(np.median(track_errors))
    report = {
        "experiment": "EM1-0406 nonsymmetric-object correspondence ablation",
        "source_archive": str(args.archive),
        "archive_sha256": _sha256(args.archive),
        "reference_frame": args.reference_frame,
        "evaluation_frames": [item["frame_id"] for item in frame_results],
        "method": {
            "without_track_loss": "dark-object mask centroid, area, and principal axis",
            "with_track_loss": "SIFT correspondences and RANSAC 2D similarity",
            "held_out_protocol": "every third verified match after spatial sorting",
            "rois_xyxy": {"reference": ROI_REFERENCE, "placed": ROI_PLACED},
        },
        "summary": {
            "frames": len(frame_results),
            "held_out_correspondences": len(mask_errors),
            "mask_only_median_error_px": mask_median,
            "mask_only_median_95ci_px": bootstrap_median_interval(mask_errors),
            "tracked_median_error_px": track_median,
            "tracked_median_95ci_px": bootstrap_median_interval(track_errors),
            "median_error_reduction_percent": 100 * (1 - track_median / mask_median),
            "median_tracked_rotation_deg": float(
                np.median([item["tracked_rotation_deg"] for item in frame_results])
            ),
        },
        "frames": frame_results,
        "limitations": [
            "This is an image-plane rotation ablation, not a full SE(3) object-pose benchmark.",
            "ROIs and the dark-object threshold were fixed after visual audit of this sequence.",
            "Full-set RANSAC rejects descriptor mismatches before the deterministic "
            "fit/test split.",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    assert audit_data is not None
    make_audit_figure(reference, *audit_data, args.output / "audit.png")
    print(json.dumps(report["summary"], indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
