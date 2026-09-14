from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from onevideo2policy.video.cotracker_adapter import CoTracker3Adapter
from onevideo2policy.video.manifest import validate_manifest
from onevideo2policy.video.point_sampling import sample_mask_points
from onevideo2policy.video.sam2_adapter import Sam2PointPrompt, Sam2VideoAdapter


@dataclass(frozen=True)
class ObjectPerception:
    masks: NDArray[np.bool_]
    seed_points_xy: NDArray[np.int64]
    tracks_xy: NDArray[np.float32]
    visible: NDArray[np.bool_]
    reseed_frames: NDArray[np.int64] | None = None


def summarize_perception(
    results: Mapping[str, ObjectPerception],
) -> dict[str, object]:
    """Compute annotation-free diagnostics without claiming ground-truth accuracy."""
    objects: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        masks = np.asarray(result.masks, dtype=bool)
        tracks = np.asarray(result.tracks_xy)
        visible = np.asarray(result.visible, dtype=bool)
        if masks.ndim != 3 or tracks.ndim != 3 or visible.shape != tracks.shape[:2]:
            raise ValueError(f"Invalid perception arrays for {name!r}")
        if len(masks) != len(tracks):
            raise ValueError(f"Mask and track frame counts differ for {name!r}")

        areas = masks.mean(axis=(1, 2))
        adjacent_ious = []
        for previous, current in zip(masks[:-1], masks[1:], strict=True):
            union = np.logical_or(previous, current).sum()
            adjacent_ious.append(
                float(np.logical_and(previous, current).sum() / union) if union else 1.0
            )

        height, width = masks.shape[1:]
        rounded = np.rint(tracks).astype(np.int64)
        in_bounds = (
            (rounded[..., 0] >= 0)
            & (rounded[..., 0] < width)
            & (rounded[..., 1] >= 0)
            & (rounded[..., 1] < height)
        )
        eligible = visible & in_bounds
        inside = np.zeros_like(eligible)
        frame_ids, point_ids = np.nonzero(eligible)
        inside[frame_ids, point_ids] = masks[
            frame_ids,
            rounded[frame_ids, point_ids, 1],
            rounded[frame_ids, point_ids, 0],
        ]
        objects[name] = {
            "mask_area_fraction_min": float(areas.min()),
            "mask_area_fraction_max": float(areas.max()),
            "mean_adjacent_mask_iou": float(np.mean(adjacent_ious))
            if adjacent_ious
            else 1.0,
            "track_survival": float(visible.mean()),
            "mean_visible_tracks": float(visible.sum(axis=1).mean()),
            "mean_tracks_inside_mask": float(inside.sum() / eligible.sum())
            if eligible.any()
            else 0.0,
        }
    return {
        "provisional": True,
        "note": "No manual ground-truth masks yet; temporal diagnostics only.",
        "objects": objects,
    }


def load_prompt_file(path: str | Path) -> dict[str, Sam2PointPrompt]:
    """Load named SAM2 point prompts from a small JSON configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Prompt file must contain a non-empty object mapping")
    prompts: dict[str, Sam2PointPrompt] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not name or not isinstance(values, dict):
            raise ValueError("Each prompt entry must have a name and mapping value")
        try:
            prompts[name] = Sam2PointPrompt(
                object_id=values["object_id"],
                frame_idx=values.get("frame_idx", 0),
                points_xy=np.asarray(values["points_xy"]),
                labels=np.asarray(values["labels"]),
            )
        except KeyError as exc:
            raise ValueError(f"Prompt {name!r} is missing {exc.args[0]}") from exc
    return prompts


def load_manifest_rgb(manifest_path: str | Path) -> NDArray[np.uint8]:
    """Load validated manifest frames as an RGB array for CoTracker."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Loading RGB frames requires: uv sync --extra video") from exc
    manifest_path = Path(manifest_path)
    validate_manifest(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    images = []
    for frame in manifest["frames"]:
        path = Path(frame["rgb"])
        path = path if path.is_absolute() else manifest_path.parent / path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode RGB frame: {path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError("All RGB frames must have the same shape")
    return np.stack(images).astype(np.uint8, copy=False)


def run_perception(
    frames: NDArray[np.uint8],
    frames_dir: str | Path,
    prompts: Mapping[str, Sam2PointPrompt],
    segmenter: Sam2VideoAdapter,
    tracker: CoTracker3Adapter,
    *,
    point_count: int,
    seed: int = 42,
    border: int = 0,
    reseed_interval: int | None = None,
) -> dict[str, ObjectPerception]:
    """Propagate masks, then track deterministic points with optional reseeding."""
    frames = np.asarray(frames, dtype=np.uint8)
    if not prompts:
        raise ValueError("At least one named object prompt is required")
    if reseed_interval is not None and reseed_interval < 2:
        raise ValueError("reseed_interval must be at least 2 frames")
    masks_by_id = segmenter.segment_video(
        frames_dir, prompts.values(), expected_frame_count=len(frames)
    )
    masks_by_name = {name: masks_by_id[prompt.object_id] for name, prompt in prompts.items()}
    return run_tracking_on_masks(
        frames,
        masks_by_name,
        prompts,
        tracker,
        point_count=point_count,
        seed=seed,
        border=border,
        reseed_interval=reseed_interval,
    )


def load_mask_directories(
    masks_dir: str | Path, object_names: list[str], *, expected_frame_count: int
) -> dict[str, NDArray[np.bool_]]:
    """Load a frozen set of lossless object-mask PNG directories."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Loading mask PNGs requires: uv sync --extra video") from exc
    masks_dir = Path(masks_dir)
    result: dict[str, NDArray[np.bool_]] = {}
    for name in object_names:
        paths = sorted((masks_dir / name).glob("*.png"))
        if len(paths) != expected_frame_count:
            raise ValueError(
                f"Expected {expected_frame_count} masks for {name!r}, found {len(paths)}"
            )
        masks = []
        for path in paths:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not decode mask: {path}")
            masks.append(mask > 0)
        result[name] = np.stack(masks)
    return result


def run_tracking_on_masks(
    frames: NDArray[np.uint8],
    masks_by_name: Mapping[str, NDArray[np.bool_]],
    prompts: Mapping[str, Sam2PointPrompt],
    tracker: CoTracker3Adapter,
    *,
    point_count: int,
    seed: int = 42,
    border: int = 0,
    reseed_interval: int | None = None,
) -> dict[str, ObjectPerception]:
    """Track points on frozen masks, optionally refreshing them at fixed windows."""
    frames = np.asarray(frames, dtype=np.uint8)
    if set(masks_by_name) != set(prompts):
        raise ValueError("Frozen mask names must exactly match prompt names")
    if reseed_interval is not None and reseed_interval < 2:
        raise ValueError("reseed_interval must be at least 2 frames")
    results: dict[str, ObjectPerception] = {}
    for name, prompt in prompts.items():
        masks = np.asarray(masks_by_name[name], dtype=bool)
        if masks.shape != frames.shape[:3]:
            raise ValueError(f"Mask dimensions do not match frames for {name!r}")
        if reseed_interval is None:
            points = sample_mask_points(
                masks[prompt.frame_idx],
                point_count,
                seed=seed + prompt.object_id,
                border=border,
            )
            tracks, visible = tracker.track(frames, points, query_frame=prompt.frame_idx)
            reseed_frames = np.asarray([prompt.frame_idx], dtype=np.int64)
        else:
            if prompt.frame_idx != 0:
                raise ValueError("Windowed reseeding currently requires frame-zero prompts")
            tracks = np.empty((len(frames), point_count, 2), dtype=np.float32)
            visible = np.zeros((len(frames), point_count), dtype=bool)
            seeds = []
            actual_reseed_frames = []
            window_starts = np.arange(0, len(frames), reseed_interval, dtype=np.int64)
            for window_id, start in enumerate(window_starts):
                stop = min(int(start) + reseed_interval, len(frames))
                window_points = None
                query_frame = 0
                for candidate in range(int(start), stop):
                    try:
                        window_points = sample_mask_points(
                            masks[candidate],
                            point_count,
                            seed=seed + prompt.object_id + window_id,
                            border=border,
                        )
                    except ValueError as exc:
                        if "eligible pixels" not in str(exc):
                            raise
                    else:
                        query_frame = candidate - int(start)
                        actual_reseed_frames.append(candidate)
                        break
                if window_points is None:
                    raise ValueError(
                        f"No frame in [{start}, {stop}) has {point_count} eligible "
                        f"mask pixels for {name!r}"
                    )
                window_tracks, window_visible = tracker.track(
                    frames[start:stop], window_points, query_frame=query_frame
                )
                tracks[start:stop] = window_tracks
                visible[start:stop] = window_visible
                seeds.append(window_points)
            points = np.stack(seeds)
            reseed_frames = np.asarray(actual_reseed_frames, dtype=np.int64)
        results[name] = ObjectPerception(
            masks, points, tracks, visible, reseed_frames=reseed_frames
        )
    return results


def save_perception_artifacts(
    results: Mapping[str, ObjectPerception], output_dir: str | Path
) -> None:
    """Save lossless masks, sampled points, and track arrays using the repository contract."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Saving mask PNGs requires: uv sync --extra video") from exc
    output_dir = Path(output_dir)
    for name, result in results.items():
        mask_dir = output_dir / "masks" / name
        mask_dir.mkdir(parents=True, exist_ok=True)
        for frame_id, mask in enumerate(result.masks):
            path = mask_dir / f"{frame_id:06d}.png"
            if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                raise OSError(f"Failed to write {path}")
        tracks_dir = output_dir / "tracks"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        np.save(tracks_dir / f"{name}_seed_points.npy", result.seed_points_xy)
        np.savez_compressed(
            tracks_dir / f"{name}.npz",
            xy=result.tracks_xy,
            visible=result.visible,
            reseed_frames=(
                result.reseed_frames
                if result.reseed_frames is not None
                else np.asarray([], dtype=np.int64)
            ),
        )


def save_perception_overlay(
    frames: NDArray[np.uint8],
    results: Mapping[str, ObjectPerception],
    output_path: str | Path,
    *,
    fps: float,
) -> None:
    """Write an RGB mask-and-track overlay for fast visual identity review."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Saving an overlay requires: uv sync --extra video") from exc
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or fps <= 0:
        raise ValueError("Frames must have shape [T,H,W,3] and fps must be positive")
    palette = [(75, 220, 90), (70, 140, 255), (255, 100, 100), (220, 180, 60)]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Could not open video writer for {output_path}")
    names = list(results)
    try:
        for frame_idx, rgb in enumerate(frames):
            overlay = rgb.copy()
            for color, (name, result) in zip(palette, results.items(), strict=False):
                mask = result.masks[frame_idx]
                overlay[mask] = (
                    0.55 * overlay[mask] + 0.45 * np.asarray(color)
                ).astype(np.uint8)
                for (x, y), is_visible in zip(
                    result.tracks_xy[frame_idx], result.visible[frame_idx], strict=True
                ):
                    if is_visible and 0 <= x < width and 0 <= y < height:
                        cv2.circle(overlay, (round(float(x)), round(float(y))), 3, color, -1)
                cv2.putText(
                    overlay,
                    name,
                    (16, 30 + 28 * names.index(name)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_perception_report(report: Mapping[str, object], output_path: str | Path) -> None:
    """Write a deterministic, machine-readable perception report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
