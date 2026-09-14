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
) -> dict[str, ObjectPerception]:
    """Propagate masks, seed deterministic points, then track each object."""
    frames = np.asarray(frames, dtype=np.uint8)
    if not prompts:
        raise ValueError("At least one named object prompt is required")
    masks_by_id = segmenter.segment_video(
        frames_dir, prompts.values(), expected_frame_count=len(frames)
    )
    results: dict[str, ObjectPerception] = {}
    for name, prompt in prompts.items():
        masks = masks_by_id[prompt.object_id]
        points = sample_mask_points(
            masks[prompt.frame_idx], point_count, seed=seed + prompt.object_id, border=border
        )
        tracks, visible = tracker.track(frames, points, query_frame=prompt.frame_idx)
        results[name] = ObjectPerception(masks, points, tracks, visible)
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
            tracks_dir / f"{name}.npz", xy=result.tracks_xy, visible=result.visible
        )
