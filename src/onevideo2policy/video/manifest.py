from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_asset(manifest_dir: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Validate frame ordering, timestamps, assets, and optional RGB-D alignment."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a mapping")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing manifest schema_version")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Manifest must contain at least one frame")
    if manifest.get("frame_count") != len(frames):
        raise ValueError("frame_count does not match the frames list")

    source_video = _resolve_asset(path.parent, manifest.get("source_video"), "source_video")
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    source_frame_count = manifest.get("source_frame_count")
    if not isinstance(source_frame_count, int) or source_frame_count < len(frames):
        raise ValueError("source_frame_count must cover every sampled frame")
    resolution = manifest.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError("resolution must be a mapping")
    width, height = resolution.get("width"), resolution.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("resolution width and height must be positive integers")

    native_fps = manifest.get("native_fps")
    sample_fps = manifest.get("sample_fps")
    if not isinstance(native_fps, (int, float)) or native_fps <= 0:
        raise ValueError("native_fps must be positive")
    if not isinstance(sample_fps, (int, float)) or not 0 < sample_fps <= native_fps + 1e-6:
        raise ValueError("sample_fps must be positive and no greater than native_fps")

    manifest_dir = path.parent
    timestamps: list[float] = []
    source_ids: list[int] = []
    depth_presence: list[bool] = []
    depth_shape: tuple[int, ...] | None = None
    for expected_id, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{expected_id}] must be a mapping")
        if frame.get("frame_id") != expected_id:
            raise ValueError("frame_id values must be contiguous and zero-based")
        source_id = frame.get("source_frame_id")
        if not isinstance(source_id, int) or source_id < 0:
            raise ValueError(f"frames[{expected_id}].source_frame_id must be non-negative")
        source_ids.append(source_id)
        timestamp = frame.get("timestamp_s")
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"frames[{expected_id}].timestamp_s must be finite and non-negative")
        timestamps.append(float(timestamp))

        rgb_path = _resolve_asset(manifest_dir, frame.get("rgb"), f"frames[{expected_id}].rgb")
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        has_depth = "depth" in frame
        depth_presence.append(has_depth)
        if has_depth:
            current_path = _resolve_asset(
                manifest_dir, frame.get("depth"), f"frames[{expected_id}].depth"
            )
            if not current_path.is_file():
                raise FileNotFoundError(current_path)
            current_depth = np.load(current_path, mmap_mode="r", allow_pickle=False)
            if current_depth.ndim != 2 or not np.issubdtype(current_depth.dtype, np.number):
                raise ValueError(f"Depth frame must be a numeric 2D array: {current_path}")
            if depth_shape is None:
                depth_shape = current_depth.shape
            elif current_depth.shape != depth_shape:
                raise ValueError("All depth frames must have the same shape")

    if len(source_ids) > 1 and any(b <= a for a, b in pairwise(source_ids)):
        raise ValueError("source_frame_id values must be strictly increasing")
    if source_ids[-1] >= source_frame_count:
        raise ValueError("source_frame_id exceeds source_frame_count")
    if len(timestamps) > 1 and any(b <= a for a, b in pairwise(timestamps)):
        raise ValueError("timestamp_s values must be strictly increasing")
    if any(depth_presence) and not all(depth_presence):
        raise ValueError("Depth paths must be present for every frame or none")
    if depth_shape is not None and depth_shape != (height, width):
        raise ValueError("Depth shape does not match manifest resolution")

    return {
        "frame_count": len(frames),
        "duration_s": timestamps[-1] - timestamps[0],
        "has_depth": all(depth_presence),
        "depth_shape": list(depth_shape) if depth_shape is not None else None,
    }
