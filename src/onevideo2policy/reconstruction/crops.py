from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def make_rgba_crop(
    rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    *,
    padding_fraction: float = 0.15,
    square: bool = True,
) -> tuple[NDArray[np.uint8], dict[str, object]]:
    """Extract a transparent object crop and its source-image geometry."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("rgb and mask must have matching [H,W] dimensions")
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be non-negative")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot crop an empty object mask")

    object_width = int(xs.max() - xs.min() + 1)
    object_height = int(ys.max() - ys.min() + 1)
    padding = int(np.ceil(max(object_width, object_height) * padding_fraction))
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(rgb.shape[1], int(xs.max()) + padding + 1)
    y1 = min(rgb.shape[0], int(ys.max()) + padding + 1)

    cropped_rgb = rgb[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]
    rgba = np.dstack((cropped_rgb, cropped_mask.astype(np.uint8) * 255))
    rgba[~cropped_mask, :3] = 0
    offset = [0, 0]
    if square and rgba.shape[0] != rgba.shape[1]:
        side = max(rgba.shape[:2])
        canvas = np.zeros((side, side, 4), dtype=np.uint8)
        top = (side - rgba.shape[0]) // 2
        left = (side - rgba.shape[1]) // 2
        canvas[top : top + rgba.shape[0], left : left + rgba.shape[1]] = rgba
        rgba = canvas
        offset = [left, top]
    metadata = {
        "source_bbox_xyxy": [x0, y0, x1, y1],
        "object_bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "canvas_offset_xy": offset,
        "crop_shape": list(rgba.shape),
        "foreground_pixels": int(mask.sum()),
    }
    return rgba, metadata


def export_rgba_crops(
    rgb: NDArray[np.uint8],
    masks: Mapping[str, NDArray[np.bool_]],
    output_dir: str | Path,
    *,
    frame_idx: int,
    padding_fraction: float = 0.15,
) -> dict[str, object]:
    """Write named RGBA crops plus a machine-readable crop manifest."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Writing RGBA crops requires: uv sync --extra video") from exc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: dict[str, object] = {}
    for name, mask in masks.items():
        rgba, metadata = make_rgba_crop(
            rgb, mask, padding_fraction=padding_fraction, square=True
        )
        path = output_dir / f"{name}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
            raise OSError(f"Failed to write {path}")
        objects[name] = {"path": path.name, **metadata}
    manifest = {"frame_idx": frame_idx, "padding_fraction": padding_fraction, "objects": objects}
    (output_dir / "crops.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_video_frame(video_path: str | Path, frame_idx: int) -> NDArray[np.uint8]:
    """Decode one exact source-video frame as RGB without extracting the full clip."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Reading source video requires: uv sync --extra video") from exc
    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok or bgr is None:
        raise ValueError(f"Could not decode frame {frame_idx} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_masks_to_frame(
    masks: Mapping[str, NDArray[np.bool_]], frame_shape: tuple[int, int]
) -> dict[str, NDArray[np.bool_]]:
    """Nearest-neighbor scale frozen masks onto their original-resolution RGB frame."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Resizing masks requires: uv sync --extra video") from exc
    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError("frame_shape must be positive")
    return {
        name: cv2.resize(
            np.asarray(mask, dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        for name, mask in masks.items()
    }
