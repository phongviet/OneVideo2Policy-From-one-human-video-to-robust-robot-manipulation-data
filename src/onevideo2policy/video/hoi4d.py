"""Import decoded HOI4D RGB-D sequences and their official motion masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def hoi4d_color_map(size: int = 10) -> NDArray[np.uint8]:
    """Return HOI4D's official Pascal-style RGB palette for motion labels."""
    if size <= 0:
        raise ValueError("size must be positive")
    palette = np.zeros((size, 3), dtype=np.uint8)
    for label in range(size):
        value = label
        for bit in range(8):
            palette[label, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[label, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[label, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
    return palette


def decode_hoi4d_motion_mask(
    mask_rgb: NDArray[np.uint8], labels: list[int] | tuple[int, ...]
) -> NDArray[np.bool_]:
    """Union requested one-based HOI4D motion labels into a binary foreground mask."""
    mask_rgb = np.asarray(mask_rgb, dtype=np.uint8)
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError("HOI4D mask must have shape [H, W, 3] in RGB order")
    if not labels or any(not isinstance(label, int) or not 0 < label < 10 for label in labels):
        raise ValueError("labels must be non-empty HOI4D indices from 1 through 9")
    palette = hoi4d_color_map()
    result = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for label in labels:
        result |= np.all(mask_rgb == palette[label], axis=-1)
    return result


def _numbered_files(directory: Path, suffixes: set[str]) -> list[Path]:
    files = [path for path in directory.iterdir() if path.suffix.lower() in suffixes]
    try:
        return sorted(files, key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"Expected zero-padded numbered files in {directory}") from exc


def import_hoi4d_sequence(
    sequence_dir: str | Path,
    output_dir: str | Path,
    *,
    source_labels: list[int],
    target_labels: list[int],
    decoded_fps: float = 15.0,
) -> dict[str, Any]:
    """Import a sequence decoded by HOI4D's official utility.

    ``sequence_dir`` retains ``align_rgb/image.mp4`` and contains decoded RGB JPEGs
    in ``align_rgb``, decoded depth PNGs in ``align_depth``, and official color-coded
    frames in ``2Dseg/mask``. Ground truth is separate from model predictions.
    """
    if decoded_fps <= 0:
        raise ValueError("decoded_fps must be positive")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("HOI4D import requires: uv sync --extra video") from exc

    sequence_dir = Path(sequence_dir)
    rgb_dir = sequence_dir / "align_rgb"
    depth_dir = sequence_dir / "align_depth"
    motion_dir = sequence_dir / "2Dseg" / "mask"
    source_video = rgb_dir / "image.mp4"
    for path in (rgb_dir, depth_dir, motion_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    rgb_files = _numbered_files(rgb_dir, {".jpg", ".jpeg"})
    depth_files = _numbered_files(depth_dir, {".png"})
    motion_files = _numbered_files(motion_dir, {".png"})
    if not rgb_files or len(rgb_files) != len(depth_files) or len(rgb_files) != len(motion_files):
        raise ValueError(
            "Decoded RGB, depth, and motion-mask frame counts must match and be non-zero"
        )
    if [path.stem for path in rgb_files] != [path.stem for path in depth_files] or [
        path.stem for path in rgb_files
    ] != [path.stem for path in motion_files]:
        raise ValueError("Decoded RGB, depth, and motion-mask filenames must align")

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    output_depth_dir = output_dir / "depth"
    source_gt_dir = output_dir / "ground_truth" / "source"
    target_gt_dir = output_dir / "ground_truth" / "target"
    for path in (frames_dir, output_depth_dir, source_gt_dir, target_gt_dir):
        path.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    resolution: tuple[int, int] | None = None
    for frame_id, (rgb_path, depth_path, motion_path) in enumerate(
        zip(rgb_files, depth_files, motion_files, strict=True)
    ):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        motion_bgr = cv2.imread(str(motion_path), cv2.IMREAD_COLOR)
        if rgb is None or depth is None or motion_bgr is None:
            raise ValueError(f"Could not decode HOI4D frame {rgb_path.stem}")
        if depth.ndim != 2:
            raise ValueError(f"HOI4D depth must be one channel: {depth_path}")
        if rgb.shape[:2] != depth.shape or rgb.shape[:2] != motion_bgr.shape[:2]:
            raise ValueError(f"HOI4D frame assets disagree in shape for {rgb_path.stem}")
        current_resolution = (rgb.shape[1], rgb.shape[0])
        if resolution is None:
            resolution = current_resolution
        elif resolution != current_resolution:
            raise ValueError("HOI4D frame resolutions must be constant")

        name = f"{frame_id:06d}"
        if not cv2.imwrite(str(frames_dir / f"{name}.jpg"), rgb):
            raise OSError(f"Could not write imported RGB frame {name}")
        np.save(output_depth_dir / f"{name}.npy", depth)
        motion_rgb = cv2.cvtColor(motion_bgr, cv2.COLOR_BGR2RGB)
        for labels, destination in ((source_labels, source_gt_dir), (target_labels, target_gt_dir)):
            binary = decode_hoi4d_motion_mask(motion_rgb, labels)
            if not cv2.imwrite(str(destination / f"{name}.png"), binary.astype(np.uint8) * 255):
                raise OSError(f"Could not write imported mask {name}")
        frames.append(
            {
                "frame_id": frame_id,
                "source_frame_id": int(rgb_path.stem),
                "timestamp_s": frame_id / decoded_fps,
                "rgb": f"frames/{name}.jpg",
                "depth": f"depth/{name}.npy",
            }
        )

    assert resolution is not None
    manifest = {
        "schema_version": 1,
        "source_video": str(source_video.resolve()),
        "source_frame_count": len(frames),
        "resolution": {"width": resolution[0], "height": resolution[1]},
        "native_fps": decoded_fps,
        "sample_fps": decoded_fps,
        "frame_count": len(frames),
        "frames": frames,
        "dataset": {
            "name": "HOI4D",
            "sequence": str(sequence_dir.resolve()),
            "ground_truth": "official 2D motion segmentation",
            "source_labels": source_labels,
            "target_labels": target_labels,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
