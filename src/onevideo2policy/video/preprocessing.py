from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def prepare_video(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float,
    depth_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Extract RGB frames at a fixed rate and write an aligned data manifest."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Video preparation requires: uv sync --extra video") from exc

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir = Path(output_dir)
    depth_dir = Path(depth_dir) if depth_dir is not None else None
    if depth_dir is not None and not depth_dir.is_dir():
        raise NotADirectoryError(depth_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if native_fps <= 0:
        capture.release()
        raise ValueError(f"Could not determine FPS for {video_path}")
    if sample_fps > native_fps + 1e-6:
        capture.release()
        raise ValueError(f"sample_fps ({sample_fps}) cannot exceed native FPS ({native_fps})")

    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step_s = 1.0 / sample_fps
    next_sample_s = 0.0
    source_index = 0
    frames: list[dict[str, Any]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_s = source_index / native_fps
        if timestamp_s + 1e-9 >= next_sample_s:
            frame_name = f"{len(frames):06d}.jpg"
            if not cv2.imwrite(str(frames_dir / frame_name), frame):
                capture.release()
                raise OSError(f"Failed to write {frames_dir / frame_name}")
            observation: dict[str, Any] = {
                "frame_id": len(frames),
                "source_frame_id": source_index,
                "timestamp_s": timestamp_s,
                "rgb": f"frames/{frame_name}",
            }
            if depth_dir is not None:
                depth_path = depth_dir / f"{source_index:06d}.npy"
                if not depth_path.is_file():
                    capture.release()
                    raise FileNotFoundError(f"Missing aligned depth frame: {depth_path}")
                observation["depth"] = os.path.relpath(depth_path.resolve(), output_dir.resolve())
            frames.append(observation)
            next_sample_s += step_s
        source_index += 1
    capture.release()

    manifest = {
        "schema_version": 1,
        "source_video": str(video_path.resolve()),
        "source_frame_count": source_frame_count,
        "resolution": {"width": width, "height": height},
        "native_fps": native_fps,
        "sample_fps": sample_fps,
        "frame_count": len(frames),
        "frames": frames,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest
