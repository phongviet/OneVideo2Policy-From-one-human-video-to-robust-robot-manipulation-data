"""Utilities for compositing MuJoCo task foreground over Gaussian scene renders."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_video_frames(path: Path, width: int, height: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def composite_task_foreground(
    image: np.ndarray, segmentation: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep registered task classes and replace unlabeled simulator background."""
    mask = (np.squeeze(segmentation) > 0).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)[..., None]
    composite = image.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8), alpha[..., 0]
