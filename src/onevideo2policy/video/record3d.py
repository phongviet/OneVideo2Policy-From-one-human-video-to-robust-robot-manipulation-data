from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Record3DMetadata:
    frame_count: int
    fps: float
    rgb_size: tuple[int, int]
    depth_size: tuple[int, int]
    intrinsics_rgb: NDArray[np.float64]
    poses: NDArray[np.float64]
    raw: dict[str, Any]


class Record3DArchive:
    """Read RGB, metric depth, intrinsics, and raw poses from a Record3D archive."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            raw = json.loads(archive.read("metadata"))
        poses = np.asarray(raw["poses"], dtype=np.float64)
        intrinsics = np.asarray(raw["K"], dtype=np.float64).reshape(3, 3).T
        self.metadata = Record3DMetadata(
            frame_count=len(poses),
            fps=float(raw["fps"]),
            rgb_size=(int(raw["h"]), int(raw["w"])),
            depth_size=(int(raw["dh"]), int(raw["dw"])),
            intrinsics_rgb=intrinsics,
            poses=poses,
            raw=raw,
        )

    def read_rgb(self, frame_id: int) -> NDArray[np.uint8]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional video extra
            raise RuntimeError("Record3D RGB decoding requires: uv sync --extra video") from exc
        self._validate_frame(frame_id)
        with zipfile.ZipFile(self.path) as archive:
            encoded = np.frombuffer(archive.read(f"rgbd/{frame_id}.jpg"), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode RGB frame {frame_id}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read_depth_m(self, frame_id: int) -> NDArray[np.float32]:
        try:
            import lzfse
        except ImportError as exc:  # pragma: no cover - optional video extra
            raise RuntimeError("Record3D depth decoding requires: uv sync --extra video") from exc
        self._validate_frame(frame_id)
        with zipfile.ZipFile(self.path) as archive:
            compressed = archive.read(f"rgbd/{frame_id}.depth")
        height, width = self.metadata.depth_size
        depth = np.frombuffer(lzfse.decompress(compressed), dtype=np.float32)
        if depth.size != height * width:
            raise ValueError(
                f"Depth frame {frame_id} has {depth.size} samples; expected {height * width}"
            )
        return depth.reshape(height, width).copy()

    def intrinsics_for_depth(self) -> NDArray[np.float64]:
        rgb_height, rgb_width = self.metadata.rgb_size
        depth_height, depth_width = self.metadata.depth_size
        scaled = self.metadata.intrinsics_rgb.copy()
        scaled[0] *= depth_width / rgb_width
        scaled[1] *= depth_height / rgb_height
        return scaled

    def _validate_frame(self, frame_id: int) -> None:
        if not 0 <= frame_id < self.metadata.frame_count:
            raise IndexError(
                f"frame_id {frame_id} outside [0, {self.metadata.frame_count - 1}]"
            )
