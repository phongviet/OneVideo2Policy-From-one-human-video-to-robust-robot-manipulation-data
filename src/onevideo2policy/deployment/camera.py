"""Synchronized dual-camera contracts for physical waypoint inference."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraFrame:
    rgb: NDArray[np.uint8]
    timestamp_s: float


class CameraInterface(Protocol):
    def read(self) -> CameraFrame: ...

    def close(self) -> None: ...


class OpenCVCamera:
    """Minimal OpenCV camera adapter with timestamped RGB output."""

    def __init__(
        self,
        device: int | str,
        *,
        width: int = 84,
        height: int = 84,
        intrinsics: NDArray[np.floating] | None = None,
        distortion: NDArray[np.floating] | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional video dependency
            raise RuntimeError("OpenCVCamera requires the video extra") from exc
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(device)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._shape = (height, width, 3)
        self._intrinsics = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._distortion = None if distortion is None else np.asarray(distortion, dtype=np.float64)
        if (self._intrinsics is None) != (self._distortion is None):
            self._capture.release()
            raise ValueError("intrinsics and distortion must be supplied together")
        if self._intrinsics is not None and (
            self._intrinsics.shape != (3, 3)
            or self._distortion.ndim != 1
            or not np.isfinite(self._intrinsics).all()
            or not np.isfinite(self._distortion).all()
        ):
            self._capture.release()
            raise ValueError("camera calibration arrays are invalid")
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"could not open camera {device!r}")

    def read(self) -> CameraFrame:
        ok, bgr = self._capture.read()
        timestamp = time.monotonic()
        if not ok or bgr is None:
            raise RuntimeError("camera read failed")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        if rgb.shape != self._shape:
            raise RuntimeError(f"camera returned {rgb.shape}; expected {self._shape}")
        if self._intrinsics is not None and self._distortion.size:
            rgb = self._cv2.undistort(rgb, self._intrinsics, self._distortion)
        return CameraFrame(np.ascontiguousarray(rgb), timestamp)

    def close(self) -> None:
        self._capture.release()


class SynchronizedCameraPair:
    def __init__(
        self,
        agent: CameraInterface,
        front: CameraInterface,
        *,
        max_age_s: float = 0.25,
        max_skew_s: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min(max_age_s, max_skew_s) <= 0:
            raise ValueError("camera timing limits must be positive")
        self.agent = agent
        self.front = front
        self.max_age_s = max_age_s
        self.max_skew_s = max_skew_s
        self.clock = clock

    def capture(self) -> tuple[CameraFrame, CameraFrame]:
        agent = self.agent.read()
        front = self.front.read()
        now = self.clock()
        for name, frame in (("agent", agent), ("front", front)):
            if frame.rgb.shape != (84, 84, 3) or frame.rgb.dtype != np.uint8:
                raise RuntimeError(f"{name} camera must return uint8 RGB [84,84,3]")
            age = now - frame.timestamp_s
            if not np.isfinite(frame.timestamp_s) or age < 0 or age > self.max_age_s:
                raise RuntimeError(f"{name} camera frame is stale or has an invalid timestamp")
        if abs(agent.timestamp_s - front.timestamp_s) > self.max_skew_s:
            raise RuntimeError("camera frame skew exceeds the synchronization limit")
        return agent, front

    def close(self) -> None:
        self.agent.close()
        self.front.close()
