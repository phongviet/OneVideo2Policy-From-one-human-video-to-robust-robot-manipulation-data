from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from onevideo2policy.video.sam2_adapter import _as_numpy

TensorFactory = Callable[[NDArray[Any], str], object]


class CoTrackerPredictor(Protocol):
    def __call__(self, video: object, **kwargs: object) -> tuple[object, object]: ...


def _torch_tensor(array: NDArray[Any], device: str) -> object:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
        raise RuntimeError("Install PyTorch and CoTracker; see docs/perception-models.md") from exc
    return torch.from_numpy(array).to(device)


class CoTracker3Adapter:
    """Offline CoTracker3 adapter using explicit ``(t, x, y)`` point queries."""

    def __init__(
        self,
        predictor: CoTrackerPredictor,
        *,
        device: str = "cuda",
        tensor_factory: TensorFactory = _torch_tensor,
    ) -> None:
        self.predictor = predictor
        self.device = device
        self.tensor_factory = tensor_factory

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, device: str = "cuda"
    ) -> CoTracker3Adapter:
        try:
            from cotracker.predictor import CoTrackerPredictor as UpstreamPredictor
        except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
            raise RuntimeError(
                "Install the pinned CoTracker dependency; see docs/perception-models.md"
            ) from exc
        predictor = UpstreamPredictor(checkpoint=str(checkpoint), offline=True, v2=False)
        predictor = predictor.to(device)
        return cls(predictor, device=device)

    def track(
        self,
        frames: NDArray[np.uint8],
        query_points: NDArray[np.floating],
        *,
        query_frame: int = 0,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        frames = np.asarray(frames)
        points = np.asarray(query_points, dtype=np.float32)
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
            raise ValueError("frames must have shape [T, H, W, 3]")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("query_points must have shape [N, 2]")
        if not np.isfinite(points).all():
            raise ValueError("query_points must be finite")
        if not 0 <= query_frame < len(frames):
            raise ValueError("query_frame is outside the video")
        height, width = frames.shape[1:3]
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= height).any()
        ):
            raise ValueError("query_points must lie inside the video frame")

        video_np = np.transpose(frames, (0, 3, 1, 2))[None].astype(np.float32, copy=False)
        query_times = np.full((len(points), 1), query_frame, dtype=np.float32)
        queries_np = np.concatenate([query_times, points], axis=1)[None]
        video = self.tensor_factory(video_np, self.device)
        queries = self.tensor_factory(queries_np, self.device)
        tracks, visibility = self.predictor(
            video, queries=queries, backward_tracking=query_frame > 0
        )

        tracks_np = _as_numpy(tracks)
        visibility_np = _as_numpy(visibility)
        if tracks_np.ndim != 4 or tracks_np.shape[0] != 1 or tracks_np.shape[-1] != 2:
            raise ValueError("CoTracker tracks must have shape [1, T, N, 2]")
        if visibility_np.ndim == 4 and visibility_np.shape[-1] == 1:
            visibility_np = visibility_np[..., 0]
        if visibility_np.shape != tracks_np.shape[:3]:
            raise ValueError("CoTracker visibility must have shape [1, T, N]")
        if not np.issubdtype(visibility_np.dtype, np.bool_):
            raise ValueError("CoTracker visibility output must be boolean")
        if tracks_np.shape[1:3] != (len(frames), len(points)):
            raise ValueError("CoTracker output dimensions do not match frames and queries")
        return (
            tracks_np[0].astype(np.float32, copy=False),
            visibility_np[0].astype(bool, copy=False),
        )
