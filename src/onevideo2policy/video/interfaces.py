from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Segmenter(Protocol):
    """Adapter boundary for SAM2 or an equivalent segmentation model."""

    def segment(self, rgb: NDArray[np.uint8], prompt: object) -> NDArray[np.bool_]: ...


class PointTracker(Protocol):
    """Adapter boundary for CoTracker3 or an equivalent point tracker."""

    def track(
        self,
        frames: NDArray[np.uint8],
        query_points: NDArray[np.floating],
        *,
        query_frame: int = 0,
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_]]: ...
