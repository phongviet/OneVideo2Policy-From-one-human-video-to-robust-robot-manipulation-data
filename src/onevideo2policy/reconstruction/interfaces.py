from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class DepthEstimator(Protocol):
    """Metric or scale-consistent depth estimator boundary."""

    def estimate(self, frames: NDArray[np.uint8]) -> NDArray[np.floating]: ...


class ObjectReconstructor(Protocol):
    """Single-view object reconstruction boundary."""

    def reconstruct(
        self, crop: NDArray[np.uint8], mask: NDArray[np.bool_], output: Path
    ) -> Path: ...
