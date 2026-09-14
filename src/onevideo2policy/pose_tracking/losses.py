from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tracked_point_loss(
    predicted_xy: NDArray[np.floating],
    observed_xy: NDArray[np.floating],
    visible: NDArray[np.bool_] | None = None,
) -> float:
    """Mean Euclidean reprojection error over visible point correspondences."""
    predicted = np.asarray(predicted_xy, dtype=np.float64)
    observed = np.asarray(observed_xy, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("Point arrays must have matching shape [N, 2]")
    valid = (
        np.ones(len(predicted), dtype=bool) if visible is None else np.asarray(visible, dtype=bool)
    )
    if valid.shape != (len(predicted),):
        raise ValueError("Visibility must have shape [N]")
    if not np.any(valid):
        raise ValueError("At least one point must be visible")
    return float(np.linalg.norm(predicted[valid] - observed[valid], axis=1).mean())


def weighted_pose_objective(
    rgb: float,
    depth: float,
    mask: float,
    track: float,
    weights: dict[str, float],
) -> float:
    """Combine the four paper-inspired alignment losses with explicit weights."""
    terms = {"rgb": rgb, "depth": depth, "mask": mask, "track": track}
    missing = terms.keys() - weights.keys()
    if missing:
        raise ValueError(f"Missing loss weights: {', '.join(sorted(missing))}")
    return float(sum(weights[name] * value for name, value in terms.items()))
