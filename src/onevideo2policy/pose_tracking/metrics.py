from __future__ import annotations

import numpy as np

from onevideo2policy.types import ObjectTrack


def translation_jitter(track: ObjectTrack) -> float:
    """RMS second finite difference of translation, in metres per frame squared."""
    translations = np.stack([pose.translation for pose in track.poses])
    if len(translations) < 3:
        return 0.0
    acceleration = np.diff(translations, n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(acceleration**2, axis=1))))


def track_survival(visibility: np.ndarray) -> float:
    """Fraction of initially visible points still visible on the final frame."""
    visible = np.asarray(visibility, dtype=bool)
    if visible.ndim != 2 or visible.shape[0] == 0:
        raise ValueError("visibility must have shape [frames, points]")
    initial = visible[0]
    if not np.any(initial):
        raise ValueError("No points are visible in the initial frame")
    return float(np.count_nonzero(visible[-1] & initial) / np.count_nonzero(initial))


def mask_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("Masks must have matching shapes")
    union = np.count_nonzero(predicted | target)
    return 1.0 if union == 0 else float(np.count_nonzero(predicted & target) / union)
