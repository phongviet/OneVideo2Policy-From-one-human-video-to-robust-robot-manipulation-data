from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_mask_points(
    mask: NDArray[np.bool_], count: int, *, seed: int = 42, border: int = 0
) -> NDArray[np.int64]:
    """Sample deterministic ``(x, y)`` points without replacement from a binary mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if count <= 0:
        raise ValueError("count must be positive")
    if border < 0:
        raise ValueError("border must be non-negative")

    eligible = mask.copy()
    if border:
        if border * 2 >= min(mask.shape):
            raise ValueError("border removes the entire mask extent")
        eligible[:border] = False
        eligible[-border:] = False
        eligible[:, :border] = False
        eligible[:, -border:] = False

    rows_yx = np.argwhere(eligible)
    if len(rows_yx) < count:
        raise ValueError(f"Mask contains {len(rows_yx)} eligible pixels; requested {count}")
    selected = rows_yx[np.random.default_rng(seed).choice(len(rows_yx), count, replace=False)]
    return selected[:, ::-1].astype(np.int64, copy=False)
