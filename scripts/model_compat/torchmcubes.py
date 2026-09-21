"""CPU marching-cubes fallback for a TripoSR experiment without CUDA nvcc.

TripoSR imports ``torchmcubes`` for mesh extraction. This drop-in replacement
uses scikit-image and leaves the model's image encoder and decoder unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.measure import marching_cubes as skimage_marching_cubes


def marching_cubes(volume: torch.Tensor, isovalue: float) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces, _, _ = skimage_marching_cubes(volume.detach().cpu().numpy(), isovalue)
    # The upstream torchmcubes output is z-y-x; TripoSR reverses it to x-y-z.
    vertices = np.ascontiguousarray(vertices[:, ::-1])
    faces = np.ascontiguousarray(faces[:, ::-1])
    return torch.from_numpy(vertices.astype(np.float32)), torch.from_numpy(faces.astype(np.int64))
