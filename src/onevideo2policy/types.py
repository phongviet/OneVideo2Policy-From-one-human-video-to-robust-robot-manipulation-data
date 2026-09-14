from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class PoseSE3:
    """A rigid transform stored as a validated 4x4 homogeneous matrix."""

    matrix: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError("Last transform row must be [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Rotation block must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("Rotation determinant must be +1")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls) -> PoseSE3:
        return cls(np.eye(4, dtype=np.float64))

    @classmethod
    def from_translation(cls, xyz: FloatArray) -> PoseSE3:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
        return cls(matrix)

    @property
    def translation(self) -> FloatArray:
        return self.matrix[:3, 3].copy()

    def inverse(self) -> PoseSE3:
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -(rotation.T @ translation)
        return PoseSE3(inverse)

    def compose(self, other: PoseSE3) -> PoseSE3:
        return PoseSE3(self.matrix @ other.matrix)


@dataclass(frozen=True)
class FrameObservation:
    frame_id: int
    timestamp_s: float
    rgb_path: Path
    source_mask_path: Path | None = None
    target_mask_path: Path | None = None


@dataclass(frozen=True)
class ObjectTrack:
    object_name: str
    timestamps_s: FloatArray
    poses: tuple[PoseSE3, ...]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) != len(self.poses):
            raise ValueError("There must be one timestamp per pose")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("Timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps_s", timestamps)


@dataclass(frozen=True)
class SkillTrajectory:
    grasp_frame: int
    transfer_start: int
    place_frame: int
    relative_poses: tuple[PoseSE3, ...]

    def __post_init__(self) -> None:
        if not (0 <= self.grasp_frame <= self.transfer_start <= self.place_frame):
            raise ValueError("Skill phase frames must be monotonically ordered")
