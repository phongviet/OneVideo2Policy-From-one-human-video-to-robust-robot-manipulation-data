from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class Sam2VideoPredictor(Protocol):
    def init_state(self, *, video_path: str) -> object: ...

    def reset_state(self, inference_state: object) -> None: ...

    def add_new_points_or_box(self, **kwargs: object) -> object: ...

    def propagate_in_video(
        self, inference_state: object
    ) -> Iterable[tuple[int, object, object]]: ...


@dataclass(frozen=True)
class Sam2PointPrompt:
    object_id: int
    points_xy: NDArray[np.floating]
    labels: NDArray[np.integer]
    frame_idx: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, int) or self.object_id < 0:
            raise ValueError("object_id must be a non-negative integer")
        if not isinstance(self.frame_idx, int) or self.frame_idx < 0:
            raise ValueError("frame_idx must be a non-negative integer")
        points = np.asarray(self.points_xy, dtype=np.float32)
        labels = np.asarray(self.labels)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_xy must have shape [N, 2]")
        if labels.shape != (len(points),):
            raise ValueError("labels must have one value per point")
        if len(points) == 0:
            raise ValueError("At least one prompt point is required")
        if not np.isfinite(points).all():
            raise ValueError("Prompt points must be finite")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Point labels must be integers")
        labels = labels.astype(np.int32, copy=False)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Point labels must be 0 (negative) or 1 (positive)")
        object.__setattr__(self, "points_xy", points)
        object.__setattr__(self, "labels", labels)


def _as_numpy(value: object) -> NDArray[Any]:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else value)


class Sam2VideoAdapter:
    """Thin adapter over Meta's SAM2 video-predictor API."""

    def __init__(self, predictor: Sam2VideoPredictor, *, mask_threshold: float = 0.0) -> None:
        self.predictor = predictor
        self.mask_threshold = mask_threshold

    @classmethod
    def from_hugging_face(
        cls,
        model_id: str = "facebook/sam2.1-hiera-small",
        *,
        device: str = "cuda",
        mask_threshold: float = 0.0,
    ) -> Sam2VideoAdapter:
        try:
            from sam2.build_sam import build_sam2_video_predictor_hf
        except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
            raise RuntimeError(
                "Install the pinned SAM2 dependency; see docs/perception-models.md"
            ) from exc
        predictor = build_sam2_video_predictor_hf(model_id, device=device)
        return cls(predictor, mask_threshold=mask_threshold)

    def segment_video(
        self,
        frames_dir: str | Path,
        prompts: Iterable[Sam2PointPrompt],
        *,
        expected_frame_count: int | None = None,
    ) -> dict[int, NDArray[np.bool_]]:
        frames_dir = Path(frames_dir)
        if not frames_dir.is_dir():
            raise NotADirectoryError(frames_dir)
        prompt_list = list(prompts)
        if not prompt_list:
            raise ValueError("At least one object prompt is required")
        object_ids = [prompt.object_id for prompt in prompt_list]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Each SAM2 prompt must use a unique object_id")

        state = self.predictor.init_state(video_path=str(frames_dir))
        self.predictor.reset_state(state)
        for prompt in prompt_list:
            self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.object_id,
                points=prompt.points_xy,
                labels=prompt.labels,
                clear_old_points=True,
                normalize_coords=False,
            )

        per_object: dict[int, dict[int, NDArray[np.bool_]]] = {obj_id: {} for obj_id in object_ids}
        for frame_idx, output_ids, mask_logits in self.predictor.propagate_in_video(state):
            ids = [int(value) for value in _as_numpy(output_ids).reshape(-1)]
            logits = _as_numpy(mask_logits)
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:, 0]
            if logits.ndim != 3 or len(ids) != logits.shape[0]:
                raise ValueError("SAM2 output must contain one [H, W] mask per object ID")
            for index, obj_id in enumerate(ids):
                if obj_id in per_object:
                    per_object[obj_id][int(frame_idx)] = logits[index] > self.mask_threshold

        masks: dict[int, NDArray[np.bool_]] = {}
        for obj_id, by_frame in per_object.items():
            if not by_frame:
                raise ValueError(f"SAM2 returned no masks for object {obj_id}")
            frame_ids = sorted(by_frame)
            if frame_ids != list(range(frame_ids[-1] + 1)):
                raise ValueError(
                    f"SAM2 masks for object {obj_id} are not contiguous from frame zero"
                )
            if expected_frame_count is not None and len(frame_ids) != expected_frame_count:
                raise ValueError(
                    f"SAM2 returned {len(frame_ids)} frames for object {obj_id}; "
                    f"expected {expected_frame_count}"
                )
            masks[obj_id] = np.stack([by_frame[index] for index in frame_ids])
        return masks
