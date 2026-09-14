from pathlib import Path

import numpy as np
import pytest

from onevideo2policy.video.cotracker_adapter import CoTracker3Adapter
from onevideo2policy.video.perception import load_prompt_file, run_perception
from onevideo2policy.video.point_sampling import sample_mask_points
from onevideo2policy.video.sam2_adapter import Sam2PointPrompt, Sam2VideoAdapter


class FakeSam2Predictor:
    def __init__(self, masks: dict[int, np.ndarray]) -> None:
        self.masks = masks
        self.prompts: list[dict[str, object]] = []
        self.reset = False

    def init_state(self, *, video_path: str) -> dict[str, str]:
        return {"video_path": video_path}

    def reset_state(self, inference_state: object) -> None:
        self.reset = True

    def add_new_points_or_box(self, **kwargs: object) -> None:
        self.prompts.append(kwargs)

    def propagate_in_video(self, inference_state: object):
        object_ids = np.asarray(sorted(self.masks))
        frame_count = next(iter(self.masks.values())).shape[0]
        for frame_idx in range(frame_count):
            logits = np.stack(
                [self.masks[obj_id][frame_idx].astype(np.float32) for obj_id in object_ids]
            )[:, None]
            yield frame_idx, object_ids, logits


class FakeCoTrackerPredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def __call__(
        self, video: np.ndarray, *, queries: np.ndarray, backward_tracking: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((video, queries, backward_tracking))
        frame_count = video.shape[1]
        points = queries[0, :, 1:]
        tracks = np.repeat(points[None, None], frame_count, axis=1)
        visible = np.ones(tracks.shape[:3], dtype=bool)
        return tracks, visible


def test_sam2_adapter_propagates_prompted_objects(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    masks = {1: np.ones((3, 4, 5), dtype=bool), 2: np.zeros((3, 4, 5), dtype=bool)}
    predictor = FakeSam2Predictor(masks)
    adapter = Sam2VideoAdapter(predictor)
    prompts = [
        Sam2PointPrompt(1, np.array([[2, 2]]), np.array([1])),
        Sam2PointPrompt(2, np.array([[0, 0]]), np.array([0])),
    ]

    result = adapter.segment_video(frames_dir, prompts, expected_frame_count=3)

    assert predictor.reset
    assert set(result) == {1, 2}
    assert result[1].shape == (3, 4, 5)
    assert predictor.prompts[0]["normalize_coords"] is False


def test_cotracker_adapter_builds_txy_queries() -> None:
    predictor = FakeCoTrackerPredictor()
    adapter = CoTracker3Adapter(predictor, device="cpu", tensor_factory=lambda x, _: x)
    frames = np.zeros((3, 6, 8, 3), dtype=np.uint8)
    points = np.array([[2, 3], [5, 1]], dtype=np.float32)

    tracks, visible = adapter.track(frames, points, query_frame=1)

    video, queries, backward = predictor.calls[0]
    assert video.shape == (1, 3, 3, 6, 8)
    assert np.array_equal(queries[0, :, 0], [1, 1])
    assert np.array_equal(queries[0, :, 1:], points)
    assert backward is True
    assert tracks.shape == (3, 2, 2)
    assert visible.shape == (3, 2)


def test_perception_runner_uses_deterministic_mask_points(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = np.zeros((3, 6, 8, 3), dtype=np.uint8)
    source_masks = np.zeros((3, 6, 8), dtype=bool)
    source_masks[:, 1:5, 2:7] = True
    sam = Sam2VideoAdapter(FakeSam2Predictor({7: source_masks}))
    cotracker_predictor = FakeCoTrackerPredictor()
    tracker = CoTracker3Adapter(
        cotracker_predictor, device="cpu", tensor_factory=lambda x, _: x
    )
    prompt = Sam2PointPrompt(7, np.array([[3, 2]]), np.array([1]))

    result = run_perception(
        frames,
        frames_dir,
        {"source": prompt},
        sam,
        tracker,
        point_count=6,
        seed=42,
        border=1,
    )["source"]

    expected = sample_mask_points(source_masks[0], 6, seed=49, border=1)
    assert np.array_equal(result.seed_points_xy, expected)
    assert np.array_equal(cotracker_predictor.calls[0][1][0, :, 1:], expected)
    assert result.tracks_xy.shape == (3, 6, 2)


def test_adapters_reject_invalid_prompts_and_queries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="labels"):
        Sam2PointPrompt(1, np.array([[0, 0]]), np.array([2]))

    tracker = CoTracker3Adapter(
        FakeCoTrackerPredictor(), device="cpu", tensor_factory=lambda x, _: x
    )
    with pytest.raises(ValueError, match="inside"):
        tracker.track(
            np.zeros((2, 4, 5, 3), dtype=np.uint8),
            np.array([[5, 0]], dtype=np.float32),
        )


def test_load_prompt_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        '{"cup": {"object_id": 3, "points_xy": [[4, 5]], "labels": [1]}}'
    )

    prompts = load_prompt_file(prompt_path)

    assert prompts["cup"].object_id == 3
    assert prompts["cup"].frame_idx == 0
    assert np.array_equal(prompts["cup"].points_xy, [[4, 5]])
