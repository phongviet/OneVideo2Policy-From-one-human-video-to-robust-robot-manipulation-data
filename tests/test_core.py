import numpy as np
import pytest

from onevideo2policy.pose_tracking.losses import tracked_point_loss
from onevideo2policy.pose_tracking.metrics import mask_iou, track_survival, translation_jitter
from onevideo2policy.skill.relative_motion import relative_trajectory
from onevideo2policy.skill.segmentation import segment_place_skill
from onevideo2policy.types import ObjectTrack, PoseSE3


def make_track(name: str, positions: list[list[float]]) -> ObjectTrack:
    return ObjectTrack(
        name,
        np.arange(len(positions), dtype=float),
        tuple(PoseSE3.from_translation(np.asarray(position)) for position in positions),
    )


def test_pose_inverse_and_relative_trajectory() -> None:
    source = make_track("source", [[1, 0, 0], [2, 0, 0]])
    target = make_track("target", [[0.5, 0, 0], [0.5, 0, 0]])
    relative = relative_trajectory(source, target)
    assert np.allclose(relative[0].translation, [0.5, 0, 0])
    assert np.allclose(relative[1].translation, [1.5, 0, 0])


def test_tracking_metrics() -> None:
    predicted = np.array([[0.0, 0.0], [3.0, 4.0]])
    observed = np.zeros((2, 2))
    assert tracked_point_loss(predicted, observed) == pytest.approx(2.5)
    assert track_survival(np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)) == 0.5
    assert mask_iou(np.array([1, 1, 0]), np.array([0, 1, 1])) == pytest.approx(1 / 3)


def test_translation_jitter_is_zero_for_constant_velocity() -> None:
    track = make_track("source", [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert translation_jitter(track) == pytest.approx(0.0)


def test_place_skill_segmentation() -> None:
    source = make_track("source", [[0, 0, 0], [0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.2, 0, 0]])
    target = make_track("target", [[0.2, 0, 0]] * 5)
    skill = segment_place_skill(source, target, stationary_speed_m_s=0.01)
    assert skill.grasp_frame == 1
    # The object arrives at frame 3 and the following interval confirms it stopped.
    assert skill.place_frame == 3


def test_invalid_rotation_is_rejected() -> None:
    bad = np.eye(4)
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        PoseSE3(bad)
