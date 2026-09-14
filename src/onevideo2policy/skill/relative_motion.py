from __future__ import annotations

from onevideo2policy.types import ObjectTrack, PoseSE3


def relative_trajectory(source: ObjectTrack, target: ObjectTrack) -> tuple[PoseSE3, ...]:
    """Compute target-to-source transforms: inverse(T_target) @ T_source."""
    if len(source.poses) != len(target.poses):
        raise ValueError("Source and target tracks must have equal length")
    if len(source.timestamps_s) and not all(source.timestamps_s == target.timestamps_s):
        raise ValueError("Source and target timestamps must match")
    return tuple(
        target_pose.inverse().compose(source_pose)
        for source_pose, target_pose in zip(source.poses, target.poses, strict=True)
    )
