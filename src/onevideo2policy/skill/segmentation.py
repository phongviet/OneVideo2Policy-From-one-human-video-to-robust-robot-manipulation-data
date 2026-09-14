from __future__ import annotations

import numpy as np

from onevideo2policy.skill.relative_motion import relative_trajectory
from onevideo2policy.types import ObjectTrack, SkillTrajectory


def segment_place_skill(
    source: ObjectTrack,
    target: ObjectTrack,
    stationary_speed_m_s: float = 0.01,
    interaction_distance_m: float = 0.03,
) -> SkillTrajectory:
    """Extract a conservative grasp/transfer/place interval from two pose tracks.

    Grasp is the first sustained source movement; place is the first later frame that
    is both stationary and within the interaction threshold of the target.
    """
    if len(source.poses) < 3 or len(source.poses) != len(target.poses):
        raise ValueError("Place segmentation requires equal tracks with at least three poses")
    xyz = np.stack([pose.translation for pose in source.poses])
    target_xyz = np.stack([pose.translation for pose in target.poses])
    dt = np.diff(source.timestamps_s)
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
    moving = np.flatnonzero(speed > stationary_speed_m_s)
    if moving.size == 0:
        raise ValueError("No source-object movement was detected")
    grasp_frame = int(moving[0])

    distance = np.linalg.norm(xyz - target_xyz, axis=1)
    candidates = [
        frame
        for frame in range(grasp_frame + 1, len(xyz))
        if distance[frame] <= interaction_distance_m
        and (frame == len(xyz) - 1 or speed[frame] <= stationary_speed_m_s)
    ]
    if not candidates:
        raise ValueError("No stationary placement near the target was detected")
    place_frame = candidates[0]
    return SkillTrajectory(
        grasp_frame=grasp_frame,
        transfer_start=min(grasp_frame + 1, place_frame),
        place_frame=place_frame,
        relative_poses=relative_trajectory(source, target),
    )
