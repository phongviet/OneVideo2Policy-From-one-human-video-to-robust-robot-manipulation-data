"""Measured ball-to-bowl robosuite task used by local generation experiments."""

from __future__ import annotations

import numpy as np
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.objects import BallObject, HollowCylinderObject
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class FixedHollowCylinderObject(HollowCylinderObject):
    def _get_geom_attrs(self):
        attributes = super()._get_geom_attrs()
        attributes["joints"] = None
        return attributes


class BallToBowl(PickPlace):
    """Panda task with the measured HOI4D ball and bowl dimensions."""

    BALL_RADIUS = 0.01916061104985851
    BOWL_OUTER_RADIUS = 0.0495615182508963
    BOWL_INNER_RADIUS = BOWL_OUTER_RADIUS - 0.004
    BOWL_HALF_HEIGHT = 0.05693338151487156 / 2

    def __init__(self, *args, **kwargs):
        kwargs["single_object_mode"] = 2
        kwargs["object_type"] = "milk"  # index zero for this one-object subclass
        super().__init__(*args, **kwargs)

    def _construct_objects(self):
        self.objects = [
            BallObject(
                name="Ball",
                size=[self.BALL_RADIUS],
                density=120.0,
                friction=[1.0, 0.01, 0.001],
                rgba=[0.95, 0.35, 0.05, 1.0],
            )
        ]

    def _construct_visual_objects(self):
        self.visual_objects = [
            FixedHollowCylinderObject(
                name="VisualBowl",
                outer_radius=self.BOWL_OUTER_RADIUS,
                inner_radius=self.BOWL_INNER_RADIUS,
                height=self.BOWL_HALF_HEIGHT,
                ngeoms=24,
                rgba=[0.15, 0.35, 0.9, 1.0],
                friction=[1.0, 0.01, 0.001],
            )
        ]

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="BallBowlSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="BallSampler",
                mujoco_objects=self.objects,
                x_range=[-0.07, 0.07],
                y_range=[-0.07, 0.07],
                rotation=0,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=0,
            )
        )
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="BowlSampler",
                mujoco_objects=self.visual_objects,
                x_range=[-self.bin_size[0] / 4, -self.bin_size[0] / 4],
                y_range=[-self.bin_size[1] / 4, -self.bin_size[1] / 4],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.bin2_pos,
                z_offset=self.BOWL_HALF_HEIGHT,
            )
        )

    @property
    def ball_position(self) -> np.ndarray:
        return self.sim.data.body_xpos[self.obj_body_id["Ball"]].copy()

    @property
    def bowl_position(self) -> np.ndarray:
        return self.sim.data.body_xpos[self.obj_body_id["VisualBowl"]].copy()

    def _check_success(self):
        ball = self.ball_position
        bowl = self.bowl_position
        horizontal_error = np.linalg.norm(ball[:2] - bowl[:2])
        below_rim = ball[2] <= bowl[2] + self.BOWL_HALF_HEIGHT + self.BALL_RADIUS
        above_floor = ball[2] >= self.bin2_pos[2] + self.BALL_RADIUS * 0.5
        grasped = self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.objects[0].contact_geoms
        )
        success = bool(
            horizontal_error <= self.BOWL_INNER_RADIUS - self.BALL_RADIUS * 0.5
            and below_rim
            and above_floor
            and not grasped
        )
        self.objects_in_bins[0] = int(success)
        return success


register_env(BallToBowl)
