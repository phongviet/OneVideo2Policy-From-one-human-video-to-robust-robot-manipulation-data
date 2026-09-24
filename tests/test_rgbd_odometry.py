import numpy as np

from onevideo2policy.pose_tracking.rgbd_odometry import accumulate_world_to_camera


def test_relative_rgbd_transforms_accumulate_in_order() -> None:
    first_to_second = np.eye(4)
    first_to_second[0, 3] = -0.1
    second_to_third = np.eye(4)
    second_to_third[1, 3] = 0.2

    world_to_camera = accumulate_world_to_camera([first_to_second, second_to_third])
    camera_to_world = np.linalg.inv(world_to_camera)

    assert world_to_camera.shape == (3, 4, 4)
    assert np.allclose(camera_to_world[1, :3, 3], [0.1, 0, 0])
    assert np.allclose(camera_to_world[2, :3, 3], [0.1, -0.2, 0])


def test_relative_rgbd_transform_shape_is_checked() -> None:
    with np.testing.assert_raises(ValueError):
        accumulate_world_to_camera([np.eye(3)])
