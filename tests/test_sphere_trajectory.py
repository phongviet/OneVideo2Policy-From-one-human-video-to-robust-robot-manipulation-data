import numpy as np

from onevideo2policy.pose_tracking.sphere_trajectory import fit_fixed_radius_sphere


def test_fixed_radius_sphere_fit_recovers_center_from_visible_surface() -> None:
    rng = np.random.default_rng(4)
    radius = 0.03
    center = np.array([0.08, -0.04, 0.9])
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions[directions[:, 2] < -0.15]
    points = center + radius * directions + rng.normal(scale=0.0005, size=directions.shape)

    fitted, median_error = fit_fixed_radius_sphere(points, radius)

    assert np.linalg.norm(fitted - center) < 0.002
    assert median_error < 0.001


def test_fixed_radius_sphere_rejects_too_few_points() -> None:
    with np.testing.assert_raises(ValueError):
        fit_fixed_radius_sphere(np.zeros((5, 3)), 0.03)
