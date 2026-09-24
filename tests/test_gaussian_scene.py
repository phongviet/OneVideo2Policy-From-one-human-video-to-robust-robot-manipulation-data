from pathlib import Path

import numpy as np

from onevideo2policy.generation.gaussian_scene import (
    GaussianScene,
    fuse_gaussian_scenes,
    initialize_gaussians_from_rgbd,
    render_gaussians,
    transform_label,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = 32, 48
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((xx * 5, yy * 7, np.full_like(xx, 120)), axis=-1).astype(np.uint8)
    depth = np.ones((height, width), dtype=np.float32)
    intrinsics = np.array([[45, 0, width / 2], [0, 45, height / 2], [0, 0, 1]])
    source = np.zeros((height, width), dtype=bool)
    source[10:22, 16:30] = True
    return rgb, depth, intrinsics, source


def test_rgbd_gaussians_round_trip_and_render(tmp_path: Path) -> None:
    rgb, depth, intrinsics, source = _fixture()
    scene = initialize_gaussians_from_rgbd(rgb, depth, intrinsics, source_mask=source, stride=2)
    scene.save(tmp_path / "scene.npz")
    restored = GaussianScene.load(tmp_path / "scene.npz")

    rendered, rendered_depth, alpha = render_gaussians(
        restored, intrinsics, np.eye(4), width=48, height=32
    )

    assert len(restored.means_m) == 24 * 16
    assert np.count_nonzero(restored.labels == 1) > 0
    assert np.mean(alpha > 0.5) > 0.9
    assert np.isclose(np.median(rendered_depth[alpha > 0.5]), 1.0)
    reference = rgb.astype(np.float32) / 255.0
    assert np.mean(np.abs(rendered[alpha > 0.5] - reference[alpha > 0.5])) < 0.06


def test_object_gaussian_group_can_be_moved() -> None:
    rgb, depth, intrinsics, source = _fixture()
    scene = initialize_gaussians_from_rgbd(rgb, depth, intrinsics, source_mask=source, stride=2)
    transform = np.eye(4)
    transform[0, 3] = 0.1

    moved = transform_label(scene, 1, transform)

    selected = scene.labels == 1
    assert np.allclose(moved.means_m[selected, 0], scene.means_m[selected, 0] + 0.1)
    assert np.array_equal(moved.means_m[~selected], scene.means_m[~selected])


def test_multiview_fusion_filters_single_view_static_points() -> None:
    rgb, depth, intrinsics, source = _fixture()
    first = initialize_gaussians_from_rgbd(rgb, depth, intrinsics, source_mask=source, stride=4)
    second = initialize_gaussians_from_rgbd(rgb, depth, intrinsics, source_mask=source, stride=4)
    # Add one transient static Gaussian to only the second observation.
    second = GaussianScene(
        means_m=np.vstack((second.means_m, [4.0, 4.0, 4.0])).astype(np.float32),
        log_scales_m=np.vstack((second.log_scales_m, second.log_scales_m[0])),
        rotations_wxyz=np.vstack((second.rotations_wxyz, second.rotations_wxyz[0])),
        opacities=np.append(second.opacities, 0.9).astype(np.float32),
        colors_rgb=np.vstack((second.colors_rgb, [1.0, 0.0, 0.0])).astype(np.float32),
        labels=np.append(second.labels, 0).astype(np.uint8),
    )

    fused = fuse_gaussian_scenes(
        [first, second],
        np.repeat(np.eye(4)[None], 2, axis=0),
        reference_index=0,
        voxel_size_m=0.02,
        min_static_observations=2,
    )

    assert not np.any(np.all(np.isclose(fused.means_m, [4, 4, 4]), axis=1))
    assert np.count_nonzero(fused.labels == 1) == np.count_nonzero(first.labels == 1)
