# Gaussian splatting stage

## Position in the pipeline

The Gaussian representation is built after calibrated geometry and before synthetic
robot-image generation:

```text
RGB-D video → masks and tracks → metric geometry and poses
            → labeled 3D Gaussian scene
            → robot and object trajectory composition
            → randomized Gaussian renders with actions
            → policy training and robustness evaluation
```

The Gaussian scene supplies appearance. Measured primitives remain the collision
geometry used by the simulator.

## Implemented local stage

`ov2p build-gaussian-scene` unprojects aligned RGB-D into explicit Gaussian means,
anisotropic scales, rotations, opacity, RGB color, and semantic labels. Its renderer
projects full 3D covariance into screen-space ellipses and performs front-to-back
alpha compositing. Source and target Gaussian groups can be transformed independently.

The verified HOI4D initialization uses source frame 89 because frame 86 has no valid
depth under the ball mask:

```bash
uv run ov2p build-gaussian-scene \
  --rgb results/gaussian_scene/inputs/00089.png \
  --depth data/raw/sources/hoi4d/HOI4D_metric/ZY20210800001_H1_C7_N14_S280_s04_T5/raw_depth/00089.png \
  --camera-info data/raw/sources/hoi4d/HOI4D_metric/ZY20210800001_H1_C7_N14_S280_s04_T5/camera/recon/split_0/info.json \
  --source-mask data/interim/hoi4d_ball_to_bowl_gate/masks/source/000001.png \
  --target-mask data/interim/hoi4d_ball_to_bowl_gate/masks/target/000001.png \
  --output results/gaussian_scene/hoi4d_ball_to_bowl_frame89 \
  --max-width 640 --stride 3
```

Measured result:

| Metric | Value |
|---|---:|
| Gaussian count | 18,420 |
| Static/source/target count | 18,211 / 36 / 173 |
| Valid-depth render coverage | 73.0% |
| Same-view PSNR | 24.18 dB |
| Verification camera translation | 3 cm |
| Verification source translation | 5 cm |

The output contains `scene.npz`, a reference render, alpha and depth images, a novel
camera view, an independently moved source render, and `report.json`.

## Metric camera trajectory and multiview fusion

The arbitrary-scale supplied camera poses were replaced with RGB-D PnP odometry over
all 72 sampled frames. Dynamic source and target masks are excluded from feature
matching. The recovered path is 0.649 m long with 0.202 m start-to-end displacement,
91.5% median inlier ratio, 0.75 px median reprojection error, and 2.1 mm median
matched-depth residual.

Seven keyframes are transformed into the metric world frame and fused at 1 cm voxel
resolution. Static voxels require observations from at least two frames; movable ball
and bowl Gaussians come only from reference frame 1 to avoid motion smearing.

| Multiview metric | Value |
|---|---:|
| Fused Gaussian count | 7,007 |
| Static/source/target count | 6,954 / 9 / 44 |
| Held-out frame | 30 |
| Held-out evaluated coverage | 67.0% |
| Held-out PSNR | 20.68 dB |

The trajectory command is `ov2p estimate-rgbd-trajectory`. Multiview fusion is
reproduced by `scripts/fuse_hoi4d_gaussians.py`.

## Remaining Gaussian work

1. Optimize Gaussian means, covariance, opacity, and color against held-out views.
2. Remove the hand and other transient pixels from the static scene.
3. Transform source Gaussians with the recovered 6D object trajectory.
4. Composite the simulated robot with correct depth ordering and shadows.
5. Render synchronized RGB, robot state, and action sequences for the four required
   policy ablations.

The present artifact proves metric multiview fusion, semantic motion, and rendering
contracts. It is not yet an optimized 3DGS or a Gaussian robot-demonstration dataset.
