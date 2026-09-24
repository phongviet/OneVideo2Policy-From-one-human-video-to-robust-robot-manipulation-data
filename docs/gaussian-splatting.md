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

## Remaining Gaussian work

1. Fuse multiple RGB-D frames in a common metric camera frame and remove duplicate
   or inconsistent Gaussians.
2. Optimize Gaussian means, covariance, opacity, and color against held-out views.
3. Remove the hand and other transient pixels from the static scene.
4. Transform source Gaussians with the recovered 6D object trajectory.
5. Composite the simulated robot with correct depth ordering and shadows.
6. Render synchronized RGB, robot state, and action sequences for the four required
   policy ablations.

The present artifact proves representation, semantic motion, and rendering contracts.
It is not yet a trained multiview 3DGS or a Gaussian robot-demonstration dataset.
