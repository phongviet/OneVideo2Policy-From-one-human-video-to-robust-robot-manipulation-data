# Roadmap and gates

## Current milestone: one video to relative trajectory

1. Record one clean 10–20 second monocular Place demonstration.
2. Sample frames and manually select source and target in frame zero.
3. Integrate SAM2 and CoTracker3 adapters; annotate 30 evaluation frames.
4. Integrate TRELLIS and VGGT adapters; visually inspect novel views and scale.
5. Optimize initial translation, rotation, and scale with RGB/depth/mask losses.
6. Warm-start each later pose from its predecessor and ablate tracked-point loss.

### Gate A — reconstruction

Both rigid objects are recognizable from useful nearby views and their proportions are
plausible. If not, simplify the objects or swap the reconstruction backend.

### Gate B — tracking

Object identity is stable, mask IoU and track survival meet `configs/place.yaml`, and
the relative trajectory has no catastrophic jumps. If not, manually refine the
initialization before adding downstream systems.

## Deferred milestones

- Robot motion: manually define the grasp transform; generate transit, grasp,
  transfer, and release stages with kinematic validation.
- Synthetic engine: randomize object pose, nearby cameras, background, tabletop
  texture, and Gaussian color-based lighting.
- Policy: train Diffusion Policy from front/side RGB and joint positions.
- Evaluation: compare one demo, 2D augmentation, geometry-only 3D augmentation, and
  full 3DGS under isolated and combined shifts.

### Gate C — generation

At least 80–90% of synthetic trajectories satisfy task and kinematic constraints.

### Gate D — learning

The policy learns the in-distribution Place task before robustness experiments.

### Gate E — value

Full or geometry-only 3D augmentation improves robustness over ordinary 2D
augmentation under at least one controlled shift. A negative result is retained and
analyzed rather than hidden.

