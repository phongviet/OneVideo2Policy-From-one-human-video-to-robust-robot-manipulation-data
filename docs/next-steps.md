# Next steps

## Current state

The 72-frame HOI4D ball-to-bowl sequence now supplies RGB, aligned sensor depth,
official motion masks, measured geometry, and a passing SAM2/CoTracker3 perception
run. The selected 6 GB model stack is:

- SAM2.1 Hiera Small for segmentation;
- CoTracker3 offline for point tracking;
- TripoSR at mesh resolution 128 for visual object proposals;
- Depth Anything V2 Metric Hypersim Small at input size 518 for depth structure;
- HOI4D RGB-D and measured primitives for scale and collision geometry;
- robosuite with the temporal visual waypoint policy for robot experiments.

TripoSR and Depth Anything have both run on the local machine. The learned HOI4D
meshes are too flat for collision use, and monocular depth has a large scale bias.
Those failures are contained by using measured geometry for physics.

The measured ball-to-bowl robosuite task is now implemented. Its scripted controller
passes 3/3 oracle-position rollouts, and the 2.61-million-parameter visual waypoint
model passes 5/5 randomized rollouts after training on 500 targeted localization
frames. Median held-out localization error is 5.7 mm and median closed-loop initial
error is 4.9 mm.

Gaussian appearance composition is also integrated. A mixed 2,000-frame locator
passes 5/5 paired clean and 5/5 Gaussian-composite rollouts. One complete composited
robot trajectory contains 350 synchronized dual-camera, state, and action samples.
That policy augmentation remains appearance-only. A separate semantic-anchor path now
registers robosuite to HOI4D metric cameras and produces a 353-sample dual-camera
demonstration with Gaussian/simulator depth ordering.

The final 4,000-frame robustness model also includes rendered ±2 cm camera jitter
and half-light data. It passes 97/100 total rollouts: 20/20 nominal, 19/20 camera,
20/20 lighting, 20/20 Gaussian appearance, and 18/20 combined.

## Work remaining

1. **Physical run:** calibrate a real robot and camera, add safety checks, then compare
   the learned policy with the scripted controller.

The correspondence-sensitive rotation ablation is complete on the asymmetric
`EM1-0406` action camera: 0.70 px median held-out error versus 34.89 px without
tracked correspondences. Its calibrated depth also closes the reconstruction
comparison: neither TripoSR nor Stable Fast 3D passes the 15% proportion gate, so
metric RGB-D geometry remains selected. The remaining item requires external evidence
absent from this workspace. Metric Gaussian insertion is now implemented from the
shared table normal, bowl center, and source direction, including a 353-sample
depth-ordered dual-camera demonstration. Physical validation needs access to the
robot, camera calibration, workspace
measurements, and an operator-approved safety envelope. The existing video cannot
recover those quantities by computation alone.

## Immediate command

Reproduce the measured-task learned policy gate:

```bash
MUJOCO_GL=egl PYTHONPATH=scripts:src .venv/bin/python \
  scripts/evaluate_ball_bowl_waypoint.py \
  --checkpoint results/policy/ball_localization_robust_4000/visual_waypoint.pt \
  --output results/evaluation/final_nominal_20 --episodes 20
```

The detailed measurements are in
[`local-model-benchmarks.md`](local-model-benchmarks.md), and the gate sequence is in
[`roadmap.md`](roadmap.md).
