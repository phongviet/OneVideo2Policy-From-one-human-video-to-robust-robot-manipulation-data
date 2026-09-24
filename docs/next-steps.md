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

## Work remaining

1. **Gaussian robot rendering:** optimize the fused seven-keyframe metric
   scene, remove transient hand pixels, and composite the robot.
2. **Controlled evaluation:** measure success across object pose, camera, background,
   and lighting shifts with confidence intervals and failure categories.
3. **Physical run:** calibrate a real robot and camera, add safety checks, then compare
   the learned policy with the scripted controller.

## Immediate command

Reproduce the measured-task learned policy gate:

```bash
MUJOCO_GL=egl PYTHONPATH=scripts .venv/bin/python \
  scripts/evaluate_ball_bowl_waypoint.py \
  --checkpoint results/policy/ball_localization_500/visual_waypoint.pt \
  --output results/evaluation/ball_localization_500_random5 --episodes 5
```

The detailed measurements are in
[`local-model-benchmarks.md`](local-model-benchmarks.md), and the gate sequence is in
[`roadmap.md`](roadmap.md).
