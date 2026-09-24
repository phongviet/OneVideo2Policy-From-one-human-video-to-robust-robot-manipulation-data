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

## Work remaining

1. **Calibrated 6D trajectory:** recover camera and object poses from HOI4D RGB-D,
   report reprojection error and translation jitter, and compare against the current
   planar proxy.
2. **Task-specific simulator:** replace robosuite's built-in can and bin with the
   measured ball and bowl assets, then retarget the demonstrated trajectory.
3. **Policy transfer:** generate successful ball-to-bowl Panda demonstrations and
   retrain the selected temporal waypoint model on those observations.
4. **Controlled evaluation:** measure success across object pose, camera, background,
   and lighting shifts with confidence intervals and failure categories.
5. **Physical run:** calibrate a real robot and camera, add safety checks, then compare
   the learned policy with the scripted controller.

## Immediate command

Prepare the selected local model input contract:

```bash
uv run ov2p prepare-model-run \
  --config configs/two_paths.yaml \
  --manifest data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --crops data/interim/hoi4d_ball_to_bowl_gate_reseed16/crops_native \
  --output results/model_bundle/hoi4d_ball_to_bowl
```

The detailed measurements are in
[`local-model-benchmarks.md`](local-model-benchmarks.md), and the gate sequence is in
[`roadmap.md`](roadmap.md).
