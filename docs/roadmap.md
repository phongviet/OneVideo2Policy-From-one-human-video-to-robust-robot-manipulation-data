# Roadmap and gates

## Current milestone

The local pipeline now runs segmentation, tracking, TripoSR reconstruction proposals,
Depth Anything V2 Small, measured collision geometry, robosuite demonstration
generation, and policy evaluation on the 6 GB machine.

Completed evidence:

1. SAM2.1 Hiera Small and CoTracker3 pass the HOI4D perception gate.
2. HOI4D RGB-D provides metric dimensions and depth reference.
3. TripoSR and Depth Anything V2 Small fit local VRAM and have measured reports.
4. Primitive collision assets and the point-robot systems path run end to end.
5. One hundred successful Panda demonstrations were generated in robosuite.
6. The temporal visual waypoint policy passes 19/20 held-out rollouts.
7. Metric RGB-D odometry and the target-relative ball trajectory cover all 72 frames.
8. A fused Gaussian scene renders the animated ball with 0.28 px median centroid error;
   held-out footprint tuning improves three independent views by 0.49 dB mean PSNR.
9. The measured ball-to-bowl simulator passes 3/3 exact-position controller rollouts.
10. The compact learned visual waypoint policy passes 5/5 randomized measured-task
    rollouts with 4.9 mm median initial XY error.
11. Gaussian appearance composition produces synchronized robot demonstrations; the
    mixed model passes 5/5 paired clean and 5/5 composited rollouts.
12. The final robustness model passes 97/100 across nominal, camera, lighting,
    Gaussian-background, and combined 20-episode conditions.
13. Semantic metric anchors register robosuite into HOI4D with a 2.7 mm table-plane
    residual and produce a 353-sample depth-ordered dual-camera Gaussian demo.
14. The physical deployment preflight validates calibration, freezes artifact hashes,
    smoke-tests checkpoint inference to 3.84 mm, and safety-checks a 149-command
    Cartesian dry run. A paired-trial validator enforces the final hardware gate.
15. Printable watertight source and target fixtures match the frozen policy geometry
    to STL float precision and include reproducible hash and manifold audits.
16. A print-ready ChArUco target and automatic detector produce native intrinsics,
    distortion, robot-frame correspondences, and held-out camera calibration inputs.

## Remaining gates

### Gate A — reconstruction quality (evaluated; learned meshes rejected for metrics)

The retained Record3D `EM1-0406` sequence provides a 282×282 asymmetric action-camera
crop and 609 aligned metric-depth samples. Its robust visible-surface extents are
81.9 × 60.6 × 35.4 mm. After isotropic longest-axis alignment, TripoSR predicts
81.9 × 69.7 × 14.5 mm and Stable Fast 3D predicts 81.9 × 68.3 × 23.7 mm. Neither
passes the frozen requirement that both secondary extents fall within 15% of the
RGB-D reference. Stable Fast 3D is closer, but is non-watertight and peaks at
6,169 MiB; TripoSR is watertight and peaks at 1,867 MiB. The project therefore uses
Record3D/HOI4D metric depth or fitted primitives for geometry and keeps TripoSR only
as a fast visual proposal. The comparison requirement is complete without promoting
an inaccurate learned mesh into physics.

### Gate B — motion quality (passed for observable image-plane rotation)

Camera and spherical-object translation pass their metric artifact checks. The retained
Record3D `EM1-0406` sequence supplies a nonsymmetric action camera with a rectangular
body and offset lens. Across 13 fully visible post-placement frames, a correspondence
constrained similarity fit reduces median error on 82 held-out points from 34.89 px
for mask-only pose fitting to 0.70 px, a 98.0% reduction. The recovered median image
plane rotation is −41.7°. This closes the tracked-point rotation ablation; full SE(3)
object-pose accuracy remains outside this 2D test.

### Gate C — task transfer (passed locally)

The measured HOI4D ball-and-bowl geometry is retargeted into robosuite. Successful
oracle and learned-policy rollouts verify grasp, transfer, release, and placement.

### Gate D — policy evidence (passed locally)

The selected waypoint policy passes 97/100 total held-out simulated rollouts across
object pose, ±2 cm camera translation, half lighting, Gaussian background, and their
combination. The combined condition passes 18/20. Physical transfer remains Gate E.

### Gate E — physical validation

The local deployment package and software preflight are complete. The preflight uses
the frozen policy checkpoint, transforms policy outputs into the robot-base frame,
enforces 1 cm steps, stationary gripper transitions,
8 cm/s speed, a 15 N force limit, workspace bounds, and the exact dual-camera training
geometry. It refuses to arm from the simulation calibration fixture.
The physical object gate also requires the 3.832 cm source and 9.912 × 5.693 cm target
geometry used to train and evaluate the selected policy; the separate 3 cm filmed
object is outside this tolerance.
The remaining evidence is a measured two-camera calibration and five paired physical
trials per controller on a real robot. Both scripted and learned controllers must
reach at least 4/5 successes with zero safety aborts or force violations. Follow
[`physical-evaluation-runbook.md`](physical-evaluation-runbook.md).
