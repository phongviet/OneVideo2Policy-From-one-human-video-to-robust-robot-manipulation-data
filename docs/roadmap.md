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
8. A fused Gaussian scene renders the animated ball with 0.28 px median centroid error.
9. The measured ball-to-bowl simulator passes 3/3 exact-position controller rollouts.
10. The compact learned visual waypoint policy passes 5/5 randomized measured-task
    rollouts with 4.9 mm median initial XY error.
11. Gaussian appearance composition produces synchronized robot demonstrations; the
    mixed model passes 5/5 paired clean and 5/5 composited rollouts.
12. The final robustness model passes 97/100 across nominal, camera, lighting,
    Gaussian-background, and combined 20-episode conditions.

## Remaining gates

### Gate A — reconstruction quality

Capture or select larger, multi-view object crops and compare reconstructed proportions
against measured geometry. Current TripoSR HOI4D meshes are too flat for physics.

### Gate B — motion quality

Camera and spherical-object translation now pass their metric artifact checks. Quantify
the effect of the 20-frame interpolated occlusion interval and ablate the tracked-point
loss on a nonsymmetric object where rotation is observable.

### Gate C — task transfer (passed locally)

The measured HOI4D ball-and-bowl geometry is retargeted into robosuite. Successful
oracle and learned-policy rollouts verify grasp, transfer, release, and placement.

### Gate D — policy evidence (passed locally)

The selected waypoint policy passes 97/100 total held-out simulated rollouts across
object pose, ±2 cm camera translation, half lighting, Gaussian background, and their
combination. The combined condition passes 18/20. Physical transfer remains Gate E.

### Gate E — physical validation

Run a small real-robot evaluation after safety checks, calibration, and a successful
simulation transfer. Compare the learned policy with the scripted controller.
