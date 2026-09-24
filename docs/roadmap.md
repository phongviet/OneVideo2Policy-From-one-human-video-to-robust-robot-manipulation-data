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

## Remaining gates

### Gate A — reconstruction quality

Capture or select larger, multi-view object crops and compare reconstructed proportions
against measured geometry. Current TripoSR HOI4D meshes are too flat for physics.

### Gate B — motion quality

Camera and spherical-object translation now pass their metric artifact checks. Quantify
the effect of the 20-frame interpolated occlusion interval and ablate the tracked-point
loss on a nonsymmetric object where rotation is observable.

### Gate C — task transfer

Retarget the measured HOI4D ball-and-bowl geometry and trajectory into robosuite and
verify grasp, transfer, release, and collision constraints.

### Gate D — policy evidence

Evaluate the selected waypoint policy on the retargeted task across held-out object,
camera, background, and lighting shifts. Report failures and confidence intervals.

### Gate E — physical validation

Run a small real-robot evaluation after safety checks, calibration, and a successful
simulation transfer. Compare the learned policy with the scripted controller.
