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

## Remaining gates

### Gate A — reconstruction quality

Capture or select larger, multi-view object crops and compare reconstructed proportions
against measured geometry. Current TripoSR HOI4D meshes are too flat for physics.

### Gate B — 6D motion

Fit camera and object poses from calibrated RGB-D, quantify reprojection error and
translation jitter, and ablate the tracked-point loss.

### Gate C — task transfer

Retarget the measured HOI4D ball-and-bowl geometry and trajectory into robosuite and
verify grasp, transfer, release, and collision constraints.

### Gate D — policy evidence

Evaluate the selected waypoint policy on the retargeted task across held-out object,
camera, background, and lighting shifts. Report failures and confidence intervals.

### Gate E — physical validation

Run a small real-robot evaluation after safety checks, calibration, and a successful
simulation transfer. Compare the learned policy with the scripted controller.
