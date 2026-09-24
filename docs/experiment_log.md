# Experiment log

## Run template

- Date / commit:
- Hypothesis:
- Configuration:
- Input video and object pair:
- Changed variable:
- Fixed variables:
- Metrics:
- Result:
- Gate decision:
- Artifacts:
- Next action:

## HOI4D ball-to-bowl perception gate

- Date / commit: 2026-09-15 / `07deb44`
- Hypothesis: Multi-point SAM2 prompts and mask-eroded CoTracker queries can pass
  the frozen perception thresholds on an egocentric Place sequence.
- Configuration: `configs/hoi4d_ball_to_bowl.yaml`
- Input video and object pair: HOI4D
  `ZY20210800001/H1/C7/N14/S280/s04/T5`, ball → bowl.
- Changed variable: corrected prompt set, then corrected four-pixel mask erosion for
  deterministic track-query sampling.
- Fixed variables: SAM2/CoTracker revisions, 72 frames at 5 FPS and 640×360,
  32 queries per object, seed 42, official HOI4D ground truth.
- Metrics: ball IoU 0.777, bowl IoU 0.920; ball track survival 0.973, bowl track
  survival 0.758; 31.1/24.3 mean visible points; zero visual identity swaps.
- Result: Both mask gates pass. Ball tracking passes. Bowl tracking misses the 0.80
  survival threshold by 0.042.
- Gate decision: **Fail.** Do not begin reconstruction from this fixture yet.
- Artifacts: `data/interim/hoi4d_ball_to_bowl_gate/` and
  `results/perception/hoi4d_ball_to_bowl/metrics.json` (Git-ignored).
- Next action: add windowed or periodic mask-guided CoTracker reseeding for long
  egocentric sequences, holding segmentation and thresholds fixed.

## HOI4D mask-guided CoTracker reseeding

- Date / commit: 2026-09-15 / pending
- Hypothesis: Refreshing deterministic queries from frozen SAM2 masks every 16
  frames will recover bowl visibility without changing segmentation or thresholds.
- Configuration: `configs/hoi4d_ball_to_bowl.yaml`
- Input video and object pair: the same 72-frame HOI4D ball → bowl fixture.
- Changed variable: CoTracker queries are refreshed in 16-frame windows. When a
  boundary mask is occluded, the first sampleable mask in that window is used and
  CoTracker runs backward to cover the preceding frames.
- Fixed variables: exact SAM2 mask PNGs, model revision and checkpoint, 32 queries,
  four-pixel erosion, seed 42, source frames, ground truth, and gate thresholds.
- Metrics: ball IoU 0.777, bowl IoU 0.920; ball track survival 0.914, bowl track
  survival 0.951; 29.3/30.4 mean visible points; zero visual identity swaps.
- Result: Every segmentation and tracking threshold passes.
- Gate decision: **Pass.** Reconstruction preparation may begin.
- Artifacts: `data/interim/hoi4d_ball_to_bowl_gate_reseed16/` and
  `results/perception/hoi4d_ball_to_bowl_reseed16/metrics.json` (Git-ignored).
- Next action: export clean RGBA object crops, then audit whether this monocular
  egocentric fixture is adequate for TRELLIS and VGGT integration.

## HOI4D reconstruction readiness

- Date / commit: 2026-09-15 / pending
- Hypothesis: Native-resolution RGB can make the retained fixture sufficient for
  the first image-to-3D smoke test without changing the passing perception masks.
- Configuration: `configs/hoi4d_ball_to_bowl.yaml`
- Changed variable: RGB crops are decoded from original 1920×1080 frame 86 while
  the frozen masks are scaled by nearest-neighbor interpolation.
- Fixed variables: object identities, selected frame, SAM2 masks, and 15% padding.
- Metrics: transparent ball crop 99×99 with 3,402 foreground pixels; bowl crop
  196×196 with 13,104 foreground pixels. Host GPU has 6 GB VRAM; official TRELLIS
  minimum is 16 GB.
- Result: deterministic crop export passes, but the ball lacks adequate image
  detail and TRELLIS cannot run on this host.
- Gate decision: **Blocked.** Gate A was not attempted.
- Artifacts: `data/interim/hoi4d_ball_to_bowl_gate_reseed16/crops_native/`
  (Git-ignored) and `docs/reconstruction-readiness.md`.
- Next action: use a 16 GB+ CUDA host and record the planned close real Place demo.

## Metric-calibrated local end-to-end systems baseline

- Date / commit: 2026-09-24 / pending
- Hypothesis: Artifact-compatible local substitutes can exercise every downstream
  stage without weakening the claim boundary for the faithful path.
- Configuration: `configs/two_paths.yaml`
- Input video and object pair: passing HOI4D ball → bowl perception artifacts.
- Changed variable: downloaded aligned depth and camera calibration replaced the
  provisional 6 cm ball, 18 cm bowl, and 0.25 m scale anchor with a fitted 3.832 cm
  ball, annotated 9.915 cm bowl, and 0.580 m first-measurable carry separation.
- Fixed variables: frozen masks, task success tolerance, seed 42, and Place semantics.
- Metrics: 72-frame proxy trajectory; 250 randomized demonstrations; 12,000 valid
  action samples; 81/100 held-out proxy rollouts successful; 0.81 success versus
  the frozen 0.80 gate; runtime about 1 second.
- Result: local generation and learning gates pass.
- Gate decision: **Pass for the systems baseline only.** This is not faithful Gate A
  or evidence of image-policy or sim-to-real robustness.
- Artifacts: `results/local_e2e/hoi4d_ball_to_bowl/` and
  `results/faithful_bundle/hoi4d_ball_to_bowl/` (Git-ignored).
- Next action: retain this run as the calibrated dataset diagnostic, obtain a true
  initial separation from a fully visible fixed-camera recording, then execute the
  faithful bundle on a 16–24 GB GPU.

## Local model replacement run

- Date / commit: 2026-09-24 / pending
- Decision: The earlier TRELLIS/VGGT external-compute plan is superseded by the
  measured local stack in `configs/two_paths.yaml`.
- Object model: TripoSR, resolution 128, chunk 4096; peak 1,869 MiB. HOI4D ball and
  bowl meshes are watertight but flattened, with minimum-to-maximum extent ratios
  0.065 and 0.085. They are visual proposals only.
- Depth model: Depth Anything V2 Metric Hypersim Small, input 518; 415 MiB peak and
  0.159 s median inference after warmup. Five-frame HOI4D sensor comparison gives
  raw mean AbsRel 0.715 and scale-aligned mean AbsRel 0.155 with median scale 0.526.
- Geometry decision: use aligned HOI4D RGB-D and measured primitives for metric scale
  and collision geometry.
- Policy decision: retain the temporal dual-view waypoint model plus phase controller,
  which passed 19/20 prior held-out robosuite rollouts.
- Artifacts: `results/model_benchmarks/hoi4d_ball_to_bowl/` (Git-ignored).

## Metric RGB-D Gaussian scene initialization

- Date / commit: 2026-09-24 / pending
- Input: HOI4D source frame 89, aligned uint16 millimetre depth, camera intrinsics,
  and passing ball/bowl masks. Frame 86 was rejected because ball pixels had no valid
  sensor depth.
- Representation: 18,420 explicit metric 3D Gaussians with anisotropic scale,
  quaternion rotation, opacity, RGB color, and static/source/target labels.
- Renderer: projected 3D covariance and front-to-back alpha compositing at 640×360.
- Metrics: 73.0% alpha coverage over valid sensor depth and 24.18 dB same-view PSNR.
- Contract checks: a 3 cm camera translation renders a novel view; a 5 cm rigid
  source transform moves only the labeled ball Gaussians.
- Result: Gaussian representation and rendering stage pass their local artifact
  contract. Multiframe fitting, transient removal, and robot compositing remain.
- Artifacts: `results/gaussian_scene/hoi4d_ball_to_bowl_frame89/` (Git-ignored).
