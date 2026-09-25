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

## Metric RGB-D odometry and Gaussian fusion

- Date / commit: 2026-09-24 / pending
- Camera method: consecutive ORB matching and RGB-D PnP with source/target masks
  excluded, followed by metric transform accumulation across all 72 frames.
- Camera metrics: 0.649 m path length, 0.202 m start-to-end displacement, 91.5%
  median inlier ratio, 0.75 px median reprojection error, and 2.1 mm median depth
  residual at matched pixels.
- Gaussian fusion: frames 1, 12, 24, 36, 48, 60, and 71; 1 cm voxels; at least two
  distinct-frame observations per static voxel; movable groups retained from frame 1.
- Fused result: 7,007 Gaussians, 67.0% evaluated held-out coverage, and 20.68 dB PSNR
  on unseen frame 30 outside source and target masks.
- Gate decision: metric camera trajectory and multiview Gaussian artifact pass. Object
  6D motion, transient hand removal, Gaussian optimization, and robot composition remain.
- Artifacts: `results/geometry/hoi4d_ball_to_bowl/rgbd_odometry/` and
  `results/gaussian_scene/hoi4d_ball_to_bowl_fused/` (Git-ignored).

## Metric ball trajectory and Gaussian animation

- Date / commit: 2026-09-24 / pending
- Method: robust known-radius sphere fitting on masked aligned depth; fits above 6 mm
  median surface error rejected; translations expressed in both camera-odometry world
  coordinates and target-relative coordinates.
- Coverage: 51 accepted and 21 interpolated frames. The longest interpolation interval
  is 20 frames. Ball rotation is identity because spherical orientation is unobservable.
- Task geometry: initial observed bowl-center separation 0.582 m; final separation
  0.019 m versus a 0.0496 m bowl radius, so the observed Place check passes.
- Gaussian animation: 72 frames at 320×180 with 69.9% mean scene coverage. Across the
  51 depth-observed frames, source centroid error is 0.28 px median and 0.41 px p90.
- Gate decision: metric translation and Gaussian object animation pass. Robot
  retargeting, robot-scene composition, and the long-occlusion sensitivity ablation remain.
- Artifacts: `results/geometry/hoi4d_ball_to_bowl/object_trajectory/` and
  `results/gaussian_scene/hoi4d_ball_to_bowl_animated/` (Git-ignored).

## Measured robosuite ball-to-bowl policy

- Date / commit: 2026-09-24 / pending
- Hypothesis: measured primitive geometry plus a compact visual source locator can
  transfer the validated waypoint controller to the filmed ball-to-bowl task locally.
- Simulator: Panda `BallToBowl`, 1.916 cm ball radius, 4.956 cm bowl outer radius,
  5.693 cm bowl height, dual 84×84 cameras, OSC pose control.
- Demonstrations: 10/10 successes in ten attempts, 3,292 synchronized samples.
- Controller ceiling: exact ball positions produce 3/3 randomized successes.
- Initial learned result: 1/5 successes; the ten trajectories provide only ten
  distinct initial positions and include conflicting carried-ball targets.
- Correction: train the waypoint model only on approach phases and render 500
  independent randomized localization frames. Freeze the unobstructed initial
  estimate during approach.
- Localization: 5.7 mm held-out median XY error, 6.3 mm mean, 11.2 mm p90.
- Final result: 5/5 randomized learned-policy successes with 4.9 mm median initial
  XY error. Model size is 2,607,715 parameters; training took 7.5 seconds locally.
- Gate decision: **Pass for task-specific local simulation and learned perception.**
  Controlled visual shifts and physical robot transfer remain separate gates.
- Artifacts: `results/simulation/robosuite_ball_bowl_10/`,
  `results/simulation/ball_localization_500/`, `results/policy/ball_localization_500/`,
  and `results/evaluation/ball_localization_500_random5/` (Git-ignored).

## Gaussian appearance policy experiment

- Date / commit: 2026-09-25 / pending
- Hypothesis: the fused HOI4D Gaussian scene can supply useful background diversity
  to the task-specific simulator without requiring a large rendering model.
- Composition: preserve MuJoCo class-mask pixels for Panda, gripper, ball, and bowl;
  replace unlabeled pixels with the animated fused Gaussian scene. Camera alignment
  is appearance-only and unregistered.
- Single-domain result: clean-only training gives 11.4 cm median XY error on Gaussian
  frames; Gaussian-only training gives 4.6 cm on clean frames.
- Reset-pose mixed result: offline median error falls to 3.1 mm clean and 4.2 mm
  Gaussian, but Gaussian closed loop reaches only 3/5 because two 9.9–13.0 mm
  outliers miss the grasp.
- Correction: add 500 clean and 500 Gaussian samples using demonstration-derived
  Panda joint poses, merge them with 1,000 reset-pose samples, and shuffle the
  episode split. The 2,000-frame model trains in 28.7 seconds.
- Final offline result: median XY error is 3.1–4.0 mm and p90 is below 8.4 mm across
  clean/Gaussian and reset/varied-pose subsets.
- Final closed loop result: 5/5 clean at 5.0 mm median initial XY error and 5/5
  Gaussian at 2.0 mm, using the same checkpoint and paired seed. All succeed on the
  first grasp attempt.
- Data contract: one complete Gaussian-composite demonstration succeeds in one
  attempt and stores 350 synchronized dual-camera, state, phase, and action samples.
- Gate decision: **Pass for Gaussian appearance augmentation.** Metric camera
  registration, depth occlusion, and scene optimization remain open.
- Artifacts: `results/simulation/ball_localization_mixed_pose_2000/`,
  `results/policy/ball_localization_mixed_pose_2000/`,
  `results/evaluation/ball_localization_mixed_pose_{clean,gaussian}_random5/`, and
  `results/simulation/robosuite_ball_bowl_gaussian_demo1/` (Git-ignored).

## Controlled visual robustness matrix

- Date / commit: 2026-09-25 / pending
- Initial selected model: 2,000 clean/Gaussian, reset/varied-pose frames.
- Nominal result: 20/20, 3.2 mm median XY error.
- Initial camera result: 11/20 under independent ±2 cm translations, 9.4 mm median
  error. This isolates camera geometry as the main remaining simulated weakness.
- Camera correction: add 500 clean and 500 Gaussian varied-pose frames rendered with
  true ±2 cm camera translations. The 3,000-frame model improves the paired camera
  result to 17/20 and scores 17/20 at half light and 20/20 on Gaussian backgrounds.
- Initial combined result: 10/20 for Gaussian + camera + half light. The existing
  photometric augmentation only scales brightness to 0.7, while the test uses 0.5.
- Combined correction: add 500 clean and 500 Gaussian camera-jitter frames rendered
  at half light. The final 4,000-frame model trains in 55.4 seconds.
- Final shared-model matrix: nominal 20/20, camera 19/20, half light 20/20,
  Gaussian background 20/20, combined 18/20. Aggregate 97/100; 95% Wilson interval
  91.5–99.0%. Combined-only interval is 69.9–97.2%.
- Gate decision: **Pass local controlled robustness.** All five point estimates meet
  the frozen 80% gate. Metric Gaussian registration and physical robot validation
  remain separate gates.
- Artifacts: `results/simulation/ball_localization_robust_4000/`,
  `results/policy/ball_localization_robust_4000/`, and
  `results/evaluation/final_*_20/` plus `robustness_combined_augmented_20/`
  (Git-ignored).

## Matched combined-shift policy ablation

- Date / commit: 2026-09-25 / pending
- Fixed condition: seed 6001, 20 episodes, randomized object pose, ±2 cm camera
  translations, half lighting, and Gaussian background.
- One-demonstration waypoint model: 1/20 successes, 5.48 cm median initial XY error.
- Clean-only 500-frame 2D model: 0/20, 10.07 cm median error. Its nominal success
  does not transfer to the combined Gaussian/camera/light domain.
- Geometry-only exact-position controller ceiling: 20/20, zero localization error.
- Final 4,000-frame learned model: 18/20, 3.73 mm median error.
- Interpretation: broad rendered coverage is necessary; the remaining two failures
  are perception outliers rather than a physics/controller ceiling. The clean-only
  row scoring below one-demo is not a monotonic data-scaling claim because both are
  far outside their training appearance and the one-demo row has one chance success.
- Artifacts: `results/evaluation/ablation_*_combined20/` (Git-ignored).

## Asymmetric-object correspondence ablation

- Date / commit: 2026-09-25 / pending
- Source: CaryxAI Everyday Manipulation 3D `EM1-0406.r3d`, 392 Record3D RGB-D
  frames with intrinsics and camera poses.
- Object: rectangular action camera with an offset circular lens; unlike the HOI4D
  ball, its image-plane rotation is observable.
- Protocol: frame 0 is the reference; frames 330–390 at stride 5 form 13 fully
  visible post-placement evaluations. SIFT/RANSAC verifies matches, two thirds fit
  a similarity pose, and every third spatially sorted match is held out.
- Ablation: mask-only centroid/area/principal-axis fitting has 34.89 px median held-out
  error (95% bootstrap CI 34.25–36.11). Correspondence-constrained fitting has
  0.70 px median error (0.57–0.91), a 98.0% reduction over 82 held-out points.
- Motion: median recovered image-plane rotation is −41.7°.
- Gate decision: **Pass for correspondence-sensitive observable rotation.** The test
  does not claim full SE(3) accuracy.
- Artifacts: `docs/experiments/em1-0406-rotation-ablation.json` and
  `docs/assets/em1-0406-rotation-ablation.png`.

## Gaussian footprint tuning and policy selection

- Date / commit: 2026-09-25 / pending
- Scene selection protocol: sweep global Gaussian scale and opacity on held-out frame
  30 using fixed baseline-alpha support; reserve frames 42, 54, and 66 for testing.
- Selected geometry: 1.1× standard-deviation scale and opacity 1.0. Selection PSNR improves
  by 0.83 dB; independent test improvements are 0.35, 0.44, and 0.69 dB.
- Downstream test: the selected 4,000-frame policy scores 18/20 on the tuned render,
  compared with its existing 20/20 on the original Gaussian render.
- Adaptation test: add 500 tuned-background, varied-Panda-pose frames and retrain the
  same 2.61-million-parameter model on 4,500 frames. It scores 17/20 on the tuned
  render, so this checkpoint is rejected.
- Selection: retain the 4,000-frame checkpoint and original Gaussian render for the
  policy. Retain the tuned scene for higher-PSNR geometry rendering. View PSNR and
  closed-loop localization robustness select different configurations.
- Artifacts: `docs/experiments/hoi4d-gaussian-footprint-tuning.json`,
  `docs/experiments/gaussian-tuning-policy-results.json`, and
  `docs/assets/hoi4d-gaussian-footprint-tuning.png`.

## Metric reconstruction proportion gate

- Date / commit: 2026-09-25 / pending
- Source: `EM1-0406` frame 0, a 282×282 action-camera crop with 609 aligned Record3D
  depth samples. Robust 2nd–98th percentile PCA extents are 81.9 × 60.6 × 35.4 mm.
- Gate: align arbitrary model scale by the longest extent, then require both secondary
  extents to be within 15% of the metric RGB-D reference.
- TripoSR-128: 81.9 × 69.7 × 14.5 mm, 58.9% maximum secondary error, watertight,
  1,867 MiB peak allocation. **Fail.**
- Stable Fast 3D: 81.9 × 68.3 × 23.7 mm, 33.1% maximum secondary error,
  non-watertight, 6,169 MiB peak allocation. **Fail.**
- Selection: use aligned RGB-D point clouds or fitted primitives for metric and
  collision geometry. Keep TripoSR-128 as the fast local visual proposal. Stable Fast
  3D is proportionally closer but does not justify its memory or topology cost here.
- Artifacts: `docs/experiments/em1-0406-metric-reference.json`,
  `docs/experiments/em1-0406-reconstruction-proportions.json`, and
  `docs/assets/em1-0406-reconstruction-proportions.png`.

## Metric Gaussian robot registration

- Date / commit: 2026-09-25 / pending
- Anchors: robust newspaper table normal, HOI4D placed-bowl center, and the projected
  source-to-target direction. These define a metric rigid transform from robosuite
  to the fused Gaussian world.
- Registration audit: rotation determinant 1.0, table-normal error 0°, target-center
  3D/reprojection error at numerical precision, 2.7 mm median table-plane residual,
  and 6.3 px rendered bowl-centroid error.
- Depth audit: 32,349 simulator foreground pixels in the still; Gaussian depth hides
  258 pixels and retains 99.2%.
- Full data contract: replay one successful 353-sample trajectory through registered
  HOI4D camera frames 24 and 36. Both 84×84 image streams match all state/action
  samples without resampling. Across the trajectory, depth ordering hides 50,161
  simulator pixels and retains 99.3–99.4% of foreground.
- Gate decision: **Pass for local metric Gaussian robot data generation.** Sparse
  scene holes and missing contact shadows limit visual fidelity; physical validation
  remains separate.
- Artifacts: `docs/experiments/metric-gaussian-robot-registration.json`,
  `docs/experiments/metric-registered-gaussian-demo.json`,
  `docs/assets/metric-gaussian-robot-registration.png`, and
  `docs/assets/metric-registered-gaussian-demo.png`.

## Physical software preflight

- Date / commit: 2026-09-25 / pending
- Frozen checkpoint SHA-256:
  `c1b5310537d85734ecef0d427a1a9a13f49d69eb617842802a044ecdcd228125`.
- Inference smoke test: predicted the retained source position within 3.84 mm.
- Motion dry run: 149 Cartesian commands completed without a stop, using 1 cm
  maximum steps and 8 cm/s maximum speed over 18.69 seconds of command time. Gripper
  close and release commands occur at stationary source and target waypoints.
- Calibration fixture: both simulated cameras pass the numeric checks at 0.5 and
  0.6 px reprojection RMSE. The preflight correctly refuses hardware readiness
  because the fixture is simulation-only and has no operator, emergency-stop, or
  physical robot verification.
- Camera transfer contract: both measured camera transforms must stay within 2 cm and
  5° of the frozen robosuite training cameras, focal lengths within 5%, and principal
  points within 2 px. A synchronized frame watchdog enforces 250 ms age and 50 ms
  inter-camera skew limits during execution.
- Negative check: a valid synthetic camera calibration with different poses and focal
  lengths is rejected before packaging for translation, rotation, and focal mismatch.
- Coordinate contract: the policy's robosuite task frame is fitted to the physical
  robot-base frame from 3D correspondences and must validate within 5 mm. Learned
  source predictions are transformed into robot coordinates before planning.
- Object contract: measured physical dimensions must remain within 10% of the
  3.832 cm source diameter and 9.912 cm outer diameter by 5.693 cm target height.
  The separately supplied 3 cm source is rejected for this frozen-policy test.
- Fixture preparation: generated watertight binary STL files for a 38.321 mm sphere
  and a 99.123 × 56.933 mm bowl. Their audited extents differ from the frozen values
  by less than 0.003 mm due to float32 STL storage, with zero boundary or nonmanifold
  edges. Final printed parts still require calliper measurements.
- Calibration capture check: generated a dimensioned 7×5 ChArUco board and exercised
  the complete detector-to-extrinsics path on seven synthetic views per camera.
  Native intrinsic RMS is 0.36–0.38 px and held-out extrinsic RMSE is 0.10–0.23 px.
  This verifies automation only; physical captures remain required.
- Trial gate test: an empty physical template is rejected; a synthetic paired fixture
  at 4/5 scripted and 4/5 learned successes passes with zero safety aborts and force
  violations. The gate also requires five shared reset IDs, counterbalanced order,
  and deployment, calibration, and checkpoint hashes. This tests the validator and
  is not robot evidence.
- Gate decision: **Pass for local deployment software.** Gate E remains open until a
  real calibration and ten paired physical trials satisfy the frozen criteria.
- Artifacts: `docs/experiments/physical-software-preflight.json`,
  `docs/experiments/physical-gate-validator-fixture.json`, and
  `docs/physical-evaluation-runbook.md`.
