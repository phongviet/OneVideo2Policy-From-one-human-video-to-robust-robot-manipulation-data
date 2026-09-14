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
