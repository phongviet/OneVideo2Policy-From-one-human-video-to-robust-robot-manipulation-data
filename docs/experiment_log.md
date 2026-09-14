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
