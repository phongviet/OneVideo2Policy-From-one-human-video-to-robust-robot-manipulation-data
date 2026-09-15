# Next-step execution plan

## Immediate objective

Produce a measured, inspectable perception artifact from one real Place demonstration:

```text
place_demo.mp4
  → sampled RGB frames
  → source/target masks
  → temporally consistent point tracks
  → segmentation and tracking report
```

This is the smallest end-to-end slice that reduces project risk. Do not begin
TRELLIS, VGGT, robot synthesis, or policy training until this slice is repeatable.

## Definition of done

The next milestone is complete when one command can process the frozen demonstration
and produce:

- a versioned run configuration;
- a frame manifest with timestamps;
- source and target masks for every sampled frame;
- tracked source and target points with per-frame visibility;
- an overlay video for visual inspection;
- metrics on 30 manually annotated frames;
- a machine-readable gate report.

The milestone passes when:

| Metric | Required value |
|---|---:|
| Mean mask IoU | ≥ 0.70 |
| Point-track survival | ≥ 0.80 |
| Visible points per object/frame | ≥ 12 |
| Object identity swaps | 0 |

These thresholds are already frozen in [`../configs/place.yaml`](../configs/place.yaml).

## Work package 1 — freeze the input

**Target: half a day**

**Status: complete for the temporary UniHand fixture.** Its RGB-D manifest passes
structural, temporal, file-presence, and depth-shape validation.

1. The active input is the imported UniHand RGB-D sequence, whose 35 native frames
   span about 1.17 seconds at 30 FPS. Treat it as a fixed-camera RGB-D baseline.
2. For a later moving-camera reconstruction run, record one 10–20 second, 30 FPS,
   1080p landscape video.
3. Keep both objects visible except for brief hand occlusion during grasp.
4. Use diffuse lighting, limited motion blur, and a mostly static camera.
5. Perform one clean sequence: approach → grasp → transfer → place → release.
6. Save the selected take as `data/raw/place_demo.mp4`.
7. Record object dimensions, approximate camera position, and capture notes in the
   experiment log.

Follow the detailed [`recording and sourcing guide`](recording-guide.md) when setting
up the scene or evaluating an external clip.

Run the existing intake step:

```bash
uv sync --extra video
uv run ov2p prepare-video data/raw/place_demo.mp4 \
  --output data/interim/place_demo \
  --fps 30 \
  --depth-dir data/raw/sources/unihand_2025-0723-07-35-56/depth
uv run ov2p validate-manifest data/interim/place_demo/manifest.json
```

**Acceptance check:** `manifest.json` exists, timestamps are monotonic, the full
action is present, and sampled frames have no long blur or exposure failures.

## Work package 2 — make perception adapters runnable

**Target: 1–2 days**

Implement concrete adapters behind the existing `Segmenter` and `PointTracker`
protocols:

1. Add a SAM2 adapter with frame-zero positive/negative point prompts.
2. Store masks as lossless single-channel PNGs.
3. Seed CoTracker3 points inside eroded source and target masks.
4. Store tracks in compressed NumPy files with `xy` and `visible` arrays.
5. Pin model revisions and installation instructions; do not vendor model weights.

The deterministic mask-point sampler and model adapters required by steps 1–3 are
implemented, covered by model-free contract tests, and verified in a real CUDA run.
The temporary UniHand fixture now has masks, tracks, an inspection overlay, and a
provisional gate report. The next task is independent manual annotation so that
mask IoU can be measured without using the predictions as ground truth.

Use a stable artifact contract:

```text
data/interim/place_demo/
├── manifest.json
├── frames/000000.jpg
├── masks/source/000000.png
├── masks/target/000000.png
├── tracks/source.npz          # xy [T,N,2], visible [T,N]
├── tracks/target.npz
└── overlays/perception.mp4
```

**Acceptance check:** re-running with the same seed and configuration preserves
artifact shapes and object identities.

## Work package 3 — add annotations and gate reporting

**Target: 1 day**

1. Select 30 evaluation frames uniformly, adding frames around grasp occlusion.
2. Manually annotate source and target masks without using the predicted masks as the
   annotation starting point.
3. Compute per-object mask IoU, track survival, and visible point counts.
4. Generate a JSON report plus a small Markdown summary and failure montage.
5. Add tests using a tiny synthetic fixture rather than committing the real video.

The frame selection, blank annotation workspace, completeness checks, IoU evaluator,
and gate logic are now implemented. Follow the
[independent annotation guide](perception-annotation.md) to create the remaining 60
human-drawn masks without exposing the annotator to SAM2 predictions.

As an alternative evaluation fixture, the project can import HOI4D RGB-D sequences
and retain their official motion masks as ground truth. Use this only after checking
that both source and destination are separately labeled; see the
[HOI4D integration guide](hoi4d-integration.md).

The retained HOI4D ball-to-bowl fixture has passed this intake check. Its config is
[`../configs/hoi4d_ball_to_bowl.yaml`](../configs/hoi4d_ball_to_bowl.yaml), and the
selection rationale and limitations are recorded in the
[data audit](data-audits/hoi4d-ball-to-bowl.md).

Its first formal perception run passed both mask-IoU thresholds but failed target
track survival (0.758 vs 0.80). A controlled 16-frame mask-guided reseeding run on
the exact frozen SAM2 masks raised target survival to 0.951 while source survival
remained above threshold at 0.914. The formal perception gate now **passes** and
reconstruction preparation is unblocked.

Suggested output:

```text
results/perception/place_demo/
├── metrics.json
├── report.md
└── failure_montage.png
```

**Decision:** pass the milestone only if every threshold is met and the overlay shows
no identity swap. Otherwise log the failure category and change one variable at a
time—prompt, point sampling, mask propagation, or video choice.

## Work package 4 — begin reconstruction only after perception passes

**Target: 2–3 days after the gate**

1. Export clean first-frame RGBA crops for both objects.
2. Integrate TRELLIS behind `ObjectReconstructor`.
3. Integrate VGGT behind `DepthEstimator`.
4. Align object scale and translation to estimated scene geometry.
5. Render a fixed novel-view turntable for each object.
6. Record recognizable geometry, missing surfaces, and scale error in the Gate A
   report.

The output of this package should be useful object representations and an honest
reconstruction audit—not photogrammetry-grade assets.

**Status: split into two paths.** The local measured-primitive baseline now executes
through trajectory recovery, synthetic generation, policy fitting, and held-out
rollout evaluation. Native crops and a hashed 12-keyframe bundle prepare the faithful
TRELLIS/VGGT path for a 16 GB+ CUDA host. The retained fixture still yields only a
99×99 ball crop, so faithful Gate A awaits the planned close real recording. See the
[two-path execution guide](two-path-execution.md) and
[reconstruction readiness audit](reconstruction-readiness.md).

## Recommended first three commits

Keep changes reviewable and experiments attributable:

1. `chore: initialize OneVideo2Policy research scaffold`
2. `feat: add reproducible video intake and dataset contract`
3. `feat: add SAM2 and CoTracker perception pipeline`

Do not combine model integration, reconstruction, and policy training in one change.

## First action

`data/raw/place_demo.mp4` is prepared from UniHand `2025-0723-07-35-56`. The
highest-value task is now annotating the 30 evaluation frames and running the
perception gate. Use the supplied depth frames for the fixed-camera baseline rather
than attempting camera-motion reconstruction. See the
[data audit](data-audits/unihand-2025-0723-07-35-56.md).

The reviewed `subject_2-20231022_201316` capture was rejected and removed; it must not
be used as `place_demo`. See its [`data audit`](data-audits/subject_2-20231022_201316.md).
