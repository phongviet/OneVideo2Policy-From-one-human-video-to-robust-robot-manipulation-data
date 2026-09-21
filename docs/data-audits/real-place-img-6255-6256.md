# Real Place captures: IMG_6255 and IMG_6256

**Decision:** Use `data/raw/IMG_6256.MOV` for the real-video perception and local
baseline. Retain `IMG_6255.MOV` as an unused take; do not replace the UniHand
`place_demo.mp4` fixture or delete either original.

Both takes are 1920×1080, approximately 29.97 FPS, with an oblique view
over a lightly textured, non-mirror tabletop. `IMG_6255.MOV` lasts 13.18 s but
the source leaves the frame during transfer, so it fails the capture rubric.
`IMG_6256.MOV` lasts 11.31 s and shows an initial hold, grasp, transfer, placement
of the round container onto the blue tray/case, release, and a final hold. Both
task objects remain in frame. Back-edge clutter does not obstruct the action.
The view is not perfectly fixed: ORB features in the static upper background
shift by a median ~48 px vertically between the first and last sampled frames.
This is tolerable for a perception smoke test but confounds an uncompensated
planar trajectory and makes the take weaker than a locked-tripod capture.

The source is roughly 200 px across in the native first frame, below the preferred
250–300 px reconstruction crop. This is a usable systems/perception take, **not**
yet evidence that paper-faithful single-view 3D reconstruction will pass Gate A.
It has no metric scene measurements or depth. The source and target dimensions,
initial separation, and camera calibration must be measured before interpreting
metric local-path outputs.

## Reproducible intake and provisional perception

```bash
uv run ov2p prepare-video data/raw/IMG_6256.MOV \
  --output data/interim/real_place_img_6256 --fps 5 --max-width 960
uv run ov2p validate-manifest data/interim/real_place_img_6256/manifest.json
uv run ov2p run-perception data/interim/real_place_img_6256/manifest.json \
  --prompts configs/real_place_img_6256_prompts.json \
  --output data/interim/real_place_img_6256_perception \
  --cotracker-checkpoint weights/cotracker3/scaled_offline.pth \
  --device cuda --points 32 --border 4 --reseed-interval 16
```

Intake produces 57 frames at 960×540 and passes manifest validation. The 6 GB
GTX 1660 Ti completed SAM2 plus CoTracker3. Provisional source/target track
survival is 0.986/0.921, with 31.5/29.5 mean visible points. Visual inspection
of the overlay found no identity swap, but source masks fluctuate under hand
occlusion. These are temporal diagnostics, **not** independent segmentation IoU.
The formal perception gate remains pending a blind annotation pass.

Native-resolution first-frame RGBA crops were also exported from the original
MOV. The source crop is 248×248 (190×150 visible-object bounding box) and the
target crop is 666×666 (512×466 bounding box). Neither proves that unseen
surfaces can be reconstructed; this mostly single oblique view provides little
useful parallax despite camera drift.

```bash
uv run ov2p prepare-perception-gate \
  data/interim/real_place_img_6256/manifest.json \
  --spec configs/real_place_img_6256_evaluation.json \
  --output results/perception/real_place_img_6256
```

The workspace was initially prepared from raw images only. Its masks were later
filled by an assisted drafting pass. A future blind annotator must use a fresh
workspace and not these masks or the SAM2 predictions for formal evaluation.

## Assisted working masks and downstream smoke test

`scripts/annotate_real_place_img_6256.py` now fills those 60 masks from the raw
frames using manually checked source centers, the colored lid and white body,
and the blue target color. All 30 overlays were visually reviewed, including
grasp and release. Because the annotator had already seen SAM2 output, these
are **working masks, not independent ground truth**. Their mean agreement with
SAM2 is 0.749 source / 0.934 target; the machine-readable
`assisted-agreement.json` explicitly marks `formal_gate_valid: false`.

The local 57-frame systems path also completes using the SAM2 masks and
`configs/real_place_img_6256_local_smoke.yaml`: 250 generated episodes and
100/100 held-out *proxy* rollouts. It now aligns static background features,
computes the source path relative to the imaged target, and interpolates eight
target-centroid frames with heavy occlusion. The largest correction versus raw
per-frame centroids is ~44 px. The source center starts outside the target and
is inside its footprint for all five final frames. Its primitive assets are a
round cylinder and shallow rectangular tray, matching object *types* but not
measured dimensions. All metric scene values remain assumptions, so this is a
wiring and pixel-space Place check, not a physical policy result.

```bash
uv run ov2p run-local-e2e \
  --config configs/real_place_img_6256_local_smoke.yaml \
  --manifest data/interim/real_place_img_6256/manifest.json \
  --masks data/interim/real_place_img_6256_perception/masks \
  --output results/local_e2e/real_place_img_6256_smoke
```

The run writes `relative_trajectory.png`, `camera_transforms.npy`, primitive
OBJ files, a proxy rollout, and `report.json`. A hashed 12-keyframe faithful
input bundle is staged at `results/faithful_bundle/real_place_img_6256/`, with
its status marked blocked by the independent gate and 16 GB+ GPU requirement.
