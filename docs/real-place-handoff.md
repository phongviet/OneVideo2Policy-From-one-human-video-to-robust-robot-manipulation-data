# Real Place local handoff

## State on 2026-09-20

The `IMG_6256.MOV` intake, SAM2/CoTracker3 perception, native object crops,
camera-compensated local systems run, and 12-keyframe faithful input bundle are
complete. The 57-frame manifest validates. The local run reproduced byte for byte
in a second output directory, including its trajectory, generated demonstrations,
ridge policy, rollout, and report. It produced 250 proxy episodes and 100/100
held-out proxy successes. These checks show reproducible local execution; they do
not measure robot success or 3D reconstruction quality.

`results/handoff/real_place_img_6256/` is a portable snapshot containing the
original MOV, all sampled frames, masks, point tracks, native crops, model
keyframes, current source code and configs, and SHA-256 checksums. The packaged
manifest uses a relative path to the original video. Prepare and verify it with:

```bash
.venv/bin/python scripts/prepare_real_place_handoff.py \
  results/handoff/real_place_img_6256
.venv/bin/python scripts/prepare_real_place_handoff.py \
  results/handoff/real_place_img_6256 --verify
```

The handoff is a source and evidence package. `faithful/run-spec.json` names the
planned TRELLIS, VGGT, robosuite, and Diffusion Policy outputs; those adapters and
the faithful pipeline have not yet been implemented or run. A larger GPU alone will
not produce a research result without that work.

## Inputs still needed

1. **Independent masks.** A fresh prediction-free workspace is at
   `results/perception/real_place_img_6256_blind/`. An annotator who has not seen
   the SAM2 results must draw 60 visible-object binary PNG masks from its 30 raw
   images. The source is the round container, excluding the hand; the target is the
   blue tray/case, excluding the source and hand. Use white for visible object
   pixels and black elsewhere. Existing assisted masks are useful for engineering
   review but cannot be reused for this formal gate. Once complete, run:

   ```bash
   .venv/bin/ov2p evaluate-perception-gate \
     --workspace results/perception/real_place_img_6256_blind \
     --predictions data/interim/real_place_img_6256_perception \
     --diagnostics data/interim/real_place_img_6256_perception/gate-report.json \
     --config configs/real_place_img_6256_local_smoke.yaml \
     --identity-swaps 0 \
     --output results/perception/real_place_img_6256_blind/metrics.json
   ```

   **2026-09-21 AI annotation update:** A separate agent without prior task
   history annotated all 60 masks from raw JPGs only. Its
   [`annotation-provenance-ai.json`](../results/perception/real_place_img_6256_blind/annotation-provenance-ai.json)
   records the frozen mask hashes and states that it did not open predicted or
   assisted masks. On these prediction-blind **AI** masks, SAM2.1 Small passes
   the configured numeric gate with source/target mean IoU 0.800/0.968;
   SAM2.1 Tiny fails source IoU at 0.465. The separate
   [`metrics-ai.json`](../results/perception/real_place_img_6256_blind/metrics-ai.json)
   and [`metrics-ai-tiny.json`](../results/perception/real_place_img_6256_blind/metrics-ai-tiny.json)
   preserve that provenance. The workspace's original `metrics.json` remains
   incomplete. These are not human-drawn masks, so the formal human ground-truth
   gate remains pending. The prior assisted agreement, 0.749 source and 0.934
   target mean IoU, is not a valid independent gate result.

2. **Physical measurements.** Record the source diameter and height, tray length,
   width, and height, and initial source-to-target center separation in centimeters.
   Record camera intrinsics or phone model if available. The values in
   `configs/real_place_img_6256_local_smoke.yaml` are placeholders, so its metric
   trajectory and primitive sizes must not be interpreted as measured quantities.

   The completed local audit at
   `results/geometry/real_place_img_6256/audit/report.json` records why this is a
   hard evidence boundary. The MOV reports an iPhone 11 Pro Max but no usable
   lens intrinsics; background alignment leaves up to 117.9 pixels of center
   shift, and the nominally static target drifts up to 42.7 pixels. Full-clip
   monocular depth varies too much to substitute for the missing scale.

3. **Stronger capture for the reconstruction claim.** This take has camera drift,
   little useful parallax, and a roughly 190×150-pixel source footprint in the
   native first frame. It is adequate for a systems smoke test, but may fail the
   recognizable-geometry gate. A steady tripod take with larger objects in frame
   is recommended before spending significant GPU time.

## Transfer check

Transfer the entire `results/handoff/real_place_img_6256/` directory or its ZIP.
On the receiving host, install the code from `code/`, then run the packaged verifier
before installing model stacks or running reconstruction. Preserve the
`handoff-manifest.json` and `faithful/run-spec.json` with all returned outputs.
