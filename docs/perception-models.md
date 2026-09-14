# Perception model setup

SAM2 and CoTracker3 are optional GPU dependencies. The core package and tests do
not import either project, and model weights must remain outside Git.

## Pinned upstream revisions

| Adapter | Repository | Revision | Model/checkpoint |
| --- | --- | --- | --- |
| SAM2 | `facebookresearch/sam2` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `facebook/sam2.1-hiera-small` |
| CoTracker3 | `facebookresearch/co-tracker` | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` | `scaled_offline.pth` |

Install a CUDA-compatible PyTorch build first. Then install the pinned source
revisions in the same environment:

```bash
python -m pip install \
  "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
python -m pip install \
  "git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
```

SAM2's Hugging Face constructor downloads its selected checkpoint on first use.
For CoTracker3, download the official `scaled_offline.pth` checkpoint to:

```text
weights/cotracker3/scaled_offline.pth
```

The `weights/` directory is Git-ignored. Do not commit downloaded checkpoints.

## Adapter flow

1. Supply positive and negative `(x, y)` prompts for each object on a selected
   frame using `Sam2PointPrompt`.
2. `Sam2VideoAdapter` propagates one binary mask per object over all video frames.
3. `run_perception` samples unique, deterministic points inside each prompted
   mask using the configured seed.
4. `CoTracker3Adapter` converts those points to upstream `(t, x, y)` queries and
   returns tracks shaped `[T, N, 2]` with visibility shaped `[T, N]`.
5. `save_perception_artifacts` writes lossless PNG masks and compressed track files
   under the repository's established artifact layout.

The adapters validate all frame, mask, query, and output dimensions before saving
results. They deliberately do not download weights or silently choose a device.

The imported UniHand fixture has reviewed frame-zero prompts in
`configs/place_prompts.json`. After installing the pinned dependencies and placing
the CoTracker checkpoint, run:

```bash
uv run ov2p run-perception data/interim/place_demo/manifest.json \
  --prompts configs/place_prompts.json \
  --cotracker-checkpoint weights/cotracker3/scaled_offline.pth \
  --output data/interim/place_demo \
  --device cuda --points 32 --seed 42 --border 4
```

The command intentionally fails early if the optional model packages, checkpoint,
input frames, prompts, or upstream output dimensions are invalid.
