# Reconstruction readiness audit

## Inputs produced

The passing HOI4D perception run can export transparent crops from the native
1920×1080 source video while reusing the frozen 640×360 SAM2 masks. Masks are
scaled with nearest-neighbor interpolation; RGB detail comes from original frame 86.

```bash
uv run ov2p export-rgba-crops \
  data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --masks data/interim/hoi4d_ball_to_bowl_gate/masks \
  --source-video data/raw/sources/hoi4d/HOI4D_release/ZY20210800001/H1/C7/N14/S280/s04/T5/align_rgb/image.mp4 \
  --output data/interim/hoi4d_ball_to_bowl_gate_reseed16/crops_native \
  --frame 0 --padding 0.15
```

| Object | Native RGBA crop | Foreground pixels | Readiness |
|---|---:|---:|---|
| Ball | 99×99 | 3,402 | Too small/blurry for a credible geometry gate |
| Bowl | 196×196 | 13,104 | Usable for an adapter smoke test, not a strong benchmark |

The crop manifest preserves source bounding boxes, canvas offsets, foreground
counts, and the original source-frame ID.

## Blocker

The current host has an NVIDIA GeForce GTX 1660 Ti with 6 GB VRAM. The official
[TRELLIS repository](https://github.com/microsoft/TRELLIS) specifies an NVIDIA GPU
with at least 16 GB for its only image-conditioned model, TRELLIS-image-large.
Installing its compiled CUDA stack and multi-gigabyte weights cannot produce a
valid local run on this host, so the integration stops before that environment
change.

VGGT integration alone would not satisfy Gate A because the planned gate requires
recognizable object reconstructions as well as scene depth/scale. It is deferred
with TRELLIS to keep the milestone atomic.

## Unblocking conditions

Provide both:

1. a CUDA host with at least 16 GB VRAM (24 GB preferred for useful margin); and
2. the planned real Place recording where each rigid object occupies substantially
   more pixels and exposes useful nearby views.

After that, pin the TRELLIS and VGGT revisions in an isolated reconstruction
environment, run one-object smoke tests, render fixed turntables, and evaluate
recognizability, missing surfaces, and scale before pose optimization.
