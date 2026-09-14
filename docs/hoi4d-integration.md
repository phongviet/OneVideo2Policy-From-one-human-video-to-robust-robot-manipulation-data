# HOI4D integration

HOI4D is supported as an RGB-D source with official 2D motion-segmentation masks.
Imported masks live in `ground_truth/`, never in `masks/`, so they cannot be
confused with SAM2 predictions or leak into the independent gate.

## Chosen candidate family

Use a released **Mug `T1`** sequence. In HOI4D's official task definitions, this is
“Pick and place.” A concrete released candidate is:

```text
ZY20210800004/H4/C2/N49/S16/s03/T1
```

For `C2` (Mug), the official motion-label map assigns 1 and 3 to mug instances and
2 to the right hand. Do not assume which mug is source or target: inspect the first
and final frames after download, then select the correct IDs for
`--source-label` and `--target-label`. If the destination is an unlabelled table,
reject the sequence for the two-object Place benchmark and choose another candidate.

## Download and decode

Download RGB video, depth video, and annotations from the official
[HOI4D project page](https://hoi4d.github.io/), preserving the sequence layout. Run
the upstream `utils/decode.py` so the sequence contains:

```text
align_rgb/image.mp4
align_rgb/00000.jpg ...
align_depth/00000.png ...
2Dseg/mask/00000.png ...
```

Store it under the Git-ignored dataset path, for example
`data/raw/sources/hoi4d/<sequence-id>/`. Then inspect the color-coded masks and run:

```bash
uv run ov2p import-hoi4d \
  data/raw/sources/hoi4d/ZY20210800004/H4/C2/N49/S16/s03/T1 \
  --output data/interim/hoi4d_mug_t1 \
  --source-label 1 \
  --target-label 3 \
  --fps 15
uv run ov2p validate-manifest data/interim/hoi4d_mug_t1/manifest.json
```

The masks are official motion labels, not SAM2 results. They are suitable as
ground truth for an evaluation fixture only after confirming the two selected labels
refer to distinct physical source and target objects throughout the selected clip.
