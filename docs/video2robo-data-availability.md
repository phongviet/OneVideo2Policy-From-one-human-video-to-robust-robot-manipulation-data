# Video2Robo public data availability

Checked on 2026-09-21 against the official project page, CVPR paper and
supplement, GitHub organization, GitHub Pages source repository, and public
Hugging Face search.

## Availability result

The image-action training dataset is not publicly released. The official
project page labels the code as "coming soon." The only repository in the
`video2robo` GitHub organization is the project website. It contains rendered
website previews, not the policy-training records described in the paper.

Missing fields required for training include:

- absolute robot joint targets;
- proprioception and end-effector trajectories;
- synchronized frame/action timestamps;
- episode metadata, success labels, and train/validation splits;
- the complete generated demonstrations;
- policy configuration and checkpoints.

The CVPR supplementary artifact is a PDF. It provides additional figures and
method details but no dataset archive. No official Video2Robo dataset was found
on Hugging Face or another linked storage service.

## Public preview recovered

The project website does publicly contain:

- six source human demonstration videos: Attach, Drum, Place, Pour, Stack, and
  Sweep;
- five rendered demonstration previews per task;
- front and side views for each preview.

This totals 66 MP4 files: six input videos and 60 rendered previews. All files
decode successfully, run at 30 FPS, and total about 9.2 MB. They are pinned to
website revision `088d3d66690506908d6386345e08845dc1203615`; the local manifest
stores the source URL, size, and SHA-256 hash of each file.

The source repository specifies no license, so these files should remain a
local research reference and should not be redistributed. They do not contain
action labels and cannot train a Video2Robo policy by themselves.

Fetch or verify them with:

```bash
.venv/bin/python scripts/fetch_video2robo_public_samples.py
```

Local output:

```text
data/raw/video2robo_public_samples/
├── manifest.json
├── attach/
├── drum/
├── place/
├── pour/
├── stack/
└── sweep/
```

## What would unlock exact training

An official release must supply the complete image-action episodes and the code
that establishes observation/action horizons and Diffusion Policy training
parameters. Until then, the practical path is to reproduce the data schema from
the paper using locally generated trajectories rather than treating the website
videos as training data.
