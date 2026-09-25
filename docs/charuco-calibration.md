# Automatic physical camera calibration

The physical workflow uses a 7×5 ChArUco board with 30 mm squares and 22 mm markers
from OpenCV dictionary `DICT_5X5_100`.

- [Print-ready PDF](../assets/calibration/charuco-7x5-30mm.pdf)
- [PNG reference](../assets/calibration/charuco-7x5-30mm.png)
- [Board manifest](../assets/calibration/charuco-manifest.json)

Print the PDF at **100% / actual size**. Disable fit-to-page scaling. Measure several
squares with callipers and reject the print if the mean square length differs from
30.0 mm by more than the measurement tolerance used for the task-frame calibration.
Mount the paper to a rigid, flat backing.

## Capture protocol

For each physical camera, collect at least seven native-resolution images spanning
the image center, edges, different depths, and moderate board tilts. Keep six or more
views in the fit split and reserve at least one distinct view for validation. Each
image must expose at least six ChArUco inner corners.

For every image, record the board-to-robot-base transform. A rigid board attached to a
robot tool can obtain this from the measured tool pose and a fixed tool-to-board
transform. A stationary board can use a separately measured fixture transform. Do not
estimate this transform from the same image pixels, because that would remove the
independent robot-frame reference.

Fill a copy of `configs/physical_charuco_capture_template.json`. Image paths are
resolved relative to the capture JSON file.

## Extract observations

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_charuco_calibration.py \
  --captures path/to/physical-charuco-captures.json \
  --output path/to/physical-calibration-observations.json
```

The extractor detects corner IDs, estimates native camera intrinsics and distortion
from all views, scales intrinsics and pixels to the frozen 84×84 policy stream, and
transforms board corners into robot-base coordinates. It preserves fit and validation
splits for extrinsic evaluation.

Then estimate the task frame and camera extrinsics:

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_physical_cameras.py \
  --observations path/to/physical-calibration-observations.json \
  --output path/to/measured-physical-calibration.json
```

The retained synthetic end-to-end check detects 7 views per camera, estimates native
intrinsics at 0.36–0.38 px RMS, and produces held-out extrinsic errors of 0.10–0.23 px.
Those numbers validate the software path and are not physical calibration evidence.
