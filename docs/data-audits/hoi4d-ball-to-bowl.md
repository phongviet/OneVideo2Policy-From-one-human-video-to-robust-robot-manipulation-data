# HOI4D ball-to-bowl audit

## Decision

**Accepted as the metric dataset candidate and as an RGB-only perception and
mask-evaluation fixture.** It contains a real two-object placement action with
official masks for both the moving ball and the destination bowl. The retained
local subset is RGB-only; restoring this sequence's official aligned depth and
camera parameters upgrades it to a complete metric reference.

| Field | Value |
| --- | --- |
| Dataset | HOI4D |
| Sequence | `ZY20210800001/H1/C7/N14/S280/s04/T5` |
| Official task | C7 Bowl, T5: Put the ball in the bowl |
| Frames / rate | 300 frames at 15 FPS (19.93 s) |
| Source | Ball, official motion label 3 |
| Target | Bowl, official motion label 1 |
| Hand | Official motion label 2; excluded from both masks |
| Source-mask coverage | 187/300 frames |
| Target-mask coverage | 300/300 frames |
| Target dimensions | 9.9177 cm long × 9.9123 cm wide × 5.6933 cm high |
| Target pose coverage | 300/300 frames, metric camera coordinates |
| Depth | Not retained: the downloaded depth archive was an incomplete shard |

The GPU gate view uses original frames 86–299, sampled at 5 FPS and resized to
640×360. This yields 72 aligned frames over 14.2 seconds. Frame 86 has the largest
initial visible ball mask, so it supplies the positive source prompt without
starting from a clipped object at the image boundary.

The action annotation contains two complete placement attempts. Their `putdown`
segments are 7.116–9.340 s and 14.725–15.818 s.

## Metric field coverage

This sequence can supply every physical field missing from the real-phone
capture. Some values are direct annotations and the ball values are recovered
from the aligned metric depth:

| Required field | Sequence source | Status in retained subset |
| --- | --- | --- |
| Source ball diameter and height | Robust 3D extent of official ball mask back-projected through aligned depth | Requires official depth and camera parameters |
| Target length, width, and height | `objpose/*.json` `dimensions` | Available: 0.0991768 m × 0.0991230 m × 0.0569334 m |
| Initial source-target center separation | Distance between the masked ball 3D centroid and annotated bowl center in the first usable rest frame | Requires official depth and camera parameters |
| Camera calibration | Official Camera Parameters package; per-frame trajectory is stored as an Open3D `output.log` in the 3D scene package | Requires official camera/3D scene files |
| Metric target trajectory | `objpose/*.json` `center`, `rotation`, and `dimensions` | Available for all 300 frames |

The bowl's frame-0 center is `(-0.114057, 0.026107, 0.861622)` m in the
annotation camera frame. Its dimensions are constant across all 300 object-pose
files. The ball is a motion-segmentation object rather than the category object,
so HOI4D does not give it a direct `objpose` entry. Its metric center and size must
be computed from its official mask and aligned depth. This is still measured
RGB-D geometry; it is not a monocular scale estimate.

To complete the local metric bundle, fetch **Depth Video** and **Camera
Parameters** from the official HOI4D project page and retain only this sequence.
Fetch the corresponding 3D scene files as well if the per-frame world-camera
trajectory is needed. The existing RGB, action, masks, and object-pose annotations
do not need to be downloaded again.

## Storage decision

The retained HOI4D subset contains 61 RGB clips with their 2D masks, actions, and
object-pose metadata (about 722 MB). The imported ball-to-bowl fixture adds about
119 MB. The three 20–23 GB bulk archives were deleted after extraction, keeping the
combined retained footprint under 1 GB and below the 20 GB budget.

## Use and limits

The retained fixture is appropriate for SAM2/CoTracker evaluation using
independently provided mask ground truth. Until its depth and camera packages are
restored, it is not a metric geometry benchmark. The complete official sequence
is suitable for metric pipeline diagnosis, but remains egocentric and differs
from the fixed phone-camera deployment view. Keep the real Place demonstration
as the final policy target.

## Perception gate result

The initial 2026-09-15 CUDA run produced mean mask IoU of 0.777 for the ball and
0.920 for the bowl, but bowl survival was only 0.758. The controlled follow-up held
the exact masks and all thresholds fixed while refreshing deterministic CoTracker
queries every 16 frames. Ball and bowl survival reached 0.914 and 0.951,
respectively, with no visual identity swap. The formal perception gate **passes**.
