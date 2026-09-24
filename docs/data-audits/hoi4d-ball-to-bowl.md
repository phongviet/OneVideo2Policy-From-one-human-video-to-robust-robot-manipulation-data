# HOI4D ball-to-bowl audit

## Decision

**Accepted as a calibrated metric carry-and-place fixture and as a perception
fixture.** It contains a real two-object placement action with official masks for
both the moving ball and the destination bowl. The local subset now includes raw
aligned depth, intrinsics, and per-frame extrinsics.

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
| Depth | 300 aligned raw 16-bit PNGs retained; integer unit is 1 mm |
| Camera calibration | 300 extrinsics plus `fx=1060.2955`, `fy=1061.5068`, `cx=971.5211`, `cy=523.2619` |

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
| Source ball diameter and height | Robust sphere fit to official ball mask back-projected through aligned depth | Available: 0.0383212 m median diameter; 10th–90th percentile 0.0377411–0.0413126 m |
| Target length, width, and height | `objpose/*.json` `dimensions` | Available: 0.0991768 m × 0.0991230 m × 0.0569334 m |
| Initial source-target center separation | Distance between fitted ball center and annotated bowl center | First measurable value is 0.579754 m at frame 87; the frame-zero ball is outside the view |
| Camera calibration | Official camera intrinsics plus the calibrated sequence package's per-frame extrinsics | Available for all 300 frames |
| Metric target trajectory | `objpose/*.json` `center`, `rotation`, and `dimensions` | Available for all 300 frames |

The bowl's frame-0 center is `(-0.114057, 0.026107, 0.861622)` m in the
annotation camera frame. Its dimensions are constant across all 300 object-pose
files. The ball is a motion-segmentation object rather than the category object,
so HOI4D does not give it a direct `objpose` entry. Its metric center and size are
computed from its official mask and aligned depth. The median fitted diameter is
3.832 cm over 51 accepted frames. The 10th–90th percentile range is 3.774–4.131 cm.

The ball first has a motion mask at frame 82 and first has usable masked depth at
frame 87. It enters from outside the image while already being carried. Therefore
the sequence does **not** contain a true initial rest-state separation. The retained
0.579754 m value is explicitly the first measurable carry-state separation and
must not be described as frame-zero ground truth.

The metric bundle was completed on 2026-09-24 from the exact-sequence archive
`ZY20210800001_H1_C7_N14_S280_s04_T5.tar.gz` and the official Camera Parameters
package. The archive SHA-256 is
`26d77fcd8e27d4c0091eaf3d71309681f0f96da61c0dadb419f22f36c4a5c7b1`.
All 300 raw depth frames are 1920×1080. Decoded RGB frames 0, 86, and 299 match
the existing official video pixel-for-pixel, and the standalone official intrinsic
matrix matches the sequence calibration exactly.

## Storage decision

The retained HOI4D subset contains 61 RGB clips with their 2D masks, actions, and
object-pose metadata (about 722 MB). The imported ball-to-bowl fixture adds about
119 MB. The three 20–23 GB bulk archives were deleted after extraction, keeping the
combined retained footprint under 1 GB and below the 20 GB budget.

## Use and limits

The fixture is appropriate for SAM2/CoTracker evaluation and calibrated metric
pipeline diagnosis. Its absent initial ball view prevents evaluating recovery of
the complete action from frame zero. It also remains egocentric and differs from
the fixed phone-camera deployment view. Keep the real Place demonstration as the
final policy target.

## Perception gate result

The initial 2026-09-15 CUDA run produced mean mask IoU of 0.777 for the ball and
0.920 for the bowl, but bowl survival was only 0.758. The controlled follow-up held
the exact masks and all thresholds fixed while refreshing deterministic CoTracker
queries every 16 frames. Ball and bowl survival reached 0.914 and 0.951,
respectively, with no visual identity swap. The formal perception gate **passes**.
