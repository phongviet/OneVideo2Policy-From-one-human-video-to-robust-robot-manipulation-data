# HOI4D ball-to-bowl audit

## Decision

**Accepted as an RGB-only perception and mask-evaluation fixture.** It contains a
real two-object placement action with official masks for both the moving ball and
the destination bowl.

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
| Depth | Not retained: the downloaded depth archive was an incomplete shard |

The GPU gate view uses original frames 86–299, sampled at 5 FPS and resized to
640×360. This yields 72 aligned frames over 14.2 seconds. Frame 86 has the largest
initial visible ball mask, so it supplies the positive source prompt without
starting from a clipped object at the image boundary.

## Storage decision

The retained HOI4D subset contains 61 RGB clips with their 2D masks, actions, and
object-pose metadata (about 722 MB). The imported ball-to-bowl fixture adds about
119 MB. The three 20–23 GB bulk archives were deleted after extraction, keeping the
combined retained footprint under 1 GB and below the 20 GB budget.

## Use and limits

The fixture is appropriate for SAM2/CoTracker evaluation using independently
provided mask ground truth. It is not the final geometry/reconstruction benchmark:
it is egocentric RGB-only, has camera motion, and lacks retained depth. Keep the
fixed-camera UniHand fixture for RGB-D adapter checks, and replace both with a
longer recorded real Place demonstration before policy training.
