# Recording and sourcing `place_demo.mp4`

## Recommended option: record it yourself

### Scene

- Use a rigid, matte, textured, or visually asymmetric source object. A small printed
  carton or a cube with different colors on each face is better than a plain cylinder.
- Use a rigid target that makes success obvious: a shallow tray or a high-contrast
  taped rectangle approximately 1.5–2× the object's footprint.
- Place the source and target 15–30 cm apart on a matte, uncluttered tabletop.
- Remove other movable objects from the frame.

### Camera

- Use the phone's rear camera in landscape orientation.
- Record at 1080p and 30 FPS with no digital zoom.
- Mount the phone on a tripod or rigid support, approximately 0.8–1.2 m from the task.
- Use a 30–45° elevated view so the top and sides of both objects remain visible.
- Lock focus and exposure if the camera app supports it.
- Use diffuse, steady lighting and avoid reflections, hard shadows, and backlighting.

### Action script

1. Hold the unchanged starting scene for two seconds with hands out of frame.
2. Reach toward the source without covering it earlier than necessary.
3. Grasp it cleanly and move it smoothly toward the target.
4. Place it fully inside or on the target, then release it.
5. Move the hand away and hold the final scene for two seconds.

Aim for 10–20 seconds total. Record three takes, inspect all three, and retain only
the cleanest take as the single benchmark demonstration.

### Reject a take when

- the camera moves or autofocus visibly pulses;
- the source or target leaves the frame;
- the hand hides the object for a long interval;
- motion blur removes object edges;
- the object is dropped, re-grasped, or adjusted after placement;
- the target relationship is visually ambiguous;
- another object moves during the action.

### Store and validate

Save the selected recording exactly as:

```text
data/raw/place_demo.mp4
```

Inspect its metadata:

```bash
ffprobe -v error \
  -show_entries format=duration:stream=width,height,r_frame_rate,codec_name \
  -of default=noprint_wrappers=1 \
  data/raw/place_demo.mp4
```

Then run the intake pipeline:

```bash
uv sync --extra video
uv run ov2p prepare-video data/raw/place_demo.mp4 \
  --output data/interim/place_demo \
  --fps 10
```

## Fallback option: search an existing dataset

Search labels and annotations before downloading full archives. Useful queries are:

```text
"putting something onto something" human action dataset
"putting something on a surface" video dataset
"pick up and place" egocentric hand object video
"place object in tray" monocular RGB demonstration
```

Potential sources:

- [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
  has explicit placement-related action classes and short clips. Its low-resolution
  videos are better for pipeline prototyping than final 3D reconstruction.
- [Ego4D Hands and Objects](https://ego4d-data.org/docs/benchmarks/hands-and-objects/)
  contains egocentric object interactions, but clips require careful filtering and
  acceptance of the dataset license.
- [EPIC-KITCHENS](https://epic-kitchens.github.io/) contains annotated egocentric
  object interactions, though kitchen clutter and unscripted actions often violate
  this benchmark's clean-scene requirements.

Before adopting any external clip, confirm its license permits the intended research,
repository media, and portfolio demo. Do not commit raw third-party video unless the
license explicitly permits redistribution.

## Candidate acceptance rubric

Accept a clip only when every required row is **yes**:

| Requirement | Required |
|---|:---:|
| One RGB viewpoint is sufficient | Yes |
| Exactly one source object and one clear target | Yes |
| Complete approach → grasp → transfer → place → release | Yes |
| Source and target remain visible except for brief grasp occlusion | Yes |
| Stable camera, exposure, and focus | Yes |
| Rigid, non-transparent, non-reflective objects | Yes |
| Objective placement success can be computed | Yes |
| Redistribution or local research use is permitted | Yes |

Prefer recording a new clip whenever any row is uncertain.

