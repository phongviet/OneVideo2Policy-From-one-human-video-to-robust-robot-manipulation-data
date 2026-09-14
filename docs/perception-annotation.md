# Independent perception annotation

The perception gate uses 30 frozen frames from the temporary UniHand sequence.
The selection keeps six evenly spaced pre-contact frames and every frame from the
hand's approach through placement. It intentionally emphasizes occlusion and fast
object motion.

## Prepare the workspace

```bash
uv run ov2p prepare-perception-gate \
  data/interim/place_demo/manifest.json \
  --spec configs/place_evaluation.json \
  --output results/perception/place_demo
```

This copies raw RGB images only. It does not copy, show, or initialize annotations
from SAM2 predictions. For each selected frame, independently draw two binary masks:

```text
results/perception/place_demo/
├── images/000000.jpg
└── annotations/
    ├── source/000000.png   # cup only; exclude the hand
    └── target/000000.png   # tray, including portions occluded by cup/hand only if visible
```

Masks must be 1280×720, single-channel PNGs. Use 255 for object foreground and 0
for background. Annotate only visible pixels and exclude cast shadows, reflections,
the person's hand, and items resting inside the tray. Do not inspect predicted masks
until all 60 annotations are frozen.

Suitable tools include CVAT, Label Studio, Supervisely, or a local polygon editor.
Export one binary PNG per object and frame using the exact paths above.

## Evaluate the frozen annotations

After inspecting the overlay and recording the number of identity swaps, run:

```bash
uv run ov2p evaluate-perception-gate \
  --workspace results/perception/place_demo \
  --predictions data/interim/place_demo \
  --diagnostics data/interim/place_demo/gate-report.json \
  --config configs/place.yaml \
  --identity-swaps 0 \
  --output results/perception/place_demo/metrics.json
```

Until all annotation files exist, the command writes an `incomplete` report and
lists every missing mask. Once complete, it calculates per-object mean mask IoU and
combines it with track survival, visible-point count, and the identity-swap review.
