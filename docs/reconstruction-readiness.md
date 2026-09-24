# Local reconstruction readiness

## Selected models

The 6 GB GTX 1660 Ti path uses models measured on this machine:

| Stage | Model | Local result |
|---|---|---|
| Object proposal | `stabilityai/TripoSR` | 128 mesh resolution and 4096 chunk size peak at about 1.87 GiB |
| Monocular depth | `depth-anything/Depth-Anything-V2-Metric-Hypersim-Small` | 518 input size peaks at 415 MiB and takes 0.16 s/frame after warmup |
| Metric geometry | HOI4D aligned RGB-D plus calibrated primitives | Used for scale and collision geometry |

SAM2.1 Hiera Small and CoTracker3 offline provide the masks and tracks consumed by
these stages. Stable Fast 3D remains an optional visual proposal, but its roughly
6.1 GiB peak allocation leaves too little margin on this GPU.

## HOI4D result

The native crops are 99×99 for the ball and 196×196 for the bowl. TripoSR produced
watertight meshes for both at 128 extraction resolution, but their minimum-to-maximum
extent ratios are 0.065 and 0.085. This severe flattening makes them unsuitable for
collision geometry. They are retained as visual proposals only.

Depth Anything V2 Metric Small was evaluated on five sampled frames against the
HOI4D sensor depth. It used a median 0.526 scale correction, with raw mean AbsRel
0.715 and independently scale-aligned mean AbsRel 0.155. The model therefore supplies
depth structure; aligned RGB-D supplies metric scale. Per-frame scale alignment is a
diagnostic and is not evidence of monocular metric recovery.

## Reproduce

```bash
.venv/bin/python scripts/benchmark_local_depth.py \
  --repo experiments/runs/model_benchmarks/third_party/Depth-Anything-V2 \
  --checkpoint "$HOME/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/snapshots/3bc65d4e14a6786a61acec16453c50e12bf5f338/depth_anything_v2_metric_hypersim_vits.pth" \
  --frames data/interim/hoi4d_ball_to_bowl_gate/frames \
  --target-masks data/interim/hoi4d_ball_to_bowl_gate/masks/target \
  --source-masks data/interim/hoi4d_ball_to_bowl_gate/masks/source \
  --output results/model_benchmarks/hoi4d_ball_to_bowl/depth_metric_small \
  --sizes 518 --frame-ids 0 18 36 59 71
```

The raw reports and generated meshes are kept under
`results/model_benchmarks/hoi4d_ball_to_bowl/` and are ignored by Git.
