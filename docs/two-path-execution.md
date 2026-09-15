# Two-path end-to-end execution

OneVideo2Policy now separates systems completion from paper-faithful evidence. Both
paths consume the same validated perception artifacts and preserve compatible stage
boundaries.

| Stage | Local path | Faithful path |
|---|---|---|
| Geometry | Deterministic sphere and open-bowl OBJ proxies | TRELLIS image-to-3D |
| Scene scale | Configured measured scene dimensions | VGGT aligned to measurements |
| Motion | Planar mask-centroid proxy plus Place lift arc | Optimized 6D object poses |
| Robot | Bounded point-robot controller | Robosuite Panda with OSC/IK |
| Data | Object-centric randomized state trajectories | Rendered robot demonstrations |
| Policy | Ridge behavior cloning | Image Diffusion Policy |
| Claim | Local artifact and control-flow validation | Full research hypothesis |

## Run the local path

This command is CPU-safe and requires only the frozen source and target masks:

```bash
uv run ov2p run-local-e2e \
  --config configs/two_paths.yaml \
  --manifest data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --masks data/interim/hoi4d_ball_to_bowl_gate/masks \
  --output results/local_e2e/hoi4d_ball_to_bowl
```

The output contains primitive OBJ assets, the recovered proxy trajectory, randomized
demonstrations, fitted policy weights, and a gate report. The retained HOI4D fixture
uses a provisional configured initial separation of 0.25 m because the dataset does
not provide that scene measurement. Replace it with the measured value from the real
recording before interpreting metric trajectory errors.

The local path proves that every repository stage can execute and exchange artifacts.
It does not establish single-view reconstruction quality, robot-arm feasibility,
image-policy robustness, or sim-to-real transfer.

## Prepare the faithful path

```bash
uv run ov2p prepare-faithful-run \
  --config configs/two_paths.yaml \
  --manifest data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --crops data/interim/hoi4d_ball_to_bowl_gate_reseed16/crops_native \
  --output results/faithful_bundle/hoi4d_ball_to_bowl
```

This creates a portable bundle with 12 uniformly sampled VGGT keyframes, both
TRELLIS crops, SHA-256 hashes, model identifiers, hardware requirements, and the
expected output contract. Copy the bundle to a 16 GB+ CUDA host; 24 GB is
recommended. High-compute results must return under the paths declared in
`run-spec.json` before they can be compared with the local baseline.

## Advancement rule

Continue development with the local path whenever a heavyweight model is
unavailable. Only use the faithful path to support the central 3D reconstruction and
robustness claims. Never substitute a local proxy result into a faithful-path table.
