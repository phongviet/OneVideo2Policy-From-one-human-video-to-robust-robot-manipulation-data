# Local end-to-end execution

OneVideo2Policy uses two compatible local paths. Both consume the validated SAM2 and
CoTracker3 artifacts.

| Stage | Systems baseline | Local model assisted path |
|---|---|---|
| Object geometry | Measured sphere and bowl primitives | TripoSR visual mesh proposals |
| Scene geometry | HOI4D aligned RGB-D | Depth Anything V2 Metric Small, aligned to RGB-D |
| Collision geometry | Measured primitives | Measured primitives; learned meshes are visual only |
| Motion | Relative mask trajectory with a Place lift arc | Metric RGB-D camera and ball translation over 72 frames |
| Robot | Point-robot baseline | robosuite Panda |
| Policy | Ridge behavior cloning | Temporal dual-view waypoint model plus phase controller |

## Run the systems baseline

```bash
uv run ov2p run-local-e2e \
  --config configs/two_paths.yaml \
  --manifest data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --masks data/interim/hoi4d_ball_to_bowl_gate/masks \
  --output results/local_e2e/hoi4d_ball_to_bowl
```

The output includes primitive OBJ assets, a recovered proxy trajectory, randomized
demonstrations, policy weights, rollout video, and a gate report. The HOI4D values
come from the downloaded sequence: a 3.832 cm fitted ball diameter, 9.915 cm bowl
diameter, and 5.693 cm bowl height.

## Prepare the local model bundle

```bash
uv run ov2p prepare-model-run \
  --config configs/two_paths.yaml \
  --manifest data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --crops data/interim/hoi4d_ball_to_bowl_gate_reseed16/crops_native \
  --output results/model_bundle/hoi4d_ball_to_bowl
```

The bundle contains 12 uniformly sampled geometry keyframes, both reconstruction
crops, SHA-256 hashes, selected model identifiers, local VRAM requirements, and the
expected output contract. The selected models fit the local 6 GB GPU.

TripoSR output must not be used as collision geometry for the current crops because
the HOI4D meshes are severely flattened. Depth Anything output must be scale-aligned
to sensor depth or a measured reference. These boundaries are encoded in
`configs/two_paths.yaml`.
