<div align="center">

# OneVideo2Policy

### From one human video to robust robot manipulation data

**A focused reproduction study of geometry-consistent synthetic demonstrations for imitation learning.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/tests-47%20passing-2EA44F)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-early%20research%20prototype-orange)](#project-status)

<img src="docs/assets/hero-placeholder.svg" width="920" alt="OneVideo2Policy project overview placeholder">

<sub>Project media placeholder — a real input-to-output demo will replace this panel after the first tracking milestone.</sub>

</div>

## The question

Robot imitation policies need diverse demonstrations, but collecting them usually
means more teleoperation time, robot access, and hardware wear. OneVideo2Policy asks:

> Can a single monocular human demonstration be converted into diverse,
> geometrically consistent synthetic robot demonstrations—and do they improve
> robustness more than ordinary 2D image augmentation?

This project narrows that question to one reproducible **Place** task: pick up a
rigid object and place it on a rigid target. The goal is not to reimplement every
component of Video2Robo or claim a new algorithm. It is to build a careful,
measurable reproduction of its central data-generation claim.

## At a glance

| | Project contract |
|---|---|
| **Input** | One 10–20 second monocular RGB human demonstration |
| **Task** | Tabletop pick, transfer, and place with rigid objects |
| **Representation** | Per-object 3D Gaussian representation and SE(3) trajectory |
| **Synthetic output** | Robot demonstrations with geometric and appearance variation |
| **Policy** | Diffusion Policy using front/side RGB and robot joint state |
| **Main comparison** | One demo vs. 2D augmentation vs. 3D geometry vs. full 3DGS |
| **Current focus** | Independent perception annotation and gate evaluation |

## System overview

```mermaid
flowchart LR
    A[One monocular<br/>human video] --> B[Segment and<br/>track objects]
    B --> C[Recover depth and<br/>object geometry]
    C --> D[Estimate 6D<br/>object trajectories]
    D --> E[Extract relative<br/>object skill]
    E --> F[Synthesize robot<br/>trajectory]
    F --> G[Render diverse<br/>3DGS demonstrations]
    G --> H[Train Diffusion<br/>Policy]
    H --> I[Evaluate controlled<br/>distribution shifts]

    classDef active fill:#e8f5ff,stroke:#1677a8,stroke-width:2px,color:#102a43;
    classDef future fill:#f6f8fa,stroke:#8c959f,stroke-dasharray:5 5,color:#57606a;
    class A,B,C,D active;
    class E,F,G,H,I future;
```

Solid nodes are the current milestone; dashed nodes are intentionally gated future
work. Heavy model integrations are added only after their upstream measurements are
stable.

## Project status

The repository now includes a runnable local end-to-end systems baseline and a
local model assisted run contract. It does **not** yet claim validated 6D motion,
physical robot success, or sim-to-real transfer.

| Component | Status | Evidence / next deliverable |
|---|:---:|---|
| Place benchmark and thresholds | ✅ | [`configs/place.yaml`](configs/place.yaml) |
| Video sampling and manifests | ✅ | `ov2p prepare-video` |
| SE(3), relative motion, tracking metrics | ✅ | Unit tested core package |
| SAM2 and CoTracker3 adapters | ✅ | CUDA run, overlay, and diagnostics complete |
| Independent perception gate | ✅ | HOI4D GT gate passes with controlled reseeding |
| Reconstruction preparation | ✅ | Native-resolution RGBA crop export verified |
| Real-scene metric geometry | 🟡 | Planar xy scale anchored by the measured 3 cm source diameter; calibrated 3D still pending |
| HOI4D metric camera and ball trajectory | ✅ | 72 frames; 0.75 px camera reprojection error; 0.28 px Gaussian ball tracking error |
| TripoSR and Depth Anything V2 Small | ✅ | Both run on 6 GB; HOI4D mesh and sensor-depth diagnostics recorded |
| Local end-to-end baseline | ✅ | 250 demos; 81/100 held-out rollouts with calibrated HOI4D geometry |
| Local model bundle | ✅ | Hashed TripoSR/depth inputs and 4–6 GB VRAM contract |
| 6D tracking ablation | ⬜ | Compare with/without tracked-point loss |
| RGB-D Gaussian scene | 🟡 | 7,007 fused Gaussians; 20.68 dB held-out render; optimization and robot composition pending |
| Task-specific simulator | ✅ | Measured 3.83 cm ball and 9.91 cm bowl; oracle controller passes 3/3 randomized rollouts |
| Synthetic demonstrations | ✅ | 10/10 measured ball-to-bowl trajectories plus 500 randomized localization frames |
| Policy benchmark | ✅ | Small learned visual waypoint policy passes 5/5 measured-task rollouts at 4.9 mm median XY error |

Legend: ✅ implemented · 🟡 contract/scaffold ready · ⬜ planned

## Method

### 1. Recover temporally coherent object motion

The source and target are segmented in the first frame and associated over time with
tracked pixels. Initial pose fitting combines rendered RGB, depth, and mask losses.
Later frames warm-start from the preceding estimate and add a correspondence term:

$$
\mathcal{L} = \lambda_c\mathcal{L}_{RGB}
+ \lambda_d\mathcal{L}_{depth}
+ \lambda_m\mathcal{L}_{mask}
+ \lambda_t\mathcal{L}_{track}.
$$

The tracking-loss ablation is a first-class experiment because symmetric objects can
look correct frame by frame while producing an unstable trajectory.

### 2. Represent the demonstrated skill as relative motion

For source and target poses $T_s(t)$ and $T_t(t)$, the task trajectory is:

$$
T_{rel}(t) = T_t(t)^{-1}T_s(t).
$$

This separates the demonstrated relationship from the original scene layout. A
manually specified object-relative grasp transform is acceptable for this study; the
research question is data diversity, not automatic grasp discovery.

### 3. Generate controlled 3D variation

Once tracking passes its gate, synthetic episodes will vary five independent factors:

- source and target pose;
- nearby camera pose and focal length;
- background;
- tabletop texture;
- Gaussian color-based lighting.

Each episode will retain aligned RGB observations, robot actions, joint positions,
object poses, and generation metadata.

## Evaluation design

The central benchmark holds the policy architecture fixed and changes only the
training data:

| Training data | ID | Object pose | Camera | Background | Lighting | Combined |
|---|---:|---:|---:|---:|---:|---:|
| One demonstration | — | — | — | — | — | — |
| + standard 2D augmentation | — | — | — | — | — | — |
| + geometry-only 3D augmentation | — | — | — | — | — | — |
| **+ full 3DGS augmentation** | — | — | — | — | — | — |

All cells are intentionally blank until measured. Planned secondary studies test
synthetic dataset scaling ($N \in \{10, 50, 100, 250, 500\}$), individual augmentation
families, and pose tracking with versus without correspondence loss.

<div align="center">
<img src="docs/assets/results-placeholder.svg" width="760" alt="Planned robustness results chart placeholder">
<br>
<sub>Results placeholder — generated from logged experiment outputs once Gate D passes.</sub>
</div>

## Quick start

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)
- A CUDA-capable GPU later, for perception and reconstruction models

### Install and verify

```bash
git clone <your-fork-or-repository-url>
cd onevideo2policy
uv sync --extra dev
uv run ov2p validate-config configs/place.yaml
uv run pytest
```

### Prepare one demonstration

Install the lightweight video extra, then sample a recording at a fixed rate. For
an RGB-D source, pass the directory containing zero-padded NumPy depth frames:

```bash
uv sync --extra video
uv run ov2p prepare-video path/to/place_demo.mp4 \
  --output data/interim/place_demo \
  --fps 30 \
  --depth-dir path/to/depth
uv run ov2p validate-manifest data/interim/place_demo/manifest.json
```

The commands write numbered RGB frames and validate a versioned `manifest.json`
containing original frame IDs, timestamps, and aligned RGB/depth paths. Raw and
generated datasets are ignored by Git. For large RGB-only clips on a small GPU,
`--max-width 960` downsamples extracted frames while preserving the original
video and recording both resolutions in the manifest.

### Run the local end-to-end baseline

After perception artifacts exist, the CPU-safe path exercises geometry, trajectory
recovery, randomized demonstration generation, policy fitting, and rollout gating:

```bash
uv run ov2p run-local-e2e \
  --config configs/two_paths.yaml \
  --manifest data/interim/hoi4d_ball_to_bowl_gate/manifest.json \
  --masks data/interim/hoi4d_ball_to_bowl_gate/masks \
  --output results/local_e2e/hoi4d_ball_to_bowl
```

See the [two-path execution guide](docs/two-path-execution.md) for the exact claim
boundary and the local model bundle. The real `IMG_6256.MOV` local
run and its camera-compensated, target-relative trajectory are documented in the
[capture audit](docs/data-audits/real-place-img-6255-6256.md). The
[local handoff](docs/real-place-handoff.md) records the verified transfer package
and the remaining independent annotation and scene measurements. The
[local model benchmarks](docs/local-model-benchmarks.md) record measured 6 GB GPU
alternatives and their current limits. The [Gaussian splatting stage](docs/gaussian-splatting.md) records the
metric RGB-D initialization, renderer contract, and remaining multiframe work.

After producing a binary source-object mask, seed tracking points reproducibly:

```bash
uv run ov2p sample-points source_mask.npy \
  --count 32 --seed 42 --border 4 \
  --output data/interim/place_demo/source_points.npy
```

## Repository structure

```text
onevideo2policy/
├── configs/                  # Frozen task, model, metric, and gate settings
├── data/                     # Local raw/interim/processed data (Git-ignored)
├── docs/                     # Roadmap, experiment log, failure analysis
├── experiments/runs/         # Local run outputs (Git-ignored)
├── results/                  # Tables, plots, and evaluation artifacts
├── scripts/                  # Thin experiment entry points
├── src/onevideo2policy/
│   ├── video/                # Frame preparation and perception protocols
│   ├── reconstruction/       # Depth/reconstruction protocols
│   ├── pose_tracking/        # Tracking losses and temporal metrics
│   ├── skill/                # Relative motion and Place phase extraction
│   ├── generation/           # Synthetic data engine milestone
│   ├── policy/               # Diffusion Policy milestone
│   └── evaluation/           # Dataset and robustness metrics
└── tests/                    # Fast, model-free unit tests
```

Optional SAM2 and CoTracker3 installation, pinned revisions, and adapter usage are
documented in [`docs/perception-models.md`](docs/perception-models.md).

The RGB-D import path for HOI4D's official motion masks is documented in
[`docs/hoi4d-integration.md`](docs/hoi4d-integration.md).

### Diagnose the robosuite image policy

The local diagnostic path records two camera views, OSC pose actions, next-step
absolute joint targets, task phases, object and end-effector trajectories, and
expert recovery states:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/generate_robosuite_demos.py \
  --output results/simulation/robosuite_can_100_dual_recovery \
  --episodes 100 --max-attempts 250 --max-steps 220 \
  --action-limit 0.8 --recovery-probability 0.03

.venv/bin/python scripts/train_diagnostic_policies.py \
  --data results/simulation/robosuite_can_100_dual_recovery/demonstrations.npz \
  --output results/policy/robosuite_controls_100

.venv/bin/python scripts/train_diagnostic_policies.py \
  --data results/simulation/robosuite_can_100_dual_recovery/demonstrations.npz \
  --output results/policy/robosuite_visual_waypoint_100 \
  --models visual_waypoint --epochs 50 --batch-size 256 --loss mse

MUJOCO_GL=egl .venv/bin/python scripts/evaluate_diagnostic_policies.py \
  --checkpoint results/policy/robosuite_visual_waypoint_100/visual_waypoint.pt \
  --output results/diagnostics/visual_waypoint_temporal_100_heldout_20 \
  --episodes 20 --max-steps 220 --seed 31415
```

The training script compares privileged state control, oracle phase-conditioned
vision, dual-view action chunking, compact diffusion, and closed-loop absolute-joint
chunk prediction. It uses phase-balanced sampling plus independent photometric
and camera perturbations. Closed-loop evaluation stores full robot, object, and
action trajectories for failure-phase analysis. The selected local policy uses
coordinate-aware dual-view waypoint estimation with temporal feedback and passes
19/20 held-out rollouts. See the
[robosuite diagnostic report](docs/robosuite-policy-diagnostics.md) for measured
results and the [Video2Robo gap audit](docs/video2robo-gap-audit.md) for the claim
boundary.

### Run the measured ball-to-bowl policy

The task-specific path uses the HOI4D-fitted ball and bowl dimensions. A cheap
localization dataset gives broader spatial coverage than repeating complete robot
trajectories:

```bash
MUJOCO_GL=egl PYTHONPATH=scripts .venv/bin/python \
  scripts/generate_ball_localization_data.py \
  --output results/simulation/ball_localization_500 --samples 500

.venv/bin/python scripts/train_diagnostic_policies.py \
  --data results/simulation/ball_localization_500/localization.npz \
  --output results/policy/ball_localization_500 \
  --models visual_waypoint --epochs 60 --batch-size 128

MUJOCO_GL=egl PYTHONPATH=scripts .venv/bin/python \
  scripts/evaluate_ball_bowl_waypoint.py \
  --checkpoint results/policy/ball_localization_500/visual_waypoint.pt \
  --output results/evaluation/ball_localization_500_random5 --episodes 5
```

This learned localization plus scripted waypoint controller succeeds in 5/5
randomized rollouts. The corresponding exact-position controller succeeds in 3/3,
isolating the earlier failure to visual localization.

## Research gates

The project uses explicit GO/NO-GO checks to prevent downstream ML from hiding an
upstream geometry failure:

1. **Reconstruction:** both objects are recognizable from useful nearby views.
2. **Tracking:** identity, mask IoU, track survival, reprojection error, and temporal
   jitter meet the frozen thresholds.
3. **Generation:** at least 80–90% of episodes satisfy task and kinematic constraints.
4. **Learning:** the policy succeeds in-distribution before robustness testing.
5. **Value:** 3D augmentation is compared honestly with 2D augmentation under
   controlled shifts—even if the result is negative.

Thresholds live in [`configs/place.yaml`](configs/place.yaml), and the complete plan
is documented in [`docs/roadmap.md`](docs/roadmap.md). The immediate, acceptance-test
driven execution plan is in [`docs/next-steps.md`](docs/next-steps.md). Before adding
data, use the [`place_demo` recording and sourcing guide](docs/recording-guide.md).

## Testing

The current tests are CPU-only and cover transform validation and composition,
relative trajectories, mask IoU, point survival, reprojection loss, translation
jitter, Place phase extraction, and configuration loading.

```bash
uv run ruff check .
uv run pytest
```

## Scope and limitations

- One rigid-object Place task comes before additional tasks.
- The input claim remains monocular RGB; depth estimates are inferred, not captured.
- Grasp pose is manually specified in the planned robot synthesis stage.
- Deformable objects, bimanual control, real-robot deployment, and VLA training are
  outside the initial scope.
- SAM2, CoTracker3, TripoSR, Depth Anything V2, Stable Fast 3D, and robosuite are not
  vendored. Their tested revisions and isolated environments are recorded in the model
  experiment artifacts.

## Acknowledgements

This project is inspired by **Video2Robo** and studies a deliberately narrower version
of its core claim. It is an independent reproduction and is not affiliated with the
original authors. Third-party model citations and licenses will be added alongside
their integrations.

## License

Released under the [MIT License](LICENSE).
