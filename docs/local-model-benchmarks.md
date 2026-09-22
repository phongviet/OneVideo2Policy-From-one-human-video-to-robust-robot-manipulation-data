# Local model experiments: real Place clip

Run on 2026-09-20 in WSL with a GeForce GTX 1660 Ti (6 GiB VRAM). The input is
the 57-frame, 960×540 `IMG_6256.MOV` sample and its frozen SAM2/CoTracker3
artifacts. The five-frame UniHand RGB-D fixture is a separate depth diagnostic.
These measurements select a **local systems configuration**, not a validated
robot policy for the filmed task.

## Selected local configuration

| Stage | Choice | Evidence and use |
| --- | --- | --- |
| Segmentation | SAM2.1 Hiera Small | Source sidewall remains visible. Tiny produced only 55% of Small's source mask area on average. |
| Tracking | Existing CoTracker3 offline | Already completed 57 frames; source/target track survival 0.986/0.921 with Small masks. |
| Depth | Depth Anything V2 Metric Hypersim Small, input size 518 | 0.165 s/frame and 415 MiB PyTorch peak allocation. More stable target depth across this clip than size 384. Use only as shape/ordering evidence until calibrated. |
| Mesh proposal | TripoSR, 128 extraction resolution, chunk 4096 | Both crops fit with memory margin; about 1.87 GiB PyTorch peak allocation and 0.6 s mesh extraction/object. Stable Fast 3D is an optional visual proposal for the blue target. |
| Collision geometry | Existing measured primitives once dimensions are supplied | Both learned models invent unseen surfaces. Current primitive dimensions are assumptions, not measurements. |
| Simulator | robosuite 1.5.1, MuJoCo 3.3.7, EGL, Panda, 84×84 camera | A headless 100-step PickPlace run completed at 16.6 steps/s. This is a throughput smoke test only. |
| Current proxy policy | Ridge behavior cloning | 100/100 successes on each of three held-out proxy rollout seeds. Small MLPs failed to grasp despite lower one-step action errors. |
| Image policy benchmark | Direct action models plus learned dual-view waypoint estimation | Direct action variants fail; the temporal visual waypoint policy passes 19/20 held-out rollouts. |

## Measured comparisons

### Perception

SAM2.1 Tiny completed the same 57-frame clip with the same prompts and
CoTracker3. Tiny and Small target masks have mean inter-model IoU 0.992.
For the source, mean IoU is 0.549 and Tiny's area is 0.549 of Small's. In the
[comparison overlay](../results/model_benchmarks/real_place_img_6256/perception_tiny/comparison.jpg),
Tiny visibly removes much of the container sidewall. Small's adjacent-frame
mask IoU was 0.661 for the source and 0.881 for the target; Tiny's was 0.604
and 0.882. These are annotation-free diagnostics, not mask accuracy.

A separate agent later drew 60 masks from raw frames without seeing either
model's predictions. Against these prediction-blind AI annotations, Small's
source/target mean IoU is 0.800/0.968 and Tiny's is 0.465/0.966. Small passes
the configured numeric perception thresholds; Tiny fails on source IoU. The
[AI annotation provenance](../results/perception/real_place_img_6256_blind/annotation-provenance-ai.json)
and [per-frame scores](../results/perception/real_place_img_6256_blind/per-frame-ai.json)
are saved locally. Human ground truth is still required for the formal gate.

### Depth

| Model and input size | Median video inference | PyTorch peak allocation | Fixture scale-aligned AbsRel | Fixture raw AbsRel | Video target depth CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metric Small, 384 | 0.078 s | 226 MiB | 0.087 | 0.829 | 0.140 |
| Metric Small, 518 | 0.165 s | 415 MiB | 0.088 | 0.654 | 0.060 |
| Relative Small, 384 | 0.075 s | 226 MiB | 0.136 | n/a | 0.165 |
| Relative Small, 518 | 0.167 s | 415 MiB | 0.128 | n/a | 0.166 |

Fixture uint16 depth is interpreted as millimetres. The scale-aligned
diagnostic uses the *ground-truth median independently on each frame*, so it
cannot demonstrate metric recovery on the real video. Relative-model output
is inverse depth in arbitrary units. The video CV also reflects camera/object
motion and changing predicted masks; it is not an accuracy metric. The metric
model's raw fixture scale is substantially biased, so no unmeasured metric
geometry should be inferred from its predictions.

### Mesh reconstruction

| Extraction resolution | Source/target faces | Extraction time | Max PyTorch allocation |
| --- | ---: | ---: | ---: |
| 128 | 20,094 / 35,986 | 0.66 / 0.57 s | 1.87 GiB |
| 256 | 87,904 / 149,568 | 4.85 / 4.38 s | 2.38 GiB |

TripoSR model inference took roughly 1–2 seconds/object after loading. The
meshes are watertight, but the [source turntable](../results/model_benchmarks/real_place_img_6256/triposr/comparison/source/turntable.jpg)
shows ragged edges and the [target turntable](../results/model_benchmarks/real_place_img_6256/triposr/comparison/target/turntable.jpg)
shows invented back geometry. The 128 and 256 mesh extents differ by about
1% or less along most axes, while 128 extracts much faster. Mesh units are
unscaled. Because this machine has no CUDA compiler, a scikit-image CPU
marching-cubes shim performs mesh extraction; the learned reconstruction runs
on CUDA.

Stable Fast 3D was then run on the same frozen source and target crops after
Hugging Face access was granted. Its required RGBA inputs contain the same RGB
pixels and segmentation as TripoSR's gray-background inputs. The official
CPU-only texture baker and UV extension were built locally; the neural model
ran on CUDA with FP16 autocast because this GPU does not support BF16.

| Texture resolution | Source/target mesh faces | Source/target runtime | Source/target PyTorch peak allocation |
| --- | ---: | ---: | ---: |
| 128 | 21,712 / 28,972 | 26.4 / 17.6 s | 6,169 / 6,114 MiB |
| 256 | 21,712 / 28,972 | 25.3 / 17.8 s | 6,169 / 6,114 MiB |

Reducing texture resolution did not reduce peak allocation; mesh geometry is
identical and 256 retains more texture detail. The [Stable Fast 3D source
turntable](../results/model_benchmarks/real_place_img_6256/sf3d/texture256/source/turntable.jpg)
shows a very thin disc: its smallest extent is only 6% of its largest. The
[target turntable](../results/model_benchmarks/real_place_img_6256/sf3d/texture256/target/turntable.jpg)
shows a smoother front/back surface than TripoSR, but still invents unseen
depth. Both Stable Fast 3D meshes are non-watertight. The near-full GPU memory
use, roughly an order of magnitude slower object processing, and source flattening keep
TripoSR as the practical local default. Neither model provides measured
collision geometry. Texture baking through the CPU bridge also means these
times are for this local adaptation, not the upstream CUDA implementation.

### Simulator and policy

Robosuite PickPlace with Panda and EGL renders 84×84 observations at 16.6
steps/s and 128×128 at 14.9 steps/s over 100 zero-action steps. These are
hardware measurements, not placement success. MuJoCo 3.13 initially failed
in robosuite's joint setup; pinning 3.3.7 resolved it.

On 250 existing synthetic point-robot demonstrations (12,000 samples), ridge
has 204 parameters, 0.00113 held-out action MAE, and 100% success across
three 100-episode rollout seeds. MLP-32 and MLP-64 have 2,820 and 7,684
parameters, lower one-step action MAE (0.00079 and 0.00069), but 0% rollout
success because they rarely attach the source. This is a useful warning
against selecting a policy by one-step validation loss. It is not an
evaluation of a physical robot or robosuite policy.

The initial robosuite experiment generated ten successful Panda can-to-bin
demonstrations and exposed a large initial-action error hidden by low aggregate
validation MAE. The corrected run then generated 100 successful demonstrations
in 100 attempts, with 18,071 samples, two 84×84 camera views, absolute joint
targets, phase labels, and 370 expert recovery corrections.

Seven local configurations were evaluated after episode-level training splits.
The privileged robot-plus-object state MLP achieved 8/20 held-out successes
(40%). Phase-conditioned vision, Huber action chunks, compact diffusion, and
absolute-joint chunks each achieved 0/5. L1 and MSE action-chunk controls each
achieved 0/3. Every direct-action image candidate remained in the approach phase. Offline MAE
therefore remains a poor selection metric: the phase model had the lowest OSC
validation MAE, while only the privileged state model had nonzero success.

The selected local configuration now uses a coordinate-aware dual-view network
to predict the can waypoint and the scripted phase controller for low-level
tracking. Label-preserving photometric training and a five-estimate temporal
median reduce held-out initial localization error to about 1.0 cm. This policy
achieves 19/20 held-out successes (95%), compared with 0/5 for the coordinate-aware
end-to-end action model. The remaining failure is a missed grasp from a visual
waypoint outlier.
The full table and tracked summary are in
[`robosuite-policy-diagnostics.md`](robosuite-policy-diagnostics.md) and
[`experiments/robosuite-policy-diagnostic-results.json`](experiments/robosuite-policy-diagnostic-results.json).

### Real-video geometry audit

The source MOV identifies an Apple iPhone 11 Pro Max and 1920×1080 recording,
but exposes no usable lens intrinsics. Full-clip Metric Depth Anything V2 Small
predictions have a source median mean of 0.907 and target median mean of 0.772
in model output units. The source-minus-target median has coefficient of
variation 0.632 across the clip. Background alignment still leaves up to
117.9 pixels of image-center shift, and the nominally static target drifts a
median 15.2 pixels and a maximum 42.7 pixels. Together with 0.654 raw AbsRel on
the UniHand RGB-D diagnostic, this is insufficient for metric geometry.

The geometry output is explicitly marked `uncalibrated_geometry_evidence` in
`results/geometry/real_place_img_6256/audit/report.json`. Physical dimensions
and camera calibration remain required for scaled collision geometry and a
metric trajectory.

The detailed comparison with the published Video2Robo collection and training
recipe is in [`video2robo-gap-audit.md`](video2robo-gap-audit.md). It records both
the ten-trajectory baseline diagnosis and the outcome of the corrected
100-demonstration experiment.

## Reproducibility and remaining evidence

Benchmark scripts live in `scripts/benchmark_local_depth.py`,
`scripts/benchmark_depth_fixture.py`, `scripts/benchmark_triposr.py`,
`scripts/benchmark_robosuite.py`, `scripts/benchmark_local_policy.py`,
`scripts/benchmark_image_bc_capacity.py`, `scripts/benchmark_sf3d.py`,
`scripts/audit_local_geometry.py`, `scripts/generate_robosuite_demos.py`,
`scripts/train_image_bc.py`, `scripts/evaluate_image_bc.py`, and
`scripts/render_mesh_turntable.py`. Raw reports, frozen inputs,
rendered comparisons, and meshes are under
`results/model_benchmarks/real_place_img_6256/` (ignored by Git). Upstream
repositories and isolated environments are under
`experiments/runs/model_benchmarks/` (ignored by Git). The current local
pipeline's independent test suite passes (47 tests).

The filmed task still needs human independent object masks, measured scene
dimensions or camera calibration, a reconstruction reference, and paired
simulator observations/actions before a robot-success comparison or image
policy training can identify a best task-level model. The existing local
end-to-end run remains a point-robot systems baseline using assumed object
dimensions. A measured 3D point set and camera intrinsics are also needed to
evaluate a PnP-based 6D trajectory, and aligned robot trajectories are needed
to compare image BC with a compact diffusion policy. The Gaussian-rendering
ablation has no local implementation or fitted candidate to benchmark.
