# Robosuite policy diagnostics

## Purpose

This experiment diagnoses the compact image behavior cloning failure and tests
the smallest local corrections suggested by the Video2Robo comparison. Results
use robosuite's built-in Panda can-to-bin task. They do not establish transfer
to the filmed objects or a physical robot.

## Ten-episode baseline diagnosis

The original dataset has 10 episodes and 2,767 samples, split by episode into
eight training and two validation episodes. Integrity checks pass for episode
boundaries, shapes, finite normalized actions, and duplicate validation images.
The aggregate one-step validation MAE of 0.0317 concealed critical errors:

- initial-step translation MAE is 0.175;
- the expert's mean initial translation is `[0.400, -0.317, 0.159]`, while the
  CNN predicts `[0.329, -0.092, 0.098]`;
- retreat has the highest validation translation MAE at 0.132;
- 15.8% of sampled cross-episode nearest neighbors in learned observation space
  have action disagreement above 0.5 L2.

Privileged state, oracle phase vision, action chunking, and compact diffusion
controls all scored 0 successes on the ten-episode data. The common failure from
stored training positions showed that more data and recovery coverage were
necessary before model architecture comparisons would be meaningful.

## Expanded experiment

The corrected collector produced 100 successful demonstrations in 100 attempts:

- 18,071 synchronized samples;
- agent and front 84×84 RGB views;
- proprioception, object state, end-effector, source, and target trajectories;
- OSC pose actions and next-step absolute joint targets;
- phase labels and episode boundaries;
- 370 expert corrections following random control perturbations.

Training used an 80/20 episode split, phase-balanced sampling, and independent
per-frame brightness, contrast, noise, ±3-degree rotation, and small translation.
All candidates were trained locally for 20 epochs. L1 and MSE action-chunking
variants were also trained for 10 epochs.

| Policy | Parameters | Offline validation metric | Held-out success |
| --- | ---: | ---: | ---: |
| Privileged robot + object state MLP | 82,439 | MAE 0.0276 | **8/20 (40%)** |
| Oracle phase-conditioned single-view vision | 335,111 | MAE 0.0198 | 0/5 |
| Dual-view OSC action chunker, Huber | 639,768 | MAE 0.0375 | 0/5 |
| Dual-view compact diffusion chunker | 944,152 | denoising MAE 0.0608 | 0/5 |
| Dual-view absolute-joint chunker, Huber | 642,848 | MAE 0.0268 | 0/5 |
| Dual-view OSC action chunker, L1 | 639,768 | MAE 0.0408 | 0/3 |
| Dual-view OSC action chunker, MSE | 639,768 | MAE 0.0461 | 0/3 |
| Phase-aware privileged state MLP | 84,487 | MAE 0.0111 | 1/5 |
| Coordinate-aware end-to-end vision | 2,697,845 | MAE 0.0111 | 0/5 |

The state result has a 95% Wilson interval of approximately 22% to 61%. The
image-policy sample sizes only establish clear failure in this setup; 0/5 has an
upper 95% Wilson bound of approximately 43%. Every direct-action image candidate
in this batch remained in approach phase for every evaluated episode. The absolute-joint candidate was
run with robosuite's absolute `JOINT_POSITION` controller, so its result is a
real closed-loop test rather than an offline-only comparison.

## Diagnosis

The expanded run changes the diagnosis. Data volume and output loss alone do not
fix the image policy: Huber, L1, MSE, action chunks, diffusion, phase labels, two
views, and absolute joint targets all fail. The nonzero state-policy score shows
that the expanded demonstrations contain a partially learnable control signal.
The gap is therefore concentrated in visual state extraction and closed-loop
robustness. The compact encoder compresses spatial observations too aggressively
for precise can localization, and the training distribution still lacks the
rendering and camera diversity used by Video2Robo.

Offline MAE ranks the oracle phase model best, yet that model scores 0/5 while
the higher-error state model scores 8/20. Checkpoint selection must continue to
use closed-loop success.

## Successful visual waypoint policy

The follow-up separates visual localization from low-level action generation.
A 2.61-million-parameter coordinate-aware encoder predicts the can's metric xyz
position from both 84×84 camera views. Training uses photometric augmentation
without synthetic image translations or rotations, because those transforms
moved image evidence while keeping world-coordinate labels fixed. The controller
updates the predicted waypoint during approach and takes a median over the five
most recent estimates, then follows the validated phase controller.

On the held-out 20 episodes, the visual estimator has approximately 1.0 cm mean
Euclidean error on initial frames. Closed-loop evaluation achieves **19/20
successes (95%)**, passing the 80% gate. Median final xy error to the bin center
is 1.83 cm. The 95% Wilson interval for the success rate is approximately 76% to
99%; more rollouts are needed for a tight population estimate.

This result fixes the local learned-vision control path. It is a hybrid policy:
object localization is learned, while Cartesian waypoint tracking, phase timing,
and the target-bin location are specified by the task controller. The end-to-end
networks that directly regress low-level actions remain failed baselines.

## Decision and next experiment

The local 80% held-out success gate passes at 95% for the learned visual waypoint
policy. This replaces the fully scripted expert as the selected local perception
and control configuration for the built-in robosuite task.

The remaining local failure is one missed grasp caused by a waypoint outlier.
Further work should add uncertainty estimates and more held-out seeds. The
filmed-scene path still requires measured geometry or camera calibration,
real-camera training data, and physical safety validation before sim-to-real
claims or robot execution.

The tracked machine-readable result is
[`experiments/robosuite-policy-diagnostic-results.json`](experiments/robosuite-policy-diagnostic-results.json).
Raw trajectories, weights, and reports remain under ignored `results/` paths.

## Measured ball-to-bowl transfer

The final local task replaces the built-in can and bin with HOI4D-derived collision
geometry: a 1.916 cm radius ball and a 4.956 cm outer-radius bowl. The scripted
expert generated 10/10 successful trajectories in ten attempts, totaling 3,292
dual-camera samples. An exact-position controller then passed 3/3 randomized
closed-loop rollouts.

The first transferred waypoint model used the full ten trajectories. It passed only
1/5 randomized rollouts because ten initial placements did not cover the workspace;
median initial 3D localization error was 5.9 cm. Training also mixed approach frames
with frames where the ball was carried or placed. Restricting the localization target
to approach phases removed that label conflict.

A targeted generator then rendered 500 independent randomized placements in 75.8
seconds. Training the same 2.61-million-parameter coordinate-aware model took 7.5
seconds. On 100 held-out frames, median XY error is 5.7 mm, mean is 6.3 mm, and the
90th percentile is 11.2 mm. The controller freezes the initial unobstructed estimate
because the ball remains stationary before grasping. This configuration passes **5/5
randomized measured-task rollouts**, with 4.9 mm median initial XY error.

These five rollouts establish a successful local integration test, while the sample
is too small for a tight reliability estimate. The next policy experiment expands
seeds and introduces camera, background, and lighting shifts.
