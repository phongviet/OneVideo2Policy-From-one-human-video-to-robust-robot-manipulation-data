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

The state result has a 95% Wilson interval of approximately 22% to 61%. The
image-policy sample sizes only establish clear failure in this setup; 0/5 has an
upper 95% Wilson bound of approximately 43%. Every image candidate remained in
approach phase for every evaluated episode. The absolute-joint candidate was
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

## Decision and next experiment

The 80% held-out success gate fails. The scripted OSC pose expert remains the
selected local controller, and the current learned policies should not be sent
to a physical robot or larger GPU unchanged.

The next model experiment should replace the compact CNN with a pretrained
spatial visual encoder or explicit object keypoints, preserve spatial feature
maps through the action head, and evaluate the privileged state policy on more
seeds as the attainable local control reference. The filmed-scene path still
requires measured geometry or camera calibration before sim-to-real claims.

The tracked machine-readable result is
[`experiments/robosuite-policy-diagnostic-results.json`](experiments/robosuite-policy-diagnostic-results.json).
Raw trajectories, weights, and reports remain under ignored `results/` paths.
