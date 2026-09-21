# Video2Robo comparison and local policy failure audit

## Scope

This audit compares the saved local robosuite image behavior-cloning run with
Video2Robo as described in the CVPR 2026 paper and supplementary material. The
authors' project page currently labels its code as "coming soon," and the paper
does not report optimizer, batch size, epoch count, observation horizon, action
horizon, or image resolution. Those details cannot yet be matched exactly.

Primary sources:

- <https://openaccess.thecvf.com/content/CVPR2026/html/Deng_Video2Robo_3DGS-based_Synthetic_Data_from_One_Video_Enables_Scalable_Robot_CVPR_2026_paper.html>
- <https://video2robo.github.io/>

The public-data search and recovered website preview inventory are documented in
[`video2robo-data-availability.md`](video2robo-data-availability.md).

## Material differences

| Component | Video2Robo | Current local experiment | Consequence |
| --- | --- | --- | --- |
| Demonstrations | Evaluates generation of 100 new demonstrations per task; generated trajectories have 100% reported generation success | Baseline: 10 episodes. Corrected run: 100 successful episodes, 80 train and 20 validation | Demonstration count now matches the reported scale, but scene diversity does not |
| Trajectory construction | Explicit Transit, Grasp, Transfer, Task Execution, and release workflow; source object is coupled rigidly to the end effector during manipulation | Similar scripted phases, but stored low-level OSC pose delta commands are learned directly | The expert works, but its phase-dependent deltas form a multimodal regression target |
| Scene variation | Object xy/yaw rearrangement plus texture, background, lighting, and camera perturbation | Object pose changes plus local per-frame photometric and small affine augmentation; renderer, table, and background remain fixed | The corrected data still lack true scene and camera diversity |
| Augmentation frequency | Appearance, lighting, and camera changes are applied to every frame | Corrected training applies independent augmentation per frame | Frequency is approximated, but the local transforms do not reproduce 3DGS rendering variation |
| Cameras | Front and side views | Corrected run stores agent and front 84×84 views | View count is addressed at low resolution |
| Policy | Diffusion Policy | One-step CNN baseline plus phase, chunking, and compact diffusion variants | Architecture class was tested locally; the small visual encoder still fails to recover usable spatial state |
| Actions | Absolute joint positions | Corrected data store both normalized OSC deltas and absolute joint targets | Both representations scored 0/5 with image policies |
| Physical deployment alignment | Synthetic and real camera/table layouts are approximately matched; two real cameras are used | No physical deployment or calibrated camera layout | The current result only tests same-simulator closed-loop control |
| Geometry | Metric scale is optimized from VGGT depth; tabletop is fitted and object 6D pose is tracked | Filmed geometry remains uncalibrated; robosuite uses its built-in can and bin | The local simulator data do not represent the filmed task |

## Evidence from the failed local policy

The ten demonstrations contain 2,767 samples, but phase 4 alone accounts for
1,090 samples (39.4%) and the final retreat phase has only 24 samples (0.9%).
The gripper is closed in 1,945 samples and open in 822. This imbalance makes a
low average one-step loss compatible with poor transition behavior.

At the first step of each stored episode, the expert's mean xyz command is
`[0.400, -0.317, 0.159]`; the learned policy predicts
`[0.329, -0.092, 0.098]`. Initial-step xyz MAE is 0.175, much worse than the
reported aggregate validation MAE of 0.0317. The predicted y command has only
29% of the expert's mean magnitude. The policy therefore fails to reach the can
reliably. Multiplying translation by three and discretizing the gripper still
produces 0/5 successes, confirming that simple output scaling is insufficient.

The baseline closed-loop score is 0/10 even within the same robosuite visual
domain. Consequently, missing photorealistic rendering and real-world domain
randomization are not the immediate cause of this failure. The immediate causes
are insufficient trajectory coverage, one-step mean regression, phase imbalance,
and delta-action compounding. The missing Video2Robo rendering and augmentation
would become critical at the later sim-to-real stage.

## Corrected experiment order

1. Generate at least 100 successful trajectories across the same object xy/yaw
   range used for evaluation. Preserve explicit phase boundaries and validate
   every trajectory before adding it to the training set.
2. Store absolute robot joint targets alongside OSC commands. Use front and side
   84×84 observations and retain proprioception.
3. Add object pose, tabletop texture, background, lighting, and camera variation.
   Make visual augmentation vary within trajectories rather than once per task.
4. Train a compact action-chunking or diffusion policy on absolute joint targets.
   Use episode-level splits and select checkpoints by closed-loop validation
   success rather than one-step MAE.
5. Compare against two controls: the scripted expert and a privileged state policy.
   Do not proceed to physical deployment until the learned policy succeeds on
   held-out robosuite initializations.
6. Only after the simulator policy passes, replace built-in geometry and rendering
   with the calibrated filmed-scene assets. Physical dimensions or reliable camera
   calibration are still required for that step.

## Corrected local experiment result

The prescribed 100-demonstration experiment is complete. It contains 18,071
samples and 370 recovery corrections. The privileged robot-plus-object state
policy succeeds in 8/20 held-out episodes. All tested image policies score zero:
phase-conditioned vision 0/5, Huber OSC chunks 0/5, compact diffusion 0/5,
absolute-joint chunks 0/5, and L1/MSE chunk controls 0/3 each.

This rules out demonstration count, output loss, temporal chunking, phase input,
and action representation as sufficient single fixes. The remaining local
bottleneck is visual state extraction plus closed-loop robustness. Video2Robo's
unavailable training implementation and 3DGS scene variation remain material
reproduction gaps. The full local table is in
[`robosuite-policy-diagnostics.md`](robosuite-policy-diagnostics.md).

## Decision

The baseline 0/10 and expanded image-policy failures do not reproduce
Video2Robo because its released description still lacks an executable training
implementation and the local experiment lacks its 3DGS scene distribution. The
next credible local experiment is a pretrained spatial encoder or explicit
object-keypoint policy, evaluated by held-out closed-loop success. Running the
existing compact CNN on a larger GPU would not address the measured bottleneck.
