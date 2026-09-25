# Physical evaluation runbook

This runbook closes Gate E with a small paired evaluation of the scripted controller
and the frozen learned waypoint policy. The repository can validate calibration,
exercise checkpoint inference, dry-run a safety-limited Cartesian plan, package the
artifacts, and score the recorded trials. Connecting a specific robot requires an
adapter for that robot's vendor SDK which implements `RobotInterface`.

## Frozen acceptance criteria

Use the same robot, cameras, workspace, source object, target bowl, and reset
distribution for both controllers.

- exactly five scripted and five learned-policy trials;
- at least four successful placements for each controller;
- zero safety aborts;
- zero trials above 15 N measured force;
- reprojection RMSE no greater than 2 px for either camera;
- measured table normal within 5 degrees of robot-base +Z;
- source diameter within 10% of 3.832 cm and target dimensions within 10% of
  9.912 cm outer diameter by 5.693 cm height;
- each camera within 2 cm and 5° of its frozen training pose, focal length within 5%,
  and principal point within 2 px;
- the learned checkpoint SHA-256 equals
  `c1b5310537d85734ecef0d427a1a9a13f49d69eb617842802a044ecdcd228125`.

Five trials only establish a smoke-level hardware result. At 4/5 successes, the 95%
Wilson interval is 37.6–96.4%, so the report must retain the interval and avoid a
high-confidence reliability claim.

## 1. Calibrate the physical setup

Install the video dependency used for camera capture and PnP calibration with
`uv sync --extra video`.

Copy `configs/physical_calibration_observations_template.json` and fill every
placeholder from measurements:

1. Record the physical robot serial.
2. Calibrate both 84×84 observation streams and enter their 3×3 intrinsics and
   distortion coefficients. Use explicit zero coefficients only for a verified
   distortion-free or already rectified stream.
3. Record at least three noncollinear correspondences between the frozen robosuite
   task frame and robot-base frame in `task_frame.fit`, plus independent validation
   correspondences. The held-out rigid-alignment RMSE must be at most 5 mm.
4. Record at least six calibration-target robot-frame 3D points and corresponding
   pixels in each camera's `fit`; use separate points in `validation`.
5. Measure the table plane, home pose, target center, source diameter, target outer
   diameter, and target height in robot-base coordinates. The previously supplied
   3 cm source does not match the frozen 3.832 cm training object and cannot be used
   for the final Gate E claim. Printable matched assets and their dimensional audit
   are in [`physical-fixtures.md`](physical-fixtures.md).
6. Set workspace bounds in robot-base coordinates inside the mechanically clear region. Keep them no wider than
   the limits in `configs/physical_safety.yaml` unless the safety review updates both.
7. Estimate the task-to-robot and camera-to-robot transforms and held-out errors:

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_physical_cameras.py \
  --observations path/to/physical-calibration-observations.json \
  --output path/to/measured-physical-calibration.json
```

8. After the operator checks the emergency stop, swept volume, tool, gripper, camera
   mounts, and low-speed mode, set `operator_approved` and
   `emergency_stop_verified` to true.

Do not copy values from the simulation fixture into a physical calibration. The
fixture exists only to exercise the software path.

## 2. Run the software preflight

From the repository root, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_physical_evaluation.py \
  --calibration path/to/measured-physical-calibration.json \
  --safety configs/physical_safety.yaml \
  --camera-reference configs/physical_camera_reference.json \
  --checkpoint results/policy/ball_localization_robust_4000/visual_waypoint.pt \
  --smoke-data results/simulation/ball_localization_robust_4000/demonstrations.npz \
  --output results/deployment/physical
```

Inspect `results/deployment/physical/preflight-report.json`. Proceed only when:

- `status` is `software_preflight_pass`;
- `hardware_ready` is true;
- `arming_blockers` is empty;
- the checkpoint smoke error is at most 0.03 m;
- `training_camera_alignment.valid` is true;
- the checkpoint, calibration, camera-reference, and safety hashes in the deployment
  manifest match the files supplied to the robot process; the manifest also hashes
  the hardware-ready preflight report itself.

The local simulation-fixture preflight is recorded in
`docs/experiments/physical-software-preflight.json`. It passes inference and command
checks but deliberately reports `hardware_ready: false`.

## 3. Connect the robot adapter

Implement `onevideo2policy.deployment.safety.RobotInterface` for the installed robot
SDK:

- `state()` must return measured Cartesian position, external force, timestamp, and
  fault state;
- `command()` must send the requested Cartesian waypoint and gripper value and block
  until the command finishes so the supervisor can check the resulting force/fault state;
- `stop()` must halt motion immediately through the SDK's supported stop path.

Route every command through `SafetySupervisor`. Keep the 1 cm Cartesian step, 8 cm/s
speed, 15 N force, configured workspace, and gripper limits active. Start with the
gripper open and the arm at the measured home pose. The included runtime changes the
gripper only while stationary, checks force after every command, and stops on stale or
unsynchronized camera frames.

The adapter factory must accept `calibration=` and `safety_config=` keyword arguments
and return an object implementing `RobotInterface`. Put its module on `PYTHONPATH`.
Run one learned trial with:

```bash
PYTHONPATH=src:path/to/adapter .venv/bin/python scripts/run_physical_trial.py \
  --deployment results/deployment/physical \
  --robot-factory my_robot_adapter:create_robot \
  --controller learned \
  --agent-camera 0 --front-camera 1 \
  --trial-id learned-1 --reset-id reset-1 --pair-order 2 \
  --output results/evaluation/physical-raw/learned-1.json \
  --execute
```

For a scripted trial, change the controller and supply the independently measured
source center with `--scripted-source-m X Y Z`. After motion, inspect bowl containment
and replace the generated null `success` value with a Boolean before merging the trial
record into the physical results file.

Stop the evaluation immediately for a robot fault, force limit, camera timeout,
calibration movement, unexpected contact, or any person entering the swept volume.
A stopped trial is a safety abort and cannot be rerun under the same trial ID.

## 4. Run paired trials

Prepare five physical reset placements before choosing the controller order. Use each
placement once for the scripted controller and once for the learned controller.
Counterbalance order across placements, for example:

| Placement | First | Second |
|---:|---|---|
| 1 | scripted | learned |
| 2 | learned | scripted |
| 3 | scripted | learned |
| 4 | learned | scripted |
| 5 | scripted | learned |

For each trial record:

- unique `trial_id`;
- shared `reset_id` for the scripted and learned runs at the same placement;
- `pair_order` equal to 1 or 2, counterbalanced across the five reset pairs;
- `controller`: `scripted` or `learned`;
- Boolean placement `success` using the same bowl containment rule;
- Boolean `safety_abort`;
- peak measured `max_force_n`;
- learned-policy localization error when a manual reference is available;
- concise notes for misses or interventions.

After editing each generated record's `success`, assemble all ten records and attach
the verified deployment hashes:

```bash
PYTHONPATH=src .venv/bin/python scripts/assemble_physical_results.py \
  --deployment results/deployment/physical \
  --trial results/evaluation/physical-raw/*.json \
  --output results/evaluation/physical-results.json
```

Retain raw camera recordings and robot logs outside Git, and put their paths or run
IDs in the notes. `configs/physical_results_template.json` documents the assembled
schema.

## 5. Validate Gate E

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_physical_results.py \
  --results path/to/physical-results.json \
  --output results/evaluation/physical-gate
```

The command exits with status 0 only when both controllers pass the frozen threshold
and all required provenance fields exist. Commit the resulting
`physical-gate-report.json` after checking it against the raw robot logs. The passing
file currently tracked at `docs/experiments/physical-gate-validator-fixture.json` is
a synthetic validator fixture, not physical evidence.
