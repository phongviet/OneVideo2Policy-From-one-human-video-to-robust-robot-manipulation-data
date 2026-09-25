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
- the learned checkpoint SHA-256 equals
  `c1b5310537d85734ecef0d427a1a9a13f49d69eb617842802a044ecdcd228125`.

Five trials only establish a smoke-level hardware result. At 4/5 successes, the 95%
Wilson interval is 37.6–96.4%, so the report must retain the interval and avoid a
high-confidence reliability claim.

## 1. Calibrate the physical setup

Copy `configs/physical_calibration_template.json` and fill every placeholder from
measurements:

1. Record the physical robot serial.
2. Calibrate both 84×84 observation streams and enter their 3×3 intrinsics.
3. Estimate each camera-to-robot-base 4×4 rigid transform using a calibration target
   at multiple robot poses.
4. Measure reprojection RMSE on held-out target observations.
5. Measure the table plane in robot coordinates and enter its unit normal and height.
6. Set workspace bounds inside the mechanically clear region. Keep them no wider than
   the limits in `configs/physical_safety.yaml` unless the safety review updates both.
7. After the operator checks the emergency stop, swept volume, tool, gripper, camera
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
  --checkpoint results/policy/ball_localization_robust_4000/visual_waypoint.pt \
  --smoke-data results/simulation/ball_localization_robust_4000/demonstrations.npz \
  --output results/deployment/physical
```

Inspect `results/deployment/physical/preflight-report.json`. Proceed only when:

- `status` is `software_preflight_pass`;
- `hardware_ready` is true;
- `arming_blockers` is empty;
- the checkpoint smoke error is at most 0.03 m;
- the checkpoint, calibration, and safety hashes in the deployment manifest match
  the files supplied to the robot process.

The local simulation-fixture preflight is recorded in
`docs/experiments/physical-software-preflight.json`. It passes inference and command
checks but deliberately reports `hardware_ready: false`.

## 3. Connect the robot adapter

Implement `onevideo2policy.deployment.safety.RobotInterface` for the installed robot
SDK:

- `state()` must return measured Cartesian position, external force, timestamp, and
  fault state;
- `command()` must send the requested Cartesian waypoint and gripper value;
- `stop()` must halt motion immediately through the SDK's supported stop path.

Route every command through `SafetySupervisor`. Keep the 1 cm Cartesian step, 8 cm/s
speed, 15 N force, configured workspace, and gripper limits active. Start with the
gripper open and the arm at the measured home pose.

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
- `controller`: `scripted` or `learned`;
- Boolean placement `success` using the same bowl containment rule;
- Boolean `safety_abort`;
- peak measured `max_force_n`;
- learned-policy localization error when a manual reference is available;
- concise notes for misses or interventions.

Store these fields in a copy of `configs/physical_results_template.json`. Retain raw
camera recordings and robot logs outside Git, and put their paths or run IDs in the
notes.

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
