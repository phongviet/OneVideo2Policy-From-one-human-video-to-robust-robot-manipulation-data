"""Load and verify a hardware-ready deployment package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from onevideo2policy.deployment.results import FROZEN_CHECKPOINT_SHA256
from onevideo2policy.deployment.safety import SafetyEnvelope


@dataclass(frozen=True)
class DeploymentPackage:
    root: Path
    calibration: dict[str, Any]
    safety_config: dict[str, Any]
    envelope: SafetyEnvelope
    checkpoint_path: Path


def load_deployment_package(root: str | Path) -> DeploymentPackage:
    """Verify hashes and arming state before exposing deployment artifacts."""
    root = Path(root)
    manifest = json.loads((root / "deployment-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("preflight_report") != "preflight-report.json":
        raise RuntimeError("manifest has an invalid preflight report path")
    report = json.loads((root / "preflight-report.json").read_text(encoding="utf-8"))
    errors = []
    required = {
        "visual_waypoint.pt",
        "calibration.json",
        "safety.yaml",
        "camera-reference.json",
        "preflight-report.json",
    }
    missing = required - set(manifest.get("files", {}))
    if missing:
        errors.append("manifest omits required files: " + ", ".join(sorted(missing)))
    for name, expected in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"artifact hash mismatch: {name}")
    if report.get("status") != "software_preflight_pass":
        errors.append("software preflight did not pass")
    if report.get("hardware_ready") is not True:
        errors.append("deployment package is not hardware-ready")
    if report.get("arming_blockers"):
        errors.append("deployment package has arming blockers")
    checkpoint_path = root / "visual_waypoint.pt"
    if sha256(checkpoint_path) != FROZEN_CHECKPOINT_SHA256:
        errors.append("checkpoint does not match the frozen policy")
    if errors:
        raise RuntimeError("; ".join(errors))

    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    safety_config = yaml.safe_load((root / "safety.yaml").read_text(encoding="utf-8"))
    envelope = SafetyEnvelope(
        workspace_bounds_m=np.asarray(safety_config["workspace_bounds_m"], dtype=np.float64),
        max_cartesian_step_m=float(safety_config["max_cartesian_step_m"]),
        max_cartesian_speed_m_s=float(safety_config["max_cartesian_speed_m_s"]),
        max_force_n=float(safety_config["max_force_n"]),
        gripper_range=tuple(safety_config["gripper_range"]),
    )
    envelope.validate()
    return DeploymentPackage(root, calibration, safety_config, envelope, checkpoint_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
