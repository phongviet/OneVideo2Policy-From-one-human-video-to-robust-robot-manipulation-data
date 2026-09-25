"""Assemble individual physical trial records with deployment provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onevideo2policy.deployment.package import load_deployment_package, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--trial", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = load_deployment_package(args.deployment)
    trials = [json.loads(path.read_text(encoding="utf-8")) for path in args.trial]
    result = {
        "schema_version": 1,
        "robot_serial": package.calibration["robot_serial"],
        "deployment_manifest_sha256": sha256(package.root / "deployment-manifest.json"),
        "calibration_sha256": sha256(package.root / "calibration.json"),
        "checkpoint_sha256": sha256(package.checkpoint_path),
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
