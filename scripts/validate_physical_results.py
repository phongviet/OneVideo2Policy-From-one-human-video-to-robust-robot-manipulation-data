"""Validate paired scripted and learned physical trials against the frozen Gate E."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onevideo2policy.deployment.results import validate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials-per-controller", default=5, type=int)
    parser.add_argument("--min-success-rate", default=0.8, type=float)
    parser.add_argument("--max-force-n", default=15.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(args.results.read_text(encoding="utf-8"))
    report = validate_results(
        raw,
        trials_per_controller=args.trials_per_controller,
        min_success_rate=args.min_success_rate,
        max_force_n=args.max_force_n,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "physical-gate-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
