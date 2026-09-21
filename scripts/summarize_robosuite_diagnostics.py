"""Combine collection, training, and closed-loop diagnostic reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collection = read(args.collection)
    training = read(args.training)
    evaluations = []
    for path in args.evaluation:
        report = read(path)
        evaluations.append(
            {
                "path": str(path),
                "policy_kind": report["policy_kind"],
                "episodes": report["episodes"],
                "successes": report["successes"],
                "success_rate": report["success_rate"],
                "training_initial_positions": report["training_initial_positions"],
                "terminal_phase_counts": report["terminal_phase_counts"],
            }
        )
    best = max(evaluations, key=lambda item: item["success_rate"], default=None)
    report = {
        "status": "complete" if evaluations else "training_only",
        "collection": {
            key: collection.get(key)
            for key in (
                "successful_episodes",
                "attempts",
                "samples",
                "recovery_samples",
                "cameras",
                "stored_action_representations",
            )
        },
        "training": training,
        "evaluations": evaluations,
        "best_closed_loop": best,
        "decision": (
            "pass learned-policy gate"
            if best and best["success_rate"] >= 0.8
            else "fail learned-policy gate; retain scripted expert"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Robosuite policy diagnostic summary",
        "",
        f"Decision: **{report['decision']}**.",
        "",
        "| Policy | Successes | Episodes | Success rate | Training starts |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in evaluations:
        lines.append(
            f"| {item['policy_kind']} | {item['successes']} | {item['episodes']} | "
            f"{item['success_rate']:.1%} | {item['training_initial_positions']} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
