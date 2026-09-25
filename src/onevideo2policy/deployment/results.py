"""Machine-checkable acceptance gate for paired physical robot trials."""

from __future__ import annotations

import math
from typing import Any

FROZEN_CHECKPOINT_SHA256 = (
    "c1b5310537d85734ecef0d427a1a9a13f49d69eb617842802a044ecdcd228125"
)


def validate_results(
    raw: dict[str, Any],
    *,
    trials_per_controller: int,
    min_success_rate: float,
    max_force_n: float,
) -> dict[str, Any]:
    """Validate paired scripted and learned trials against the physical gate."""
    errors = []
    if not raw.get("robot_serial") or str(raw["robot_serial"]).startswith("REPLACE_"):
        errors.append("physical robot serial is missing")
    for key in ("calibration_sha256", "checkpoint_sha256"):
        value = str(raw.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            errors.append(f"{key} must be a SHA-256 digest")
    if raw.get("checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256:
        errors.append("checkpoint_sha256 does not match the frozen policy")

    trials = raw.get("trials", [])
    if not isinstance(trials, list):
        errors.append("trials must be a list")
        trials = []
    invalid_entries = [index for index, item in enumerate(trials) if not isinstance(item, dict)]
    if invalid_entries:
        errors.append(f"trial entries must be objects: {invalid_entries}")
        trials = [item for item in trials if isinstance(item, dict)]
    ids = [item.get("trial_id") for item in trials]
    if any(not isinstance(trial_id, str) or not trial_id for trial_id in ids):
        errors.append("every trial requires a nonempty string trial_id")
    valid_ids = [trial_id for trial_id in ids if isinstance(trial_id, str) and trial_id]
    if len(valid_ids) != len(set(valid_ids)):
        errors.append("trial IDs must be unique")

    summaries = {}
    for controller in ("scripted", "learned"):
        selected = [item for item in trials if item.get("controller") == controller]
        if len(selected) != trials_per_controller:
            errors.append(
                f"{controller} requires {trials_per_controller} trials; found {len(selected)}"
            )
        malformed = [
            item.get("trial_id")
            for item in selected
            if type(item.get("success")) is not bool
            or type(item.get("safety_abort")) is not bool
            or not math.isfinite(_force(item.get("max_force_n")))
        ]
        if malformed:
            errors.append(f"{controller} trials have invalid required fields: {malformed}")
        successes = sum(item.get("success") is True for item in selected)
        safety_aborts = sum(item.get("safety_abort") is True for item in selected)
        force_violations = sum(_force(item.get("max_force_n")) > max_force_n for item in selected)
        rate = successes / len(selected) if selected else 0.0
        summaries[controller] = {
            "trials": len(selected),
            "successes": successes,
            "success_rate": rate,
            "wilson_95": wilson_interval(successes, len(selected)),
            "safety_aborts": safety_aborts,
            "force_violations": force_violations,
            "passes": bool(
                len(selected) == trials_per_controller
                and rate >= min_success_rate
                and safety_aborts == 0
                and force_violations == 0
            ),
        }

    unknown = sorted(
        {str(item.get("controller")) for item in trials} - {"scripted", "learned"}
    )
    if unknown:
        errors.append("unknown controllers: " + ", ".join(unknown))
    passed = not errors and all(summary["passes"] for summary in summaries.values())
    return {
        "schema_version": 1,
        "status": "pass" if passed else "incomplete_or_failed",
        "gate": {
            "trials_per_controller": trials_per_controller,
            "min_success_rate": min_success_rate,
            "max_force_n": max_force_n,
            "zero_safety_aborts": True,
        },
        "controllers": summaries,
        "errors": errors,
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    """Return a two-sided Wilson score interval for a Bernoulli success rate."""
    if trials == 0:
        return None
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    radius = z / denominator * math.sqrt(
        rate * (1 - rate) / trials + z**2 / (4 * trials**2)
    )
    return [center - radius, center + radius]


def _force(value: Any) -> float:
    try:
        force = float(value)
    except (TypeError, ValueError):
        return math.inf
    return force
