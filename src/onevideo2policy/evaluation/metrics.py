from __future__ import annotations

from collections.abc import Iterable


def valid_episode_rate(validity: Iterable[bool]) -> float:
    values = list(validity)
    if not values:
        raise ValueError("At least one episode is required")
    return sum(values) / len(values)


def success_rate(successes: Iterable[bool]) -> float:
    return valid_episode_rate(successes)
