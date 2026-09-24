"""Merge compatible NPZ demonstration or localization datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = [np.load(path, allow_pickle=False) for path in args.inputs]
    try:
        keys = set(loaded[0].files) - {"episode_ends"}
        if any(set(item.files) - {"episode_ends"} != keys for item in loaded[1:]):
            raise ValueError("Input datasets must have matching array keys")
        arrays = {key: np.concatenate([item[key] for item in loaded]) for key in sorted(keys)}
        ends = []
        offset = 0
        for item in loaded:
            ends.append(item["episode_ends"].astype(np.int64) + offset)
            offset += len(item[next(iter(keys))])
        arrays["episode_ends"] = np.concatenate(ends)
    finally:
        for item in loaded:
            item.close()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "demonstrations.npz", **arrays)
    report = {
        "status": "pass",
        "inputs": [str(path) for path in args.inputs],
        "samples": offset,
        "episodes": len(arrays["episode_ends"]),
        "keys": sorted(arrays),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
