"""Generate dimension-checked printable meshes for the frozen physical task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onevideo2policy.deployment.physical_assets import (
    BALL_DIAMETER_M,
    BOWL_HEIGHT_M,
    BOWL_OUTER_DIAMETER_M,
    BOWL_WALL_M,
    audit_binary_stl,
    bowl_triangles,
    sha256,
    sphere_triangles,
    write_binary_stl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/physical"))
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--rings", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.segments < 16 or args.rings < 8 or args.segments % 4:
        raise ValueError("segments must be a multiple of four >=16 and rings must be >=8")
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "source-ball-38.321mm.stl": sphere_triangles(
            BALL_DIAMETER_M / 2, args.segments, args.rings
        ),
        "target-bowl-99.123x56.933mm.stl": bowl_triangles(
            BOWL_OUTER_DIAMETER_M / 2, BOWL_HEIGHT_M, BOWL_WALL_M, args.segments
        ),
    }
    report = {
        "schema_version": 1,
        "units": "millimeters in STL coordinates",
        "source": {"diameter_m": BALL_DIAMETER_M, "color": "matte orange"},
        "target": {
            "outer_diameter_m": BOWL_OUTER_DIAMETER_M,
            "height_m": BOWL_HEIGHT_M,
            "wall_m": BOWL_WALL_M,
            "color": "matte blue",
        },
        "files": {},
        "usage": "Printable geometry fixture; verify final printed dimensions before Gate E.",
    }
    for name, triangles_m in outputs.items():
        path = args.output / name
        write_binary_stl(path, triangles_m * 1000)
        audit = audit_binary_stl(path)
        if not audit["watertight"]:
            raise RuntimeError(f"generated mesh is not watertight: {name}")
        report["files"][name] = {"sha256": sha256(path), **audit}
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
