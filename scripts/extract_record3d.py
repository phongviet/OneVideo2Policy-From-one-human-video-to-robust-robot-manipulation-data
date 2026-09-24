#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from onevideo2policy.video.record3d import Record3DArchive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract reproducible RGB-D frames from Record3D")
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install the video extra: uv sync --extra video") from exc
    args = parse_args()
    source = Record3DArchive(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    for frame_id in args.frames:
        rgb = source.read_rgb(frame_id)
        depth = source.read_depth_m(frame_id)
        cv2.imwrite(
            str(args.output / f"rgb_{frame_id:06d}.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        np.save(args.output / f"depth_m_{frame_id:06d}.npy", depth)
    metadata = {
        "source": str(args.archive),
        "frames": args.frames,
        "fps": source.metadata.fps,
        "rgb_size": source.metadata.rgb_size,
        "depth_size": source.metadata.depth_size,
        "intrinsics_rgb": source.metadata.intrinsics_rgb.tolist(),
        "intrinsics_depth": source.intrinsics_for_depth().tolist(),
        "poses_raw": source.metadata.poses[args.frames].tolist(),
        "depth_unit": "metres",
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
