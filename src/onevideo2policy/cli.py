from __future__ import annotations

import argparse
import json
from pathlib import Path

from onevideo2policy.config import load_config
from onevideo2policy.video.manifest import validate_manifest
from onevideo2policy.video.point_sampling import sample_mask_points
from onevideo2policy.video.preprocessing import prepare_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ov2p", description="OneVideo2Policy utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate an experiment YAML")
    validate.add_argument("config", type=Path)

    prepare = subparsers.add_parser("prepare-video", help="Extract sampled RGB frames")
    prepare.add_argument("video", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--fps", required=True, type=float)
    prepare.add_argument("--depth-dir", type=Path)

    manifest = subparsers.add_parser("validate-manifest", help="Validate an RGB/RGB-D manifest")
    manifest.add_argument("manifest", type=Path)

    points = subparsers.add_parser("sample-points", help="Sample deterministic points from a mask")
    points.add_argument("mask", type=Path, help="Boolean mask stored as a NumPy .npy file")
    points.add_argument("--count", required=True, type=int)
    points.add_argument("--seed", default=42, type=int)
    points.add_argument("--border", default=0, type=int)
    points.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"Valid configuration: {config['project']['name']}")
    elif args.command == "prepare-video":
        manifest = prepare_video(args.video, args.output, args.fps, args.depth_dir)
        print(json.dumps({"manifest": str(args.output / "manifest.json"), **manifest}, indent=2))
    elif args.command == "validate-manifest":
        print(json.dumps(validate_manifest(args.manifest), indent=2))
    elif args.command == "sample-points":
        import numpy as np

        mask = np.load(args.mask, allow_pickle=False)
        sampled = sample_mask_points(mask, args.count, seed=args.seed, border=args.border)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, sampled)
        print(f"Saved {len(sampled)} points to {args.output}")


if __name__ == "__main__":
    main()
