from __future__ import annotations

import argparse
import json
from pathlib import Path

from onevideo2policy.config import load_config
from onevideo2policy.evaluation.perception_gate import (
    evaluate_annotation_workspace,
    prepare_annotation_workspace,
)
from onevideo2policy.video.cotracker_adapter import CoTracker3Adapter
from onevideo2policy.video.hoi4d import import_hoi4d_rgb_video, import_hoi4d_sequence
from onevideo2policy.video.manifest import validate_manifest
from onevideo2policy.video.perception import (
    load_manifest_rgb,
    load_prompt_file,
    run_perception,
    save_perception_artifacts,
    save_perception_overlay,
    save_perception_report,
    summarize_perception,
)
from onevideo2policy.video.point_sampling import sample_mask_points
from onevideo2policy.video.preprocessing import prepare_video
from onevideo2policy.video.sam2_adapter import Sam2VideoAdapter


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

    perception = subparsers.add_parser(
        "run-perception", help="Run SAM2 followed by deterministic CoTracker3 queries"
    )
    perception.add_argument("manifest", type=Path)
    perception.add_argument("--prompts", required=True, type=Path)
    perception.add_argument("--output", required=True, type=Path)
    perception.add_argument("--sam-model", default="facebook/sam2.1-hiera-small")
    perception.add_argument("--cotracker-checkpoint", required=True, type=Path)
    perception.add_argument("--device", default="cuda")
    perception.add_argument("--points", default=32, type=int)
    perception.add_argument("--seed", default=42, type=int)
    perception.add_argument("--border", default=4, type=int)

    annotation = subparsers.add_parser(
        "prepare-perception-gate", help="Create a prediction-independent mask workspace"
    )
    annotation.add_argument("manifest", type=Path)
    annotation.add_argument("--spec", required=True, type=Path)
    annotation.add_argument("--output", required=True, type=Path)

    gate = subparsers.add_parser(
        "evaluate-perception-gate", help="Score complete manual masks against predictions"
    )
    gate.add_argument("--workspace", required=True, type=Path)
    gate.add_argument("--predictions", required=True, type=Path)
    gate.add_argument("--diagnostics", required=True, type=Path)
    gate.add_argument("--config", required=True, type=Path)
    gate.add_argument("--identity-swaps", required=True, type=int)
    gate.add_argument("--output", required=True, type=Path)

    hoi4d = subparsers.add_parser(
        "import-hoi4d", help="Import decoded HOI4D RGB-D and motion masks"
    )
    hoi4d.add_argument("sequence", type=Path, help="Decoded HOI4D sequence directory")
    hoi4d.add_argument(
        "--annotations", type=Path, help="Separate official annotation sequence directory"
    )
    hoi4d.add_argument("--output", required=True, type=Path)
    hoi4d.add_argument("--source-label", required=True, action="append", type=int)
    hoi4d.add_argument("--target-label", required=True, action="append", type=int)
    hoi4d.add_argument("--fps", default=15.0, type=float, help="Official decoded frame rate")
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
    elif args.command == "prepare-perception-gate":
        workspace = prepare_annotation_workspace(args.manifest, args.spec, args.output)
        print(
            json.dumps(
                {
                    "workspace": str(args.output / "workspace.json"),
                    "frames": len(workspace["frames"]),
                    "objects": workspace["objects"],
                },
                indent=2,
            )
        )
    elif args.command == "evaluate-perception-gate":
        config = load_config(args.config)
        report = evaluate_annotation_workspace(
            args.workspace,
            args.predictions,
            args.diagnostics,
            min_mask_iou=float(config["gates"]["min_mask_iou"]),
            min_track_survival=float(config["gates"]["min_track_survival"]),
            min_visible_points=int(config["tracking"]["min_visible_points"]),
            identity_swaps=args.identity_swaps,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "import-hoi4d":
        if args.annotations is not None:
            manifest = import_hoi4d_rgb_video(
                args.sequence,
                args.annotations,
                args.output,
                source_labels=args.source_label,
                target_labels=args.target_label,
            )
        else:
            manifest = import_hoi4d_sequence(
                args.sequence,
                args.output,
                source_labels=args.source_label,
                target_labels=args.target_label,
                decoded_fps=args.fps,
            )
        print(json.dumps({"manifest": str(args.output / "manifest.json"), **manifest}, indent=2))
    elif args.command == "run-perception":
        if not args.cotracker_checkpoint.is_file():
            raise FileNotFoundError(args.cotracker_checkpoint)
        frames = load_manifest_rgb(args.manifest)
        prompts = load_prompt_file(args.prompts)
        segmenter = Sam2VideoAdapter.from_hugging_face(args.sam_model, device=args.device)
        tracker = CoTracker3Adapter.from_checkpoint(
            args.cotracker_checkpoint, device=args.device
        )
        results = run_perception(
            frames,
            args.manifest.parent / "frames",
            prompts,
            segmenter,
            tracker,
            point_count=args.points,
            seed=args.seed,
            border=args.border,
        )
        save_perception_artifacts(results, args.output)
        with args.manifest.open(encoding="utf-8") as stream:
            sample_fps = float(json.load(stream)["sample_fps"])
        save_perception_overlay(
            frames, results, args.output / "overlays" / "perception.mp4", fps=sample_fps
        )
        report = summarize_perception(results)
        save_perception_report(report, args.output / "gate-report.json")
        summary = {
            name: {
                "frames": len(result.masks),
                "points": len(result.seed_points_xy),
                "track_shape": list(result.tracks_xy.shape),
            }
            for name, result in results.items()
        }
        summary["report"] = report
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
