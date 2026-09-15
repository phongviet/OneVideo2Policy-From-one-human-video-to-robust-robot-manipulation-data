from __future__ import annotations

import argparse
import json
from pathlib import Path

from onevideo2policy.config import load_config
from onevideo2policy.evaluation.perception_gate import (
    evaluate_annotation_workspace,
    evaluate_ground_truth_directories,
    prepare_annotation_workspace,
)
from onevideo2policy.pipeline import prepare_faithful_bundle, run_local_end_to_end
from onevideo2policy.reconstruction.crops import (
    export_rgba_crops,
    load_video_frame,
    resize_masks_to_frame,
)
from onevideo2policy.video.cotracker_adapter import CoTracker3Adapter
from onevideo2policy.video.hoi4d import import_hoi4d_rgb_video, import_hoi4d_sequence
from onevideo2policy.video.manifest import validate_manifest
from onevideo2policy.video.perception import (
    load_manifest_rgb,
    load_mask_directories,
    load_prompt_file,
    run_perception,
    run_tracking_on_masks,
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
    perception.add_argument(
        "--reseed-interval",
        type=int,
        help="Deterministically refresh mask points every N frames",
    )

    retrack = subparsers.add_parser(
        "retrack-perception", help="Run CoTracker3 on an exact frozen set of mask PNGs"
    )
    retrack.add_argument("manifest", type=Path)
    retrack.add_argument("--masks", required=True, type=Path)
    retrack.add_argument("--prompts", required=True, type=Path)
    retrack.add_argument("--output", required=True, type=Path)
    retrack.add_argument("--cotracker-checkpoint", required=True, type=Path)
    retrack.add_argument("--device", default="cuda")
    retrack.add_argument("--points", default=32, type=int)
    retrack.add_argument("--seed", default=42, type=int)
    retrack.add_argument("--border", default=4, type=int)
    retrack.add_argument("--reseed-interval", required=True, type=int)

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

    dataset_gate = subparsers.add_parser(
        "evaluate-dataset-gate", help="Score predictions against dataset-provided masks"
    )
    dataset_gate.add_argument("--ground-truth", required=True, type=Path)
    dataset_gate.add_argument("--predictions", required=True, type=Path)
    dataset_gate.add_argument("--diagnostics", required=True, type=Path)
    dataset_gate.add_argument("--config", required=True, type=Path)
    dataset_gate.add_argument("--identity-swaps", required=True, type=int)
    dataset_gate.add_argument("--output", required=True, type=Path)

    crops = subparsers.add_parser(
        "export-rgba-crops", help="Export square transparent crops from frozen masks"
    )
    crops.add_argument("manifest", type=Path)
    crops.add_argument("--masks", required=True, type=Path)
    crops.add_argument("--output", required=True, type=Path)
    crops.add_argument("--frame", default=0, type=int)
    crops.add_argument("--padding", default=0.15, type=float)
    crops.add_argument(
        "--source-video",
        type=Path,
        help="Use the manifest's original source-frame ID at native video resolution",
    )

    local_e2e = subparsers.add_parser(
        "run-local-e2e", help="Run the CPU-safe measured-primitive Place baseline"
    )
    local_e2e.add_argument("--config", required=True, type=Path)
    local_e2e.add_argument("--manifest", required=True, type=Path)
    local_e2e.add_argument("--masks", required=True, type=Path)
    local_e2e.add_argument("--output", required=True, type=Path)

    faithful = subparsers.add_parser(
        "prepare-faithful-run", help="Bundle hashed TRELLIS/VGGT inputs for a GPU host"
    )
    faithful.add_argument("--config", required=True, type=Path)
    faithful.add_argument("--manifest", required=True, type=Path)
    faithful.add_argument("--crops", required=True, type=Path)
    faithful.add_argument("--output", required=True, type=Path)

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
    hoi4d.add_argument("--max-width", type=int, help="Resize archive-native RGB and masks")
    hoi4d.add_argument("--start-frame", default=0, type=int)
    hoi4d.add_argument("--end-frame", type=int)
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
    elif args.command == "evaluate-dataset-gate":
        config = load_config(args.config)
        report = evaluate_ground_truth_directories(
            args.ground_truth,
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
    elif args.command == "export-rgba-crops":
        frames = load_manifest_rgb(args.manifest)
        if not 0 <= args.frame < len(frames):
            raise ValueError("Crop frame is outside the manifest")
        object_names = sorted(path.name for path in args.masks.iterdir() if path.is_dir())
        masks = load_mask_directories(args.masks, object_names, expected_frame_count=len(frames))
        rgb = frames[args.frame]
        frame_masks = {name: values[args.frame] for name, values in masks.items()}
        source_frame_id = None
        if args.source_video is not None:
            with args.manifest.open(encoding="utf-8") as stream:
                source_frame_id = int(json.load(stream)["frames"][args.frame]["source_frame_id"])
            rgb = load_video_frame(args.source_video, source_frame_id)
            frame_masks = resize_masks_to_frame(frame_masks, rgb.shape[:2])
        manifest = export_rgba_crops(
            rgb,
            frame_masks,
            args.output,
            frame_idx=args.frame,
            padding_fraction=args.padding,
        )
        if source_frame_id is not None:
            manifest["source_frame_id"] = source_frame_id
            (args.output / "crops.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(manifest, indent=2))
    elif args.command == "run-local-e2e":
        config = load_config(args.config)
        with args.manifest.open(encoding="utf-8") as stream:
            frame_count = len(json.load(stream)["frames"])
        masks = load_mask_directories(
            args.masks, ["source", "target"], expected_frame_count=frame_count
        )
        report = run_local_end_to_end(masks["source"], masks["target"], args.output, config=config)
        print(json.dumps(report, indent=2))
    elif args.command == "prepare-faithful-run":
        config = load_config(args.config)
        spec = prepare_faithful_bundle(args.manifest, args.crops, args.output, config=config)
        print(json.dumps(spec, indent=2))
    elif args.command == "import-hoi4d":
        if args.annotations is not None:
            manifest = import_hoi4d_rgb_video(
                args.sequence,
                args.annotations,
                args.output,
                source_labels=args.source_label,
                target_labels=args.target_label,
                sample_fps=args.fps,
                max_width=args.max_width,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
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
    elif args.command in {"run-perception", "retrack-perception"}:
        if not args.cotracker_checkpoint.is_file():
            raise FileNotFoundError(args.cotracker_checkpoint)
        frames = load_manifest_rgb(args.manifest)
        prompts = load_prompt_file(args.prompts)
        tracker = CoTracker3Adapter.from_checkpoint(args.cotracker_checkpoint, device=args.device)
        if args.command == "run-perception":
            segmenter = Sam2VideoAdapter.from_hugging_face(args.sam_model, device=args.device)
            results = run_perception(
                frames,
                args.manifest.parent / "frames",
                prompts,
                segmenter,
                tracker,
                point_count=args.points,
                seed=args.seed,
                border=args.border,
                reseed_interval=args.reseed_interval,
            )
        else:
            masks = load_mask_directories(
                args.masks, list(prompts), expected_frame_count=len(frames)
            )
            results = run_tracking_on_masks(
                frames,
                masks,
                prompts,
                tracker,
                point_count=args.points,
                seed=args.seed,
                border=args.border,
                reseed_interval=args.reseed_interval,
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
                "points_per_window": int(result.seed_points_xy.shape[-2]),
                "reseed_frames": (
                    result.reseed_frames.tolist() if result.reseed_frames is not None else []
                ),
                "track_shape": list(result.tracks_xy.shape),
            }
            for name, result in results.items()
        }
        summary["report"] = report
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
