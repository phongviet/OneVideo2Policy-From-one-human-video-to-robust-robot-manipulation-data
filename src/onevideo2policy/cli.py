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
from onevideo2policy.generation.gaussian_scene import (
    initialize_gaussians_from_rgbd,
    render_gaussians,
    transform_label,
)
from onevideo2policy.pipeline import prepare_model_bundle, run_local_end_to_end
from onevideo2policy.pose_tracking.rgbd_odometry import estimate_rgbd_trajectory
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
    prepare.add_argument("--max-width", type=int, help="Downsample wide RGB video for inference")

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

    model_run = subparsers.add_parser(
        "prepare-model-run", help="Bundle inputs for the selected local reconstruction models"
    )
    model_run.add_argument("--config", required=True, type=Path)
    model_run.add_argument("--manifest", required=True, type=Path)
    model_run.add_argument("--crops", required=True, type=Path)
    model_run.add_argument("--output", required=True, type=Path)

    gaussian_scene = subparsers.add_parser(
        "build-gaussian-scene", help="Initialize and render a metric 3D Gaussian scene from RGB-D"
    )
    gaussian_scene.add_argument("--rgb", required=True, type=Path)
    gaussian_scene.add_argument("--depth", required=True, type=Path)
    gaussian_scene.add_argument("--camera-info", required=True, type=Path)
    gaussian_scene.add_argument("--source-mask", type=Path)
    gaussian_scene.add_argument("--target-mask", type=Path)
    gaussian_scene.add_argument("--output", required=True, type=Path)
    gaussian_scene.add_argument("--max-width", default=640, type=int)
    gaussian_scene.add_argument("--stride", default=3, type=int)

    rgbd_trajectory = subparsers.add_parser(
        "estimate-rgbd-trajectory", help="Estimate a metric camera trajectory with RGB-D PnP"
    )
    rgbd_trajectory.add_argument("--manifest", required=True, type=Path)
    rgbd_trajectory.add_argument("--depth-dir", required=True, type=Path)
    rgbd_trajectory.add_argument("--camera-info", required=True, type=Path)
    rgbd_trajectory.add_argument("--masks", required=True, type=Path)
    rgbd_trajectory.add_argument("--output", required=True, type=Path)

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
        manifest = prepare_video(args.video, args.output, args.fps, args.depth_dir, args.max_width)
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
        frames = (
            load_manifest_rgb(args.manifest)
            if config["paths"]["local"].get("camera_compensation") == "background_affine"
            else None
        )
        report = run_local_end_to_end(
            masks["source"], masks["target"], args.output, config=config, frames=frames
        )
        print(json.dumps(report, indent=2))
    elif args.command == "prepare-model-run":
        config = load_config(args.config)
        spec = prepare_model_bundle(args.manifest, args.crops, args.output, config=config)
        print(json.dumps(spec, indent=2))
    elif args.command == "build-gaussian-scene":
        import cv2
        import numpy as np

        rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_mm is None:
            raise FileNotFoundError("RGB or depth input could not be decoded")
        if rgb_bgr.shape[:2] != depth_mm.shape:
            raise ValueError("RGB and depth dimensions must match")
        with args.camera_info.open(encoding="utf-8") as stream:
            camera_info = json.load(stream)
        calibration = camera_info["crop_intrinsic"]
        intrinsics = np.array(
            [
                [calibration["fx"], 0, calibration["cx"]],
                [0, calibration["fy"], calibration["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        original_height, original_width = depth_mm.shape
        scale = min(1.0, args.max_width / original_width)
        width, height = round(original_width * scale), round(original_height * scale)
        rgb_bgr = cv2.resize(rgb_bgr, (width, height), interpolation=cv2.INTER_AREA)
        depth_m = cv2.resize(
            depth_mm.astype(np.float32) / 1000.0,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        intrinsics[:2] *= scale

        def load_mask(path: Path | None) -> np.ndarray | None:
            if path is None:
                return None
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(path)
            return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0

        scene = initialize_gaussians_from_rgbd(
            cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            depth_m,
            intrinsics,
            source_mask=load_mask(args.source_mask),
            target_mask=load_mask(args.target_mask),
            stride=args.stride,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        scene.save(args.output / "scene.npz")
        rendered, rendered_depth, alpha = render_gaussians(
            scene, intrinsics, np.eye(4), width=width, height=height
        )
        cv2.imwrite(
            str(args.output / "reference-render.png"),
            cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(args.output / "alpha.png"), (alpha * 255).astype(np.uint8))
        valid_depth = rendered_depth > 0
        depth_visual = np.zeros_like(rendered_depth, dtype=np.uint8)
        if np.any(valid_depth):
            low, high = np.percentile(rendered_depth[valid_depth], [2, 98])
            depth_visual[valid_depth] = np.clip(
                (rendered_depth[valid_depth] - low) / max(high - low, 1e-6) * 255, 0, 255
            )
        cv2.imwrite(str(args.output / "depth.png"), depth_visual)
        novel_camera = np.eye(4)
        novel_camera[0, 3] = -0.03
        novel_render, _, _ = render_gaussians(
            scene, intrinsics, novel_camera, width=width, height=height
        )
        cv2.imwrite(
            str(args.output / "novel-view-3cm.png"),
            cv2.cvtColor((novel_render * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        source_transform = np.eye(4)
        source_transform[0, 3] = 0.05
        moved_scene = transform_label(scene, 1, source_transform)
        moved_render, _, _ = render_gaussians(
            moved_scene, intrinsics, np.eye(4), width=width, height=height
        )
        cv2.imwrite(
            str(args.output / "source-shift-5cm.png"),
            cv2.cvtColor((moved_render * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        reference = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        evaluated = alpha > 0.5
        mse = float(np.mean((rendered[evaluated] - reference[evaluated]) ** 2))
        report = {
            "schema_version": 1,
            "representation": "metric RGB-D initialized anisotropic 3D Gaussians",
            "gaussian_count": len(scene.means_m),
            "resolution": {"width": width, "height": height},
            "stride": args.stride,
            "label_counts": {
                "static": int(np.count_nonzero(scene.labels == 0)),
                "source": int(np.count_nonzero(scene.labels == 1)),
                "target": int(np.count_nonzero(scene.labels == 2)),
            },
            "reference_render": {
                "coverage_alpha_gt_0_5": float(np.mean(evaluated)),
                "mse": mse,
                "psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
            },
            "verification_renders": {
                "novel_camera_translation_m": [-0.03, 0.0, 0.0],
                "source_translation_m": [0.05, 0.0, 0.0],
            },
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    elif args.command == "estimate-rgbd-trajectory":
        import cv2
        import numpy as np

        with args.manifest.open(encoding="utf-8") as stream:
            manifest_data = json.load(stream)
        with args.camera_info.open(encoding="utf-8") as stream:
            camera_info = json.load(stream)
        width = int(manifest_data["resolution"]["width"])
        height = int(manifest_data["resolution"]["height"])
        calibration = camera_info["crop_intrinsic"]
        first_source_id = int(manifest_data["frames"][0]["source_frame_id"])
        first_raw_depth = cv2.imread(
            str(args.depth_dir / f"{first_source_id:05d}.png"), cv2.IMREAD_UNCHANGED
        )
        if first_raw_depth is None:
            raise FileNotFoundError("first sensor depth frame")
        scale_x, scale_y = width / first_raw_depth.shape[1], height / first_raw_depth.shape[0]
        intrinsics = np.array(
            [
                [calibration["fx"] * scale_x, 0, calibration["cx"] * scale_x],
                [0, calibration["fy"] * scale_y, calibration["cy"] * scale_y],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        rgb_frames = []
        depth_frames = []
        exclusions = []
        source_frame_ids = []
        for frame in manifest_data["frames"]:
            rgb_path = Path(frame["rgb"])
            if not rgb_path.is_absolute():
                rgb_path = args.manifest.parent / rgb_path
            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(rgb_path)
            source_id = int(frame["source_frame_id"])
            raw_depth = cv2.imread(
                str(args.depth_dir / f"{source_id:05d}.png"), cv2.IMREAD_UNCHANGED
            )
            if raw_depth is None:
                raise FileNotFoundError(args.depth_dir / f"{source_id:05d}.png")
            frame_id = int(frame["frame_id"])
            masks = []
            for name in ("source", "target"):
                mask = cv2.imread(
                    str(args.masks / name / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE
                )
                if mask is None:
                    raise FileNotFoundError(args.masks / name / f"{frame_id:06d}.png")
                masks.append(mask > 0)
            rgb_frames.append(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
            depth_frames.append(
                cv2.resize(
                    raw_depth.astype(np.float32) / 1000.0,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            exclusions.append(masks[0] | masks[1])
            source_frame_ids.append(source_id)
        world_to_camera, steps = estimate_rgbd_trajectory(
            rgb_frames, depth_frames, intrinsics, exclusion_masks=exclusions
        )
        camera_to_world = np.linalg.inv(world_to_camera)
        positions = camera_to_world[:, :3, 3]
        step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output / "camera-trajectory.npz",
            schema_version=np.asarray(1),
            world_to_camera=world_to_camera,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            source_frame_ids=np.asarray(source_frame_ids),
        )
        report = {
            "schema_version": 1,
            "method": "ORB RGB-D PnP with dynamic object mask exclusion",
            "units": "metres",
            "frame_count": len(rgb_frames),
            "path_length_m": float(np.sum(step_distances)),
            "start_to_end_m": float(np.linalg.norm(positions[-1] - positions[0])),
            "median_step_translation_m": float(np.median(step_distances)),
            "median_inlier_ratio": float(np.median([step["inlier_ratio"] for step in steps])),
            "median_reprojection_error_px": float(
                np.median([step["median_reprojection_error_px"] for step in steps])
            ),
            "median_depth_residual_m": float(
                np.median([step["median_depth_residual_m"] for step in steps])
            ),
            "steps": steps,
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in report.items() if key != "steps"}, indent=2))
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
