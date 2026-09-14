from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from onevideo2policy.video.manifest import validate_manifest


def load_evaluation_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen frame/object selection for a perception gate."""
    with Path(path).open(encoding="utf-8") as stream:
        spec = json.load(stream)
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Evaluation spec must use schema_version 1")
    objects = spec.get("objects")
    frame_ids = spec.get("frame_ids")
    if (
        not isinstance(objects, list)
        or not objects
        or any(not isinstance(name, str) or not name for name in objects)
        or len(set(objects)) != len(objects)
    ):
        raise ValueError("objects must contain unique, non-empty names")
    if (
        not isinstance(frame_ids, list)
        or not frame_ids
        or any(not isinstance(frame_id, int) or frame_id < 0 for frame_id in frame_ids)
        or frame_ids != sorted(set(frame_ids))
    ):
        raise ValueError("frame_ids must contain sorted, unique, non-negative integers")
    return spec


def prepare_annotation_workspace(
    manifest_path: str | Path,
    spec_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy only raw RGB frames into a blank, prediction-independent workspace."""
    manifest_path = Path(manifest_path)
    validate_manifest(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    spec = load_evaluation_spec(spec_path)
    if spec["frame_ids"][-1] >= manifest["frame_count"]:
        raise ValueError("Evaluation frame exceeds manifest frame_count")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["objects"]:
        (output_dir / "annotations" / name).mkdir(parents=True, exist_ok=True)

    frames = {frame["frame_id"]: frame for frame in manifest["frames"]}
    items = []
    for frame_id in spec["frame_ids"]:
        source = Path(frames[frame_id]["rgb"])
        source = source if source.is_absolute() else manifest_path.parent / source
        destination = images_dir / f"{frame_id:06d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        items.append(
            {
                "frame_id": frame_id,
                "image": str(destination.relative_to(output_dir)),
                "annotations": {
                    name: f"annotations/{name}/{frame_id:06d}.png"
                    for name in spec["objects"]
                },
            }
        )

    workspace = {
        "schema_version": 1,
        "name": spec.get("name", "perception-gate"),
        "source_manifest": str(manifest_path.resolve()),
        "resolution": manifest["resolution"],
        "objects": spec["objects"],
        "frames": items,
        "annotation_policy": (
            "Draw binary masks from the raw images only; do not view or initialize from "
            "predicted masks. White is foreground and black is background."
        ),
    }
    (output_dir / "workspace.json").write_text(
        json.dumps(workspace, indent=2) + "\n", encoding="utf-8"
    )
    return workspace


def binary_mask_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate binary intersection over union, including the empty-mask case."""
    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if predicted.shape != ground_truth.shape or predicted.ndim != 2:
        raise ValueError("Masks must be equally shaped 2D arrays")
    union = np.logical_or(predicted, ground_truth).sum()
    return float(np.logical_and(predicted, ground_truth).sum() / union) if union else 1.0


def build_gate_report(
    ious_by_object: Mapping[str, Sequence[float]],
    diagnostics: Mapping[str, Mapping[str, float]],
    *,
    min_mask_iou: float,
    min_track_survival: float,
    min_visible_points: int,
    identity_swaps: int,
) -> dict[str, Any]:
    """Build a final report only from independent IoUs and stored track diagnostics."""
    objects: dict[str, dict[str, Any]] = {}
    for name, values_sequence in ious_by_object.items():
        values = np.asarray(values_sequence, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"IoUs for {name!r} must be a non-empty finite sequence")
        if not ((0 <= values) & (values <= 1)).all():
            raise ValueError(f"IoUs for {name!r} must be between zero and one")
        current = diagnostics[name]
        mean_iou = float(values.mean())
        track_survival = float(current["track_survival"])
        mean_visible = float(current["mean_visible_tracks"])
        object_pass = (
            mean_iou >= min_mask_iou
            and track_survival >= min_track_survival
            and mean_visible >= min_visible_points
        )
        objects[name] = {
            "annotated_frames": len(values),
            "mean_mask_iou": mean_iou,
            "min_mask_iou": float(values.min()),
            "track_survival": track_survival,
            "mean_visible_tracks": mean_visible,
            "pass": object_pass,
        }
    passed = all(item["pass"] for item in objects.values()) and identity_swaps == 0
    return {
        "status": "pass" if passed else "fail",
        "provisional": False,
        "thresholds": {
            "min_mask_iou": min_mask_iou,
            "min_track_survival": min_track_survival,
            "min_visible_points": min_visible_points,
            "max_identity_swaps": 0,
        },
        "identity_swaps": identity_swaps,
        "objects": objects,
    }


def evaluate_annotation_workspace(
    workspace_dir: str | Path,
    predictions_dir: str | Path,
    diagnostics_path: str | Path,
    *,
    min_mask_iou: float,
    min_track_survival: float,
    min_visible_points: int,
    identity_swaps: int,
) -> dict[str, Any]:
    """Evaluate complete PNG annotations, or return an explicit incomplete report."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Evaluating masks requires: uv sync --extra video") from exc
    workspace_dir = Path(workspace_dir)
    with (workspace_dir / "workspace.json").open(encoding="utf-8") as stream:
        workspace = json.load(stream)
    with Path(diagnostics_path).open(encoding="utf-8") as stream:
        diagnostics_report = json.load(stream)

    missing = []
    pairs: dict[str, list[tuple[Path, Path]]] = {name: [] for name in workspace["objects"]}
    for frame in workspace["frames"]:
        for name in workspace["objects"]:
            ground_truth = workspace_dir / frame["annotations"][name]
            predicted = Path(predictions_dir) / "masks" / name / f"{frame['frame_id']:06d}.png"
            if not ground_truth.is_file():
                missing.append(str(ground_truth.relative_to(workspace_dir)))
            elif not predicted.is_file():
                raise FileNotFoundError(predicted)
            else:
                pairs[name].append((predicted, ground_truth))
    if missing:
        return {
            "status": "incomplete",
            "provisional": True,
            "missing_annotations": missing,
            "completed_annotations": sum(len(values) for values in pairs.values()),
            "required_annotations": len(workspace["frames"]) * len(workspace["objects"]),
        }

    ious: dict[str, list[float]] = {name: [] for name in workspace["objects"]}
    expected_shape = (
        int(workspace["resolution"]["height"]),
        int(workspace["resolution"]["width"]),
    )
    for name, object_pairs in pairs.items():
        for predicted_path, ground_truth_path in object_pairs:
            predicted = cv2.imread(str(predicted_path), cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)
            if predicted is None or ground_truth is None:
                raise ValueError("Could not decode a prediction or annotation mask")
            if predicted.shape != expected_shape or ground_truth.shape != expected_shape:
                raise ValueError("Mask shape does not match workspace resolution")
            ious[name].append(binary_mask_iou(predicted > 0, ground_truth > 0))
    return build_gate_report(
        ious,
        diagnostics_report["objects"],
        min_mask_iou=min_mask_iou,
        min_track_survival=min_track_survival,
        min_visible_points=min_visible_points,
        identity_swaps=identity_swaps,
    )
