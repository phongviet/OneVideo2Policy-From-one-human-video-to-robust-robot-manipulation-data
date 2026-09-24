from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_PATHS = (
    ("task", "name"),
    ("task", "source_object"),
    ("task", "target_object"),
    ("video", "input"),
    ("video", "sample_fps"),
    ("tracking", "segmenter"),
    ("tracking", "point_tracker"),
    ("gates", "min_mask_iou"),
)

LOCAL_PATH_KEYS = (
    "target_diameter_m",
    "target_height_m",
    "source_radius_m",
    "lift_height_m",
    "randomization_m",
    "train_episodes",
    "eval_episodes",
    "steps",
    "min_success_rate",
)

FAITHFUL_PATH_KEYS = (
    "reconstruction",
    "geometry",
    "simulator",
    "policy",
    "keyframes",
    "min_vram_gb",
    "recommended_vram_gb",
)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing: list[str] = []
    for section, key in REQUIRED_PATHS:
        if not isinstance(config.get(section), dict) or key not in config[section]:
            missing.append(f"{section}.{key}")
    if missing:
        raise ValueError("Missing required configuration values: " + ", ".join(missing))

    fps = config["video"]["sample_fps"]
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("video.sample_fps must be positive")
    iou = config["gates"]["min_mask_iou"]
    if not isinstance(iou, (int, float)) or not 0 <= iou <= 1:
        raise ValueError("gates.min_mask_iou must be between 0 and 1")

    paths = config.get("paths")
    if paths is not None:
        if not isinstance(paths, dict):
            raise ValueError("paths must be a mapping")
        for name, keys in (("local", LOCAL_PATH_KEYS), ("faithful", FAITHFUL_PATH_KEYS)):
            section = paths.get(name)
            if not isinstance(section, dict):
                raise ValueError(f"paths.{name} must be a mapping")
            missing_keys = [key for key in keys if key not in section]
            if missing_keys:
                raise ValueError(
                    f"paths.{name} is missing required values: {', '.join(missing_keys)}"
                )
        if not 0 <= float(paths["local"]["min_success_rate"]) <= 1:
            raise ValueError("paths.local.min_success_rate must be between 0 and 1")
        local = paths["local"]
        if local.get("camera_compensation", "none") not in ("none", "background_affine"):
            raise ValueError("paths.local.camera_compensation is unsupported")
        source_primitive = local.get("source_primitive", "uv_sphere")
        target_primitive = local.get("target_primitive", "open_bowl")
        if source_primitive not in ("uv_sphere", "cylinder"):
            raise ValueError("paths.local.source_primitive is unsupported")
        if target_primitive not in ("open_bowl", "rectangular_tray"):
            raise ValueError("paths.local.target_primitive is unsupported")
        required_dimensions = (["source_height_m"] if source_primitive == "cylinder" else []) + (
            ["target_length_m", "target_width_m"] if target_primitive == "rectangular_tray" else []
        )
        for key in required_dimensions:
            if key not in local or not isinstance(local[key], (int, float)) or local[key] <= 0:
                raise ValueError(f"paths.local.{key} must be positive")
        if "source_diameter_m" in local:
            diameter = float(local["source_diameter_m"])
            radius = float(local["source_radius_m"])
            if diameter <= 0:
                raise ValueError("paths.local.source_diameter_m must be positive")
            if abs(diameter - 2 * radius) > 1e-9:
                raise ValueError("paths.local source diameter must equal twice its radius")
        if int(paths["faithful"]["recommended_vram_gb"]) < int(paths["faithful"]["min_vram_gb"]):
            raise ValueError("faithful recommended VRAM must be at least its minimum")
