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
