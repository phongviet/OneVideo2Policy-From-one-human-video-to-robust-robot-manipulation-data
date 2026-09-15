from pathlib import Path

import pytest

from onevideo2policy.config import load_config, validate_config


def test_default_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "place.yaml")
    assert config["task"]["name"] == "place"
    assert config["video"]["monocular_rgb_only"] is False
    assert config["video"]["capture"] == "realsense_l515_fixed_camera"


def test_two_path_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "two_paths.yaml")

    assert config["paths"]["local"]["policy"] == "ridge_behavior_cloning"
    assert config["paths"]["faithful"]["recommended_vram_gb"] == 24


def test_two_path_config_requires_both_paths() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "two_paths.yaml")
    del config["paths"]["faithful"]

    with pytest.raises(ValueError, match="paths.faithful"):
        validate_config(config)
