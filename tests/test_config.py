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
    assert config["paths"]["model_assisted"]["recommended_vram_gb"] == 6


def test_two_path_config_requires_both_paths() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "two_paths.yaml")
    del config["paths"]["model_assisted"]

    with pytest.raises(ValueError, match="paths.model_assisted"):
        validate_config(config)


def test_real_place_config_requires_case_dimensions() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "real_place_img_6256_local_smoke.yaml"
    )
    del config["paths"]["local"]["target_width_m"]

    with pytest.raises(ValueError, match="paths.local.target_width_m"):
        validate_config(config)


def test_real_place_source_diameter_matches_radius() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "real_place_img_6256_local_smoke.yaml"
    )
    config["paths"]["local"]["source_radius_m"] = 0.03

    with pytest.raises(ValueError, match="diameter must equal twice its radius"):
        validate_config(config)
