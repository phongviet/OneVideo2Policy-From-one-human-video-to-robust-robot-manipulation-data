from pathlib import Path

from onevideo2policy.config import load_config


def test_default_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "place.yaml")
    assert config["task"]["name"] == "place"
    assert config["video"]["monocular_rgb_only"] is False
    assert config["video"]["capture"] == "realsense_l515_fixed_camera"
