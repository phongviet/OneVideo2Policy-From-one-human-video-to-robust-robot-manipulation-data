from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def predict_source_position(
    checkpoint_path: str | Path,
    agent_rgb: NDArray[np.uint8],
    front_rgb: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """Load the frozen compact waypoint model and infer metric source xyz."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment extra
        raise RuntimeError("Waypoint inference requires PyTorch") from exc

    class SpatialVisionEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(8, 32, 5, stride=2, padding=2),
                torch.nn.SiLU(),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
                torch.nn.SiLU(),
                torch.nn.Flatten(),
            )

        def forward(self, image: object) -> object:
            batch, _, height, width = image.shape
            y = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
            x = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack((xx, yy)).expand(batch, -1, -1, -1)
            return self.network(torch.cat((image, coordinates), dim=1))

    class WaypointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = SpatialVisionEncoder()
            self.head = torch.nn.Sequential(
                torch.nn.Linear(128 * 6 * 6, 512),
                torch.nn.SiLU(),
                torch.nn.Linear(512, 3),
            )

        def forward(self, image: object) -> object:
            return self.head(self.vision(image))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "visual_waypoint" or not checkpoint.get("dual_camera"):
        raise ValueError("checkpoint is not the dual-camera visual waypoint model")
    if agent_rgb.shape != (84, 84, 3) or front_rgb.shape != agent_rgb.shape:
        raise ValueError("deployment images must both have shape [84,84,3]")
    model = WaypointModel().eval()
    model.load_state_dict(checkpoint["model_state"])
    image = np.concatenate((agent_rgb, front_rgb), axis=-1)
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255
    with torch.inference_mode():
        normalized = model(tensor)[0].numpy()
    return (
        normalized * np.asarray(checkpoint["can_position_std"], dtype=np.float32)
        + np.asarray(checkpoint["can_position_mean"], dtype=np.float32)
    )
