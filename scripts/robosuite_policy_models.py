"""Small policy models used by local robosuite diagnostics."""

from __future__ import annotations

import torch


class VisionEncoder(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class StatePolicy(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PhaseStatePolicy(torch.nn.Module):
    """Privileged control ceiling with the expert phase made explicit."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim + 8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(phase.long(), num_classes=8).float()
        return self.network(torch.cat((state, one_hot), dim=-1))


class SpatialVisionEncoder(torch.nn.Module):
    """Coordinate-aware encoder that retains a 6x6 spatial feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(channels + 2, 32, 5, stride=2, padding=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        y = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
        x = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy)).expand(batch, -1, -1, -1)
        return self.network(torch.cat((image, coordinates), dim=1))


class SpatialPhasePolicy(torch.nn.Module):
    """Predict normalized object state from pixels before phase-aware control."""

    def __init__(
        self,
        proprio_dim: int,
        object_dim: int,
        action_dim: int,
        image_channels: int = 6,
    ) -> None:
        super().__init__()
        self.vision = SpatialVisionEncoder(image_channels)
        self.object_head = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 6, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, object_dim),
        )
        self.control = torch.nn.Sequential(
            torch.nn.Linear(proprio_dim + object_dim + 8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
            torch.nn.Tanh(),
        )

    def forward(
        self, image: torch.Tensor, proprio: torch.Tensor, phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_state = self.object_head(self.vision(image))
        one_hot = torch.nn.functional.one_hot(phase.long(), num_classes=8).float()
        action = self.control(torch.cat((proprio, object_state, one_hot), dim=-1))
        return action, object_state


class VisualWaypointPolicy(torch.nn.Module):
    """Estimate the source object's metric position from two RGB views."""

    def __init__(self, image_channels: int = 6) -> None:
        super().__init__()
        self.vision = SpatialVisionEncoder(image_channels)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 6, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 3),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.vision(image))


class PhaseImagePolicy(torch.nn.Module):
    def __init__(self, proprio_dim: int, action_dim: int, image_channels: int = 3) -> None:
        super().__init__()
        self.vision = VisionEncoder(image_channels)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4 + proprio_dim + 8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
            torch.nn.Tanh(),
        )

    def forward(
        self, image: torch.Tensor, proprio: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(phase.long(), num_classes=8).float()
        return self.head(torch.cat((self.vision(image), proprio, one_hot), dim=-1))


class ChunkImagePolicy(torch.nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        horizon: int,
        image_channels: int = 6,
        bounded: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.vision = VisionEncoder(image_channels)
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(64 * 4 * 4 + proprio_dim, 384),
            torch.nn.ReLU(),
            torch.nn.Linear(384, 384),
            torch.nn.ReLU(),
            torch.nn.Linear(384, action_dim * horizon),
        ]
        if bounded:
            layers.append(torch.nn.Tanh())
        self.head = torch.nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        output = self.head(torch.cat((self.vision(image), proprio), dim=-1))
        return output.reshape(-1, self.horizon, self.action_dim)


class DiffusionChunkPolicy(torch.nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        horizon: int,
        diffusion_steps: int = 20,
        image_channels: int = 6,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.diffusion_steps = diffusion_steps
        self.vision = VisionEncoder(image_channels)
        self.time_embedding = torch.nn.Embedding(diffusion_steps, 32)
        self.denoiser = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4 + proprio_dim + action_dim * horizon + 32, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, action_dim * horizon),
        )

    def encode(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.vision(image), proprio), dim=-1)

    def forward_condition(
        self, condition: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        flattened = noisy_action.reshape(len(noisy_action), -1)
        output = self.denoiser(
            torch.cat((condition, flattened, self.time_embedding(timestep)), dim=-1)
        )
        return output.reshape(-1, self.horizon, self.action_dim)

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_condition(self.encode(image, proprio), noisy_action, timestep)


def diffusion_schedule(steps: int, device: torch.device | str) -> dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}


@torch.inference_mode()
def sample_diffusion_actions(
    model: DiffusionChunkPolicy,
    image: torch.Tensor,
    proprio: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    schedule = diffusion_schedule(model.diffusion_steps, image.device)
    condition = model.encode(image, proprio)
    shape = (len(image), model.horizon, model.action_dim)
    action = torch.randn(shape, device=image.device, generator=generator)
    for step in reversed(range(model.diffusion_steps)):
        timestep = torch.full((len(image),), step, device=image.device, dtype=torch.long)
        predicted_noise = model.forward_condition(condition, action, timestep)
        alpha = schedule["alphas"][step]
        alpha_bar = schedule["alpha_bars"][step]
        mean = (action - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(
            alpha
        )
        if step:
            noise = torch.randn(shape, device=image.device, generator=generator)
            action = mean + torch.sqrt(schedule["betas"][step]) * noise
        else:
            action = mean
    return action.clamp(-1, 1)
