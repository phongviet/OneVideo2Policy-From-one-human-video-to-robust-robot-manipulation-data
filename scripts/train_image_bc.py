"""Train a compact RGB-plus-proprio behavior-cloning policy on robosuite demos."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch


class ImageBC(torch.nn.Module):
    def __init__(self, proprio_dim: int, action_dim: int) -> None:
        super().__init__()
        self.vision = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4 + proprio_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.vision(image), proprio), dim=-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.data)
    images = data["images"]
    proprio = data["proprio"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    ends = data["episode_ends"]
    starts = np.r_[0, ends[:-1]]
    if len(ends) < 3:
        raise ValueError("At least three episodes are required for an episode-level split")
    split = max(1, int(round(len(ends) * 0.8)))
    split = min(split, len(ends) - 1)
    train_ids = np.concatenate([np.arange(starts[i], ends[i]) for i in range(split)])
    val_ids = np.concatenate([np.arange(starts[i], ends[i]) for i in range(split, len(ends))])
    mean, std = proprio[train_ids].mean(0), np.maximum(proprio[train_ids].std(0), 1e-4)
    model = ImageBC(proprio.shape[1], actions.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    started = time.perf_counter()
    history = []
    best_validation_mae = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        losses = []
        batch_count = int(np.ceil(len(train_ids) / args.batch_size))
        for batch in np.array_split(np.random.permutation(train_ids), batch_count):
            rgb = torch.from_numpy(images[batch]).to(device=device, dtype=torch.float32)
            rgb = rgb.permute(0, 3, 1, 2) / 255
            state = torch.from_numpy((proprio[batch] - mean) / std).to(device)
            target = torch.from_numpy(actions[batch]).to(device)
            prediction = model(rgb, state)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_predictions = []
        with torch.inference_mode():
            for batch in np.array_split(val_ids, int(np.ceil(len(val_ids) / args.batch_size))):
                rgb = torch.from_numpy(images[batch]).to(device=device, dtype=torch.float32)
                state = torch.from_numpy((proprio[batch] - mean) / std).to(device)
                val_predictions.append(model(rgb.permute(0, 3, 1, 2) / 255, state).cpu().numpy())
        predicted = np.concatenate(val_predictions)
        validation_mae = float(np.mean(np.abs(predicted - actions[val_ids])))
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": float(np.mean(losses)),
                "validation_mae": validation_mae,
            }
        )
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_state = copy.deepcopy(model.state_dict())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "proprio_mean": mean,
            "proprio_std": std,
            "proprio_dim": proprio.shape[1],
            "action_dim": actions.shape[1],
        },
        args.output / "checkpoint.pt",
    )
    best = min(history, key=lambda row: row["validation_mae"])
    report = {
        "status": "supervised_fit_only",
        "device": device,
        "episodes": int(len(ends)),
        "train_episodes": int(split),
        "validation_episodes": int(len(ends) - split),
        "samples": int(len(actions)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best,
        "final_epoch": history[-1],
        "elapsed_seconds": time.perf_counter() - started,
        "limitation": (
            "Validation is one-step action error; closed-loop robosuite success is untested."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
