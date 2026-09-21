"""Compare compact policies on the existing synthetic proxy demonstrations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from onevideo2policy.pipeline import (
    _features,
    _phase,
    fit_ridge_policy,
    generate_local_demonstrations,
)


def rollout(predict, offset: np.ndarray, *, seed: int, episodes: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    successes = 0
    ever_attached = 0
    final_errors = []
    for _ in range(episodes):
        target = np.r_[rng.uniform(-0.06, 0.06, 2), 0.0]
        source = target + offset + np.r_[rng.uniform(-0.06, 0.06, 2), 0.0]
        eef = source + [0, 0, 0.15]
        attached = False
        did_attach = False
        for step in range(48):
            phase = _phase(step, 48)
            action = predict(_features(source, target, eef, phase))
            action[:3] = np.clip(action[:3], -0.03, 0.03)
            eef += action[:3]
            if action[3] > 0 and np.linalg.norm(eef - (source + [0, 0, 0.02])) < 0.05:
                attached = True
                did_attach = True
            if attached:
                source = eef - [0, 0, 0.02]
            if action[3] < 0 and phase == 4:
                attached = False
        error = float(np.linalg.norm(source[:2] - target[:2]))
        successes += error <= 0.03
        ever_attached += did_attach
        final_errors.append(error)
    return {
        "success_rate": float(successes / episodes),
        "attach_rate": float(ever_attached / episodes),
        "median_final_xy_error_m": float(np.median(final_errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = np.load(args.demonstrations)
    train_x = data["observations"].astype(np.float32)
    train_y = data["actions"].astype(np.float32)
    offset = np.load(args.proxy)["source_xyz_m"][0].astype(np.float64)
    offset[2] = 0.0
    test_x, test_y = generate_local_demonstrations(
        offset, episodes=50, steps=48, randomization_m=0.06, seed=10042
    )
    start = time.perf_counter()
    weights = fit_ridge_policy(train_x, train_y)
    ridge_time = time.perf_counter() - start

    def ridge_predict(x):
        return (np.r_[x, 1.0] @ weights).copy()

    report = {
        "scope": "synthetic bounded point robot; no robot dynamics or visual policy",
        "train_episodes": 250,
        "test_episodes": 50,
        "ridge": {
            "fit_seconds": ridge_time,
            "parameters": int(weights.size),
            "heldout_action_mae": float(
                np.mean(np.abs(np.c_[test_x, np.ones(len(test_x))] @ weights - test_y))
            ),
            "rollouts": [
                rollout(ridge_predict, offset, seed=seed) for seed in (10042, 20042, 30042)
            ],
        },
    }
    x_mean, x_std = train_x.mean(0), np.maximum(train_x.std(0), 1e-4)
    x_train = torch.from_numpy((train_x - x_mean) / x_std)
    y_train = torch.from_numpy(train_y)
    x_test = torch.from_numpy((test_x - x_mean) / x_std)
    for width in (32, 64):
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[1], width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, train_y.shape[1]),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        start = time.perf_counter()
        model.train()
        for _ in range(60):
            for batch in torch.randperm(len(x_train)).split(256):
                prediction = model(x_train[batch])
                loss = torch.nn.functional.mse_loss(prediction[:, :3], y_train[batch, :3])
                loss += torch.nn.functional.mse_loss(prediction[:, 3], y_train[batch, 3])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()

        def predict(x, trained_model=model):
            with torch.inference_mode():
                inp = torch.from_numpy(((x - x_mean) / x_std).astype(np.float32))
                return trained_model(inp).numpy().astype(np.float64)

        with torch.inference_mode():
            test_pred = model(x_test).numpy()
        report[f"mlp_{width}"] = {
            "fit_seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters()),
            "heldout_action_mae": float(np.mean(np.abs(test_pred - test_y))),
            "rollouts": [rollout(predict, offset, seed=seed) for seed in (10042, 20042, 30042)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
