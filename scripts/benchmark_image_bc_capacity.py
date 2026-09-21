"""Measure whether a compact image behavior-cloning network fits on the local GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this capacity benchmark")
    torch.manual_seed(42)
    results = {}
    for size, batch in ((84, 32), (128, 16)):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 7),
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        images = torch.rand(batch, 3, size, size, device="cuda")
        targets = torch.rand(batch, 7, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(args.steps + 10):
            start = time.perf_counter()
            prediction = model(images)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= 10:
                times.append(time.perf_counter() - start)
        results[f"{size}_batch_{batch}"] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "median_train_step_ms": 1000 * statistics.median(times),
            "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        del model, optimizer, images, targets
        torch.cuda.empty_cache()
    report = {
        "device": torch.cuda.get_device_name(),
        "scope": "synthetic tensors, compute and memory only; no learned task performance",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
