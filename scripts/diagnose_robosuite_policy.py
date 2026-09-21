"""Audit robosuite demonstrations and a compact image-BC checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_image_bc import ImageBC  # noqa: E402


def episode_ids(ends: np.ndarray) -> np.ndarray:
    ids = np.empty(int(ends[-1]), dtype=np.int32)
    start = 0
    for episode, end in enumerate(ends):
        ids[start:end] = episode
        start = int(end)
    return ids


def infer(
    model: ImageBC,
    images: np.ndarray,
    proprio: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    predictions, embeddings = [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = slice(start, start + batch_size)
            rgb = torch.from_numpy(images[batch]).to(device=device, dtype=torch.float32)
            rgb = rgb.permute(0, 3, 1, 2) / 255
            state = torch.from_numpy((proprio[batch] - mean) / std).to(device)
            vision = model.vision(rgb)
            predictions.append(model.head(torch.cat((vision, state), dim=-1)).cpu().numpy())
            embeddings.append(torch.cat((vision, state), dim=-1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def image_hashes(images: np.ndarray) -> list[str]:
    return [hashlib.sha256(image.tobytes()).hexdigest() for image in images]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.data)
    ends = data["episode_ends"].astype(int)
    starts = np.r_[0, ends[:-1]]
    ep_ids = episode_ids(ends)
    images = data["images"]
    proprio = data["proprio"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    phases = data["phases"].astype(int)
    split = min(max(1, int(round(len(ends) * 0.8))), len(ends) - 1)
    train_mask = ep_ids < split
    validation_mask = ~train_mask
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ImageBC(checkpoint["proprio_dim"], checkpoint["action_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions, embeddings = infer(
        model,
        images,
        proprio,
        np.asarray(checkpoint["proprio_mean"], dtype=np.float32),
        np.asarray(checkpoint["proprio_std"], dtype=np.float32),
        device,
    )
    phase_metrics = {}
    for phase in np.unique(phases):
        mask = (phases == phase) & validation_mask
        phase_metrics[str(int(phase))] = {
            "validation_samples": int(mask.sum()),
            "action_mae": float(np.abs(predictions[mask] - actions[mask]).mean()),
            "translation_mae": float(np.abs(predictions[mask, :3] - actions[mask, :3]).mean()),
            "gripper_sign_accuracy": float(
                (np.sign(predictions[mask, -1]) == np.sign(actions[mask, -1])).mean()
            ),
        }
    initial_target = actions[starts]
    initial_prediction = predictions[starts]
    can_positions = data["can_positions"] if "can_positions" in data else data["objects"][:, 7:10]
    initial_can = can_positions[starts]
    hashes = image_hashes(images)
    train_hashes = {hashes[i] for i in np.flatnonzero(train_mask)}
    duplicate_validation_frames = sum(
        hashes[i] in train_hashes for i in np.flatnonzero(validation_mask)
    )
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(len(actions), min(1000, len(actions)), replace=False)
    standardized = embeddings[sample_ids]
    standardized = (standardized - standardized.mean(0)) / np.maximum(standardized.std(0), 1e-4)
    squared_norm = np.sum(standardized**2, axis=1)
    distances = squared_norm[:, None] + squared_norm[None, :] - 2 * standardized @ standardized.T
    distances = np.maximum(distances, 0)
    same_episode = ep_ids[sample_ids, None] == ep_ids[sample_ids][None, :]
    distances[same_episode] = np.inf
    nearest = distances.argmin(1)
    neighbor_action_gap = np.linalg.norm(actions[sample_ids] - actions[sample_ids[nearest]], axis=1)
    integrity = {
        "episode_ends_strictly_increasing": bool(np.all(np.diff(ends) > 0)),
        "sample_counts_match": bool(
            all(len(data[key]) == ends[-1] for key in data.files if key != "episode_ends")
        ),
        "actions_finite": bool(np.isfinite(actions).all()),
        "actions_within_normalized_limits": bool((np.abs(actions) <= 1.0001).all()),
        "duplicate_validation_frames_in_training": int(duplicate_validation_frames),
        "dual_camera_shape_match": (
            bool(data["images_front"].shape == images.shape) if "images_front" in data else None
        ),
    }
    report = {
        "status": "diagnostic_complete",
        "device": device,
        "episodes": int(len(ends)),
        "samples": int(len(actions)),
        "train_episodes": int(split),
        "validation_episodes": int(len(ends) - split),
        "phase_counts": {
            str(int(phase)): int((phases == phase).sum()) for phase in np.unique(phases)
        },
        "phase_metrics": phase_metrics,
        "initial_step": {
            "translation_mae": float(
                np.abs(initial_prediction[:, :3] - initial_target[:, :3]).mean()
            ),
            "mean_expert_translation": initial_target[:, :3].mean(0).tolist(),
            "mean_predicted_translation": initial_prediction[:, :3].mean(0).tolist(),
            "gripper_sign_accuracy": float(
                (np.sign(initial_prediction[:, -1]) == np.sign(initial_target[:, -1])).mean()
            ),
        },
        "coverage": {
            "initial_can_xyz_min": initial_can.min(0).tolist(),
            "initial_can_xyz_max": initial_can.max(0).tolist(),
            "initial_can_xy_span": np.ptp(initial_can[:, :2], axis=0).tolist(),
        },
        "ambiguity": {
            "sampled_cross_episode_neighbors": int(len(sample_ids)),
            "median_nearest_embedding_distance": float(
                np.median(np.sqrt(distances[np.arange(len(sample_ids)), nearest]))
            ),
            "median_neighbor_action_l2_gap": float(np.median(neighbor_action_gap)),
            "p90_neighbor_action_l2_gap": float(np.quantile(neighbor_action_gap, 0.9)),
            "fraction_neighbor_action_gap_over_0_5": float((neighbor_action_gap > 0.5).mean()),
        },
        "integrity": integrity,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    phase_names = list(phase_metrics)
    values = [phase_metrics[name]["translation_mae"] for name in phase_names]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(phase_names, values)
    axis.set(xlabel="Expert phase", ylabel="Validation translation MAE")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "phase_translation_mae.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    limit = min(int(ends[split]), int(ends[split - 1]) + 180)
    ids = np.arange(int(starts[split]), limit)
    for dimension, axis in enumerate(axes):
        axis.plot(actions[ids, dimension], label="expert", linewidth=1.5)
        axis.plot(predictions[ids, dimension], label="prediction", linewidth=1)
        axis.set_ylabel("xyz"[dimension])
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("Validation step")
    fig.tight_layout()
    fig.savefig(args.output / "validation_action_trace.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
