"""Train state, phase-conditioned, and action-chunking diagnostic policies."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robosuite_policy_models import (  # noqa: E402
    ChunkImagePolicy,
    DiffusionChunkPolicy,
    PhaseImagePolicy,
    PhaseStatePolicy,
    SpatialPhasePolicy,
    StatePolicy,
    VisualWaypointPolicy,
    diffusion_schedule,
)


def normalize(array: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = array[ids].mean(0).astype(np.float32)
    std = np.maximum(array[ids].std(0), 1e-4).astype(np.float32)
    return mean, std


def image_tensor(
    images: np.ndarray,
    images_front: np.ndarray | None,
    ids: np.ndarray,
    device: str,
    *,
    dual_camera: bool,
    augment: bool,
    geometric_augment: bool = True,
) -> torch.Tensor:
    arrays = [images[ids]]
    if dual_camera:
        if images_front is None:
            raise ValueError("Dual-camera training requested without front-view images")
        arrays.append(images_front[ids])
    image = torch.from_numpy(np.concatenate(arrays, axis=-1)).to(device=device, dtype=torch.float32)
    image = image.permute(0, 3, 1, 2) / 255
    if augment:
        scale = torch.empty((len(ids), 1, 1, 1), device=device).uniform_(0.7, 1.3)
        shift = torch.empty((len(ids), 1, 1, 1), device=device).uniform_(-0.15, 0.15)
        image = (image * scale + shift + torch.randn_like(image) * 0.02).clamp(0, 1)
        if geometric_augment:
            angle = torch.empty(len(ids), device=device).uniform_(-3, 3) * torch.pi / 180
            translation = torch.empty((len(ids), 2), device=device).uniform_(-0.05, 0.05)
            cosine, sine = torch.cos(angle), torch.sin(angle)
            transform = torch.zeros((len(ids), 2, 3), device=device)
            transform[:, 0, 0] = cosine
            transform[:, 0, 1] = -sine
            transform[:, 1, 0] = sine
            transform[:, 1, 1] = cosine
            transform[:, :, 2] = translation
            grid = torch.nn.functional.affine_grid(transform, image.shape, align_corners=False)
            image = torch.nn.functional.grid_sample(
                image, grid, mode="bilinear", padding_mode="border", align_corners=False
            )
    return image


def chunk_indices(ends: np.ndarray, horizon: int) -> np.ndarray:
    result = np.empty((int(ends[-1]), horizon), dtype=np.int64)
    start = 0
    for end in ends:
        ids = np.arange(start, int(end))
        result[start:end] = np.minimum(ids[:, None] + np.arange(horizon), int(end) - 1)
        start = int(end)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        default=(
            "state,phase_state,visual_waypoint,spatial_phase,phase,chunk,diffusion,absolute_chunk"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--loss", choices=("huber", "l1", "mse"), default="huber")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.data)
    ends = data["episode_ends"].astype(int)
    starts = np.r_[0, ends[:-1]]
    split = min(max(1, int(round(len(ends) * 0.8))), len(ends) - 1)
    train_ids = np.concatenate([np.arange(starts[i], ends[i]) for i in range(split)])
    val_ids = np.concatenate([np.arange(starts[i], ends[i]) for i in range(split, len(ends))])
    proprio = data["proprio"].astype(np.float32)
    images = data["images"]
    images_front = data["images_front"] if "images_front" in data else None
    objects = data["objects"].astype(np.float32)
    state = np.concatenate((proprio, objects), axis=1)
    phases = data["phases"].astype(np.int64)
    actions = data["actions"].astype(np.float32)
    p_mean, p_std = normalize(proprio, train_ids)
    s_mean, s_std = normalize(state, train_ids)
    o_mean, o_std = normalize(objects, train_ids)
    c_mean, c_std = normalize(data["can_positions"].astype(np.float32), train_ids)
    has_dual = "images_front" in data
    chunks = chunk_indices(ends, args.horizon)
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    reports = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in requested:
        if name == "absolute_chunk" and "joint_targets" not in data:
            reports[name] = {"status": "skipped", "reason": "joint_targets missing"}
            continue
        if name == "state":
            model = StatePolicy(state.shape[1], actions.shape[1]).to(device)
            target_array = actions
        elif name == "phase_state":
            model = PhaseStatePolicy(state.shape[1], actions.shape[1]).to(device)
            target_array = actions
        elif name == "spatial_phase":
            model = SpatialPhasePolicy(
                proprio.shape[1], objects.shape[1], actions.shape[1], 6 if has_dual else 3
            ).to(device)
            target_array = actions
        elif name == "visual_waypoint":
            model = VisualWaypointPolicy(6 if has_dual else 3).to(device)
            target_array = data["can_positions"].astype(np.float32)
        elif name == "phase":
            model = PhaseImagePolicy(proprio.shape[1], actions.shape[1], 3).to(device)
            target_array = actions
        elif name == "chunk":
            model = ChunkImagePolicy(
                proprio.shape[1], actions.shape[1], args.horizon, 6 if has_dual else 3
            ).to(device)
            target_array = actions
        elif name == "diffusion":
            model = DiffusionChunkPolicy(
                proprio.shape[1],
                actions.shape[1],
                args.horizon,
                image_channels=6 if has_dual else 3,
            ).to(device)
            target_array = actions
        elif name == "absolute_chunk":
            target_array = data["joint_targets"].astype(np.float32)
            model = ChunkImagePolicy(
                proprio.shape[1],
                target_array.shape[1],
                args.horizon,
                6 if has_dual else 3,
                bounded=False,
            ).to(device)
        else:
            raise ValueError(f"Unknown model: {name}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        phase_counts = np.maximum(np.bincount(phases[train_ids], minlength=8), 1)
        weights = 1 / phase_counts[phases[train_ids]]
        weights /= weights.sum()
        best_mae, best_state, history = float("inf"), None, []
        started = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            sampled = np.random.choice(train_ids, len(train_ids), replace=True, p=weights)
            losses = []
            for ids in np.array_split(sampled, int(np.ceil(len(sampled) / args.batch_size))):
                object_loss = None
                if name == "state":
                    inputs = torch.from_numpy((state[ids] - s_mean) / s_std).to(device)
                    prediction = model(inputs)
                    target = torch.from_numpy(target_array[ids]).to(device)
                elif name == "phase_state":
                    inputs = torch.from_numpy((state[ids] - s_mean) / s_std).to(device)
                    phase = torch.from_numpy(phases[ids]).to(device)
                    prediction = model(inputs, phase)
                    target = torch.from_numpy(target_array[ids]).to(device)
                elif name == "spatial_phase":
                    rgb = image_tensor(
                        images, images_front, ids, device, dual_camera=has_dual, augment=True
                    )
                    prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                    phase = torch.from_numpy(phases[ids]).to(device)
                    prediction, predicted_object = model(rgb, prop, phase)
                    target = torch.from_numpy(target_array[ids]).to(device)
                    object_target = torch.from_numpy((objects[ids] - o_mean) / o_std).to(device)
                    object_loss = torch.nn.functional.mse_loss(predicted_object, object_target)
                elif name == "visual_waypoint":
                    rgb = image_tensor(
                        images,
                        images_front,
                        ids,
                        device,
                        dual_camera=has_dual,
                        augment=True,
                        geometric_augment=False,
                    )
                    prediction = model(rgb)
                    target = torch.from_numpy((target_array[ids] - c_mean) / c_std).to(device)
                elif name == "phase":
                    rgb = image_tensor(
                        images, images_front, ids, device, dual_camera=False, augment=True
                    )
                    prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                    phase = torch.from_numpy(phases[ids]).to(device)
                    prediction = model(rgb, prop, phase)
                    target = torch.from_numpy(target_array[ids]).to(device)
                elif name in ("chunk", "absolute_chunk"):
                    rgb = image_tensor(
                        images, images_front, ids, device, dual_camera=has_dual, augment=True
                    )
                    prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                    prediction = model(rgb, prop)
                    target = torch.from_numpy(target_array[chunks[ids]]).to(device)
                else:
                    rgb = image_tensor(
                        images, images_front, ids, device, dual_camera=has_dual, augment=True
                    )
                    prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                    target = torch.from_numpy(target_array[chunks[ids]]).to(device)
                    timestep = torch.randint(model.diffusion_steps, (len(ids),), device=device)
                    schedule = diffusion_schedule(model.diffusion_steps, device)
                    noise = torch.randn_like(target)
                    alpha_bar = schedule["alpha_bars"][timestep, None, None]
                    noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1 - alpha_bar) * noise
                    prediction = model(rgb, prop, noisy, timestep)
                if name == "diffusion" or args.loss == "mse":
                    loss = torch.nn.functional.mse_loss(
                        prediction, noise if name == "diffusion" else target
                    )
                elif args.loss == "l1":
                    loss = torch.nn.functional.l1_loss(prediction, target)
                else:
                    loss = torch.nn.functional.huber_loss(prediction, target)
                if object_loss is not None:
                    loss = loss + 0.1 * object_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            predictions, targets = [], []
            with torch.inference_mode():
                for ids in np.array_split(val_ids, int(np.ceil(len(val_ids) / args.batch_size))):
                    if name == "state":
                        inputs = torch.from_numpy((state[ids] - s_mean) / s_std).to(device)
                        prediction = model(inputs)
                        target = target_array[ids]
                    elif name == "phase_state":
                        inputs = torch.from_numpy((state[ids] - s_mean) / s_std).to(device)
                        prediction = model(inputs, torch.from_numpy(phases[ids]).to(device))
                        target = target_array[ids]
                    elif name == "spatial_phase":
                        rgb = image_tensor(
                            images,
                            images_front,
                            ids,
                            device,
                            dual_camera=has_dual,
                            augment=False,
                        )
                        prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                        prediction, _ = model(rgb, prop, torch.from_numpy(phases[ids]).to(device))
                        target = target_array[ids]
                    elif name == "visual_waypoint":
                        rgb = image_tensor(
                            images,
                            images_front,
                            ids,
                            device,
                            dual_camera=has_dual,
                            augment=False,
                        )
                        prediction = model(rgb)
                        target = (target_array[ids] - c_mean) / c_std
                    elif name == "phase":
                        rgb = image_tensor(
                            images,
                            images_front,
                            ids,
                            device,
                            dual_camera=False,
                            augment=False,
                        )
                        prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                        prediction = model(rgb, prop, torch.from_numpy(phases[ids]).to(device))
                        target = target_array[ids]
                    elif name in ("chunk", "absolute_chunk"):
                        rgb = image_tensor(
                            images,
                            images_front,
                            ids,
                            device,
                            dual_camera=has_dual,
                            augment=False,
                        )
                        prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                        prediction = model(rgb, prop)
                        target = target_array[chunks[ids]]
                    else:
                        rgb = image_tensor(
                            images,
                            images_front,
                            ids,
                            device,
                            dual_camera=has_dual,
                            augment=False,
                        )
                        prop = torch.from_numpy((proprio[ids] - p_mean) / p_std).to(device)
                        target_tensor = torch.from_numpy(target_array[chunks[ids]]).to(device)
                        timestep = torch.full(
                            (len(ids),),
                            model.diffusion_steps // 2,
                            device=device,
                            dtype=torch.long,
                        )
                        schedule = diffusion_schedule(model.diffusion_steps, device)
                        noise = torch.zeros_like(target_tensor)
                        alpha_bar = schedule["alpha_bars"][timestep, None, None]
                        noisy = torch.sqrt(alpha_bar) * target_tensor
                        prediction = model(rgb, prop, noisy, timestep)
                        target = noise.cpu().numpy()
                    predictions.append(prediction.cpu().numpy())
                    targets.append(target)
            mae = float(np.abs(np.concatenate(predictions) - np.concatenate(targets)).mean())
            history.append(
                {"epoch": epoch + 1, "train_huber": float(np.mean(losses)), "val_mae": mae}
            )
            if mae < best_mae:
                best_mae = mae
                best_state = copy.deepcopy(model.state_dict())
        checkpoint = {
            "model_state": best_state,
            "kind": name,
            "proprio_dim": proprio.shape[1],
            "state_dim": state.shape[1],
            "action_dim": target_array.shape[1],
            "horizon": args.horizon,
            "dual_camera": has_dual,
            "proprio_mean": p_mean,
            "proprio_std": p_std,
            "state_mean": s_mean,
            "state_std": s_std,
            "object_dim": objects.shape[1],
            "object_mean": o_mean,
            "object_std": o_std,
            "can_position_mean": c_mean,
            "can_position_std": c_std,
        }
        torch.save(checkpoint, args.output / f"{name}.pt")
        report = {
            "status": "trained",
            "kind": name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_validation_mae": best_mae,
            "best_epoch": min(history, key=lambda item: item["val_mae"])["epoch"],
            "train_episodes": split,
            "validation_episodes": len(ends) - split,
            "dual_camera": (
                has_dual
                if name
                in (
                    "visual_waypoint",
                    "spatial_phase",
                    "chunk",
                    "diffusion",
                    "absolute_chunk",
                )
                else False
            ),
            "phase_balanced_sampling": True,
            "photometric_augmentation": name
            in (
                "visual_waypoint",
                "spatial_phase",
                "phase",
                "chunk",
                "diffusion",
                "absolute_chunk",
            ),
            "geometric_augmentation": name
            in ("spatial_phase", "phase", "chunk", "diffusion", "absolute_chunk"),
            "validation_metric": (
                "normalized_can_position_mae"
                if name == "visual_waypoint"
                else "diffusion_denoising_mae"
                if name == "diffusion"
                else "action_mae"
            ),
            "loss": "diffusion_noise_mse" if name == "diffusion" else args.loss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        reports[name] = report
        (args.output / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    (args.output / "training_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
