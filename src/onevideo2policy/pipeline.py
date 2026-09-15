from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_faithful_bundle(
    manifest_path: str | Path,
    crops_dir: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create a portable, hashed TRELLIS/VGGT input bundle on any machine."""
    manifest_path = Path(manifest_path)
    crops_dir = Path(crops_dir)
    output_dir = Path(output_dir)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Manifest must contain at least two frames")
    faithful = config["paths"]["faithful"]
    keyframe_count = min(int(faithful["keyframes"]), len(frames))
    indices = np.unique(np.linspace(0, len(frames) - 1, keyframe_count).round().astype(int))
    bundle_frames = output_dir / "frames"
    bundle_crops = output_dir / "crops"
    bundle_frames.mkdir(parents=True, exist_ok=True)
    bundle_crops.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for bundle_id, frame_idx in enumerate(indices):
        source = Path(frames[int(frame_idx)]["rgb"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = bundle_frames / f"{bundle_id:06d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        files.append(
            {
                "role": "vggt_keyframe",
                "frame_idx": int(frame_idx),
                "path": destination.relative_to(output_dir).as_posix(),
                "sha256": _sha256(destination),
            }
        )
    for name in ("source", "target"):
        source = crops_dir / f"{name}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = bundle_crops / source.name
        shutil.copy2(source, destination)
        files.append(
            {
                "role": f"trellis_{name}",
                "path": destination.relative_to(output_dir).as_posix(),
                "sha256": _sha256(destination),
            }
        )
    spec: dict[str, Any] = {
        "schema_version": 1,
        "path": "faithful",
        "status": "ready_for_external_compute",
        "models": {
            "object_reconstruction": faithful["reconstruction"],
            "scene_geometry": faithful["geometry"],
            "simulator": faithful["simulator"],
            "policy": faithful["policy"],
        },
        "hardware": {
            "min_vram_gb": int(faithful["min_vram_gb"]),
            "recommended_vram_gb": int(faithful["recommended_vram_gb"]),
        },
        "inputs": files,
        "expected_outputs": [
            "reconstruction/source.glb",
            "reconstruction/target.glb",
            "geometry/predictions.npz",
            "simulation/demonstrations.npz",
            "policy/checkpoint",
            "report.json",
        ],
    }
    (output_dir / "run-spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def _centroids(masks: NDArray[np.bool_]) -> NDArray[np.float64]:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or len(masks) < 2:
        raise ValueError("masks must have shape [T,H,W] with at least two frames")
    centers = np.full((len(masks), 2), np.nan, dtype=np.float64)
    for frame_idx, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs):
            centers[frame_idx] = [xs.mean(), ys.mean()]
    valid = np.flatnonzero(np.isfinite(centers[:, 0]))
    if not len(valid):
        raise ValueError("At least one non-empty mask is required")
    frame_ids = np.arange(len(masks))
    for axis in range(2):
        centers[:, axis] = np.interp(frame_ids, valid, centers[valid, axis])
    return centers


def recover_planar_proxy(
    source_masks: NDArray[np.bool_],
    target_masks: NDArray[np.bool_],
    *,
    target_diameter_m: float,
    lift_height_m: float,
    initial_separation_m: float | None = None,
) -> dict[str, NDArray[np.float64] | float]:
    """Recover a metric planar proxy trajectory from relative mask centroids.

    This local baseline intentionally does not claim monocular 6D reconstruction.
    Scale comes from a measured target diameter and height is a conservative Place arc.
    """
    if target_diameter_m <= 0 or lift_height_m <= 0:
        raise ValueError("target diameter and lift height must be positive")
    source_masks = np.asarray(source_masks, dtype=bool)
    target_masks = np.asarray(target_masks, dtype=bool)
    if source_masks.shape != target_masks.shape:
        raise ValueError("source and target masks must have matching shapes")
    source_xy = _centroids(source_masks)
    target_xy = _centroids(target_masks)
    diameters = []
    for mask in target_masks:
        ys, xs = np.nonzero(mask)
        if len(xs):
            diameters.append(max(float(np.ptp(xs) + 1), float(np.ptp(ys) + 1)))
    pixel_diameter = float(np.median(diameters))
    if initial_separation_m is not None:
        if initial_separation_m <= 0:
            raise ValueError("initial separation must be positive")
        initial_pixel_separation = float(np.linalg.norm(source_xy[0] - target_xy[0]))
        if initial_pixel_separation == 0:
            raise ValueError("initial source and target centroids must differ")
        metres_per_pixel = initial_separation_m / initial_pixel_separation
        scale_anchor = "configured_initial_separation"
    else:
        metres_per_pixel = target_diameter_m / pixel_diameter
        scale_anchor = "configured_target_diameter"
    relative_xy = (source_xy - target_xy) * metres_per_pixel
    relative_xy[:, 1] *= -1
    progress = np.linspace(0.0, 1.0, len(source_masks))
    z = lift_height_m * np.sin(np.pi * progress) ** 2
    xyz = np.column_stack((relative_xy, z))
    return {
        "source_xyz_m": xyz,
        "target_xyz_m": np.zeros_like(xyz),
        "metres_per_pixel": metres_per_pixel,
        "scale_anchor": scale_anchor,
    }


def _write_obj(
    path: Path, vertices: list[tuple[float, float, float]], faces: list[list[int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# OneVideo2Policy deterministic primitive"]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
    lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sphere_obj(
    path: str | Path, *, radius_m: float, rings: int = 12, segments: int = 24
) -> Path:
    """Write a compact UV-sphere collision/visual proxy as Wavefront OBJ."""
    if radius_m <= 0 or rings < 3 or segments < 3:
        raise ValueError("sphere dimensions and tessellation must be positive")
    vertices = [(0.0, 0.0, radius_m), (0.0, 0.0, -radius_m)]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        for segment in range(segments):
            theta = 2 * np.pi * segment / segments
            vertices.append(
                (
                    radius_m * np.sin(phi) * np.cos(theta),
                    radius_m * np.sin(phi) * np.sin(theta),
                    radius_m * np.cos(phi),
                )
            )
    faces: list[list[int]] = []
    first_ring = 3
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append([1, first_ring + segment, first_ring + nxt])
    for ring in range(rings - 2):
        a = first_ring + ring * segments
        b = a + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.extend([[a + segment, b + segment, b + nxt], [a + segment, b + nxt, a + nxt]])
    last_ring = first_ring + (rings - 2) * segments
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append([2, last_ring + nxt, last_ring + segment])
    output = Path(path)
    _write_obj(output, vertices, faces)
    return output


def write_bowl_obj(
    path: str | Path,
    *,
    outer_radius_m: float,
    height_m: float,
    wall_m: float = 0.006,
    segments: int = 32,
) -> Path:
    """Write an open tapered bowl proxy with inner and outer wall surfaces."""
    if min(outer_radius_m, height_m, wall_m) <= 0 or wall_m >= outer_radius_m:
        raise ValueError("invalid bowl dimensions")
    vertices: list[tuple[float, float, float]] = []
    inner = outer_radius_m - wall_m
    bottom_outer = outer_radius_m * 0.65
    bottom_inner = max(bottom_outer - wall_m, wall_m)
    for radius, z in (
        (bottom_outer, 0.0),
        (outer_radius_m, height_m),
        (inner, height_m),
        (bottom_inner, wall_m),
    ):
        for segment in range(segments):
            theta = 2 * np.pi * segment / segments
            vertices.append((radius * np.cos(theta), radius * np.sin(theta), z))
    faces: list[list[int]] = []
    for ring_a, ring_b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        for segment in range(segments):
            nxt = (segment + 1) % segments
            a, an = ring_a * segments + segment + 1, ring_a * segments + nxt + 1
            b, bn = ring_b * segments + segment + 1, ring_b * segments + nxt + 1
            faces.append([a, b, bn, an])
    output = Path(path)
    _write_obj(output, vertices, faces)
    return output


def _phase(step: int, steps: int) -> int:
    fraction = step / max(steps - 1, 1)
    return (
        0
        if fraction < 0.2
        else 1
        if fraction < 0.35
        else 2
        if fraction < 0.7
        else 3
        if fraction < 0.88
        else 4
    )


def _features(
    source: NDArray[np.float64], target: NDArray[np.float64], eef: NDArray[np.float64], phase: int
) -> NDArray[np.float64]:
    one_hot = np.eye(5, dtype=np.float64)[phase]
    relative = np.concatenate((source - target, eef - source, eef - target))
    phase_conditioned = np.outer(one_hot, relative).ravel()
    return np.concatenate((phase_conditioned, one_hot))


def _expert_action(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    eef: NDArray[np.float64],
    phase: int,
    max_step_m: float,
) -> NDArray[np.float64]:
    if phase == 0:
        waypoint, grip = source + [0, 0, 0.10], -1.0
    elif phase == 1:
        waypoint, grip = source + [0, 0, 0.02], 1.0
    elif phase == 2:
        waypoint, grip = target + [0, 0, 0.10], 1.0
    elif phase == 3:
        waypoint, grip = target + [0, 0, 0.02], 1.0
    else:
        waypoint, grip = target + [0, 0, 0.12], -1.0
    delta = np.clip(waypoint - eef, -max_step_m, max_step_m)
    return np.r_[delta, grip]


def generate_local_demonstrations(
    initial_offset_m: NDArray[np.floating],
    *,
    episodes: int,
    steps: int,
    randomization_m: float,
    seed: int,
    max_step_m: float = 0.03,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate kinematically bounded object-centric Place demonstrations."""
    if episodes <= 0 or steps < 10 or randomization_m < 0:
        raise ValueError("invalid demonstration settings")
    rng = np.random.default_rng(seed)
    base = np.asarray(initial_offset_m, dtype=np.float64).copy()
    base[2] = 0
    observations, actions = [], []
    for _ in range(episodes):
        target = np.r_[rng.uniform(-randomization_m, randomization_m, 2), 0.0]
        source = target + base + np.r_[rng.uniform(-randomization_m, randomization_m, 2), 0.0]
        eef = source + [0, 0, 0.15]
        attached = False
        for step in range(steps):
            phase = _phase(step, steps)
            observations.append(_features(source, target, eef, phase))
            action = _expert_action(source, target, eef, phase, max_step_m)
            actions.append(action)
            eef = eef + action[:3]
            if action[3] > 0 and np.linalg.norm(eef - (source + [0, 0, 0.02])) < 0.04:
                attached = True
            if attached:
                source = eef - [0, 0, 0.02]
            if action[3] < 0 and phase == 4:
                attached = False
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def fit_ridge_policy(
    observations: NDArray[np.floating],
    actions: NDArray[np.floating],
    *,
    regularization: float = 1e-4,
) -> NDArray[np.float64]:
    """Fit a deterministic linear behavior-cloning baseline with a bias term."""
    x = np.asarray(observations, dtype=np.float64)
    y = np.asarray(actions, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("observations and actions must be aligned matrices")
    design = np.column_stack((x, np.ones(len(x))))
    gram = design.T @ design + regularization * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ y)


def evaluate_local_policy(
    weights: NDArray[np.floating],
    initial_offset_m: NDArray[np.floating],
    *,
    episodes: int,
    steps: int,
    randomization_m: float,
    seed: int,
    success_xy_m: float,
    max_step_m: float = 0.03,
) -> dict[str, float | int]:
    """Roll out the local policy in the same bounded point-robot dynamics."""
    rng = np.random.default_rng(seed)
    base = np.asarray(initial_offset_m, dtype=np.float64).copy()
    base[2] = 0
    successes = 0
    for _ in range(episodes):
        target = np.r_[rng.uniform(-randomization_m, randomization_m, 2), 0.0]
        source = target + base + np.r_[rng.uniform(-randomization_m, randomization_m, 2), 0.0]
        eef = source + [0, 0, 0.15]
        attached = False
        for step in range(steps):
            phase = _phase(step, steps)
            feature = np.r_[_features(source, target, eef, phase), 1.0]
            action = feature @ np.asarray(weights)
            action[:3] = np.clip(action[:3], -max_step_m, max_step_m)
            eef += action[:3]
            if action[3] > 0 and np.linalg.norm(eef - (source + [0, 0, 0.02])) < 0.05:
                attached = True
            if attached:
                source = eef - [0, 0, 0.02]
            if action[3] < 0 and phase == 4:
                attached = False
        if np.linalg.norm(source[:2] - target[:2]) <= success_xy_m:
            successes += 1
    return {"episodes": episodes, "successes": successes, "success_rate": successes / episodes}


def save_local_rollout_video(
    weights: NDArray[np.floating],
    initial_offset_m: NDArray[np.floating],
    output_path: str | Path,
    *,
    steps: int,
    fps: float = 10.0,
) -> None:
    """Render one top-down policy rollout for inspection of the local baseline."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Rendering rollout video requires: uv sync --extra video") from exc
    target = np.zeros(3, dtype=np.float64)
    source = np.asarray(initial_offset_m, dtype=np.float64).copy()
    source[2] = 0
    eef = source + [0, 0, 0.15]
    attached = False
    states = []
    for step in range(steps):
        phase = _phase(step, steps)
        states.append((source.copy(), target.copy(), eef.copy(), phase, attached))
        feature = np.r_[_features(source, target, eef, phase), 1.0]
        action = feature @ np.asarray(weights)
        action[:3] = np.clip(action[:3], -0.03, 0.03)
        eef += action[:3]
        if action[3] > 0 and np.linalg.norm(eef - (source + [0, 0, 0.02])) < 0.05:
            attached = True
        if attached:
            source = eef - [0, 0, 0.02]
        if action[3] < 0 and phase == 4:
            attached = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Could not open video writer for {output_path}")

    def pixel(xy: NDArray[np.floating]) -> tuple[int, int]:
        return (
            round(width / 2 + float(xy[0]) * 900),
            round(height / 2 - float(xy[1]) * 900),
        )

    phase_names = ["approach", "grasp", "transfer", "place", "retreat"]
    try:
        for source_xyz, target_xyz, eef_xyz, phase, is_attached in states:
            canvas = np.full((height, width, 3), (242, 240, 232), dtype=np.uint8)
            cv2.circle(canvas, pixel(target_xyz[:2]), 50, (110, 75, 35), 8)
            cv2.circle(canvas, pixel(source_xyz[:2]), 18, (80, 170, 240), -1)
            ex, ey = pixel(eef_xyz[:2])
            cv2.drawMarker(canvas, (ex, ey), (40, 40, 40), cv2.MARKER_CROSS, 24, 3)
            label = f"phase: {phase_names[phase]}  z={eef_xyz[2]:.3f}m  attached={is_attached}"
            cv2.putText(
                canvas,
                label,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()


def run_local_end_to_end(
    source_masks: NDArray[np.bool_],
    target_masks: NDArray[np.bool_],
    output_dir: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the CPU-safe local systems baseline and persist every stage artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local = config["paths"]["local"]
    proxy = recover_planar_proxy(
        source_masks,
        target_masks,
        target_diameter_m=float(local["target_diameter_m"]),
        lift_height_m=float(local["lift_height_m"]),
        initial_separation_m=(
            float(local["initial_separation_m"]) if "initial_separation_m" in local else None
        ),
    )
    xyz = np.asarray(proxy["source_xyz_m"])
    initial_offset = xyz[0]
    np.savez_compressed(output_dir / "proxy_trajectory.npz", **proxy)
    write_sphere_obj(output_dir / "assets" / "source.obj", radius_m=float(local["source_radius_m"]))
    write_bowl_obj(
        output_dir / "assets" / "target.obj",
        outer_radius_m=float(local["target_diameter_m"]) / 2,
        height_m=float(local["target_height_m"]),
    )
    observations, actions = generate_local_demonstrations(
        initial_offset,
        episodes=int(local["train_episodes"]),
        steps=int(local["steps"]),
        randomization_m=float(local["randomization_m"]),
        seed=int(config["project"]["seed"]),
    )
    episode_ends = np.arange(
        int(local["steps"]), len(observations) + 1, int(local["steps"]), dtype=np.int64
    )
    np.savez_compressed(
        output_dir / "demonstrations.npz",
        observations=observations,
        actions=actions,
        episode_ends=episode_ends,
    )
    finite_samples = np.isfinite(observations).all(axis=1) & np.isfinite(actions).all(axis=1)
    bounded_samples = (np.abs(actions[:, :3]) <= 0.03 + 1e-7).all(axis=1)
    valid_sample_rate = float(np.mean(finite_samples & bounded_samples))
    weights = fit_ridge_policy(observations, actions)
    np.savez_compressed(output_dir / "policy.npz", weights=weights)
    save_local_rollout_video(
        weights,
        initial_offset,
        output_dir / "rollout.mp4",
        steps=int(local["steps"]),
    )
    evaluation = evaluate_local_policy(
        weights,
        initial_offset,
        episodes=int(local["eval_episodes"]),
        steps=int(local["steps"]),
        randomization_m=float(local["randomization_m"]),
        seed=int(config["project"]["seed"]) + 10_000,
        success_xy_m=float(config["task"]["success"]["max_xy_error_m"]),
    )
    threshold = float(local["min_success_rate"])
    generation_pass = valid_sample_rate >= 0.90
    learning_pass = evaluation["success_rate"] >= threshold
    report: dict[str, Any] = {
        "status": "pass" if generation_pass and learning_pass else "fail",
        "path": "local",
        "fidelity": "systems_baseline",
        "claims": {
            "supported": "artifact-compatible end-to-end execution on local hardware",
            "unsupported": "paper-faithful 3D reconstruction or sim-to-real robustness",
        },
        "geometry": {"source": "uv_sphere", "target": "open_bowl"},
        "trajectory": {
            "method": "configured-scale planar mask-centroid proxy",
            "frames": len(xyz),
            "metres_per_pixel": proxy["metres_per_pixel"],
            "scale_anchor": proxy["scale_anchor"],
        },
        "dataset": {
            "episodes": int(local["train_episodes"]),
            "steps_per_episode": int(local["steps"]),
            "samples": len(observations),
            "valid_sample_rate": valid_sample_rate,
        },
        "policy": {"type": "ridge_behavior_cloning", **evaluation},
        "gates": {
            "generation": {
                "min_valid_sample_rate": 0.90,
                "pass": generation_pass,
            },
            "learning": {
                "min_success_rate": threshold,
                "pass": learning_pass,
            },
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
