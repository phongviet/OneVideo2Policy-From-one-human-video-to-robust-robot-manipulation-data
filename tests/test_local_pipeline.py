from pathlib import Path

import numpy as np

from onevideo2policy.pipeline import (
    evaluate_local_policy,
    fit_ridge_policy,
    generate_local_demonstrations,
    prepare_faithful_bundle,
    recover_planar_proxy,
    run_local_end_to_end,
    write_bowl_obj,
    write_sphere_obj,
)


def _moving_masks() -> tuple[np.ndarray, np.ndarray]:
    source = np.zeros((12, 32, 48), dtype=bool)
    target = np.zeros_like(source)
    target[:, 12:24, 30:44] = True
    for frame in range(12):
        x = round(4 + frame * 27 / 11)
        source[frame, 14:20, x : x + 6] = True
    source[5:7] = False
    return source, target


def test_planar_proxy_interpolates_occlusion_and_uses_measured_scale() -> None:
    source, target = _moving_masks()

    result = recover_planar_proxy(source, target, target_diameter_m=0.14, lift_height_m=0.1)

    xyz = result["source_xyz_m"]
    assert xyz.shape == (12, 3)
    assert np.isfinite(xyz).all()
    assert result["metres_per_pixel"] == 0.01
    assert xyz[0, 2] == 0
    assert xyz[6, 2] > 0.09


def test_primitive_meshes_are_written(tmp_path: Path) -> None:
    sphere = write_sphere_obj(tmp_path / "sphere.obj", radius_m=0.03)
    bowl = write_bowl_obj(tmp_path / "bowl.obj", outer_radius_m=0.09, height_m=0.04)

    assert sphere.read_text().count("\nv ") > 100
    assert bowl.read_text().count("\nf ") > 100


def test_local_policy_learns_generated_place_rollouts() -> None:
    offset = np.array([-0.18, 0.04, 0.0])
    observations, actions = generate_local_demonstrations(
        offset, episodes=100, steps=48, randomization_m=0.05, seed=3
    )
    weights = fit_ridge_policy(observations, actions)

    report = evaluate_local_policy(
        weights,
        offset,
        episodes=30,
        steps=48,
        randomization_m=0.05,
        seed=99,
        success_xy_m=0.03,
    )

    assert report["success_rate"] >= 0.8


def test_local_end_to_end_writes_contract(tmp_path: Path) -> None:
    source, target = _moving_masks()
    config = {
        "project": {"seed": 4},
        "task": {"success": {"max_xy_error_m": 0.03}},
        "paths": {
            "local": {
                "target_diameter_m": 0.14,
                "target_height_m": 0.04,
                "source_radius_m": 0.025,
                "lift_height_m": 0.10,
                "randomization_m": 0.04,
                "train_episodes": 50,
                "eval_episodes": 20,
                "steps": 48,
                "min_success_rate": 0.8,
            }
        },
    }

    report = run_local_end_to_end(source, target, tmp_path, config=config)

    assert report["status"] == "pass"
    assert report["gates"]["generation"]["pass"] is True
    assert report["gates"]["learning"]["pass"] is True
    assert (tmp_path / "assets/source.obj").is_file()
    assert (tmp_path / "demonstrations.npz").is_file()
    assert (tmp_path / "policy.npz").is_file()
    assert (tmp_path / "rollout.mp4").stat().st_size > 0
    assert (tmp_path / "report.json").is_file()


def test_faithful_bundle_selects_and_hashes_inputs(tmp_path: Path) -> None:
    import json

    source_dir = tmp_path / "source"
    frames_dir = source_dir / "frames"
    crops_dir = source_dir / "crops"
    frames_dir.mkdir(parents=True)
    crops_dir.mkdir()
    frames = []
    for frame_idx in range(5):
        path = frames_dir / f"{frame_idx:06d}.jpg"
        path.write_bytes(f"frame-{frame_idx}".encode())
        frames.append({"rgb": f"frames/{path.name}"})
    (source_dir / "manifest.json").write_text(json.dumps({"frames": frames}))
    for name in ("source", "target"):
        (crops_dir / f"{name}.png").write_bytes(name.encode())
    config = {
        "paths": {
            "faithful": {
                "keyframes": 3,
                "reconstruction": "trellis",
                "geometry": "vggt",
                "simulator": "robosuite",
                "policy": "diffusion_policy",
                "min_vram_gb": 16,
                "recommended_vram_gb": 24,
            }
        }
    }

    spec = prepare_faithful_bundle(
        source_dir / "manifest.json", crops_dir, tmp_path / "bundle", config=config
    )

    keyframes = [item for item in spec["inputs"] if item["role"] == "vggt_keyframe"]
    assert spec["status"] == "ready_for_external_compute"
    assert len(keyframes) == 3
    assert all(len(item["sha256"]) == 64 for item in spec["inputs"])
    assert (tmp_path / "bundle/run-spec.json").is_file()
