"""Run Stable Fast 3D on the frozen Place crops with a CPU texture-baking bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sf3d.system import SF3D
from texture_baker import TextureBaker


class CpuTextureBaker(TextureBaker):
    """Bridge the official CPU baker extension to a CUDA reconstruction model."""

    def rasterize(self, uv, face_indices, bake_resolution):
        raster = super().rasterize(
            uv.float().cpu().contiguous(),
            face_indices.int().cpu().contiguous(),
            bake_resolution,
        )
        return raster.to(uv.device)

    def interpolate(self, attr, raster, face_indices):
        values = super().interpolate(
            attr.float().cpu().contiguous(),
            raster.float().cpu().contiguous(),
            face_indices.int().cpu().contiguous(),
        )
        return values.to(attr.device)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="stabilityai/stable-fast-3d")
    parser.add_argument("--texture-resolution", type=int, default=256)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stable Fast 3D CUDA benchmark requires a GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = SF3D.from_pretrained(
        args.model, config_name="config.yaml", weight_name="model.safetensors"
    )
    model.to("cuda").eval()
    model.baker = CpuTextureBaker()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    load_peak = torch.cuda.max_memory_allocated() / 2**20
    records = {}
    for name in ("source", "target"):
        image_path = args.inputs / f"{name}_rgba.png"
        image = Image.open(image_path).convert("RGBA")
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            if args.precision == "fp16":
                with torch.autocast("cuda", dtype=torch.float16):
                    mesh, _ = model.run_image(
                        image, bake_resolution=args.texture_resolution, remesh="none"
                    )
            else:
                mesh, _ = model.run_image(
                    image, bake_resolution=args.texture_resolution, remesh="none"
                )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        mesh_path = folder / "mesh.glb"
        mesh.export(mesh_path, include_normals=True)
        records[name] = {
            "input_sha256": sha256(image_path),
            "mesh_sha256": sha256(mesh_path),
            "seconds": seconds,
            "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "mesh_extents_unscaled": np.asarray(mesh.extents).tolist(),
        }
    report = {
        "model": args.model,
        "precision": args.precision,
        "texture_resolution": args.texture_resolution,
        "texture_baker": "official CPU extension with CUDA tensor transfer bridge",
        "model_load_seconds": load_seconds,
        "model_load_peak_gpu_allocated_mb": load_peak,
        "objects": records,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
