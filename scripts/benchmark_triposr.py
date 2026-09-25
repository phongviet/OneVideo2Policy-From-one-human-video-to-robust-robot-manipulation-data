"""Reconstruct the same two Place crops with TripoSR and record local resource use."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tsr.system import TSR


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="stabilityai/TripoSR")
    parser.add_argument("--mesh-resolution", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--skip-renders", action="store_true")
    parser.add_argument("--names", nargs="+", default=["source", "target"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device).eval()
    synchronize()
    load_seconds = time.perf_counter() - start
    load_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    objects = {}
    for name in args.names:
        image_path = args.inputs / f"{name}_gray.png"
        image = Image.open(image_path).convert("RGB")
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            code = model([image], device=device)
        synchronize()
        inference_seconds = time.perf_counter() - start
        inference_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        start = time.perf_counter()
        with torch.inference_mode():
            mesh = model.extract_mesh(code, True, resolution=args.mesh_resolution)[0]
        synchronize()
        extraction_seconds = time.perf_counter() - start
        mesh_path = folder / "mesh.glb"
        mesh.export(mesh_path)
        extraction_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        render_seconds = None
        if not args.skip_renders:
            start = time.perf_counter()
            with torch.inference_mode():
                renders = model.render(code, n_views=8, return_type="pil")[0]
            synchronize()
            render_seconds = time.perf_counter() - start
            for idx, frame in enumerate(renders):
                frame.save(folder / f"view_{idx:02d}.png")
            sheet = Image.new("RGB", (4 * 256, 2 * 256), "white")
            for idx, frame in enumerate(renders):
                sheet.paste(frame.resize((256, 256)), ((idx % 4) * 256, (idx // 4) * 256))
            sheet.save(folder / "turntable.jpg")
        objects[name] = {
            "input_sha256": sha256(image_path),
            "mesh_sha256": sha256(mesh_path),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "mesh_extents_unscaled": np.asarray(mesh.extents).tolist(),
            "inference_seconds": inference_seconds,
            "extraction_seconds": extraction_seconds,
            "render_seconds": render_seconds,
            "inference_peak_gpu_allocated_mb": inference_peak_mb,
            "extraction_peak_gpu_allocated_mb": extraction_peak_mb,
        }
    report = {
        "model": args.model,
        "device": device,
        "mesh_resolution": args.mesh_resolution,
        "chunk_size": args.chunk_size,
        "mesh_extractor": "scikit-image CPU fallback for torchmcubes; neural model runs on GPU",
        "model_load_seconds": load_seconds,
        "model_load_peak_gpu_allocated_mb": load_peak_mb,
        "objects": objects,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
