"""Package and verify the current real Place inputs for a larger CUDA host."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from onevideo2policy.video.manifest import validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(package: Path) -> dict[str, object]:
    with (package / "handoff-manifest.json").open(encoding="utf-8") as stream:
        record = json.load(stream)
    files = record["files"]
    for relative, expected in files.items():
        path = package / relative
        if path.resolve().is_relative_to(package.resolve()) is False or not path.is_file():
            raise ValueError(f"Missing or unsafe handoff file: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")

    manifest_path = package / "source/manifest.json"
    manifest_check = validate_manifest(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    count = manifest_check["frame_count"]
    shape = (manifest["resolution"]["height"], manifest["resolution"]["width"])
    for name in ("source", "target"):
        masks = sorted((package / "source/masks" / name).glob("*.png"))
        if [path.stem for path in masks] != [f"{i:06d}" for i in range(count)]:
            raise ValueError(f"Mask sequence is incomplete: {name}")
        for path in masks:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != shape:
                raise ValueError(f"Mask shape or decode failed: {path}")
        with np.load(package / "source/tracks" / f"{name}.npz", allow_pickle=False) as tracks:
            if tracks["xy"].shape[:2] != tracks["visible"].shape:
                raise ValueError(f"Track visibility shape mismatch: {name}")
            if tracks["xy"].shape[0] != count or tracks["xy"].shape[2] != 2:
                raise ValueError(f"Track sequence shape mismatch: {name}")
    with (package / "faithful/run-spec.json").open(encoding="utf-8") as stream:
        spec = json.load(stream)
    for item in spec["inputs"]:
        path = package / "faithful" / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Faithful input checksum mismatch: {item['path']}")
    return {"status": "verified", "files": len(files), "frames": count, "resolution": shape}


def prepare(package: Path) -> dict[str, object]:
    if package.exists():
        raise FileExistsError(package)
    source = package / "source"
    source.mkdir(parents=True)
    interim = ROOT / "data/interim/real_place_img_6256"
    perception = ROOT / "data/interim/real_place_img_6256_perception"
    shutil.copy2(ROOT / "data/raw/IMG_6256.MOV", source / "IMG_6256.MOV")
    shutil.copytree(interim / "frames", source / "frames")
    shutil.copytree(perception / "masks", source / "masks")
    shutil.copytree(perception / "tracks", source / "tracks")
    shutil.copytree(perception / "crops_native", source / "crops_native")
    shutil.copy2(perception / "gate-report.json", source / "gate-report.json")
    with (interim / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest["source_video"] = "IMG_6256.MOV"
    (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    shutil.copytree(ROOT / "results/faithful_bundle/real_place_img_6256", package / "faithful")
    code = package / "code"
    code.mkdir()
    for directory in ("src", "scripts", "configs", "tests", "docs"):
        shutil.copytree(
            ROOT / directory,
            code / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for filename in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, code / filename)
    shutil.copy2(
        ROOT / "results/local_e2e/real_place_img_6256_smoke/report.json",
        package / "local-report.json",
    )
    shutil.copy2(
        ROOT / "results/perception/real_place_img_6256/assisted-agreement.json",
        package / "assisted-agreement.json",
    )
    record = {
        "schema_version": 1,
        "source": "IMG_6256.MOV",
        "claim_status": (
            "Local systems smoke only; independent perception gate and metric scene "
            "measurements pending"
        ),
        "files": {
            path.relative_to(package).as_posix(): sha256(path)
            for path in sorted(package.rglob("*"))
            if path.is_file()
        },
    }
    (package / "handoff-manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return verify(package)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--verify", action="store_true", help="Verify an existing package")
    args = parser.parse_args()
    result = verify(args.package) if args.verify else prepare(args.package)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
