"""Fetch the public Video2Robo website samples with provenance and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "088d3d66690506908d6386345e08845dc1203615"
REPOSITORY = "https://github.com/video2robo/video2robo.github.io"
RAW_ROOT = f"https://raw.githubusercontent.com/video2robo/video2robo.github.io/{REVISION}"
TASKS = ("attach", "drum", "place", "pour", "stack", "sweep")


def sample_specs() -> list[dict[str, str | int | None]]:
    specs: list[dict[str, str | int | None]] = []
    for task in TASKS:
        specs.append(
            {
                "task": task,
                "kind": "input_human_video",
                "sample": None,
                "view": None,
                "path": f"assets/mp4/{task}/0__{task}.mp4",
            }
        )
        for sample in range(5):
            for camera_index, view in enumerate(("front", "side")):
                specs.append(
                    {
                        "task": task,
                        "kind": "rendered_demonstration_preview",
                        "sample": sample,
                        "view": view,
                        "path": f"assets/mp4/{task}/{sample:06d}_{camera_index}.mp4",
                    }
                )
    return specs


def download(spec: dict[str, str | int | None], output: Path) -> dict:
    relative_path = str(spec["path"])
    destination = output / relative_path.removeprefix("assets/mp4/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_url = f"{RAW_ROOT}/{relative_path}"
    if not destination.exists():
        request = urllib.request.Request(source_url, headers={"User-Agent": "onevideo2policy"})
        with urllib.request.urlopen(request) as response:
            destination.write_bytes(response.read())
    payload = destination.read_bytes()
    return {
        **spec,
        "local_path": destination.relative_to(output).as_posix(),
        "source_url": source_url,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/video2robo_public_samples"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    specs = sample_specs()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        files = list(executor.map(lambda spec: download(spec, args.output), specs))
    manifest = {
        "status": "public_preview_only",
        "repository": REPOSITORY,
        "revision": REVISION,
        "project_page": "https://video2robo.github.io/",
        "license": "not specified in the source repository",
        "files": files,
        "counts": {
            "tasks": len(TASKS),
            "input_human_videos": sum(row["kind"] == "input_human_video" for row in files),
            "rendered_preview_videos": sum(
                row["kind"] == "rendered_demonstration_preview" for row in files
            ),
        },
        "not_included": [
            "robot actions or absolute joint targets",
            "proprioception or end-effector trajectories",
            "episode metadata and training splits",
            "the full generated training set",
            "policy checkpoints or training configuration",
        ],
        "use_limitation": (
            "These website videos are suitable for visual reference only and are not "
            "a trainable reproduction of the Video2Robo image-action dataset."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({**manifest["counts"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
