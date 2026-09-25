"""Compare arbitrary-scale single-image meshes with a metric RGB-D extent reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-reference", required=True, type=Path)
    parser.add_argument("--triposr-report", required=True, type=Path)
    parser.add_argument("--sf3d-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object-name", default="action_camera")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metric_report = read_json(args.metric_reference)
    metric_mm = np.asarray(metric_report["metric_extents_m_descending"]) * 1000
    records = []
    for name, path in [("TripoSR-128", args.triposr_report), ("Stable Fast 3D", args.sf3d_report)]:
        source = read_json(path)
        mesh = source["objects"][args.object_name]
        unscaled = np.sort(np.asarray(mesh["mesh_extents_unscaled"], dtype=float))[::-1]
        scaled_mm = unscaled * metric_mm[0] / unscaled[0]
        ratio = unscaled / unscaled[0]
        dimension_error = np.abs(scaled_mm - metric_mm)
        relative_error = dimension_error / metric_mm
        records.append(
            {
                "model": name,
                "report": str(path),
                "report_sha256": sha256(path),
                "watertight": mesh["watertight"],
                "peak_gpu_allocated_mb": model_peak(source, mesh),
                "mesh_extents_unscaled_descending": unscaled.tolist(),
                "scale_aligned_extents_mm_descending": scaled_mm.tolist(),
                "extent_ratios_descending": ratio.tolist(),
                "absolute_dimension_error_mm": dimension_error.tolist(),
                "secondary_ratio_mae": float(
                    np.mean(np.abs(ratio[1:] - metric_mm[1:] / metric_mm[0]))
                ),
                "max_secondary_relative_dimension_error": float(np.max(relative_error[1:])),
                "passes_15_percent_secondary_extent_gate": bool(np.all(relative_error[1:] <= 0.15)),
            }
        )
    report = {
        "schema_version": 1,
        "experiment": "EM1-0406 action-camera reconstruction proportion gate",
        "metric_reference": str(args.metric_reference),
        "metric_reference_sha256": sha256(args.metric_reference),
        "metric_extents_mm_descending": metric_mm.tolist(),
        "metric_extent_ratios_descending": (metric_mm / metric_mm[0]).tolist(),
        "alignment": "isotropic scale chosen to match the largest metric extent",
        "acceptance_gate": "each secondary scale-aligned extent within 15% of RGB-D reference",
        "models": records,
        "decision": {
            "learned_mesh_gate_passed": any(
                item["passes_15_percent_secondary_extent_gate"] for item in records
            ),
            "visual_mesh": "TripoSR-128 for local speed and watertight output",
            "metric_geometry": "Record3D RGB-D point cloud or fitted primitive",
            "reason": (
                "Stable Fast 3D is proportionally closer but non-watertight and uses 6.17 GiB; "
                "neither learned mesh passes the metric extent gate."
            ),
        },
        "limitations": metric_report["limitations"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_comparison(
        metric_mm, records, args.metric_reference.parent, args.output / "comparison.png"
    )
    print(json.dumps(report, indent=2))


def model_peak(report: dict[str, Any], mesh: dict[str, Any]) -> float:
    candidates = [
        mesh.get("peak_gpu_allocated_mb"),
        mesh.get("extraction_peak_gpu_allocated_mb"),
        mesh.get("inference_peak_gpu_allocated_mb"),
        report.get("model_load_peak_gpu_allocated_mb"),
    ]
    return float(max(value for value in candidates if value is not None))


def plot_comparison(
    metric_mm: np.ndarray,
    records: list[dict[str, Any]],
    input_folder: Path,
    output: Path,
) -> None:
    figure, (image_axis, chart_axis) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.5]}
    )
    image_axis.imshow(Image.open(input_folder / "action_camera_rgba.png"))
    image_axis.set_title("Audited Record3D crop")
    image_axis.axis("off")
    x = np.arange(3)
    width = 0.25
    chart_axis.bar(x - width, metric_mm, width, label="RGB-D reference")
    for index, record in enumerate(records):
        chart_axis.bar(
            x + index * width,
            record["scale_aligned_extents_mm_descending"],
            width,
            label=record["model"],
        )
    chart_axis.set_xticks(x, ["long", "middle", "thickness"])
    chart_axis.set_ylabel("Scale-aligned extent (mm)")
    chart_axis.set_title("Proportion check against metric depth")
    chart_axis.legend(fontsize=8)
    chart_axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
