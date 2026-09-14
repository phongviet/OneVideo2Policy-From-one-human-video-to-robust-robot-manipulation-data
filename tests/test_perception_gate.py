import json
from pathlib import Path

import numpy as np
import pytest

from onevideo2policy.evaluation.perception_gate import (
    binary_mask_iou,
    build_gate_report,
    evaluate_ground_truth_directories,
    load_evaluation_spec,
)


def test_load_evaluation_spec_rejects_duplicate_frames(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "objects": ["source"], "frame_ids": [0, 0]}
        )
    )

    try:
        load_evaluation_spec(path)
    except ValueError as exc:
        assert "frame_ids" in str(exc)
    else:
        raise AssertionError("Duplicate frames must be rejected")


def test_binary_mask_iou() -> None:
    predicted = np.array([[1, 1], [0, 0]], dtype=bool)
    ground_truth = np.array([[1, 0], [1, 0]], dtype=bool)

    assert binary_mask_iou(predicted, ground_truth) == 1 / 3
    assert binary_mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0


def test_gate_report_applies_all_thresholds() -> None:
    diagnostics = {
        "source": {"track_survival": 0.92, "mean_visible_tracks": 29.5},
        "target": {"track_survival": 0.96, "mean_visible_tracks": 30.9},
    }

    passing = build_gate_report(
        {"source": [0.8, 0.9], "target": [0.7, 0.8]},
        diagnostics,
        min_mask_iou=0.7,
        min_track_survival=0.8,
        min_visible_points=12,
        identity_swaps=0,
    )
    swapped = build_gate_report(
        {"source": [0.8], "target": [0.8]},
        diagnostics,
        min_mask_iou=0.7,
        min_track_survival=0.8,
        min_visible_points=12,
        identity_swaps=1,
    )

    assert passing["status"] == "pass"
    assert passing["provisional"] is False
    assert swapped["status"] == "fail"


def test_dataset_ground_truth_evaluation(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    ground_truth = tmp_path / "ground_truth" / "source"
    predictions = tmp_path / "predictions" / "masks" / "source"
    ground_truth.mkdir(parents=True)
    predictions.mkdir(parents=True)
    mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    assert cv2.imwrite(str(ground_truth / "000000.png"), mask)
    assert cv2.imwrite(str(predictions / "000000.png"), mask)
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "objects": {
                    "source": {"track_survival": 0.9, "mean_visible_tracks": 20.0}
                }
            }
        )
    )

    report = evaluate_ground_truth_directories(
        tmp_path / "ground_truth",
        tmp_path / "predictions",
        diagnostics,
        min_mask_iou=0.7,
        min_track_survival=0.8,
        min_visible_points=12,
        identity_swaps=0,
    )

    assert report["status"] == "pass"
    assert report["objects"]["source"]["mean_mask_iou"] == 1.0
