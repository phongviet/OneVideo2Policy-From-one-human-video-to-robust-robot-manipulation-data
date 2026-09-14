import json
from pathlib import Path

import numpy as np
import pytest

from onevideo2policy.video.manifest import validate_manifest
from onevideo2policy.video.point_sampling import sample_mask_points


def test_validate_rgbd_manifest(tmp_path: Path) -> None:
    (tmp_path / "source.mp4").touch()
    (tmp_path / "frames").mkdir()
    (tmp_path / "depth").mkdir()
    frames = []
    for frame_id in range(3):
        (tmp_path / "frames" / f"{frame_id:06d}.jpg").touch()
        np.save(tmp_path / "depth" / f"{frame_id:06d}.npy", np.ones((4, 5), dtype=np.uint16))
        frames.append(
            {
                "frame_id": frame_id,
                "source_frame_id": frame_id,
                "timestamp_s": frame_id / 30,
                "rgb": f"frames/{frame_id:06d}.jpg",
                "depth": f"depth/{frame_id:06d}.npy",
            }
        )
    manifest = {
        "schema_version": 1,
        "source_video": "source.mp4",
        "source_frame_count": 3,
        "resolution": {"width": 5, "height": 4},
        "native_fps": 30.0,
        "sample_fps": 30.0,
        "frame_count": 3,
        "frames": frames,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    report = validate_manifest(path)

    assert report == {
        "frame_count": 3,
        "duration_s": pytest.approx(2 / 30),
        "has_depth": True,
        "depth_shape": [4, 5],
    }


def test_manifest_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    (tmp_path / "source.mp4").touch()
    rgb = tmp_path / "frame.jpg"
    rgb.touch()
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_video": "source.mp4",
                "source_frame_count": 2,
                "resolution": {"width": 5, "height": 4},
                "native_fps": 30,
                "sample_fps": 30,
                "frame_count": 2,
                "frames": [
                    {"frame_id": 0, "source_frame_id": 0, "timestamp_s": 0.0, "rgb": "frame.jpg"},
                    {"frame_id": 1, "source_frame_id": 1, "timestamp_s": 0.0, "rgb": "frame.jpg"},
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="timestamp_s"):
        validate_manifest(path)


def test_point_sampling_is_deterministic_and_xy_ordered() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 2:5] = True

    first = sample_mask_points(mask, 5, seed=7)
    second = sample_mask_points(mask, 5, seed=7)

    assert np.array_equal(first, second)
    assert len(np.unique(first, axis=0)) == 5
    assert all(mask[y, x] for x, y in first)


def test_point_sampling_rejects_insufficient_pixels() -> None:
    with pytest.raises(ValueError, match="eligible pixels"):
        sample_mask_points(np.eye(3, dtype=bool), 4)


def test_point_sampling_border_erodes_object_boundary() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:8] = True

    points = sample_mask_points(mask, 9, seed=7, border=2)

    assert {(int(x), int(y)) for x, y in points} == {
        (x, y) for y in range(3, 6) for x in range(3, 6)
    }
