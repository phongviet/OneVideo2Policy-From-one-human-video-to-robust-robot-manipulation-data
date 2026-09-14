from pathlib import Path

import cv2
import numpy as np
import pytest

from onevideo2policy.reconstruction.crops import (
    export_rgba_crops,
    make_rgba_crop,
    resize_masks_to_frame,
)


def test_make_rgba_crop_is_square_and_transparent() -> None:
    rgb = np.full((8, 12, 3), [20, 40, 60], dtype=np.uint8)
    mask = np.zeros((8, 12), dtype=bool)
    mask[2:6, 4:10] = True

    rgba, metadata = make_rgba_crop(rgb, mask, padding_fraction=0)

    assert rgba.shape == (6, 6, 4)
    assert np.all(rgba[1:5, :, 3] == 255)
    assert np.all(rgba[[0, 5], :, 3] == 0)
    assert metadata["object_bbox_xyxy"] == [4, 2, 10, 6]
    assert metadata["foreground_pixels"] == 24


def test_export_rgba_crops_writes_lossless_manifest(tmp_path: Path) -> None:
    rgb = np.full((5, 6, 3), 100, dtype=np.uint8)
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 2:5] = True

    manifest = export_rgba_crops(rgb, {"ball": mask}, tmp_path, frame_idx=7)
    written = cv2.imread(str(tmp_path / "ball.png"), cv2.IMREAD_UNCHANGED)

    assert manifest["frame_idx"] == 7
    assert written.shape[-1] == 4
    assert (tmp_path / "crops.json").is_file()


def test_make_rgba_crop_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_rgba_crop(np.zeros((4, 5, 3), dtype=np.uint8), np.zeros((4, 5), dtype=bool))


def test_resize_masks_to_frame_uses_nearest_neighbor() -> None:
    mask = np.array([[True, False], [False, False]])

    resized = resize_masks_to_frame({"object": mask}, (4, 4))["object"]

    assert resized.dtype == bool
    assert resized[:2, :2].all()
    assert not resized[2:, :].any()
