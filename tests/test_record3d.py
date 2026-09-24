from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from onevideo2policy.video.record3d import Record3DArchive


def test_record3d_metadata_and_scaled_intrinsics(tmp_path) -> None:
    archive_path = tmp_path / "sample.r3d"
    metadata = {
        "fps": 30,
        "h": 1920,
        "w": 1440,
        "dh": 256,
        "dw": 192,
        "K": [1200, 0, 0, 0, 1200, 0, 720, 960, 1],
        "poses": [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0.1, 0, 0]],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("metadata", json.dumps(metadata))
    source = Record3DArchive(archive_path)
    assert source.metadata.frame_count == 2
    assert source.metadata.rgb_size == (1920, 1440)
    assert np.allclose(
        source.metadata.intrinsics_rgb,
        [[1200, 0, 720], [0, 1200, 960], [0, 0, 1]],
    )
    assert np.allclose(
        source.intrinsics_for_depth(),
        [[160, 0, 96], [0, 160, 128], [0, 0, 1]],
    )
    with pytest.raises(IndexError):
        source.read_depth_m(2)
