import numpy as np
import pytest

from onevideo2policy.video.hoi4d import (
    decode_hoi4d_motion_mask,
    hoi4d_color_map,
    import_hoi4d_sequence,
)
from onevideo2policy.video.manifest import validate_manifest


def test_hoi4d_motion_mask_uses_official_palette() -> None:
    palette = hoi4d_color_map()
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 1] = palette[1]
    image[1, 2] = palette[3]

    mask = decode_hoi4d_motion_mask(image, [1, 3])

    assert np.array_equal(mask, [[False, True, False], [False, False, True]])


def test_hoi4d_motion_mask_rejects_unusable_labels() -> None:
    with pytest.raises(ValueError, match="labels"):
        decode_hoi4d_motion_mask(np.zeros((2, 2, 3), dtype=np.uint8), [0])


def test_import_hoi4d_sequence_preserves_ground_truth(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    sequence = tmp_path / "sequence"
    rgb_dir = sequence / "align_rgb"
    depth_dir = sequence / "align_depth"
    motion_dir = sequence / "2Dseg" / "mask"
    for path in (rgb_dir, depth_dir, motion_dir):
        path.mkdir(parents=True)
    (rgb_dir / "image.mp4").touch()
    palette = hoi4d_color_map()
    for frame_id in range(2):
        rgb = np.full((4, 5, 3), 50 + frame_id, dtype=np.uint8)
        depth = np.full((4, 5), 1000 + frame_id, dtype=np.uint16)
        motion = np.zeros((4, 5, 3), dtype=np.uint8)
        motion[0, 1] = palette[1]
        motion[3, 4] = palette[3]
        assert cv2.imwrite(str(rgb_dir / f"{frame_id:05d}.jpg"), rgb)
        assert cv2.imwrite(str(depth_dir / f"{frame_id:05d}.png"), depth)
        assert cv2.imwrite(
            str(motion_dir / f"{frame_id:05d}.png"), cv2.cvtColor(motion, cv2.COLOR_RGB2BGR)
        )

    output = tmp_path / "imported"
    manifest = import_hoi4d_sequence(
        sequence, output, source_labels=[1], target_labels=[3], decoded_fps=15
    )

    assert manifest["dataset"]["name"] == "HOI4D"
    assert validate_manifest(output / "manifest.json")["has_depth"] is True
    source = cv2.imread(str(output / "ground_truth/source/000000.png"), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(output / "ground_truth/target/000000.png"), cv2.IMREAD_GRAYSCALE)
    assert source[0, 1] == 255
    assert target[3, 4] == 255
