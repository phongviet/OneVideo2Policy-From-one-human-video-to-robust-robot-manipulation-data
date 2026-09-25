"""Generate the dimensioned ChArUco calibration board used by the physical workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_M = 0.03
MARKER_LENGTH_M = 0.022
DICTIONARY_NAME = "DICT_5X5_100"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/calibration"))
    parser.add_argument("--pixels-per-square", type=int, default=240)
    args = parser.parse_args()
    if args.pixels_per_square < 80:
        raise ValueError("pixels per square must be at least 80")
    args.output.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary
    )
    size_px = (SQUARES_X * args.pixels_per_square, SQUARES_Y * args.pixels_per_square)
    image = board.generateImage(size_px, marginSize=0, borderBits=1)
    png = args.output / "charuco-7x5-30mm.png"
    pdf = args.output / "charuco-7x5-30mm.pdf"
    if not cv2.imwrite(str(png), image):
        raise RuntimeError("failed to write ChArUco PNG")
    width_m = SQUARES_X * SQUARE_LENGTH_M
    height_m = SQUARES_Y * SQUARE_LENGTH_M
    figure = plt.figure(figsize=(width_m / 0.0254, height_m / 0.0254), frameon=False)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.axis("off")
    figure.savefig(
        pdf,
        format="pdf",
        dpi=args.pixels_per_square / (SQUARE_LENGTH_M / 0.0254),
        bbox_inches=None,
        pad_inches=0,
        metadata={"Creator": "OneVideo2Policy", "CreationDate": None},
    )
    plt.close(figure)
    report = {
        "schema_version": 1,
        "dictionary": DICTIONARY_NAME,
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_m": SQUARE_LENGTH_M,
        "marker_length_m": MARKER_LENGTH_M,
        "printed_width_m": width_m,
        "printed_height_m": height_m,
        "inner_corners": int(len(board.getChessboardCorners())),
        "files": {
            png.name: {"sha256": sha256(png), "pixels": list(size_px)},
            pdf.name: {"sha256": sha256(pdf), "page_size_mm": [width_m * 1000, height_m * 1000]},
        },
        "print_instruction": (
            "Print PDF at 100% / actual size and verify square length with callipers."
        ),
    }
    (args.output / "charuco-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
