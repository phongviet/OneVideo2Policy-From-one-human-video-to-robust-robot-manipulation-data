"""Create color-assisted working masks from the raw IMG_6256 frames.

These are deliberately NOT independent ground-truth masks: the annotator has
already viewed the model predictions. Do not use them for a formal IoU gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path("results/perception/real_place_img_6256")
PREDICTIONS = Path("data/interim/real_place_img_6256_perception/masks")

# Approximate label centers read from the raw-frame contact sheet and checked
# against color connected components. They are not SAM2 predictions.
LABEL_CENTERS = {
    0: (602, 235),
    2: (594, 244),
    4: (592, 246),
    6: (590, 250),
    8: (603, 253),
    10: (602, 267),
    12: (600, 269),
    14: (600, 274),
    16: (600, 272),
    18: (606, 218),
    20: (663, 97),
    22: (680, 128),
    24: (641, 139),
    26: (493, 149),
    28: (423, 147),
    30: (401, 149),
    32: (382, 200),
    34: (363, 297),
    36: (375, 307),
    37: (377, 308),
    38: (377, 306),
    40: (360, 293),
    42: (329, 300),
    44: (307, 300),
    46: (284, 297),
    48: (254, 294),
    50: (260, 284),
    52: (253, 269),
    54: (248, 268),
    56: (250, 282),
}


def label_candidates(hsv: np.ndarray) -> np.ndarray:
    hue, saturation, value = cv2.split(hsv)
    pink = (hue >= 135) & (hue <= 175) & (saturation >= 55) & (value >= 100)
    yellow = (hue >= 18) & (hue <= 38) & (saturation >= 100) & (value >= 110)
    mask = (pink | yellow).astype(np.uint8)
    mask[:50] = 0
    mask[410:] = 0
    mask[:, :120] = 0
    mask[:, 700:] = 0
    return mask


def source_mask(hsv: np.ndarray, frame_id: int) -> np.ndarray:
    """Trace the visible round container from its colored lid and white body."""
    x, y = LABEL_CENTERS[frame_id]
    hue, saturation, value = cv2.split(hsv)
    lid = label_candidates(hsv).astype(bool)
    white_body = (saturation < 40) & (value > 120)
    blue_lid = (hue >= 75) & (hue <= 115) & (saturation >= 55) & (value > 115)
    candidate = (lid | white_body | blue_lid).astype(np.uint8)
    # The hand is not foreground. The physical container fits this local outline
    # throughout the clip; color selection clips off fingers crossing the outline.
    outline = np.zeros(candidate.shape, np.uint8)
    cv2.ellipse(outline, (x + 8, y + 10), (49, 39), 0, 0, 360, 1, -1)
    candidate &= outline
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # Keep components belonging to the object, including split patches under fingers.
    count, labels, stats, centers = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros(candidate.shape, np.uint8)
    for component in range(1, count):
        center = centers[component]
        area = stats[component, cv2.CC_STAT_AREA]
        if area >= 25 and np.linalg.norm(center - (x, y)) < 52:
            keep[labels == component] = 1
    return keep


def target_mask(hsv: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Trace visible blue target pixels without hallucinating hand-covered areas."""
    hue, saturation, value = cv2.split(hsv)
    blue = (hue >= 88) & (hue <= 122) & (saturation >= 45) & (value >= 45)
    mask = blue.astype(np.uint8)
    mask[:130] = 0
    mask[:, :80] = 0
    mask[:, 450:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # The blue printed patch on the container is not part of the target.
    mask[cv2.dilate(source, np.ones((3, 3), np.uint8)) > 0] = 0
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Replace this script's masks")
    args = parser.parse_args()
    workspace = json.loads((WORKSPACE / "workspace.json").read_text())
    expected = {item["frame_id"] for item in workspace["frames"]}
    if expected != set(LABEL_CENTERS):
        raise ValueError("Label center map does not match the frozen evaluation frames")
    review_frames = []
    for item in workspace["frames"]:
        frame = cv2.imread(str(WORKSPACE / item["image"]))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        source = source_mask(hsv, item["frame_id"])
        target = target_mask(hsv, source)
        for name, mask in {"source": source, "target": target}.items():
            destination = WORKSPACE / item["annotations"][name]
            if destination.exists() and not args.replace:
                raise FileExistsError(f"Refusing to overwrite annotation: {destination}")
            if not cv2.imwrite(str(destination), mask * 255):
                raise OSError(f"Could not write {destination}")
            print(item["frame_id"], name, int(mask.sum()))
        review = frame.copy()
        for mask, color in ((source, (0, 255, 0)), (target, (255, 0, 0))):
            occupied = mask > 0
            review[occupied] = (0.45 * review[occupied] + 0.55 * np.array(color)).astype(np.uint8)
        cv2.putText(
            review,
            str(item["frame_id"]),
            (15, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        review_frames.append(cv2.resize(review, (480, 270)))
    montage = np.vstack([np.hstack(review_frames[index : index + 5]) for index in range(0, 30, 5)])
    if not cv2.imwrite(str(WORKSPACE / "working-masks-review.jpg"), montage):
        raise OSError("Could not write working-mask review montage")
    (WORKSPACE / "annotation-provenance.json").write_text(
        json.dumps(
            {
                "status": "assisted_working_masks_not_independent_ground_truth",
                "method": "Raw-frame color segmentation with manually checked source centers",
                "warning": (
                    "The annotator had already seen SAM2 predictions. Agreement with "
                    "those predictions must not be used as a formal perception gate."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    agreement: dict[str, dict[str, float]] = {}
    for name in ("source", "target"):
        scores = []
        for item in workspace["frames"]:
            frame_name = f"{item['frame_id']:06d}.png"
            annotated = cv2.imread(str(WORKSPACE / item["annotations"][name]), 0) > 0
            predicted = cv2.imread(str(PREDICTIONS / name / frame_name), 0) > 0
            intersection = np.logical_and(annotated, predicted).sum()
            union = np.logical_or(annotated, predicted).sum()
            scores.append(float(intersection / union) if union else 1.0)
        agreement[name] = {"mean_iou": float(np.mean(scores)), "min_iou": float(min(scores))}
    (WORKSPACE / "assisted-agreement.json").write_text(
        json.dumps(
            {
                "formal_gate_valid": False,
                "reason": "Mask drafting was not blind to the predictions",
                "frames": len(workspace["frames"]),
                "objects": agreement,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
