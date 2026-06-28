"""
Persian Banking OCR — Layout Analysis (Phase 2)
Detects and crops individual fields from a preprocessed cheque image.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


FIELD_TEMPLATES = {
    "date":           {"y_ratio": (0.17, 0.28), "x_ratio": (0.05, 0.75)},
    "amount_numeric": {"y_ratio": (0.30, 0.42), "x_ratio": (0.05, 0.75)},
    "amount_words":   {"y_ratio": (0.43, 0.56), "x_ratio": (0.05, 0.90)},
    "account":        {"y_ratio": (0.57, 0.69), "x_ratio": (0.05, 0.75)},
    "payee":          {"y_ratio": (0.70, 0.82), "x_ratio": (0.05, 0.75)},
}


@dataclass
class FieldCrop:
    name: str
    image: np.ndarray
    bbox: tuple          # (x, y, w, h) in original coords


def crop_fields(binary: np.ndarray) -> List[FieldCrop]:
    """
    Crop known field regions using layout template ratios.
    Falls back to horizontal-line detection for unknown layouts.
    """
    h, w = binary.shape[:2]
    crops = []

    for name, ratios in FIELD_TEMPLATES.items():
        y1 = int(ratios["y_ratio"][0] * h)
        y2 = int(ratios["y_ratio"][1] * h)
        x1 = int(ratios["x_ratio"][0] * w)
        x2 = int(ratios["x_ratio"][1] * w)

        roi = binary[y1:y2, x1:x2]

        # Padding
        pad = 4
        roi_padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad,
                                        cv2.BORDER_CONSTANT, value=255)
        crops.append(FieldCrop(name=name, image=roi_padded,
                               bbox=(x1, y1, x2 - x1, y2 - y1)))

    return crops


def detect_lines(binary: np.ndarray) -> List[int]:
    """Detect horizontal ruling lines (y-positions)."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1] // 2, 1))
    morph   = cv2.morphologyEx(~binary, cv2.MORPH_OPEN, kernel)
    ys      = np.where(morph.sum(axis=1) > binary.shape[1] * 0.4)[0]
    if len(ys) == 0:
        return []
    # Cluster close y-values
    groups, g = [[ys[0]]], [ys[0]]
    for y in ys[1:]:
        if y - g[-1] < 5:
            g.append(y)
        else:
            groups.append(g := [y])
    return [int(np.mean(g)) for g in groups]


if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        samples = list(Path("data/synthetic").glob("*_clean.png"))
        path = str(samples[0]) if samples else None

    if not path:
        print("Usage: python layout.py <image_path>")
        sys.exit(1)

    img    = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 8)
    crops  = crop_fields(binary)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for c in crops:
        fname = out_dir / f"{Path(path).stem}_{c.name}.png"
        cv2.imwrite(str(fname), c.image)
        results.append({"field": c.name, "bbox": list(c.bbox), "saved": str(fname)})

    print(json.dumps(results, indent=2))
