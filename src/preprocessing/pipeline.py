"""
Persian Banking OCR — Preprocessing Pipeline
Phase 1: Deskewing, shadow removal, adaptive thresholding
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreprocessResult:
    cleaned: np.ndarray
    skew_angle: float
    had_shadow: bool
    original_shape: tuple


def detect_skew(gray: np.ndarray) -> float:
    """Hough-transform based skew detection."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:30]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def remove_shadow(gray: np.ndarray) -> np.ndarray:
    """Background normalisation using large morphological kernel."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    bg = cv2.dilate(gray, kernel)
    bg = cv2.GaussianBlur(bg, (51, 51), 0)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding — handles uneven illumination."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21, C=8
    )


def preprocess(image_path: str) -> PreprocessResult:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    original_shape = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phase 1a — skew correction
    angle = detect_skew(gray)
    gray  = deskew(gray, angle)

    # Phase 1b — shadow detection (std of local patches)
    h8, w8 = (gray.shape[0] // 8) * 8, (gray.shape[1] // 8) * 8; gray_crop = gray[:h8, :w8]; patches = gray_crop.reshape(8, h8 // 8, 8, w8 // 8)
    local_std = patches.std(axis=(1, 3))
    had_shadow = bool(local_std.std() > 12)

    if had_shadow:
        gray = remove_shadow(gray)

    # Phase 1c — binarise
    cleaned = adaptive_threshold(gray)

    return PreprocessResult(
        cleaned=cleaned,
        skew_angle=angle,
        had_shadow=had_shadow,
        original_shape=original_shape,
    )


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        # Quick test on a synthetic sample
        samples = list(Path("data/synthetic").glob("*_degraded.png"))
        if not samples:
            print("No samples found — run scripts/generate_synthetic_data.py first")
            sys.exit(1)
        path = str(samples[0])

    result = preprocess(path)
    out = Path(path).stem + "_preprocessed.png"
    cv2.imwrite(out, result.cleaned)
    print(json.dumps({
        "input":       path,
        "output":      out,
        "skew_angle":  round(result.skew_angle, 2),
        "had_shadow":  result.had_shadow,
        "shape":       list(result.original_shape),
    }, ensure_ascii=False, indent=2))
