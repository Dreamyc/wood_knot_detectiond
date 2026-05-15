from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    score: float
    area: int


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def _components_from_mask(mask: np.ndarray, min_area: int, max_area: int) -> list[Detection]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections: list[Detection] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if w < 6 or h < 6:
            continue
        aspect = w / float(h)
        if aspect < 0.32 or aspect > 3.10:
            continue
        extent = area / float(w * h)
        if extent < 0.30:
            continue
        score = min(1.0, 0.45 + extent * 0.45 + min(area / max(min_area * 4, 1), 1.0) * 0.10)
        detections.append(Detection((int(x), int(y), int(x + w), int(y + h)), score, int(area)))
    return sorted(detections, key=lambda det: det.score, reverse=True)


def detect_knots(image: Image.Image) -> list[Detection]:
    """Detect dark brown knot regions while rejecting blue tree-pith markings."""

    rgb = _as_rgb_array(image)
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hue, sat, val = cv2.split(hsv)

    brown_hue = (hue <= 28) | (hue >= 170)
    dark_enough = gray <= 118
    value_limited = val <= 150
    saturated = sat >= 55
    not_blue = ~((hue >= 85) & (hue <= 135) & (sat >= 40))
    mask = (brown_hue & dark_enough & value_limited & saturated & not_blue).astype(np.uint8) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    min_area = max(35, int(width * height * 0.0012))
    max_area = max(min_area + 1, int(width * height * 0.09))
    return _components_from_mask(mask, min_area=min_area, max_area=max_area)


def baseline_detect_knots(image: Image.Image) -> list[Detection]:
    """Simple grayscale threshold baseline used for comparison."""

    rgb = _as_rgb_array(image)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold = min(132, int(np.percentile(gray, 22)))
    mask = (gray <= threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    min_area = max(35, int(width * height * 0.0012))
    max_area = max(min_area + 1, int(width * height * 0.12))
    return _components_from_mask(mask, min_area=min_area, max_area=max_area)
