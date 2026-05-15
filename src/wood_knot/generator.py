from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Region:
    label: str
    bbox: BBox


@dataclass(frozen=True)
class SyntheticSample:
    image: Image.Image
    knots: list[Region]
    pith: Region
    seed: int


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _avoid_overlap(candidate: BBox, regions: Iterable[Region], max_iou: float = 0.03) -> bool:
    return all(_bbox_iou(candidate, region.bbox) <= max_iou for region in regions)


def _wood_background(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    grain = 11 * np.sin(2 * math.pi * (6.5 * y + 0.75 * x))
    slow_wave = 8 * np.sin(2 * math.pi * (1.7 * y - 0.25 * x))
    noise = rng.normal(0, 5, size=(height, width))
    tone = grain + slow_wave + noise
    base = np.dstack(
        [
            177 + tone,
            126 + 0.62 * tone,
            73 + 0.35 * tone,
        ]
    )
    arr = np.clip(base, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, "RGB")

    draw = ImageDraw.Draw(image, "RGBA")
    line_rng = random.Random(seed + 1000)
    for row in range(8, height, 12):
        points = []
        phase = line_rng.random() * math.tau
        for col in range(0, width + 8, 8):
            offset = math.sin(col / 26 + phase) * 3 + line_rng.uniform(-1.5, 1.5)
            points.append((col, row + offset))
        color = (98, 61, 34, line_rng.randint(28, 52))
        draw.line(points, fill=color, width=line_rng.choice([1, 1, 2]))
    return image.filter(ImageFilter.SMOOTH_MORE)


def _draw_pith(draw: ImageDraw.ImageDraw, width: int, height: int) -> Region:
    cx = int(width * 0.66)
    cy = int(height * 0.52)
    rx = max(16, int(width * 0.065))
    ry = max(22, int(height * 0.22))
    bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
    draw.ellipse(bbox, fill=(39, 76, 135), outline=(18, 48, 95), width=3)
    inner = (cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2)
    draw.ellipse(inner, outline=(98, 148, 210), width=2)
    return Region("pith", bbox)


def _draw_knot(draw: ImageDraw.ImageDraw, bbox: BBox, rng: random.Random) -> None:
    x1, y1, x2, y2 = bbox
    shadow = (x1 + 2, y1 + 2, x2 + 3, y2 + 3)
    draw.ellipse(shadow, fill=(55, 35, 22))
    draw.ellipse(bbox, fill=(87, 48, 27), outline=(43, 25, 15), width=2)
    inset = max(3, min(x2 - x1, y2 - y1) // 5)
    ring = (x1 + inset, y1 + inset, x2 - inset, y2 - inset)
    draw.ellipse(ring, outline=(137, 78, 39), width=2)
    core_inset = max(5, min(x2 - x1, y2 - y1) // 3)
    core = (x1 + core_inset, y1 + core_inset, x2 - core_inset, y2 - core_inset)
    if core[2] > core[0] and core[3] > core[1]:
        draw.ellipse(core, fill=(38, 24, 16))

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    for _ in range(4):
        angle = rng.uniform(0, math.tau)
        radius = rng.uniform(0.35, 0.48) * min(x2 - x1, y2 - y1)
        ex = cx + math.cos(angle) * radius
        ey = cy + math.sin(angle) * radius
        draw.line((cx, cy, ex, ey), fill=(45, 27, 16), width=1)


def generate_synthetic_board(
    width: int = 640,
    height: int = 360,
    knot_count: int = 5,
    seed: int = 1,
) -> SyntheticSample:
    """Generate a wood-board image with labeled knots and a tree-pith distractor."""

    if width < 120 or height < 90:
        raise ValueError("width and height must be large enough for labeled regions")
    if knot_count < 1:
        raise ValueError("knot_count must be positive")

    rng = random.Random(seed)
    image = _wood_background(width, height, seed)
    draw = ImageDraw.Draw(image)
    pith = _draw_pith(draw, width, height)

    regions: list[Region] = [pith]
    knots: list[Region] = []
    attempts = 0
    while len(knots) < knot_count and attempts < knot_count * 80:
        attempts += 1
        rx = rng.randint(max(10, width // 25), max(14, width // 13))
        ry = rng.randint(max(8, height // 28), max(12, height // 12))
        cx = rng.randint(rx + 8, width - rx - 8)
        cy = rng.randint(ry + 8, height - ry - 8)
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        if not _avoid_overlap(bbox, regions):
            continue
        _draw_knot(draw, bbox, rng)
        region = Region("knot", bbox)
        knots.append(region)
        regions.append(region)

    if len(knots) != knot_count:
        raise RuntimeError("could not place all knots without overlap; use a larger image")

    return SyntheticSample(image=image, knots=knots, pith=pith, seed=seed)
