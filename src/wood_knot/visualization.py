from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .detector import Detection
from .generator import Region


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 12)
    except OSError:
        return ImageFont.load_default()


def draw_annotations(
    image: Image.Image,
    ground_truth: Iterable[Region],
    pith: Region,
    detections: Iterable[Detection],
    output_path: Path,
) -> None:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _font()

    for region in ground_truth:
        draw.rectangle(region.bbox, outline=(20, 170, 70), width=3)
        draw.text((region.bbox[0] + 3, region.bbox[1] + 3), "GT knot", fill=(20, 120, 50), font=font)

    draw.rectangle(pith.bbox, outline=(45, 125, 235), width=3)
    draw.text((pith.bbox[0] + 3, pith.bbox[1] + 3), "tree pith", fill=(20, 70, 180), font=font)

    for detection in detections:
        draw.rectangle(detection.bbox, outline=(220, 40, 35), width=2)
        draw.text(
            (detection.bbox[0] + 3, max(0, detection.bbox[1] - 14)),
            f"knot {detection.score:.2f}",
            fill=(200, 25, 25),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def draw_metric_chart(summary: dict[str, dict[str, float]], output_path: Path) -> None:
    width, height = 760, 420
    image = Image.new("RGB", (width, height), (250, 250, 246))
    draw = ImageDraw.Draw(image)
    font = _font()
    title_font = font

    draw.text((30, 24), "Detection performance comparison", fill=(25, 28, 33), font=title_font)
    metrics = ["precision", "recall", "f1", "mean_iou"]
    labels = {"rule": "Rule/OpenCV", "baseline": "Gray baseline"}
    colors = {"rule": (32, 126, 91), "baseline": (183, 88, 49)}
    left = 110
    top = 84
    chart_width = 560
    group_gap = 56
    bar_width = 42
    baseline_y = 340
    scale = 230

    draw.line((left - 10, baseline_y, left + chart_width + 20, baseline_y), fill=(70, 70, 70), width=1)
    for i in range(6):
        y = baseline_y - i * scale / 5
        draw.line((left - 10, y, left + chart_width + 20, y), fill=(222, 222, 216), width=1)
        draw.text((46, y - 7), f"{i / 5:.1f}", fill=(80, 80, 80), font=font)

    for index, metric in enumerate(metrics):
        group_x = left + index * (2 * bar_width + group_gap)
        for offset, method in enumerate(["rule", "baseline"]):
            value = summary[method][metric]
            x1 = group_x + offset * (bar_width + 8)
            y1 = baseline_y - value * scale
            draw.rectangle((x1, y1, x1 + bar_width, baseline_y), fill=colors[method])
            draw.text((x1 - 3, y1 - 17), f"{value:.2f}", fill=(30, 30, 30), font=font)
        draw.text((group_x - 8, baseline_y + 14), metric, fill=(45, 45, 45), font=font)

    legend_x = 510
    for index, method in enumerate(["rule", "baseline"]):
        y = 38 + index * 22
        draw.rectangle((legend_x, y, legend_x + 14, y + 14), fill=colors[method])
        draw.text((legend_x + 22, y - 1), labels[method], fill=(35, 35, 35), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
