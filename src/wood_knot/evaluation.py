from __future__ import annotations

from typing import Iterable, Protocol


BBox = tuple[int, int, int, int]


class DetectionLike(Protocol):
    bbox: BBox
    score: float


def compute_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_detections(
    ground_truth: Iterable[BBox],
    detections: Iterable[DetectionLike],
    iou_threshold: float = 0.30,
) -> dict[str, float]:
    gt_boxes = list(ground_truth)
    dets = sorted(list(detections), key=lambda item: item.score, reverse=True)
    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    tp = 0
    fp = 0

    for detection in dets:
        best_index = -1
        best_iou = 0.0
        for index, gt in enumerate(gt_boxes):
            if index in matched_gt:
                continue
            iou = compute_iou(detection.bbox, gt)
            if iou > best_iou:
                best_index = index
                best_iou = iou
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_index)
            matched_ious.append(best_iou)
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def average_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0}
    keys = ("precision", "recall", "f1", "mean_iou")
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
