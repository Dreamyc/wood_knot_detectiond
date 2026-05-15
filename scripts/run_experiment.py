from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wood_knot.detector import baseline_detect_knots, detect_knots
from wood_knot.evaluation import average_metrics, evaluate_detections
from wood_knot.generator import generate_synthetic_board
from wood_knot.report import write_docx_report
from wood_knot.visualization import draw_annotations, draw_metric_chart


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs"
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, float]] = []
    baseline_rows: list[dict[str, float]] = []

    for index, seed in enumerate(range(1, 25), start=1):
        knot_count = 3 + seed % 3
        sample = generate_synthetic_board(width=640, height=360, knot_count=knot_count, seed=seed)
        gt_boxes = [region.bbox for region in sample.knots]

        rule_detections = detect_knots(sample.image)
        baseline_detections = baseline_detect_knots(sample.image)
        rule_metrics = evaluate_detections(gt_boxes, rule_detections)
        baseline_metrics = evaluate_detections(gt_boxes, baseline_detections)
        rule_rows.append(rule_metrics)
        baseline_rows.append(baseline_metrics)

        rows.append(
            {
                "seed": seed,
                "knot_count": knot_count,
                "rule_detections": len(rule_detections),
                "baseline_detections": len(baseline_detections),
                "rule_precision": rule_metrics["precision"],
                "rule_recall": rule_metrics["recall"],
                "rule_f1": rule_metrics["f1"],
                "rule_mean_iou": rule_metrics["mean_iou"],
                "baseline_precision": baseline_metrics["precision"],
                "baseline_recall": baseline_metrics["recall"],
                "baseline_f1": baseline_metrics["f1"],
                "baseline_mean_iou": baseline_metrics["mean_iou"],
            }
        )

        if index <= 6:
            sample.image.save(sample_dir / f"sample_{seed:02d}_input.png")
            draw_annotations(
                sample.image,
                sample.knots,
                sample.pith,
                rule_detections,
                sample_dir / f"sample_{seed:02d}_annotated.png",
            )

    summary = {"rule": average_metrics(rule_rows), "baseline": average_metrics(baseline_rows)}

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    draw_metric_chart(summary, output_dir / "comparison_chart.png")
    write_docx_report(summary, output_dir / "课程设计报告-木材结疤检测系统.docx")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
