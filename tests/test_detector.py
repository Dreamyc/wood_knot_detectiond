import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wood_knot.detector import baseline_detect_knots, detect_knots
from wood_knot.evaluation import compute_iou, evaluate_detections
from wood_knot.generator import generate_synthetic_board


class WoodKnotDetectorTests(unittest.TestCase):
    def test_generator_creates_labeled_board(self):
        sample = generate_synthetic_board(width=220, height=140, knot_count=4, seed=7)

        self.assertEqual(sample.image.size, (220, 140))
        self.assertEqual(len(sample.knots), 4)
        self.assertEqual(sample.pith.label, "pith")
        for knot in sample.knots:
            x1, y1, x2, y2 = knot.bbox
            self.assertGreater(x2 - x1, 10)
            self.assertGreater(y2 - y1, 8)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 220)
            self.assertLessEqual(y2, 140)

    def test_rule_detector_finds_knots_without_flagging_blue_pith(self):
        sample = generate_synthetic_board(width=260, height=160, knot_count=3, seed=11)

        detections = detect_knots(sample.image)
        metrics = evaluate_detections([region.bbox for region in sample.knots], detections)

        self.assertGreaterEqual(metrics["recall"], 0.66)
        self.assertGreaterEqual(metrics["precision"], 0.75)
        self.assertTrue(
            all(compute_iou(det.bbox, sample.pith.bbox) < 0.10 for det in detections),
            "tree pith must not be reported as a knot",
        )

    def test_rule_detector_beats_gray_threshold_baseline_on_pith_sample(self):
        sample = generate_synthetic_board(width=260, height=160, knot_count=4, seed=23)

        rule_metrics = evaluate_detections(
            [region.bbox for region in sample.knots],
            detect_knots(sample.image),
        )
        baseline_metrics = evaluate_detections(
            [region.bbox for region in sample.knots],
            baseline_detect_knots(sample.image),
        )

        self.assertGreater(rule_metrics["f1"], baseline_metrics["f1"])
        self.assertGreaterEqual(rule_metrics["f1"], 0.70)

    def test_evaluation_matches_each_ground_truth_once(self):
        ground_truth = [(10, 10, 40, 40), (70, 20, 105, 60)]
        detections = [
            type("DetectionLike", (), {"bbox": (12, 12, 38, 38), "score": 0.9})(),
            type("DetectionLike", (), {"bbox": (68, 18, 108, 62), "score": 0.8})(),
            type("DetectionLike", (), {"bbox": (130, 20, 160, 50), "score": 0.7})(),
        ]

        metrics = evaluate_detections(ground_truth, detections, iou_threshold=0.30)

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
