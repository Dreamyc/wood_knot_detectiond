"""Wood knot detection course-design package."""

from .detector import Detection, baseline_detect_knots, detect_knots
from .evaluation import compute_iou, evaluate_detections
from .generator import Region, SyntheticSample, generate_synthetic_board

__all__ = [
    "Detection",
    "Region",
    "SyntheticSample",
    "baseline_detect_knots",
    "compute_iou",
    "detect_knots",
    "evaluate_detections",
    "generate_synthetic_board",
]
