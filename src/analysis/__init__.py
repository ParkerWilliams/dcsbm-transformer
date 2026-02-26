"""Analysis module for predictive horizon and statistical analysis.

Provides event extraction from evaluation output, AUROC-based predictive
horizon computation, and statistical controls.
"""

from src.analysis.auroc_horizon import (
    auroc_from_groups,
    compute_auroc_curve,
    compute_predictive_horizon,
    run_auroc_analysis,
    run_shuffle_control,
)
from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)

__all__ = [
    "AnalysisEvent",
    "auroc_from_groups",
    "compute_auroc_curve",
    "compute_predictive_horizon",
    "extract_events",
    "filter_contaminated_events",
    "run_auroc_analysis",
    "run_shuffle_control",
    "stratify_by_r",
]
