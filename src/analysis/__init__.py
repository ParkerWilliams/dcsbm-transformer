"""Analysis module for predictive horizon and statistical analysis.

Provides event extraction from evaluation output, AUROC-based predictive
horizon computation, and statistical controls.
"""

from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)

__all__ = [
    "AnalysisEvent",
    "extract_events",
    "filter_contaminated_events",
    "stratify_by_r",
]
