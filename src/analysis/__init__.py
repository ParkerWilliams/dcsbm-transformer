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
from src.analysis.statistical_controls import (
    apply_statistical_controls,
    auroc_with_bootstrap_ci,
    cohens_d,
    compute_cohens_d_by_lookback,
    compute_correlation_matrix,
    compute_headline_comparison,
    compute_metric_ranking,
    holm_bonferroni,
)

__all__ = [
    "AnalysisEvent",
    "apply_statistical_controls",
    "auroc_from_groups",
    "auroc_with_bootstrap_ci",
    "cohens_d",
    "compute_auroc_curve",
    "compute_cohens_d_by_lookback",
    "compute_correlation_matrix",
    "compute_headline_comparison",
    "compute_metric_ranking",
    "compute_predictive_horizon",
    "extract_events",
    "filter_contaminated_events",
    "holm_bonferroni",
    "run_auroc_analysis",
    "run_shuffle_control",
    "stratify_by_r",
]
