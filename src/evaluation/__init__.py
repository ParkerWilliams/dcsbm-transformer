"""Evaluation package for behavioral classification and SVD metric collection."""

from src.evaluation.svd_metrics import (
    CONDITION_CAP,
    EPS,
    compute_all_metrics,
    condition_number,
    grassmannian_distance,
    guard_matrix_for_svd,
    rank1_residual_norm,
    read_write_alignment,
    spectral_entropy,
    spectral_gap_1_2,
    spectral_gap_2_3,
    spectral_gap_4_5,
    stable_rank,
)

__all__ = [
    "EPS",
    "CONDITION_CAP",
    "guard_matrix_for_svd",
    "stable_rank",
    "spectral_entropy",
    "spectral_gap_1_2",
    "spectral_gap_2_3",
    "spectral_gap_4_5",
    "condition_number",
    "rank1_residual_norm",
    "read_write_alignment",
    "grassmannian_distance",
    "compute_all_metrics",
]
