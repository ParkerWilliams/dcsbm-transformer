"""Null model baseline: jumper-free walk generation, position-matched drift extraction,
and Marchenko-Pastur random matrix reference.

Standalone analysis module -- takes a trained model + graph and produces null analysis
independently. Can run null analysis on any existing experiment. Does NOT modify the
evaluation pipeline (per CONTEXT.md locked decision).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import quad
from scipy.stats import kstest

from src.config.experiment import ExperimentConfig
from src.evaluation.pipeline import EvaluationResult, fused_evaluate
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.walk.generator import generate_batch_unguided_walks

log = logging.getLogger(__name__)


def generate_null_walks(
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    n_walks: int,
    seed: int = 9999,
) -> np.ndarray:
    """Generate walks that never encounter a jumper vertex.

    Uses column-filtered adjacency to prevent walks from reaching jumper vertices.
    Same walk length as config.training.walk_length.

    Args:
        graph_data: DCSBM graph (same graph used for training/evaluation).
        jumpers: List of JumperInfo from the experiment.
        config: ExperimentConfig for walk_length.
        n_walks: Number of null walks to generate (typically 5x violation count).
        seed: Random seed for null walk generation.

    Returns:
        Array of shape (n_walks, walk_length) with dtype int32.
    """
    walk_length = config.training.walk_length
    jumper_set = {j.vertex_id for j in jumpers}

    if not jumper_set:
        log.warning("No jumpers provided; null walks are standard unguided walks")
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, graph_data.n, size=n_walks, dtype=np.int32)
        return generate_batch_unguided_walks(
            starts, walk_length, rng,
            graph_data.adjacency.indptr, graph_data.adjacency.indices,
        )

    # 1. Build filtered adjacency: zero out COLUMNS for jumper vertices
    #    This prevents walks from reaching jumper vertices.
    adj = graph_data.adjacency.copy().tolil()
    for v in jumper_set:
        adj[:, v] = 0
    filtered_csr = adj.tocsr()
    filtered_csr.eliminate_zeros()

    # 2. Find valid start vertices (non-jumper with out-degree > 0)
    degrees = np.diff(filtered_csr.indptr)
    valid_mask = np.ones(graph_data.n, dtype=bool)
    for v in jumper_set:
        valid_mask[v] = False
    valid_mask &= degrees > 0
    valid_vertices = np.where(valid_mask)[0]

    if len(valid_vertices) < 10:
        log.warning(
            "Only %d valid start vertices after jumper filtering; "
            "falling back to discard approach",
            len(valid_vertices),
        )
        return _generate_null_walks_discard(
            graph_data, jumper_set, walk_length, n_walks, seed
        )

    # 3. Generate walks on filtered graph
    rng = np.random.default_rng(seed)
    starts = rng.choice(valid_vertices, size=n_walks, replace=True).astype(np.int32)

    walks = generate_batch_unguided_walks(
        starts, walk_length, rng,
        filtered_csr.indptr, filtered_csr.indices,
    )

    # 4. Verify: no walk contains a jumper vertex
    visited = set(walks.flatten().tolist())
    violating = visited & jumper_set
    if violating:
        raise RuntimeError(
            f"Null walk verification failed: walks visit jumper vertices {violating}"
        )

    return walks


def _generate_null_walks_discard(
    graph_data: GraphData,
    jumper_set: set[int],
    walk_length: int,
    n_walks: int,
    seed: int,
) -> np.ndarray:
    """Fallback: generate on full graph, discard walks visiting jumpers.

    Used when the filtered graph has too few valid start vertices.
    """
    rng = np.random.default_rng(seed)
    indptr = graph_data.adjacency.indptr
    indices = graph_data.adjacency.indices

    collected = []
    max_attempts = n_walks * 20  # safety limit
    attempts = 0

    while len(collected) < n_walks and attempts < max_attempts:
        batch_size = min(n_walks * 2, n_walks - len(collected) + 100)
        starts = rng.integers(0, graph_data.n, size=batch_size, dtype=np.int32)
        batch = generate_batch_unguided_walks(
            starts, walk_length, rng, indptr, indices,
        )
        for i in range(batch.shape[0]):
            if not set(batch[i].tolist()) & jumper_set:
                collected.append(batch[i])
                if len(collected) >= n_walks:
                    break
        attempts += batch_size

    if len(collected) < n_walks:
        raise RuntimeError(
            f"Could not generate {n_walks} null walks after {max_attempts} attempts; "
            f"only collected {len(collected)}"
        )

    return np.stack(collected[:n_walks], axis=0)


def run_null_evaluation(
    model: nn.Module,
    null_walks: np.ndarray,
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    device: torch.device,
    batch_size: int = 32,
) -> EvaluationResult:
    """Evaluate null walks through the trained model to collect SVD metrics.

    Calls fused_evaluate() on the null walks. Since null walks contain no
    jumper vertices, all behavioral labels will be NOT_APPLICABLE.

    Args:
        model: Trained TransformerLM in eval mode.
        null_walks: Null walk array from generate_null_walks().
        graph_data: Same graph as used for training.
        jumpers: Same jumpers as the experiment (needed for fused_evaluate API).
        config: ExperimentConfig.
        device: Computation device.
        batch_size: Batch size for evaluation.

    Returns:
        EvaluationResult with SVD metrics for null walks.
    """
    return fused_evaluate(
        model, null_walks, graph_data, jumpers, config, device, batch_size
    )


def extract_position_matched_drift(
    metric_array: np.ndarray,
    event_positions: list[int],
    max_lookback: int,
) -> dict[int, np.ndarray]:
    """Extract metric values at position-matched lookback distances.

    For each event position and each lookback j=1..max_lookback, extracts
    the metric value at position (event_position - j) across all sequences.
    Pools across all event positions.

    Args:
        metric_array: Shape [n_sequences, max_steps-1]. Metric values per step.
        event_positions: Absolute positions corresponding to violation resolution_steps.
        max_lookback: Maximum lookback distance (typically r).

    Returns:
        Dict mapping lookback distance j -> array of metric values across all
        sequences and all event positions, NaN-filtered.
    """
    n_steps = metric_array.shape[1]
    result: dict[int, np.ndarray] = {}

    for j in range(1, max_lookback + 1):
        all_values = []
        for pos in event_positions:
            idx = pos - j
            if 0 <= idx < n_steps:
                all_values.append(metric_array[:, idx])

        if all_values:
            pooled = np.concatenate(all_values)
            finite_mask = np.isfinite(pooled)
            result[j] = pooled[finite_mask]
        else:
            result[j] = np.array([], dtype=np.float32)

    return result


def marchenko_pastur_pdf(x: float, gamma: float, sigma2: float = 1.0) -> float:
    """Marchenko-Pastur probability density function.

    For the distribution of eigenvalues of (1/n) X^T X where X is m x n
    with iid entries of variance sigma^2 and gamma = m/n.

    f(x) = sqrt((lam+ - x)(x - lam-)) / (2 * pi * sigma^2 * gamma * x)
    for x in [lam-, lam+], else 0.

    Args:
        x: Point at which to evaluate the PDF.
        gamma: Aspect ratio m/n.
        sigma2: Variance of the entries.

    Returns:
        PDF value at x.
    """
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2

    if x <= lam_minus or x >= lam_plus:
        return 0.0

    numerator = np.sqrt((lam_plus - x) * (x - lam_minus))
    denominator = 2.0 * np.pi * sigma2 * gamma * x
    return float(numerator / denominator)


def marchenko_pastur_cdf(x, gamma: float, sigma2: float = 1.0):
    """Marchenko-Pastur cumulative distribution function.

    Computed by numerical integration of the PDF from lambda_minus to x.
    Handles both scalar and array inputs (scipy.stats.kstest passes arrays).

    Args:
        x: Point(s) at which to evaluate the CDF. Scalar or array.
        gamma: Aspect ratio.
        sigma2: Variance parameter.

    Returns:
        CDF value(s) at x. Same shape as input.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    scalar_input = x_arr.ndim == 0
    x_arr = np.atleast_1d(x_arr)

    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2

    result = np.zeros_like(x_arr, dtype=np.float64)

    for i, xi in enumerate(x_arr):
        if xi <= lam_minus:
            result[i] = 0.0
        elif xi >= lam_plus:
            result[i] = 1.0
        else:
            val, _ = quad(marchenko_pastur_pdf, lam_minus, float(xi), args=(gamma, sigma2))
            result[i] = val

    if scalar_input:
        return float(result[0])
    return result


def run_mp_ks_test(
    singular_values: np.ndarray,
    gamma: float,
) -> dict:
    """Compare empirical QK^T singular values against Marchenko-Pastur distribution.

    Calibrates sigma^2 from the mean of squared singular values using the
    MP mean formula: E[lambda] = sigma^2 * (1 + gamma), so
    sigma^2 = mean(sv^2) / (1 + gamma).

    Uses scipy.stats.kstest with a callable CDF.

    Args:
        singular_values: 1D array of empirical singular values from QK^T.
        gamma: Aspect ratio w / d_model.

    Returns:
        Dict with: ks_statistic, ks_p_value, gamma, sigma2, lambda_minus, lambda_plus.
    """
    sv_squared = singular_values.astype(np.float64) ** 2

    # Calibrate sigma^2 from data
    sigma2 = float(np.mean(sv_squared)) / (1.0 + gamma)

    # KS test against MP CDF
    ks_result = kstest(
        sv_squared,
        lambda x: marchenko_pastur_cdf(x, gamma, sigma2),
    )

    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

    return {
        "ks_statistic": float(ks_result.statistic),
        "ks_p_value": float(ks_result.pvalue),
        "gamma": float(gamma),
        "sigma2": float(sigma2),
        "lambda_minus": float(lam_minus),
        "lambda_plus": float(lam_plus),
    }
