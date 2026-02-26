"""SVD computational overhead benchmarking (OVHD-01, OVHD-02, OVHD-03).

Compares wall-clock timing and accuracy of full SVD, randomized SVD,
and values-only SVD for each target matrix type (QK^T, WvWo, AVWo).
Uses CUDA events for GPU timing with warmup iterations.
"""

import logging
import time

import numpy as np
import torch

from src.config.experiment import ExperimentConfig

log = logging.getLogger(__name__)


def _full_svd(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full SVD via torch.linalg.svd (economy mode).

    Args:
        M: Input matrix tensor.

    Returns:
        Tuple of (U, S, Vh).
    """
    return torch.linalg.svd(M, full_matrices=False)


def _randomized_svd(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD via torch.svd_lowrank.

    Note: torch.svd_lowrank returns (U, S, V) where V is not transposed.

    Args:
        M: Input matrix tensor.

    Returns:
        Tuple of (U, S, V). V is the right singular vectors (not transposed).
    """
    q = min(M.shape[-2], M.shape[-1])
    U, S, V = torch.svd_lowrank(M, q=q)
    return U, S, V


def _values_only_svd(M: torch.Tensor) -> torch.Tensor:
    """Singular values only via torch.linalg.svdvals.

    Args:
        M: Input matrix tensor.

    Returns:
        Singular values tensor S.
    """
    return torch.linalg.svdvals(M)


def _time_svd_method(
    matrix: torch.Tensor,
    method_fn: callable,
    n_warmup: int = 5,
    n_timed: int = 20,
) -> float:
    """Time an SVD method with warmup iterations.

    Uses CUDA events for GPU timing when available, otherwise
    uses time.perf_counter() for CPU timing.

    Args:
        matrix: Input matrix tensor.
        method_fn: Callable that takes a matrix and returns SVD results.
        n_warmup: Number of warmup iterations.
        n_timed: Number of timed iterations.

    Returns:
        Average time per call in milliseconds.
    """
    use_cuda = matrix.is_cuda

    # Warmup
    for _ in range(n_warmup):
        method_fn(matrix)

    if use_cuda:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(n_timed):
            method_fn(matrix)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        return elapsed_ms / n_timed
    else:
        start = time.perf_counter()
        for _ in range(n_timed):
            method_fn(matrix)
        end = time.perf_counter()

        elapsed_s = end - start
        return (elapsed_s / n_timed) * 1000.0  # Convert to ms


def _compare_accuracy(
    reference_S: torch.Tensor,
    approx_S: torch.Tensor,
) -> dict[str, float]:
    """Compare accuracy of approximate singular values against reference.

    Args:
        reference_S: Singular values from full SVD (ground truth).
        approx_S: Singular values from approximate method.

    Returns:
        Dict with 'frob_error' (relative Frobenius error) and
        'sv_correlation' (Pearson correlation of singular values).
    """
    ref_norm = torch.norm(reference_S).item()
    if ref_norm == 0:
        return {"frob_error": float("nan"), "sv_correlation": float("nan")}

    # Align lengths (randomized may return fewer values)
    min_len = min(len(reference_S), len(approx_S))
    ref = reference_S[:min_len].float()
    approx = approx_S[:min_len].float()

    # Relative Frobenius error on singular values
    frob_error = (torch.norm(ref - approx) / torch.norm(ref)).item()

    # Pearson correlation
    if min_len < 2:
        sv_correlation = 1.0 if frob_error == 0 else float("nan")
    else:
        corr_matrix = torch.corrcoef(torch.stack([ref, approx]))
        sv_correlation = corr_matrix[0, 1].item()
        if not np.isfinite(sv_correlation):
            sv_correlation = float("nan")

    return {"frob_error": float(frob_error), "sv_correlation": float(sv_correlation)}


def benchmark_svd_for_target(
    target_name: str,
    matrix_shape: tuple[int, int],
    device: torch.device | None = None,
    n_warmup: int = 5,
    n_timed: int = 20,
    seed: int = 42,
) -> dict:
    """Benchmark all SVD methods for a single target at given matrix dimensions.

    Args:
        target_name: Name of the SVD target (e.g., 'qkt', 'wvwo', 'avwo').
        matrix_shape: Shape of the matrix (rows, cols).
        device: Device for computation. Defaults to CPU.
        n_warmup: Number of warmup iterations.
        n_timed: Number of timed iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with timing and accuracy results for all three SVD methods.
    """
    if device is None:
        device = torch.device("cpu")

    # Generate reproducible random matrix
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    matrix = torch.randn(*matrix_shape, generator=gen, device="cpu").to(device)

    # Time all three methods
    full_ms = _time_svd_method(matrix, _full_svd, n_warmup, n_timed)
    rand_ms = _time_svd_method(matrix, _randomized_svd, n_warmup, n_timed)
    vals_ms = _time_svd_method(matrix, _values_only_svd, n_warmup, n_timed)

    # Compute accuracy comparison using full SVD as reference
    _, ref_S, _ = _full_svd(matrix)
    _, rand_S, _ = _randomized_svd(matrix)
    vals_S = _values_only_svd(matrix)

    rand_accuracy = _compare_accuracy(ref_S, rand_S)
    vals_accuracy = _compare_accuracy(ref_S, vals_S)

    return {
        "target": target_name,
        "matrix_shape": list(matrix_shape),
        "full_svd_ms": float(full_ms),
        "randomized_svd_ms": float(rand_ms),
        "values_only_ms": float(vals_ms),
        "randomized_frob_error": rand_accuracy["frob_error"],
        "randomized_sv_correlation": rand_accuracy["sv_correlation"],
        "values_only_sv_correlation": vals_accuracy["sv_correlation"],
    }


def run_svd_benchmark(
    config: ExperimentConfig,
    device: torch.device | None = None,
    n_warmup: int = 5,
    n_timed: int = 20,
) -> dict:
    """Orchestrate SVD benchmarking across all targets.

    Determines matrix dimensions from the experiment configuration and
    benchmarks all three SVD methods for each target.

    Args:
        config: Experiment configuration with model and training parameters.
        device: Device for computation. Defaults to CPU.
        n_warmup: Number of warmup iterations.
        n_timed: Number of timed iterations.

    Returns:
        Nested dict with timing and accuracy results for all targets,
        plus summary statistics.
    """
    if device is None:
        device = torch.device("cpu")

    w = config.training.w
    d_model = config.model.d_model
    n_layers = config.model.n_layers

    # Matrix dimensions per target
    target_shapes = {
        "qkt": (w, w),
        "wvwo": (d_model, d_model),
        "avwo": (w, d_model),
    }

    by_target: dict[str, dict] = {}
    for target_name, shape in target_shapes.items():
        log.info("Benchmarking SVD for %s (%s)", target_name, shape)
        result = benchmark_svd_for_target(
            target_name, shape, device, n_warmup, n_timed
        )
        by_target[target_name] = result

    # Summary: total SVD cost per step (all targets * n_layers)
    total_full = sum(t["full_svd_ms"] for t in by_target.values()) * n_layers
    total_rand = sum(t["randomized_svd_ms"] for t in by_target.values()) * n_layers
    total_vals = sum(t["values_only_ms"] for t in by_target.values()) * n_layers

    method_totals = {
        "full_svd": total_full,
        "randomized_svd": total_rand,
        "values_only": total_vals,
    }
    fastest_method = min(method_totals, key=method_totals.get)

    return {
        "config": {
            "n_warmup": n_warmup,
            "n_timed": n_timed,
            "device": str(device),
        },
        "by_target": by_target,
        "summary": {
            "total_svd_ms_per_step": float(total_full),
            "fastest_method": fastest_method,
            "method_totals_ms": {k: float(v) for k, v in method_totals.items()},
        },
    }
