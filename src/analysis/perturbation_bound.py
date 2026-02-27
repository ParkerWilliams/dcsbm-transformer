"""Controlled perturbation experiments for softmax filtering bound verification.

Injects perturbations of controlled magnitude into QK^T at selected steps,
recomputes softmax -> AV -> AVWo, and measures the actual spectral change
vs the theoretical bound from the derivation in docs/softmax_bound.tex.

Standalone analysis module -- does NOT modify the evaluation pipeline.
"""

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.experiment import ExperimentConfig
from src.model.types import ExtractionMode

log = logging.getLogger(__name__)


def compute_theoretical_bound(
    qkt_fro_norm: float,
    v_spectral_norm: float,
    wo_spectral_norm: float,
    d_k: int,
    epsilon: float,
) -> float:
    """Compute the theoretical bound from Theorem 6.1.

    The bound is: epsilon * ||QK^T||_F * ||V||_2 * ||W_O||_2 / (2 * sqrt(d_k))

    When epsilon is measured relative to the SCALED QK^T (i.e., QK^T/sqrt(d_k)),
    the sqrt(d_k) factors cancel and the bound simplifies to:
    epsilon * ||QK^T_scaled||_F * ||V||_2 * ||W_O||_2 / 2

    Args:
        qkt_fro_norm: Frobenius norm of the (scaled) QK^T matrix.
        v_spectral_norm: Spectral norm (top singular value) of V.
        wo_spectral_norm: Spectral norm of W_O.
        d_k: Key dimension (d_model for single-head).
        epsilon: Relative perturbation magnitude.

    Returns:
        Theoretical upper bound on ||Delta(AVWo)||_F.
    """
    # When working with the scaled QK^T (as extracted from the model),
    # the 1/sqrt(d_k) is already applied, so the bound is:
    # epsilon * ||QK^T_scaled||_F * ||V||_2 * ||W_O||_2 / 2
    return epsilon * qkt_fro_norm * v_spectral_norm * wo_spectral_norm / 2.0


def inject_perturbation(
    qkt_scaled: torch.Tensor,
    values: torch.Tensor,
    wo_weight: torch.Tensor,
    perturbation: torch.Tensor,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    """Inject perturbation into scaled QK^T and recompute AVWo.

    Args:
        qkt_scaled: Scaled QK^T matrix [T, T] (already divided by sqrt(d_k)).
        values: Value matrix [T, D].
        wo_weight: Output projection weight [D, D].
        perturbation: Perturbation to add to scaled QK^T [T, T].
        causal_mask: Boolean lower-triangular mask [T, T].

    Returns:
        Perturbed AVWo matrix [T, D].
    """
    # Perturbed scaled scores
    qkt_perturbed = qkt_scaled + perturbation

    # Apply causal mask (fill future positions with -inf for softmax)
    qkt_masked = qkt_perturbed.masked_fill(~causal_mask, float("-inf"))

    # Softmax to get attention weights
    A_perturbed = F.softmax(qkt_masked, dim=-1)

    # Handle NaN from all-masked rows (first row in some edge cases)
    A_perturbed = torch.nan_to_num(A_perturbed, nan=0.0)

    # AV
    av_perturbed = A_perturbed @ values

    # AVWo
    avwo_perturbed = av_perturbed @ wo_weight.T

    return avwo_perturbed


def compute_spectral_change(
    avwo_original: torch.Tensor,
    avwo_perturbed: torch.Tensor,
) -> float:
    """Measure spectral change between original and perturbed AVWo.

    Computes ||sigma(perturbed) - sigma(original)||_2 where sigma
    denotes the sorted singular value vector.

    Args:
        avwo_original: Original AVWo matrix [T, D].
        avwo_perturbed: Perturbed AVWo matrix [T, D].

    Returns:
        L2 norm of singular value difference vector.
    """
    sigma_orig = torch.linalg.svdvals(avwo_original)
    sigma_pert = torch.linalg.svdvals(avwo_perturbed)

    # Pad shorter vector if shapes differ (shouldn't happen for same-shape matrices)
    min_len = min(len(sigma_orig), len(sigma_pert))
    diff = sigma_pert[:min_len] - sigma_orig[:min_len]

    return float(torch.linalg.norm(diff).item())


def generate_adversarial_direction(
    qkt_scaled: torch.Tensor,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    """Generate adversarial perturbation direction aligned with top singular vector.

    The adversarial direction is u_1 @ v_1^T from the SVD of QK^T,
    which amplifies the dominant attention pattern.

    Args:
        qkt_scaled: Scaled QK^T matrix [T, T].
        causal_mask: Boolean lower-triangular mask [T, T].

    Returns:
        Unit Frobenius norm direction matrix [T, T], zero in upper triangle.
    """
    # SVD of the causally masked QK^T
    qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
    U, S, Vh = torch.linalg.svd(qkt_masked, full_matrices=False)

    # Rank-1 direction from top singular vectors
    u1 = U[:, 0]  # [T]
    v1 = Vh[0, :]  # [T]
    direction = torch.outer(u1, v1)  # [T, T]

    # Apply causal mask
    direction = direction.masked_fill(~causal_mask, 0.0)

    # Normalize to unit Frobenius norm
    fro_norm = torch.linalg.norm(direction, "fro")
    if fro_norm > 1e-12:
        direction = direction / fro_norm

    return direction


def generate_random_direction(
    T: int,
    causal_mask: torch.Tensor,
    generator: torch.Generator,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a random perturbation direction with unit Frobenius norm.

    Args:
        T: Sequence length (matrix dimension).
        causal_mask: Boolean lower-triangular mask [T, T].
        generator: Torch random generator for reproducibility.
        device: Device for the tensor.

    Returns:
        Unit Frobenius norm direction matrix [T, T], zero in upper triangle.
    """
    # Generate on CPU (generator is always CPU) then move to target device
    direction = torch.randn(T, T, generator=generator)
    if device is not None and device.type != "cpu":
        direction = direction.to(device)

    # Apply causal mask
    direction = direction.masked_fill(~causal_mask, 0.0)

    # Normalize to unit Frobenius norm
    fro_norm = torch.linalg.norm(direction, "fro")
    if fro_norm > 1e-12:
        direction = direction / fro_norm

    return direction


def run_perturbation_at_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    magnitudes: list[float],
    n_random: int,
    seed: int,
) -> dict[str, Any]:
    """Run perturbation experiments at a single generation step.

    Args:
        model: TransformerLM model in eval mode.
        input_ids: Input token IDs [1, T].
        layer_idx: Which transformer layer to analyze.
        magnitudes: List of epsilon values (relative perturbation magnitudes).
        n_random: Number of random directions per magnitude.
        seed: Random seed for reproducibility.

    Returns:
        Dict with per-magnitude results including adversarial and random
        spectral change ratios (actual / theoretical bound).
    """
    device = input_ids.device
    d_model = model.d_model

    # Forward pass with extraction
    with torch.no_grad():
        output = model(input_ids, mode=ExtractionMode.SVD_TARGETS)

    # Extract internals for the target layer (batch dim 0, head 0)
    # Multi-head output shape: [B, n_layers, n_heads, T, T]
    # For perturbation bound analysis, use head 0 (single-head or first head)
    qkt_scaled = output.qkt[0, layer_idx, 0].clone()  # [T, T], already scaled by 1/sqrt(d_k)
    V = output.values[0, layer_idx, 0].clone()  # [T, d_head]
    A_orig = output.attention_weights[0, layer_idx, 0].clone()  # [T, T]
    Wo = model.blocks[layer_idx].attention.W_o.weight.detach().clone()  # [D, D]

    T = qkt_scaled.shape[0]

    # Build causal mask
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    # Compute original AVWo
    AV_orig = A_orig @ V  # [T, D]
    avwo_orig = AV_orig @ Wo.T  # [T, D]

    # Compute norms for the theoretical bound
    qkt_fro = float(torch.linalg.norm(qkt_scaled.masked_fill(~causal_mask, 0.0), "fro").item())
    v_spec = float(torch.linalg.svdvals(V)[0].item())
    wo_spec = float(torch.linalg.svdvals(Wo)[0].item())

    # Generate adversarial direction
    adv_direction = generate_adversarial_direction(qkt_scaled, causal_mask)

    # Random generator (always CPU -- generate_random_direction creates on CPU then moves)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    results: dict[str, Any] = {}

    for eps in magnitudes:
        # Theoretical bound
        bound = compute_theoretical_bound(qkt_fro, v_spec, wo_spec, d_model, eps)

        # --- Adversarial perturbation ---
        adv_perturbation = eps * qkt_fro * adv_direction
        avwo_adv = inject_perturbation(qkt_scaled, V, Wo, adv_perturbation, causal_mask)
        adv_change = compute_spectral_change(avwo_orig, avwo_adv)
        adv_ratio = adv_change / bound if bound > 1e-15 else 0.0

        # --- Random perturbations ---
        random_ratios = []
        for _ in range(n_random):
            rand_dir = generate_random_direction(T, causal_mask, gen, device=device)
            rand_perturbation = eps * qkt_fro * rand_dir
            avwo_rand = inject_perturbation(qkt_scaled, V, Wo, rand_perturbation, causal_mask)
            rand_change = compute_spectral_change(avwo_orig, avwo_rand)
            rand_ratio = rand_change / bound if bound > 1e-15 else 0.0
            random_ratios.append(rand_ratio)

        results[str(eps)] = {
            "adversarial_ratio": adv_ratio,
            "random_ratios": random_ratios,
            "theoretical_bound": bound,
        }

    return results


def run_perturbation_experiment(
    model: nn.Module,
    eval_walks: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    layer_idx: int = 0,
    magnitudes: list[float] | None = None,
    n_random: int = 20,
    n_steps: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Top-level perturbation experiment orchestrator.

    Selects steps uniformly from eval walks, runs perturbation experiments
    at each step, and aggregates results across all steps.

    Args:
        model: Trained TransformerLM in eval mode.
        eval_walks: Evaluation walk array [N, walk_length].
        config: ExperimentConfig for context window etc.
        device: Computation device.
        layer_idx: Which transformer layer to analyze.
        magnitudes: List of epsilon values. Default: [0.01, 0.05, 0.10, 0.25].
        n_random: Number of random directions per (step, magnitude).
        n_steps: Number of steps to sample for experiments.
        seed: Random seed.

    Returns:
        Dict ready for result.json["metrics"]["perturbation_bound"].
    """
    if magnitudes is None:
        magnitudes = [0.01, 0.05, 0.10, 0.25]

    model.eval()
    w = config.training.w
    d_model = config.model.d_model
    n_walks = eval_walks.shape[0]

    # Select steps: use different walks and positions >= w
    rng = np.random.default_rng(seed)
    step_configs = []

    for _ in range(n_steps):
        walk_idx = rng.integers(0, n_walks)
        # Use a position that gives us a full context window
        walk = eval_walks[walk_idx]
        # Take the first w tokens as input (positions 0..w-1)
        input_tokens = walk[:w]
        step_configs.append(input_tokens)

    # Aggregate results across steps
    all_results_by_magnitude: dict[str, dict] = {}
    for eps in magnitudes:
        all_results_by_magnitude[str(eps)] = {
            "adversarial_ratios": [],
            "random_ratios": [],
            "bounds": [],
        }

    for step_idx, input_tokens in enumerate(step_configs):
        input_ids = torch.tensor(
            input_tokens.reshape(1, -1), dtype=torch.long, device=device
        )

        try:
            step_result = run_perturbation_at_step(
                model, input_ids, layer_idx, magnitudes, n_random,
                seed=seed + step_idx,
            )
        except Exception as e:
            log.warning("Perturbation experiment failed at step %d: %s", step_idx, e)
            continue

        for eps in magnitudes:
            eps_key = str(eps)
            if eps_key in step_result:
                all_results_by_magnitude[eps_key]["adversarial_ratios"].append(
                    step_result[eps_key]["adversarial_ratio"]
                )
                all_results_by_magnitude[eps_key]["random_ratios"].extend(
                    step_result[eps_key]["random_ratios"]
                )
                all_results_by_magnitude[eps_key]["bounds"].append(
                    step_result[eps_key]["theoretical_bound"]
                )

    # Aggregate into summary statistics
    all_ratios = []  # For overall tightness ratio
    total_n = 0
    total_exceeding = 0

    by_magnitude: dict[str, dict] = {}

    for eps in magnitudes:
        eps_key = str(eps)
        data = all_results_by_magnitude[eps_key]

        adv_ratios = data["adversarial_ratios"]
        rand_ratios = data["random_ratios"]
        bounds = data["bounds"]

        # Adversarial summary
        adv_n = len(adv_ratios)
        adv_exceeding = sum(1 for r in adv_ratios if r > 1.0)
        adv_mean = float(np.mean(adv_ratios)) if adv_ratios else 0.0
        adv_max = float(np.max(adv_ratios)) if adv_ratios else 0.0

        # Random summary
        rand_n = len(rand_ratios)
        rand_exceeding = sum(1 for r in rand_ratios if r > 1.0)
        rand_mean = float(np.mean(rand_ratios)) if rand_ratios else 0.0
        rand_max = float(np.max(rand_ratios)) if rand_ratios else 0.0

        by_magnitude[eps_key] = {
            "adversarial": {
                "mean_ratio": adv_mean,
                "max_ratio": adv_max,
                "n_exceeding_bound": adv_exceeding,
                "n_total": adv_n,
            },
            "random": {
                "mean_ratio": rand_mean,
                "max_ratio": rand_max,
                "n_exceeding_bound": rand_exceeding,
                "n_total": rand_n,
            },
            "theoretical_bound_value_mean": float(np.mean(bounds)) if bounds else 0.0,
        }

        # Accumulate for overall stats
        all_ratios.extend(adv_ratios)
        all_ratios.extend(rand_ratios)
        total_n += adv_n + rand_n
        total_exceeding += adv_exceeding + rand_exceeding

    # Overall metrics
    tightness_ratio = float(np.median(all_ratios)) if all_ratios else 0.0
    violation_rate = total_exceeding / total_n if total_n > 0 else 0.0
    bound_verified = violation_rate < 0.05

    return {
        "config": {
            "magnitudes": magnitudes,
            "n_random_directions": n_random,
            "n_steps": n_steps,
            "layer_idx": layer_idx,
            "seed": seed,
            "d_model": d_model,
        },
        "theoretical_bound_formula": (
            "epsilon * ||QK^T||_F * ||V||_2 * ||W_O||_2 / (2 * sqrt(d_k))"
        ),
        "by_magnitude": by_magnitude,
        "tightness_ratio": tightness_ratio,
        "violation_rate": violation_rate,
        "bound_verified": bound_verified,
    }
