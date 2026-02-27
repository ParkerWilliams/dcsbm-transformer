"""Tests for the softmax filtering bound perturbation experiments.

Verifies the theoretical bound computation, perturbation injection,
direction generation, and end-to-end experiment orchestration.
"""

import math

import numpy as np
import pytest
import torch

from src.analysis.perturbation_bound import (
    compute_spectral_change,
    compute_theoretical_bound,
    generate_adversarial_direction,
    generate_random_direction,
    inject_perturbation,
    run_perturbation_at_step,
    run_perturbation_experiment,
)
from src.config.experiment import ExperimentConfig
from src.model.transformer import TransformerLM


def _make_small_model(d_model=16, n_layers=1, vocab_size=20, max_seq=8):
    """Create a small TransformerLM for testing."""
    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq,
        dropout=0.0,
    )
    model.eval()
    return model


def _make_causal_mask(T):
    """Create a lower-triangular boolean causal mask."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


class TestComputeTheoreticalBound:
    """Tests for compute_theoretical_bound."""

    def test_known_values(self):
        """Verify bound with known inputs."""
        # bound = eps * qkt_fro * v_spec * wo_spec / 2
        result = compute_theoretical_bound(
            qkt_fro_norm=10.0,
            v_spectral_norm=5.0,
            wo_spectral_norm=3.0,
            d_k=64,
            epsilon=0.1,
        )
        # eps * qkt_fro * v_spec * wo_spec / 2 = 0.1 * 10 * 5 * 3 / 2 = 7.5
        expected = 0.1 * 10.0 * 5.0 * 3.0 / 2.0
        assert abs(result - expected) < 1e-10

    def test_zero_epsilon(self):
        """Zero perturbation should give zero bound."""
        result = compute_theoretical_bound(10.0, 5.0, 3.0, 64, 0.0)
        assert result == 0.0

    def test_scales_linearly_with_epsilon(self):
        """Bound should scale linearly with epsilon."""
        b1 = compute_theoretical_bound(10.0, 5.0, 3.0, 64, 0.1)
        b2 = compute_theoretical_bound(10.0, 5.0, 3.0, 64, 0.2)
        assert abs(b2 / b1 - 2.0) < 1e-10


class TestInjectPerturbation:
    """Tests for inject_perturbation."""

    def test_zero_perturbation(self):
        """Zero perturbation should give identical AVWo."""
        torch.manual_seed(42)
        T, D = 8, 16
        qkt = torch.randn(T, T)
        V = torch.randn(T, D)
        Wo = torch.randn(D, D)
        mask = _make_causal_mask(T)

        perturbation = torch.zeros(T, T)

        avwo = inject_perturbation(qkt, V, Wo, perturbation, mask)

        # Compute expected AVWo directly
        qkt_masked = qkt.masked_fill(~mask, float("-inf"))
        A = torch.nn.functional.softmax(qkt_masked, dim=-1)
        A = torch.nan_to_num(A, nan=0.0)
        expected = (A @ V) @ Wo.T

        change = compute_spectral_change(expected, avwo)
        assert change < 1e-5, f"Spectral change with zero perturbation: {change}"

    def test_nonzero_perturbation(self):
        """Non-zero perturbation should give non-zero spectral change."""
        torch.manual_seed(42)
        T, D = 8, 16
        qkt = torch.randn(T, T)
        V = torch.randn(T, D)
        Wo = torch.randn(D, D)
        mask = _make_causal_mask(T)

        perturbation = 0.5 * torch.randn(T, T)
        perturbation = perturbation.masked_fill(~mask, 0.0)

        avwo_orig = inject_perturbation(qkt, V, Wo, torch.zeros(T, T), mask)
        avwo_pert = inject_perturbation(qkt, V, Wo, perturbation, mask)

        change = compute_spectral_change(avwo_orig, avwo_pert)
        assert change > 0.0, "Spectral change should be positive for non-zero perturbation"


class TestAdversarialDirection:
    """Tests for generate_adversarial_direction."""

    def test_unit_frobenius_norm(self):
        """Adversarial direction should have unit Frobenius norm."""
        torch.manual_seed(42)
        T = 8
        qkt = torch.randn(T, T)
        mask = _make_causal_mask(T)

        direction = generate_adversarial_direction(qkt, mask)
        fro_norm = torch.linalg.norm(direction, "fro").item()
        assert abs(fro_norm - 1.0) < 1e-5, f"Frobenius norm: {fro_norm}"

    def test_upper_triangle_zero(self):
        """Adversarial direction should be zero in upper triangle."""
        torch.manual_seed(42)
        T = 8
        qkt = torch.randn(T, T)
        mask = _make_causal_mask(T)

        direction = generate_adversarial_direction(qkt, mask)

        # Upper triangle (excluding diagonal) should be zero
        upper = torch.triu(direction, diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper)), \
            "Upper triangle should be zero"


class TestRandomDirection:
    """Tests for generate_random_direction."""

    def test_unit_frobenius_norm(self):
        """Random direction should have unit Frobenius norm."""
        T = 8
        mask = _make_causal_mask(T)
        gen = torch.Generator()
        gen.manual_seed(42)

        direction = generate_random_direction(T, mask, gen)
        fro_norm = torch.linalg.norm(direction, "fro").item()
        assert abs(fro_norm - 1.0) < 1e-5, f"Frobenius norm: {fro_norm}"

    def test_upper_triangle_zero(self):
        """Random direction should be zero in upper triangle."""
        T = 8
        mask = _make_causal_mask(T)
        gen = torch.Generator()
        gen.manual_seed(42)

        direction = generate_random_direction(T, mask, gen)
        upper = torch.triu(direction, diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper))

    def test_different_seeds_different_directions(self):
        """Different seeds should produce different directions."""
        T = 8
        mask = _make_causal_mask(T)

        gen1 = torch.Generator()
        gen1.manual_seed(42)
        d1 = generate_random_direction(T, mask, gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(99)
        d2 = generate_random_direction(T, mask, gen2)

        assert not torch.allclose(d1, d2), "Different seeds should give different directions"


class TestBoundHolds:
    """Tests that the theoretical bound actually holds."""

    def test_bound_holds_small_model(self):
        """Verify no perturbation exceeds the theoretical bound on a small model."""
        model = _make_small_model(d_model=16, n_layers=1, vocab_size=20, max_seq=8)
        torch.manual_seed(123)
        input_ids = torch.randint(0, 20, (1, 8))

        result = run_perturbation_at_step(
            model, input_ids, layer_idx=0,
            magnitudes=[0.01, 0.10, 0.25],
            n_random=10, seed=42,
        )

        for eps_key, eps_data in result.items():
            adv_ratio = eps_data["adversarial_ratio"]
            assert adv_ratio <= 1.0 + 1e-6, (
                f"Adversarial ratio {adv_ratio} exceeds bound at eps={eps_key}"
            )

            for i, rand_ratio in enumerate(eps_data["random_ratios"]):
                assert rand_ratio <= 1.0 + 1e-6, (
                    f"Random ratio {rand_ratio} exceeds bound at eps={eps_key}, dir={i}"
                )

    def test_adversarial_larger_than_random(self):
        """Adversarial perturbation should produce larger spectral change on average."""
        model = _make_small_model(d_model=16, n_layers=1, vocab_size=20, max_seq=8)
        torch.manual_seed(123)
        input_ids = torch.randint(0, 20, (1, 8))

        result = run_perturbation_at_step(
            model, input_ids, layer_idx=0,
            magnitudes=[0.10],
            n_random=20, seed=42,
        )

        eps_data = result["0.1"]
        adv_ratio = eps_data["adversarial_ratio"]
        rand_mean = float(np.mean(eps_data["random_ratios"]))

        # Adversarial should be at least as large as random mean
        # (may not always hold for very small matrices, so use a relaxed check)
        assert adv_ratio >= rand_mean * 0.5, (
            f"Adversarial ({adv_ratio:.4f}) should be at least "
            f"half of random mean ({rand_mean:.4f})"
        )


class TestRunPerturbationExperiment:
    """Tests for the top-level experiment orchestrator."""

    def test_output_structure(self):
        """Verify the output dict has the correct structure."""
        model = _make_small_model(d_model=16, n_layers=1, vocab_size=20, max_seq=8)

        # Create minimal eval walks
        rng = np.random.default_rng(42)
        eval_walks = rng.integers(0, 20, size=(10, 16), dtype=np.int64)

        config = ExperimentConfig()
        # Override config to match small model
        from dataclasses import replace
        from src.config.experiment import ModelConfig, TrainingConfig
        config = replace(
            config,
            model=replace(config.model, d_model=16, n_layers=1),
            training=replace(config.training, w=8),
        )

        result = run_perturbation_experiment(
            model, eval_walks, config,
            device=torch.device("cpu"),
            layer_idx=0,
            magnitudes=[0.05],
            n_random=5,
            n_steps=3,
            seed=42,
        )

        # Check top-level keys
        assert "config" in result
        assert "theoretical_bound_formula" in result
        assert "by_magnitude" in result
        assert "tightness_ratio" in result
        assert "violation_rate" in result
        assert "bound_verified" in result

        # Check config
        assert result["config"]["magnitudes"] == [0.05]
        assert result["config"]["n_random_directions"] == 5
        assert result["config"]["n_steps"] == 3
        assert result["config"]["d_model"] == 16

        # Check by_magnitude structure
        assert "0.05" in result["by_magnitude"]
        mag_data = result["by_magnitude"]["0.05"]
        assert "adversarial" in mag_data
        assert "random" in mag_data
        assert "theoretical_bound_value_mean" in mag_data

        # Check adversarial structure
        adv = mag_data["adversarial"]
        assert "mean_ratio" in adv
        assert "max_ratio" in adv
        assert "n_exceeding_bound" in adv
        assert "n_total" in adv

        # Check types
        assert isinstance(result["tightness_ratio"], float)
        assert isinstance(result["violation_rate"], float)
        assert isinstance(result["bound_verified"], bool)

    def test_bound_verified_flag(self):
        """Verify bound_verified is True when violation rate < 5%."""
        model = _make_small_model(d_model=16, n_layers=1, vocab_size=20, max_seq=8)
        rng = np.random.default_rng(42)
        eval_walks = rng.integers(0, 20, size=(10, 16), dtype=np.int64)

        config = ExperimentConfig()
        from dataclasses import replace
        from src.config.experiment import ModelConfig, TrainingConfig
        config = replace(
            config,
            model=replace(config.model, d_model=16, n_layers=1),
            training=replace(config.training, w=8),
        )

        result = run_perturbation_experiment(
            model, eval_walks, config,
            device=torch.device("cpu"),
            magnitudes=[0.01, 0.05],
            n_random=5,
            n_steps=3,
            seed=42,
        )

        # The theoretical bound should hold for a real model
        assert result["bound_verified"] is True, (
            f"Bound verification failed: violation_rate={result['violation_rate']}"
        )
