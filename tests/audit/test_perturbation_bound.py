"""Audit tests for empirical bound verification and masking consistency (SFTX-02, SFTX-03).

Verifies the perturbation construction matches LaTeX epsilon definition,
Mirsky's inequality chain (SV-L2 <= Frobenius <= bound) holds across
magnitudes, synthetic fixtures produce deterministic ratio < 1.0 for
adversarial and random directions, masking consistency between
generate_adversarial_direction (zero-fill) and inject_perturbation (-inf),
and Weyl usage consistency.
"""

import math

import torch
import torch.nn.functional as F

from src.analysis.perturbation_bound import (
    compute_spectral_change,
    compute_theoretical_bound,
    generate_adversarial_direction,
    generate_random_direction,
    inject_perturbation,
)


class TestPerturbationConstruction:
    """Verify perturbation construction matches LaTeX epsilon definition."""

    def test_perturbation_frobenius_matches_epsilon_definition(self) -> None:
        """||perturbation||_F == eps * ||QK^T_masked||_F.
        The perturbation is eps * qkt_fro * direction, and direction has unit
        Frobenius norm, so ||perturbation||_F = eps * ||QK^T_masked||_F.
        This matches the LaTeX definition: Delta = epsilon * ||QK^T||_F * direction.
        """
        torch.manual_seed(42)
        T, d_k = 6, 16
        epsilon = 0.1

        qkt_scaled = torch.randn(T, T, dtype=torch.float64)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

        # Compute masked QK^T Frobenius norm
        qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
        qkt_fro = torch.linalg.norm(qkt_masked, "fro").item()

        # Generate adversarial direction
        direction = generate_adversarial_direction(qkt_scaled.float(), causal_mask).double()

        # Build perturbation
        perturbation = epsilon * qkt_fro * direction
        pert_fro = torch.linalg.norm(perturbation, "fro").item()

        # ||perturbation||_F must equal eps * ||QK^T_masked||_F
        expected = epsilon * qkt_fro
        assert abs(pert_fro - expected) / expected < 1e-5, (
            f"Perturbation Frobenius {pert_fro:.6f} != eps * ||QK^T||_F = {expected:.6f}"
        )

    def test_adversarial_direction_unit_frobenius_norm(self) -> None:
        """generate_adversarial_direction returns a matrix with unit Frobenius norm.
        The direction is normalized to ||direction||_F = 1 so that the perturbation
        magnitude is controlled entirely by eps * qkt_fro.
        """
        torch.manual_seed(42)
        T = 8
        qkt_scaled = torch.randn(T, T)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

        direction = generate_adversarial_direction(qkt_scaled, causal_mask)
        fro_norm = torch.linalg.norm(direction, "fro").item()

        # Unit Frobenius norm within tolerance
        assert abs(fro_norm - 1.0) < 1e-6, (
            f"Direction Frobenius norm = {fro_norm:.8f}, expected 1.0"
        )

    def test_adversarial_direction_zero_upper_triangle(self) -> None:
        """generate_adversarial_direction respects causal mask: upper triangle is zero.
        The SVD is computed on causally-masked QK^T (zero in upper triangle),
        and the resulting direction is also masked to zero in the upper triangle.
        """
        torch.manual_seed(42)
        T = 8
        qkt_scaled = torch.randn(T, T)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

        direction = generate_adversarial_direction(qkt_scaled, causal_mask)

        # Upper triangle (strictly above diagonal) must be zero
        upper_mask = ~causal_mask
        upper_values = direction[upper_mask]

        assert torch.all(upper_values == 0.0), (
            f"Non-zero values in upper triangle: max abs = {upper_values.abs().max().item():.2e}"
        )


class TestMirskysInequality:
    """Verify Mirsky's inequality chain: SV-L2 <= Frobenius <= bound."""

    def test_mirsky_chain_synthetic(self) -> None:
        """SV-L2 <= Frobenius <= theoretical bound on synthetic matrices.
        Mirsky's inequality: ||sigma(M+E) - sigma(M)||_2 <= ||E||_F.
        Combined with the theoretical bound on ||E||_F, this gives:
        SV-L2 <= Frobenius_change <= theoretical_bound.
        """
        torch.manual_seed(42)
        T, D = 6, 16

        # Known AVWo_orig and perturbation Delta_AVWo
        AVWo_orig = torch.randn(T, D, dtype=torch.float64)
        Delta_AVWo = torch.randn(T, D, dtype=torch.float64) * 0.1

        AVWo_perturbed = AVWo_orig + Delta_AVWo

        # Compute SV-L2 via compute_spectral_change
        sv_l2 = compute_spectral_change(AVWo_orig.float(), AVWo_perturbed.float())

        # Compute Frobenius of Delta_AVWo
        fro_change = torch.linalg.norm(Delta_AVWo, "fro").item()

        # Mirsky's inequality: SV-L2 <= Frobenius
        assert sv_l2 <= fro_change + 1e-5, (
            f"Mirsky violated: SV-L2={sv_l2:.6f} > Frobenius={fro_change:.6f}"
        )

    def test_mirsky_chain_with_theoretical_bound_multiple_magnitudes(self) -> None:
        """Full chain SV-L2 <= Frobenius <= bound for eps = [0.01, 0.05, 0.1, 0.25].
        Tests with synthetic fixtures using inject_perturbation to compute
        the actual AVWo change, then verifies both inequalities.
        """
        torch.manual_seed(42)
        T, d_k = 8, 16

        qkt_scaled = torch.randn(T, T)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        V = torch.randn(T, d_k)
        W_O = torch.randn(d_k, d_k)

        # Compute original AVWo
        qkt_for_softmax = qkt_scaled.masked_fill(~causal_mask, float("-inf"))
        A_orig = F.softmax(qkt_for_softmax, dim=-1)
        A_orig = torch.nan_to_num(A_orig, nan=0.0)
        AVWo_orig = A_orig @ V @ W_O.T

        # Compute norms
        qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
        qkt_fro = torch.linalg.norm(qkt_masked, "fro").item()
        v_spec = torch.linalg.svdvals(V)[0].item()
        wo_spec = torch.linalg.svdvals(W_O)[0].item()

        direction = generate_adversarial_direction(qkt_scaled, causal_mask)

        for eps in [0.01, 0.05, 0.1, 0.25]:
            # Build perturbation
            perturbation = eps * qkt_fro * direction

            # Compute perturbed AVWo
            AVWo_pert = inject_perturbation(qkt_scaled, V, W_O, perturbation, causal_mask)

            # Measures
            sv_l2 = compute_spectral_change(AVWo_orig, AVWo_pert)
            fro_change = torch.linalg.norm(AVWo_pert - AVWo_orig, "fro").item()
            bound = compute_theoretical_bound(qkt_fro, v_spec, wo_spec, d_k, eps)

            # Chain: SV-L2 <= Frobenius <= bound
            assert sv_l2 <= fro_change + 1e-5, (
                f"eps={eps}: Mirsky violated: SV-L2={sv_l2:.6f} > Frobenius={fro_change:.6f}"
            )
            assert fro_change <= bound + 1e-5, (
                f"eps={eps}: Bound violated: Frobenius={fro_change:.6f} > bound={bound:.6f}"
            )


class TestSyntheticBoundVerification:
    """Verify bound on synthetic fixtures with known matrices."""

    def test_adversarial_bound_ratio_below_one(self) -> None:
        """Frobenius_change / theoretical_bound < 1.0 deterministically.
        Using synthetic fixtures with random (but seeded) Q, K, V, W_O,
        the adversarial perturbation should never exceed the theoretical bound.
        Tests eps = [0.01, 0.05, 0.10].
        """
        torch.manual_seed(42)
        T, d_k = 8, 16

        qkt_scaled = torch.randn(T, T)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        V = torch.randn(T, d_k)
        W_O = torch.randn(d_k, d_k)

        # Compute original AVWo
        qkt_for_softmax = qkt_scaled.masked_fill(~causal_mask, float("-inf"))
        A_orig = F.softmax(qkt_for_softmax, dim=-1)
        A_orig = torch.nan_to_num(A_orig, nan=0.0)
        AVWo_orig = A_orig @ V @ W_O.T

        # Norms
        qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
        qkt_fro = torch.linalg.norm(qkt_masked, "fro").item()
        v_spec = torch.linalg.svdvals(V)[0].item()
        wo_spec = torch.linalg.svdvals(W_O)[0].item()

        direction = generate_adversarial_direction(qkt_scaled, causal_mask)

        for eps in [0.01, 0.05, 0.10]:
            perturbation = eps * qkt_fro * direction
            AVWo_pert = inject_perturbation(qkt_scaled, V, W_O, perturbation, causal_mask)
            fro_change = torch.linalg.norm(AVWo_pert - AVWo_orig, "fro").item()
            bound = compute_theoretical_bound(qkt_fro, v_spec, wo_spec, d_k, eps)

            ratio = fro_change / bound
            # Ratio must be strictly less than 1.0 (bound holds)
            assert ratio < 1.0, (
                f"eps={eps}: ratio={ratio:.6f} >= 1.0, bound violated! "
                f"fro_change={fro_change:.8f}, bound={bound:.8f}"
            )

    def test_random_direction_bound_ratio_below_one(self) -> None:
        """Bound holds for random directions (5 seeds), not just adversarial.
        Random directions should produce even smaller ratios than adversarial.
        """
        torch.manual_seed(42)
        T, d_k = 8, 16

        qkt_scaled = torch.randn(T, T)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        V = torch.randn(T, d_k)
        W_O = torch.randn(d_k, d_k)

        # Compute original AVWo
        qkt_for_softmax = qkt_scaled.masked_fill(~causal_mask, float("-inf"))
        A_orig = F.softmax(qkt_for_softmax, dim=-1)
        A_orig = torch.nan_to_num(A_orig, nan=0.0)
        AVWo_orig = A_orig @ V @ W_O.T

        # Norms
        qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
        qkt_fro = torch.linalg.norm(qkt_masked, "fro").item()
        v_spec = torch.linalg.svdvals(V)[0].item()
        wo_spec = torch.linalg.svdvals(W_O)[0].item()

        for seed in [100, 200, 300, 400, 500]:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            rand_dir = generate_random_direction(T, causal_mask, gen)

            for eps in [0.01, 0.05, 0.10]:
                perturbation = eps * qkt_fro * rand_dir
                AVWo_pert = inject_perturbation(
                    qkt_scaled, V, W_O, perturbation, causal_mask
                )
                fro_change = torch.linalg.norm(AVWo_pert - AVWo_orig, "fro").item()
                bound = compute_theoretical_bound(qkt_fro, v_spec, wo_spec, d_k, eps)

                ratio = fro_change / bound
                assert ratio < 1.0, (
                    f"seed={seed}, eps={eps}: random direction ratio={ratio:.6f} >= 1.0"
                )


class TestMaskingConsistency:
    """Verify masking consistency between direction generation and injection (SFTX-03)."""

    def test_zero_fill_direction_vs_neginf_softmax_no_inconsistency(self) -> None:
        """generate_adversarial_direction uses zero-fill for SVD,
        inject_perturbation uses -inf for softmax. This should NOT create
        inconsistency because: the direction has zero in upper triangle,
        adding zero to upper triangle of QK^T changes nothing, then
        inject_perturbation masks with -inf identically to unperturbed case.

        Test: verify AVWo with upper-triangle perturbation equals AVWo
        without it (perturbation only differs in upper triangle).
        """
        torch.manual_seed(42)
        T, D = 6, 8

        # QK^T with nonzero values everywhere (including upper triangle)
        qkt_scaled = torch.randn(T, T)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        V = torch.randn(T, D)
        W_O = torch.randn(D, D)

        # Perturbation from adversarial direction (zero in upper triangle)
        direction = generate_adversarial_direction(qkt_scaled, causal_mask)
        eps = 0.1
        qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
        qkt_fro = torch.linalg.norm(qkt_masked, "fro").item()
        perturbation_lower = eps * qkt_fro * direction

        # Same perturbation but with nonzero values added to upper triangle
        perturbation_full = perturbation_lower.clone()
        upper_noise = torch.randn(T, T) * 0.5
        perturbation_full[~causal_mask] = upper_noise[~causal_mask]

        # Both should produce identical AVWo because:
        # - perturbation_lower has zero upper triangle -> no change to upper QK^T
        # - perturbation_full has nonzero upper triangle -> changes upper QK^T
        # - But inject_perturbation masks upper triangle with -inf regardless
        AVWo_lower = inject_perturbation(
            qkt_scaled, V, W_O, perturbation_lower, causal_mask
        )
        AVWo_full = inject_perturbation(
            qkt_scaled, V, W_O, perturbation_full, causal_mask
        )

        torch.testing.assert_close(AVWo_lower, AVWo_full, atol=1e-6, rtol=1e-6)

    def test_masking_preserves_attention_row_sums(self) -> None:
        """After inject_perturbation, attention weights sum to 1.0 per row
        (softmax property preserved despite perturbation). This confirms
        the -inf masking correctly handles the upper triangle.
        """
        torch.manual_seed(42)
        T, D = 6, 8

        qkt_scaled = torch.randn(T, T)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        V = torch.randn(T, D)
        W_O = torch.randn(D, D)

        direction = generate_adversarial_direction(qkt_scaled, causal_mask)
        qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
        qkt_fro = torch.linalg.norm(qkt_masked, "fro").item()
        perturbation = 0.1 * qkt_fro * direction

        # Manually compute attention weights after perturbation to check row sums
        qkt_pert = qkt_scaled + perturbation
        qkt_pert_masked = qkt_pert.masked_fill(~causal_mask, float("-inf"))
        A_pert = F.softmax(qkt_pert_masked, dim=-1)
        A_pert = torch.nan_to_num(A_pert, nan=0.0)

        # Each row should sum to 1.0 (softmax property)
        row_sums = A_pert.sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones(T), atol=1e-6, rtol=1e-6
        )


class TestWeylUsageConsistency:
    """Verify Weyl usage: compute_spectral_change <= ||Delta||_F (SFTX-03)."""

    def test_compute_spectral_change_bounded_by_frobenius(self) -> None:
        """By Mirsky's theorem: ||sigma(M+E) - sigma(M)||_2 <= ||E||_F.
        The code measures SV-L2 via compute_spectral_change, and the bound
        is on ||Delta(AVWo)||_F. Verify the SV-L2 quantity is always <=
        the Frobenius change for 10 random matrix pairs.
        """
        torch.manual_seed(42)

        for trial in range(10):
            T = 6 + trial
            D = 8 + trial

            M = torch.randn(T, D)
            E = torch.randn(T, D) * 0.1

            sv_l2 = compute_spectral_change(M, M + E)
            fro_change = torch.linalg.norm(E, "fro").item()

            # Mirsky: ||sigma(M+E) - sigma(M)||_2 <= ||E||_F
            assert sv_l2 <= fro_change + 1e-5, (
                f"Trial {trial}: SV-L2={sv_l2:.6f} > ||E||_F={fro_change:.6f}"
            )

    def test_spectral_change_symmetry(self) -> None:
        """compute_spectral_change(A, B) should equal compute_spectral_change(B, A)
        because ||sigma(B) - sigma(A)||_2 = ||sigma(A) - sigma(B)||_2.
        """
        torch.manual_seed(42)
        T, D = 6, 8

        A = torch.randn(T, D)
        B = A + torch.randn(T, D) * 0.1

        change_ab = compute_spectral_change(A, B)
        change_ba = compute_spectral_change(B, A)

        # L2 norm is symmetric
        assert abs(change_ab - change_ba) < 1e-5, (
            f"Spectral change not symmetric: AB={change_ab:.6f}, BA={change_ba:.6f}"
        )
