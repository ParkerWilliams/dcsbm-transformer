"""Audit tests for softmax filtering bound derivation chain (SFTX-01, SFTX-03).

Verifies the mathematical correctness of the LaTeX derivation in
docs/softmax_bound.tex step by step: softmax Lipschitz constant 1/2,
submultiplicativity, Weyl's inequality, sqrt(d_k) cancellation between
LaTeX and code, compute_theoretical_bound formula, three-stage composition,
and bound assumptions (causal mask, V/W_O fixed, single-head).
"""

import inspect
import math
import textwrap

import torch
import torch.nn.functional as F

from src.analysis.perturbation_bound import (
    compute_theoretical_bound,
    generate_adversarial_direction,
    inject_perturbation,
)


class TestSoftmaxLipschitz:
    """Verify softmax Lipschitz constant 1/2 (Lemma 3.4 / Prop 3.7 in LaTeX)."""

    def test_softmax_jacobian_formula(self) -> None:
        """J(z) = diag(p) - pp^T.
        The softmax Jacobian at z is exactly diag(p) - p @ p^T where p = softmax(z).
        Verify by comparing against finite-difference Jacobian (central differences).
        """
        torch.manual_seed(42)
        z = torch.randn(5, dtype=torch.float64)
        p = F.softmax(z, dim=0)

        # Analytical Jacobian: J = diag(p) - pp^T
        J_analytical = torch.diag(p) - torch.outer(p, p)

        # Finite-difference Jacobian (central differences)
        eps = 1e-7
        n = len(z)
        J_fd = torch.zeros(n, n, dtype=torch.float64)
        for j in range(n):
            z_plus = z.clone()
            z_minus = z.clone()
            z_plus[j] += eps
            z_minus[j] -= eps
            J_fd[:, j] = (F.softmax(z_plus, dim=0) - F.softmax(z_minus, dim=0)) / (2 * eps)

        # Finite differences match analytical Jacobian within tolerance
        torch.testing.assert_close(J_analytical, J_fd, atol=1e-6, rtol=1e-5)

    def test_softmax_lipschitz_bound_half(self) -> None:
        """||softmax(z+d) - softmax(z)||_2 <= 0.5 * ||d||_2.
        The softmax function has global Lipschitz constant 1/2 in the L2 norm
        (Gao & Pavel 2017). Verify with 100 random trials across sizes 5, 10, 50.
        """
        torch.manual_seed(42)
        for n in [5, 10, 50]:
            for _ in range(100):
                z = torch.randn(n, dtype=torch.float64)
                delta = torch.randn(n, dtype=torch.float64) * 0.5

                diff = F.softmax(z + delta, dim=0) - F.softmax(z, dim=0)
                ratio = torch.linalg.norm(diff).item() / torch.linalg.norm(delta).item()

                # Lipschitz constant 1/2 means ratio <= 0.5
                assert ratio <= 0.5 + 1e-10, (
                    f"Softmax Lipschitz violated: ratio={ratio:.6f} > 0.5 "
                    f"for n={n}"
                )

    def test_softmax_lipschitz_tightness_at_uniform(self) -> None:
        """Lipschitz ratio approaches 1/2 when z produces uniform distribution.
        At z = 0 (uniform p = 1/n), the Jacobian J = diag(p) - pp^T has all
        non-zero eigenvalues equal to 1/n. For infinitesimal perturbation
        orthogonal to the 1-vector, the local ratio equals 1/n.
        Verify this local spectral norm and that it is always <= 1/2 (global).
        """
        for n in [10, 50, 200]:
            # Uniform distribution: z = 0
            z = torch.zeros(n, dtype=torch.float64)

            # Max-eigenvalue direction of Jacobian: orthogonal to 1-vector
            # Use [1, -1, 0, 0, ...] / sqrt(2) (orthogonal to 1-vector, unit norm)
            delta = torch.zeros(n, dtype=torch.float64)
            delta[0] = 1.0
            delta[1] = -1.0
            delta = delta / torch.linalg.norm(delta)

            # Use very small perturbation to stay in linear regime
            eps = 1e-6
            delta_scaled = eps * delta

            diff = F.softmax(z + delta_scaled, dim=0) - F.softmax(z, dim=0)
            ratio = torch.linalg.norm(diff).item() / torch.linalg.norm(delta_scaled).item()

            # At uniform p = 1/n, the Jacobian eigenvalues (non-zero) are all 1/n.
            # So the spectral norm of J is 1/n, and the local ratio is 1/n.
            expected_local = 1.0 / n
            assert abs(ratio - expected_local) < 0.02, (
                f"At uniform (n={n}), expected local ratio ~{expected_local:.4f}, "
                f"got {ratio:.4f}"
            )

            # The local ratio must always be <= global Lipschitz constant 1/2
            assert ratio <= 0.5 + 1e-10, (
                f"Local ratio {ratio:.6f} exceeds global Lipschitz constant 1/2"
            )


class TestSubmultiplicativity:
    """Verify submultiplicativity: ||AB||_F <= ||A||_F * ||B||_2 (Lemma 2.2)."""

    def test_frobenius_spectral_submult(self) -> None:
        """||AB||_F <= ||A||_F * ||B||_2.
        Submultiplicativity of Frobenius norm with spectral norm.
        Use a 4x3 A and 3x5 B with known values.
        """
        torch.manual_seed(42)
        A = torch.randn(4, 3, dtype=torch.float64)
        B = torch.randn(3, 5, dtype=torch.float64)

        ab_fro = torch.linalg.norm(A @ B, "fro").item()
        a_fro = torch.linalg.norm(A, "fro").item()
        b_spec = torch.linalg.svdvals(B)[0].item()

        # ||AB||_F <= ||A||_F * ||B||_2
        assert ab_fro <= a_fro * b_spec + 1e-10, (
            f"Submultiplicativity violated: ||AB||_F={ab_fro:.6f} > "
            f"||A||_F * ||B||_2 = {a_fro * b_spec:.6f}"
        )

    def test_spectral_frobenius_submult(self) -> None:
        """||AB||_F <= ||A||_2 * ||B||_F.
        Symmetric submultiplicativity bound (Lemma 2.2, second inequality).
        """
        torch.manual_seed(42)
        A = torch.randn(4, 3, dtype=torch.float64)
        B = torch.randn(3, 5, dtype=torch.float64)

        ab_fro = torch.linalg.norm(A @ B, "fro").item()
        a_spec = torch.linalg.svdvals(A)[0].item()
        b_fro = torch.linalg.norm(B, "fro").item()

        # ||AB||_F <= ||A||_2 * ||B||_F
        assert ab_fro <= a_spec * b_fro + 1e-10, (
            f"Symmetric submultiplicativity violated: ||AB||_F={ab_fro:.6f} > "
            f"||A||_2 * ||B||_F = {a_spec * b_fro:.6f}"
        )


class TestWeylInequality:
    """Verify Weyl's inequality: |sigma_i(M+E) - sigma_i(M)| <= ||E||_2 (Lemma 2.4)."""

    def test_weyl_inequality_known_svs(self) -> None:
        """Construct M with known singular values [10, 5, 1] and verify
        |sigma_i(M+E) - sigma_i(M)| <= ||E||_2 for all i.
        Uses diagonal M for exact known singular values.
        """
        torch.manual_seed(42)
        # M with known singular values [10, 5, 1]
        M = torch.diag(torch.tensor([10.0, 5.0, 1.0], dtype=torch.float64))

        # Random perturbation E
        E = torch.randn(3, 3, dtype=torch.float64) * 0.1

        sigma_M = torch.linalg.svdvals(M)
        sigma_ME = torch.linalg.svdvals(M + E)
        e_spec = torch.linalg.svdvals(E)[0].item()

        for i in range(3):
            diff = abs(sigma_ME[i].item() - sigma_M[i].item())
            # Weyl's inequality: each SV can change by at most ||E||_2
            assert diff <= e_spec + 1e-10, (
                f"Weyl violated at i={i}: |sigma_{i}(M+E) - sigma_{i}(M)| = "
                f"{diff:.6f} > ||E||_2 = {e_spec:.6f}"
            )


class TestSqrtDkCancellation:
    """Verify sqrt(d_k) cancellation between LaTeX and code (SFTX-03 key test)."""

    def test_sqrt_dk_cancellation_algebraic(self) -> None:
        """LaTeX bound: eps * ||QK^T_unscaled||_F * ||V||_2 * ||W_O||_2 / (2*sqrt(d_k))
        Code bound:   eps * ||QK^T_scaled||_F   * ||V||_2 * ||W_O||_2 / 2
        Since QK^T_scaled = QK^T_unscaled / sqrt(d_k), these MUST be identical.
        This proves the sqrt(d_k) cancellation is correct.
        """
        torch.manual_seed(42)
        d_k = 64
        T = 8
        epsilon = 0.05

        Q = torch.randn(T, d_k, dtype=torch.float64)
        K = torch.randn(T, d_k, dtype=torch.float64)
        V = torch.randn(T, d_k, dtype=torch.float64)
        W_O = torch.randn(d_k, d_k, dtype=torch.float64)

        # Unscaled and scaled QK^T
        qkt_unscaled = Q @ K.T  # [T, T]
        qkt_scaled = qkt_unscaled / math.sqrt(d_k)  # [T, T]

        qkt_unscaled_fro = torch.linalg.norm(qkt_unscaled, "fro").item()
        qkt_scaled_fro = torch.linalg.norm(qkt_scaled, "fro").item()
        v_spec = torch.linalg.svdvals(V)[0].item()
        wo_spec = torch.linalg.svdvals(W_O)[0].item()

        # LaTeX way (unscaled): eps * ||QK^T||_F * ||V||_2 * ||W_O||_2 / (2*sqrt(d_k))
        bound_latex = epsilon * qkt_unscaled_fro * v_spec * wo_spec / (2.0 * math.sqrt(d_k))

        # Code way (scaled): eps * ||QK^T_scaled||_F * ||V||_2 * ||W_O||_2 / 2
        bound_code = epsilon * qkt_scaled_fro * v_spec * wo_spec / 2.0

        # The two formulations must be identical (cancellation of sqrt(d_k))
        assert abs(bound_latex - bound_code) < 1e-12, (
            f"sqrt(d_k) cancellation failed: LaTeX={bound_latex:.15f}, "
            f"Code={bound_code:.15f}, diff={abs(bound_latex - bound_code):.2e}"
        )

    def test_compute_theoretical_bound_matches_formula(self) -> None:
        """compute_theoretical_bound must return eps * qkt_fro * v_spec * wo_spec / 2.
        The function takes the SCALED QK^T norm (sqrt(d_k) already applied),
        so the division by 2 (not 2*sqrt(d_k)) is the correct formula.
        """
        qkt_fro = 12.5
        v_spec = 3.0
        wo_spec = 2.0
        d_k = 64
        epsilon = 0.1

        result = compute_theoretical_bound(qkt_fro, v_spec, wo_spec, d_k, epsilon)
        expected = epsilon * qkt_fro * v_spec * wo_spec / 2.0
        # = 0.1 * 12.5 * 3.0 * 2.0 / 2.0 = 3.75

        assert abs(result - expected) < 1e-12, (
            f"compute_theoretical_bound mismatch: got {result}, expected {expected}"
        )
        assert abs(result - 3.75) < 1e-12, (
            f"Hand-computed value mismatch: got {result}, expected 3.75"
        )


class TestDerivationChain:
    """Verify end-to-end derivation chain with concrete small matrices."""

    def test_three_stage_composition(self) -> None:
        """Given concrete T=4, d_k=8 matrices, verify each stage bound:
        Stage 1: ||Delta_A||_F <= eps * ||QK^T||_F / (2*sqrt(d_k))
        Stage 2: ||Delta_A * V||_F <= ||Delta_A||_F * ||V||_2
        Stage 3: ||Delta_AV * W_O^T||_F <= ||Delta_AV||_F * ||W_O||_2
        Composed: ||Delta_AVWo||_F <= Theorem 6.1 bound
        """
        torch.manual_seed(42)
        T, d_k = 4, 8
        epsilon = 0.05

        Q = torch.randn(T, d_k, dtype=torch.float64)
        K = torch.randn(T, d_k, dtype=torch.float64)
        V = torch.randn(T, d_k, dtype=torch.float64)
        W_O = torch.randn(d_k, d_k, dtype=torch.float64)

        # Compute scaled scores and apply causal mask
        qkt_scaled = (Q @ K.T) / math.sqrt(d_k)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

        # Perturbation: epsilon * ||QK^T_masked||_F * adversarial_direction
        qkt_masked = qkt_scaled.masked_fill(~causal_mask, 0.0)
        qkt_fro = torch.linalg.norm(qkt_masked, "fro").item()

        # Simple perturbation direction (unit Frobenius norm, lower triangular)
        direction = torch.randn(T, T, dtype=torch.float64)
        direction = direction.masked_fill(~causal_mask, 0.0)
        direction = direction / torch.linalg.norm(direction, "fro")

        perturbation = epsilon * qkt_fro * direction  # [T, T]

        # Original attention weights
        qkt_for_softmax = qkt_scaled.masked_fill(~causal_mask, float("-inf"))
        A_orig = F.softmax(qkt_for_softmax, dim=-1)
        A_orig = torch.nan_to_num(A_orig, nan=0.0)

        # Perturbed attention weights
        qkt_pert = (qkt_scaled + perturbation).masked_fill(~causal_mask, float("-inf"))
        A_pert = F.softmax(qkt_pert, dim=-1)
        A_pert = torch.nan_to_num(A_pert, nan=0.0)

        # === Stage 1 ===
        delta_A = A_pert - A_orig
        delta_A_fro = torch.linalg.norm(delta_A, "fro").item()
        stage1_bound = epsilon * qkt_fro / 2.0  # Using scaled QK^T, sqrt(d_k) cancelled

        # ||Delta_A||_F must be <= eps * ||QK^T_scaled||_F / 2
        assert delta_A_fro <= stage1_bound + 1e-10, (
            f"Stage 1 violated: ||Delta_A||_F={delta_A_fro:.8f} > "
            f"bound={stage1_bound:.8f}"
        )

        # === Stage 2 ===
        delta_AV = delta_A @ V
        delta_AV_fro = torch.linalg.norm(delta_AV, "fro").item()
        v_spec = torch.linalg.svdvals(V)[0].item()
        stage2_bound = delta_A_fro * v_spec

        # ||Delta_A * V||_F must be <= ||Delta_A||_F * ||V||_2
        assert delta_AV_fro <= stage2_bound + 1e-10, (
            f"Stage 2 violated: ||Delta_AV||_F={delta_AV_fro:.8f} > "
            f"bound={stage2_bound:.8f}"
        )

        # === Stage 3 ===
        delta_AVWo = delta_AV @ W_O.T
        delta_AVWo_fro = torch.linalg.norm(delta_AVWo, "fro").item()
        wo_spec = torch.linalg.svdvals(W_O)[0].item()
        stage3_bound = delta_AV_fro * wo_spec

        # ||Delta_AV * W_O^T||_F must be <= ||Delta_AV||_F * ||W_O||_2
        assert delta_AVWo_fro <= stage3_bound + 1e-10, (
            f"Stage 3 violated: ||Delta_AVWo||_F={delta_AVWo_fro:.8f} > "
            f"bound={stage3_bound:.8f}"
        )

        # === Composed (Theorem 6.1) ===
        theorem_bound = epsilon * qkt_fro * v_spec * wo_spec / 2.0

        # ||Delta_AVWo||_F must be <= full Theorem 6.1 bound
        assert delta_AVWo_fro <= theorem_bound + 1e-10, (
            f"Theorem 6.1 violated: ||Delta_AVWo||_F={delta_AVWo_fro:.8f} > "
            f"bound={theorem_bound:.8f}"
        )


class TestBoundAssumptions:
    """Verify bound assumptions are respected in code (SFTX-03)."""

    def test_causal_masking_nullifies_upper_triangle(self) -> None:
        """inject_perturbation applies -inf causal mask before softmax, so
        perturbation values in the upper triangle of QK^T do not affect the output.
        Verify: perturbation with nonzero upper triangle produces same result
        as perturbation with zeroed upper triangle.
        """
        torch.manual_seed(42)
        T, D = 6, 8

        qkt_scaled = torch.randn(T, T)
        V = torch.randn(T, D)
        W_O = torch.randn(D, D)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

        # Perturbation with nonzero upper triangle
        pert_full = torch.randn(T, T) * 0.1

        # Same perturbation with zeroed upper triangle
        pert_lower = pert_full.masked_fill(~causal_mask, 0.0)

        # Both should produce identical AVWo because -inf masking in inject_perturbation
        # zeros out any upper-triangle contribution in softmax
        avwo_full = inject_perturbation(qkt_scaled, V, W_O, pert_full, causal_mask)
        avwo_lower = inject_perturbation(qkt_scaled, V, W_O, pert_lower, causal_mask)

        torch.testing.assert_close(avwo_full, avwo_lower, atol=1e-6, rtol=1e-6)

    def test_v_and_wo_unchanged_after_perturbation(self) -> None:
        """V and W_O must be held fixed during perturbation (not modified in place).
        The bound derivation assumes V and W_O are constants, not variables.
        """
        torch.manual_seed(42)
        T, D = 6, 8

        qkt_scaled = torch.randn(T, T)
        V = torch.randn(T, D)
        W_O = torch.randn(D, D)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

        # Save copies before perturbation
        V_before = V.clone()
        W_O_before = W_O.clone()

        # Two different perturbations
        pert1 = torch.randn(T, T) * 0.05
        pert2 = torch.randn(T, T) * 0.1

        inject_perturbation(qkt_scaled, V, W_O, pert1, causal_mask)
        inject_perturbation(qkt_scaled, V, W_O, pert2, causal_mask)

        # V and W_O must be unchanged after both calls
        torch.testing.assert_close(V, V_before, atol=0.0, rtol=0.0)
        torch.testing.assert_close(W_O, W_O_before, atol=0.0, rtol=0.0)

    def test_single_head_assumption_index_zero(self) -> None:
        """run_perturbation_at_step uses head index 0: qkt_scaled = output.qkt[0, layer_idx, 0].
        The perturbation bound derivation assumes single-head attention.
        Verify via source code inspection that head index 0 is used.
        """
        # Read the source code of run_perturbation_at_step and verify the indexing
        source = inspect.getsource(
            __import__(
                "src.analysis.perturbation_bound",
                fromlist=["run_perturbation_at_step"],
            ).run_perturbation_at_step
        )

        # Verify that qkt is extracted with head index 0: output.qkt[0, layer_idx, 0]
        assert "output.qkt[0, layer_idx, 0]" in source, (
            "Expected qkt extraction with head index 0 "
            "(output.qkt[0, layer_idx, 0]) not found in source"
        )

        # Verify that V is extracted with head index 0: output.values[0, layer_idx, 0]
        assert "output.values[0, layer_idx, 0]" in source, (
            "Expected V extraction with head index 0 "
            "(output.values[0, layer_idx, 0]) not found in source"
        )

        # Verify that attention weights use head index 0
        assert "output.attention_weights[0, layer_idx, 0]" in source, (
            "Expected attention_weights extraction with head index 0 "
            "not found in source"
        )
