"""Audit tests for WvWo and AVWo matrix constructions (SVD-02).

Verifies the mathematical correctness of the OV circuit (WvWo = Wv_h.T @ Wo_h.T)
and the net residual update (AVWo = (A_h @ V_h) @ Wo_h^T) for both single-head
and 4-head configurations, including correct weight index slicing.
"""

import torch

from src.model.transformer import TransformerLM
from src.evaluation.pipeline import _compute_avwo_for_layer


class TestWvWoSingleHead:
    """Verify OV circuit WvWo = Wv_h.T @ Wo_h.T for single-head config."""

    def test_wvwo_single_head_formula(self) -> None:
        """For single-head (n_heads=1), the OV circuit is:
          Wv_h = Wv.weight (full [d_model, d_model] matrix)
          Wo_h = Wo.weight (full [d_model, d_model] matrix)
          WvWo = Wv_h.T @ Wo_h.T
        nn.Linear stores weight as [out_features, in_features], so:
          Wv.weight is [d_model, d_model], Wo.weight is [d_model, d_model].
        """
        torch.manual_seed(42)
        d_model = 8

        model = TransformerLM(vocab_size=10, d_model=d_model, n_layers=1, n_heads=1, max_seq_len=4, dropout=0.0)

        # Set known weights for Wv and Wo
        Wv_known = torch.randn(d_model, d_model) * 0.1
        Wo_known = torch.randn(d_model, d_model) * 0.1
        with torch.no_grad():
            model.blocks[0].attention.W_v.weight.copy_(Wv_known)
            model.blocks[0].attention.W_o.weight.copy_(Wo_known)

        # Manual computation: WvWo = Wv.T @ Wo.T
        # For single-head: Wv_h = Wv_known[0:8, :] = Wv_known, Wo_h = Wo_known[:, 0:8] = Wo_known
        wvwo_manual = Wv_known.T @ Wo_known.T  # [d_model, d_model]

        # Extract from model
        wvwo = model.get_wvwo()  # [n_layers=1, n_heads=1, d_model, d_model]
        assert wvwo.shape == (1, 1, d_model, d_model), (
            f"Expected shape (1, 1, {d_model}, {d_model}), got {wvwo.shape}"
        )

        torch.testing.assert_close(
            wvwo[0, 0],
            wvwo_manual,
            atol=1e-5,
            rtol=1e-5,
        )


class TestWvWoMultiHead:
    """Verify per-head OV circuit WvWo_h = Wv_h.T @ Wo_h.T for n_heads=4."""

    def test_wvwo_multi_head_formula(self) -> None:
        """For each head h (0..3) with d_model=16, n_heads=4, d_head=4:
          Wv_h = Wv.weight[h*4:(h+1)*4, :]  -- [d_head=4, d_model=16]
          Wo_h = Wo.weight[:, h*4:(h+1)*4]  -- [d_model=16, d_head=4]
          WvWo_h = Wv_h.T @ Wo_h.T          -- [d_model=16, d_model=16]
        This maps: input -> value head h -> output projection head h -> residual.
        """
        torch.manual_seed(42)
        d_model = 16
        n_heads = 4
        d_head = d_model // n_heads  # 4

        model = TransformerLM(vocab_size=10, d_model=d_model, n_layers=1, n_heads=n_heads, max_seq_len=4, dropout=0.0)

        # Set known weights
        Wv_known = torch.randn(d_model, d_model) * 0.1
        Wo_known = torch.randn(d_model, d_model) * 0.1
        with torch.no_grad():
            model.blocks[0].attention.W_v.weight.copy_(Wv_known)
            model.blocks[0].attention.W_o.weight.copy_(Wo_known)

        # Extract from model
        wvwo = model.get_wvwo()  # [1, 4, 16, 16]
        assert wvwo.shape == (1, n_heads, d_model, d_model), (
            f"Expected shape (1, {n_heads}, {d_model}, {d_model}), got {wvwo.shape}"
        )

        # Manually compute and verify each head
        for h in range(n_heads):
            start = h * d_head
            end = (h + 1) * d_head
            # Per-head value projection slice
            Wv_h = Wv_known[start:end, :]  # [4, 16]
            # Per-head output projection slice
            Wo_h = Wo_known[:, start:end]  # [16, 4]
            # OV circuit
            wvwo_h_manual = Wv_h.T @ Wo_h.T  # [16, 16]

            torch.testing.assert_close(
                wvwo[0, h],
                wvwo_h_manual,
                atol=1e-5,
                rtol=1e-5,
            )


class TestAVWoSingleHead:
    """Verify AVWo = (A @ V) @ Wo.T for single-head config."""

    def test_avwo_single_head_formula(self) -> None:
        """For single-head: AVWo = (A @ V) @ Wo.weight.T.
        A is attention weights [B, T, T], V is value matrix [B, T, d_model].
        AV = A @ V gives [B, T, d_model], then AVWo = AV @ Wo.T = [B, T, d_model].
        """
        torch.manual_seed(42)
        d_model = 8
        T = 4

        model = TransformerLM(vocab_size=10, d_model=d_model, n_layers=1, n_heads=1, max_seq_len=T, dropout=0.0)

        # Set known Wo weight
        Wo_known = torch.randn(d_model, d_model) * 0.1
        with torch.no_grad():
            model.blocks[0].attention.W_o.weight.copy_(Wo_known)

        # Known attention weights (lower triangular, rows sum to 1) and values
        A = torch.zeros(1, T, T)
        for i in range(T):
            # Uniform attention over causal positions
            A[0, i, : i + 1] = 1.0 / (i + 1)

        V = torch.randn(1, T, d_model)

        # Manual computation
        AV = A @ V  # [1, T, d_model]
        avwo_manual = AV @ Wo_known.T  # [1, T, d_model]

        # Compute via pipeline function
        avwo_computed = _compute_avwo_for_layer(
            attention_weights=A, values=V, model=model, layer_idx=0,
            head_idx=0, n_heads=1,
        )

        torch.testing.assert_close(
            avwo_computed,
            avwo_manual,
            atol=1e-5,
            rtol=1e-5,
        )


class TestAVWoMultiHead:
    """Verify per-head AVWo_h = (A_h @ V_h) @ Wo_h^T for n_heads=4."""

    def test_avwo_multi_head_formula(self) -> None:
        """For multi-head with d_model=16, n_heads=4, d_head=4:
          AV_h = A_h @ V_h  -- [B, T, d_head=4]
          Wo_h = Wo.weight[:, h*4:(h+1)*4]  -- [d_model=16, d_head=4]
          AVWo_h = AV_h @ Wo_h.T  -- [B, T, d_model=16]
        This gives head h's contribution to the residual stream.
        """
        torch.manual_seed(42)
        d_model = 16
        n_heads = 4
        d_head = d_model // n_heads  # 4
        T = 4

        model = TransformerLM(vocab_size=10, d_model=d_model, n_layers=1, n_heads=n_heads, max_seq_len=T, dropout=0.0)

        # Set known Wo weight
        Wo_known = torch.randn(d_model, d_model) * 0.1
        with torch.no_grad():
            model.blocks[0].attention.W_o.weight.copy_(Wo_known)

        for h in range(n_heads):
            start = h * d_head
            end = (h + 1) * d_head

            # Known per-head attention weights and values
            A_h = torch.zeros(1, T, T)
            for i in range(T):
                A_h[0, i, : i + 1] = 1.0 / (i + 1)

            V_h = torch.randn(1, T, d_head)

            # Manual computation for this head
            AV_h = A_h @ V_h  # [1, T, d_head=4]
            Wo_h = Wo_known[:, start:end]  # [d_model=16, d_head=4]
            avwo_h_manual = AV_h @ Wo_h.T  # [1, T, d_model=16]

            # Compute via pipeline function
            avwo_h_computed = _compute_avwo_for_layer(
                attention_weights=A_h, values=V_h, model=model, layer_idx=0,
                head_idx=h, n_heads=n_heads,
            )

            torch.testing.assert_close(
                avwo_h_computed,
                avwo_h_manual,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_avwo_multi_head_output_shape(self) -> None:
        """AVWo_h must have shape [B, T, d_model] regardless of head index.
        Each head maps its d_head-dimensional output to the full d_model residual stream.
        """
        torch.manual_seed(42)
        d_model = 16
        n_heads = 4
        d_head = d_model // n_heads
        T = 4

        model = TransformerLM(vocab_size=10, d_model=d_model, n_layers=1, n_heads=n_heads, max_seq_len=T, dropout=0.0)

        A_h = torch.zeros(1, T, T)
        for i in range(T):
            A_h[0, i, : i + 1] = 1.0 / (i + 1)

        V_h = torch.randn(1, T, d_head)

        for h in range(n_heads):
            avwo = _compute_avwo_for_layer(
                attention_weights=A_h, values=V_h, model=model, layer_idx=0,
                head_idx=h, n_heads=n_heads,
            )
            assert avwo.shape == (1, T, d_model), (
                f"Head {h}: expected shape (1, {T}, {d_model}), got {avwo.shape}"
            )
