"""SVD metric functions with numerical guards for attention matrix analysis.

Implements 8+1 pure SVD metric functions that operate on singular values
or full SVD decompositions (U, S, Vh). Each function is independently
testable against analytically known matrices.
"""

import torch

# Numerical constants
EPS = 1e-12
CONDITION_CAP = 1e6


def guard_matrix_for_svd(M: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Clamp non-finite values before SVD computation.

    Args:
        M: Input matrix tensor of arbitrary shape.

    Returns:
        Tuple of (cleaned_matrix, guard_activated). If guard_activated is True,
        the input contained NaN or Inf values that were clamped.
    """
    has_nonfinite = not torch.isfinite(M).all().item()
    if has_nonfinite:
        M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    return M, has_nonfinite


def stable_rank(S: torch.Tensor) -> torch.Tensor:
    """Stable rank: ||M||^2_F / ||M||^2_2 = sum(s_i^2) / s_1^2.

    Args:
        S: Singular values tensor with shape [..., k].

    Returns:
        Stable rank scalar(s) with shape [...].
    """
    s_sq = S**2
    return s_sq.sum(dim=-1) / (s_sq[..., 0] + EPS)


def spectral_entropy(S: torch.Tensor) -> torch.Tensor:
    """Spectral entropy: -sum(p_i * log(p_i)) where p_i = sigma_i / sum(sigma).

    Args:
        S: Singular values tensor with shape [..., k].

    Returns:
        Spectral entropy scalar(s) with shape [...].
    """
    p = S / (S.sum(dim=-1, keepdim=True) + EPS)
    ent = -(p * torch.log(p + EPS)).sum(dim=-1)
    return torch.clamp(ent, min=0.0)


def spectral_gap_1_2(S: torch.Tensor) -> torch.Tensor:
    """Spectral gap between 1st and 2nd singular values: sigma_1 - sigma_2.

    Args:
        S: Singular values tensor with shape [..., k] where k >= 2.

    Returns:
        Spectral gap scalar(s) with shape [...].
    """
    return S[..., 0] - S[..., 1]


def spectral_gap_2_3(S: torch.Tensor) -> torch.Tensor:
    """Spectral gap between 2nd and 3rd singular values: sigma_2 - sigma_3.

    Args:
        S: Singular values tensor with shape [..., k] where k >= 3.

    Returns:
        Spectral gap scalar(s) with shape [...].
    """
    return S[..., 1] - S[..., 2]


def spectral_gap_4_5(S: torch.Tensor) -> torch.Tensor:
    """Spectral gap between 4th and 5th singular values: sigma_4 - sigma_5.

    Args:
        S: Singular values tensor with shape [..., k] where k >= 5.

    Returns:
        Spectral gap scalar(s) with shape [...].
    """
    return S[..., 3] - S[..., 4]


def condition_number(S: torch.Tensor) -> torch.Tensor:
    """Condition number: sigma_1 / sigma_n, capped at CONDITION_CAP.

    Args:
        S: Singular values tensor with shape [..., k].

    Returns:
        Condition number scalar(s) with shape [...].
    """
    raw = S[..., 0] / (S[..., -1] + EPS)
    return torch.clamp(raw, max=CONDITION_CAP)


def rank1_residual_norm(
    U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor
) -> torch.Tensor:
    """Rank-1 residual norm: ||M - sigma_1 * u_1 * v_1^T||_F / ||M||_F.

    Args:
        U: Left singular vectors with shape [..., m, k].
        S: Singular values with shape [..., k].
        Vh: Right singular vectors (conjugate transpose) with shape [..., k, n].

    Returns:
        Rank-1 residual norm scalar(s) with shape [...].
    """
    # Frobenius norm of M: sqrt(sum(s_i^2))
    fro_M = torch.sqrt((S**2).sum(dim=-1) + EPS)
    # Frobenius norm of residual: sqrt(sum(s_i^2 for i>=1))
    fro_residual = torch.sqrt((S[..., 1:] ** 2).sum(dim=-1) + EPS)
    return fro_residual / (fro_M + EPS)


def read_write_alignment(U: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    """Read-write alignment: |cos(angle)| between top left and right singular vectors.

    Measures alignment between the dominant read (U[:, 0]) and write (Vh[0, :])
    directions of a matrix. Primarily meaningful for WvWo (OV circuit).

    Args:
        U: Left singular vectors with shape [..., m, k].
        Vh: Right singular vectors with shape [..., k, n].

    Returns:
        Alignment scalar(s) in [0, 1] with shape [...].
    """
    u1 = U[..., :, 0]  # top left singular vector [..., m]
    v1 = Vh[..., 0, :]  # top right singular vector [..., n]
    dot = torch.sum(u1 * v1, dim=-1)
    return torch.abs(dot)


def grassmannian_distance(
    U_prev: torch.Tensor, U_curr: torch.Tensor, k: int = 2
) -> torch.Tensor:
    """Grassmannian distance between k-dimensional subspaces.

    Computes the geodesic distance on the Grassmann manifold between the
    subspaces spanned by the top-k columns of U_prev and U_curr.

    Args:
        U_prev: Left singular vectors from previous step, shape [..., m, p].
        U_curr: Left singular vectors from current step, shape [..., m, p].
        k: Subspace dimension (default 2).

    Returns:
        Grassmannian distance scalar(s) with shape [...].
    """
    # Principal angles via SVD of U_prev^T @ U_curr (top-k columns)
    cos_angles = torch.linalg.svdvals(
        U_prev[..., :, :k].transpose(-2, -1) @ U_curr[..., :, :k]
    )
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    angles = torch.arccos(cos_angles)
    return torch.sqrt((angles**2).sum(dim=-1))


def compute_all_metrics(
    S: torch.Tensor,
    U: torch.Tensor | None = None,
    Vh: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all applicable SVD metrics from singular values and optional vectors.

    Args:
        S: Singular values tensor with shape [..., k].
        U: Optional left singular vectors with shape [..., m, k].
        Vh: Optional right singular vectors with shape [..., k, n].

    Returns:
        Dict mapping metric names to tensor values.
    """
    metrics: dict[str, torch.Tensor] = {}

    # Singular-value-only metrics (always computed)
    metrics["stable_rank"] = stable_rank(S)
    metrics["spectral_entropy"] = spectral_entropy(S)

    k = S.shape[-1]
    if k >= 2:
        metrics["spectral_gap_1_2"] = spectral_gap_1_2(S)
    if k >= 3:
        metrics["spectral_gap_2_3"] = spectral_gap_2_3(S)
    if k >= 5:
        metrics["spectral_gap_4_5"] = spectral_gap_4_5(S)

    metrics["condition_number"] = condition_number(S)

    # Full-SVD metrics (only when U and Vh provided)
    if U is not None and Vh is not None:
        metrics["rank1_residual_norm"] = rank1_residual_norm(U, S, Vh)
        # read_write_alignment only meaningful when U and Vh have compatible
        # last dimensions (square or matching m==n in the original matrix)
        u_dim = U.shape[-2]  # m (rows of original matrix)
        v_dim = Vh.shape[-1]  # n (cols of original matrix)
        if u_dim == v_dim:
            metrics["read_write_alignment"] = read_write_alignment(U, Vh)

    return metrics
