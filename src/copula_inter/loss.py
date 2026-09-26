"""Losses and metrics for Gaussian copula models.

copula_nll: inter-instance copula NLL for R = eps I + W W^T, via the Matrix
    Determinant Lemma and Woodbury, O(N r^2):
        log|R| = N log(eps) + log|M|,  M = I + W^T W / eps
        R^{-1} z = (z - W M^{-1} W^T z / eps) / eps
        L = 0.5 (log|R| + z^T R^{-1} z - z^T z) / N
y_space_nll: Sklar total = copula NLL + marginal NLL, dense or low-rank.
woodbury_nll: NLL of N(mu, diag(D) + V V^T), O(d r^2).
indep_normal_nll, marginal_nll, oracle_copula_nll, gp_oracle_y_nll,
energy_score, kl_gaussian, plot_prediction_comparison.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from copula_inter.correlation_factory import LowRankCorrelationFactor


def _safe_cholesky(K: torch.Tensor, max_attempts: int = 8) -> torch.Tensor:
    """Cholesky of K + jitter I with adaptive jitter.

    Non-finite matrix slices are replaced by identity; otherwise K is symmetrized
    and jitter grows from 1e-6 towards 1e-1 until the factorization succeeds.

    Args:
        K: (..., n, n) symmetric matrix.
        max_attempts: number of jitter increases.

    Returns:
        (..., n, n) lower-triangular factor.
    """
    n = K.shape[-1]
    eye = torch.eye(n, dtype=K.dtype, device=K.device)

    # Replace non-finite slices with identity on-device (no host sync).
    finite = torch.isfinite(K).flatten(-2).all(-1)[..., None, None]
    K = torch.where(finite, K, eye)

    # Symmetrize to eliminate floating-point asymmetry from batched matmuls.
    K = 0.5 * (K + K.transpose(-2, -1))

    jitter = 1e-6
    for _ in range(max_attempts):
        try:
            return torch.linalg.cholesky(K + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter *= 10
    raise RuntimeError(
        f"Cholesky failed after {max_attempts} attempts "
        f"(final jitter={jitter:.0e}, shape={K.shape[-2:]}, "
        f"min_eig≈{torch.linalg.eigvalsh(K).min().item():.3e})."
    )


def woodbury_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    D: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Mean NLL of N(mu, diag(D) + V V^T) via Woodbury.

    Args:
        y: (B, N, d) targets.
        mu: (B, N, d) mean.
        D: (B, N, d) positive diagonal variances.
        V: (B, N, d, r) low-rank factor.

    Returns:
        Scalar NLL averaged over B*N instances.
    """
    r_vec = y - mu  # (B, N, d)
    r = V.shape[-1]  # rank

    # Fail early on non-finite D or V.
    if not (torch.isfinite(D).all() and torch.isfinite(V).all()):
        import warnings

        warnings.warn(
            f"woodbury_nll: non-finite values in D or V "
            f"(D nan={D.isnan().sum().item()} inf={D.isinf().sum().item()}, "
            f"V nan={V.isnan().sum().item()} inf={V.isinf().sum().item()}). "
            "This typically indicates a gradient explosion — check your LR / clip_grad_norm.",
            RuntimeWarning,
            stacklevel=2,
        )

    # D^{-1} r  — reused in both the quadratic term and the V^T D^{-1} r product
    D_inv_r = r_vec / D  # (B, N, d)

    # D^{-1} V  — reused in the capacitance matrix
    D_inv_V = V / D.unsqueeze(-1)  # (B, N, d, r)

    # Capacitance M = I_r + V^T D^{-1} V, symmetrized.
    M_raw = torch.matmul(V.transpose(-2, -1), D_inv_V)  # (B, N, r, r)
    M = torch.eye(r, dtype=V.dtype, device=V.device) + 0.5 * (M_raw + M_raw.transpose(-2, -1))
    # Cholesky of M for stable solve and log-det
    L_M = _safe_cholesky(M)  # (B, N, r, r)

    # V^T D^{-1} r = V^T (D_inv_r)                         # (B, N, r)
    VT_Dinv_r = torch.matmul(V.transpose(-2, -1), D_inv_r.unsqueeze(-1)).squeeze(-1)

    # M^{-1} (V^T D^{-1} r) via two triangular solves (cholesky_solve lacks sm_75 kernels in PyTorch 2.11+cu130).
    rhs = VT_Dinv_r.unsqueeze(-1)  # (B, N, r, 1)
    tmp = torch.linalg.solve_triangular(L_M, rhs, upper=False)
    Minv_VT_Dinv_r = torch.linalg.solve_triangular(L_M.transpose(-2, -1), tmp, upper=True).squeeze(-1)  # (B, N, r)

    # Quadratic form  r^T D^{-1} r - (V^T D^{-1} r)^T M^{-1} (V^T D^{-1} r)
    quad = (
        (r_vec * D_inv_r).sum(-1)  # (B, N)
        - (VT_Dinv_r * Minv_VT_Dinv_r).sum(-1)  # (B, N)
    )

    # Log-determinant  log|D| + log|M|  (Sylvester/Matrix Determinant Lemma)
    log_det_D = D.log().sum(-1)  # (B, N)
    log_det_M = (
        2.0 * L_M.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (B, N)
    )
    log_det = log_det_D + log_det_M

    # NLL = 0.5 * (d log 2π + log|Σ| + quadratic)
    d_size = y.shape[-1]
    nll = 0.5 * (d_size * math.log(2.0 * math.pi) + log_det + quad)

    return nll.mean()


def indep_normal_nll(z: torch.Tensor) -> torch.Tensor:
    """Mean NLL of z under i.i.d. N(0, 1): d/2 log(2 pi) + ||z||^2 / 2.

    copula_nll = woodbury_nll(z; 0, R) - indep_normal_nll(z).

    Args:
        z: (B, N, d) or (B, d).

    Returns:
        Scalar averaged over instances.
    """
    d_size = z.shape[-1]
    return 0.5 * (d_size * math.log(2.0 * math.pi) + (z**2).sum(-1)).mean()


def marginal_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Mean NLL under independent marginals N(mu, diag(D)) (woodbury_nll with V = 0).

    Args:
        y, mu, D: (B, N, d).

    Returns:
        Scalar NLL averaged over B*N instances.
    """
    d_size = y.shape[-1]
    log_det_D = D.log().sum(dim=-1)  # (B, N)
    quad_form = ((y - mu) ** 2 / D).sum(dim=-1)  # (B, N)
    nll = 0.5 * (d_size * math.log(2.0 * math.pi) + log_det_D + quad_form)
    return nll.mean()


def plot_prediction_comparison(
    mu_pred: torch.Tensor,
    D_pred: torch.Tensor,
    V_pred: torch.Tensor,
    mu_true: torch.Tensor,
    D_true: torch.Tensor,
    V_true: torch.Tensor,
    batch_idx: int = 0,
    n_instances: int = 3,
    mu_tabicl: torch.Tensor | None = None,
):
    """Plot predicted vs oracle covariance and mean for n_instances instances of one batch element.

    Columns: oracle Sigma, predicted Sigma, |difference|, mu (with optional
    TabICL base), |mu difference|.

    Args:
        mu_pred, mu_true: (B, N, d).
        D_pred, D_true: (B, N, d).
        V_pred, V_true: (B, N, d, r).
        batch_idx: batch element to plot.
        n_instances: rows to plot.
        mu_tabicl: optional (B, N, d) TabICL predictions.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(mu_pred.shape, D_pred.shape, V_pred.shape, "pred")
    print(mu_true.shape, D_true.shape, V_true.shape, "true")
    print(mu_tabicl.shape if mu_tabicl is not None else None, "tabicl")

    N = D_pred.shape[1]
    n_instances = min(n_instances, N)
    indices = np.linspace(0, N - 1, n_instances, dtype=int)

    fig, axes = plt.subplots(n_instances, 5, figsize=(26, 5 * n_instances))
    if n_instances == 1:
        axes = axes[np.newaxis, :]

    for row, inst_idx in enumerate(indices):
        # ---- Covariance ----
        Sp_V = V_pred[batch_idx, inst_idx]
        Sp_D = torch.diag(D_pred[batch_idx, inst_idx])
        Sigma_pred = (Sp_D + Sp_V @ Sp_V.T).detach().cpu().numpy()

        St_V = V_true[batch_idx, inst_idx]
        St_D = torch.diag(D_true[batch_idx, inst_idx])
        Sigma_true = (St_D + St_V @ St_V.T).detach().cpu().numpy()

        cov_max = max(np.abs(Sigma_true).max(), np.abs(Sigma_pred).max())

        sns.heatmap(
            Sigma_true,
            ax=axes[row, 0],
            cmap="coolwarm",
            center=0,
            vmin=-cov_max,
            vmax=cov_max,
            square=True,
        )
        axes[row, 0].set_title(rf"Oracle $\Sigma^*$ (inst {inst_idx})")

        sns.heatmap(
            Sigma_pred,
            ax=axes[row, 1],
            cmap="coolwarm",
            center=0,
            vmin=-cov_max,
            vmax=cov_max,
            square=True,
        )
        axes[row, 1].set_title(rf"Predicted $\hat{{\Sigma}}$ (inst {inst_idx})")

        sns.heatmap(np.abs(Sigma_true - Sigma_pred), ax=axes[row, 2], cmap="Reds", square=True)
        axes[row, 2].set_title(rf"$|\Sigma^* - \hat{{\Sigma}}|$ (inst {inst_idx})")

        # ---- Mean ----
        mu_t = mu_true[batch_idx, inst_idx].detach().cpu().numpy()  # (d,)
        mu_p = mu_pred[batch_idx, inst_idx].detach().cpu().numpy()  # (d,)
        d = len(mu_t)
        dims = np.arange(d)

        ax = axes[row, 3]
        if mu_tabicl is not None:
            mu_b = mu_tabicl[batch_idx, inst_idx].detach().cpu().numpy()  # (d,)
            width = 0.25
            ax.bar(
                dims - width,
                mu_t,
                width,
                label=r"Oracle $\mu^*$",
                color="#2563EB",
                alpha=0.8,
            )
            ax.bar(
                dims,
                mu_p,
                width,
                label=r"Predicted $\hat{\mu}$",
                color="#EA580C",
                alpha=0.8,
            )
            ax.bar(
                dims + width,
                mu_b,
                width,
                label=r"TabICL $\mu_{\rm base}$",
                color="#16A34A",
                alpha=0.8,
            )
        else:
            width = 0.35
            ax.bar(
                dims - width / 2,
                mu_t,
                width,
                label=r"Oracle $\mu^*$",
                color="#2563EB",
                alpha=0.8,
            )
            ax.bar(
                dims + width / 2,
                mu_p,
                width,
                label=r"Predicted $\hat{\mu}$",
                color="#EA580C",
                alpha=0.8,
            )
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(dims)
        ax.set_xlabel("dim")
        ax.set_title(rf"Mean (inst {inst_idx})")
        ax.legend(fontsize=8)

        ax = axes[row, 4]
        if mu_tabicl is not None:
            width = 0.35
            ax.bar(
                dims - width / 2,
                np.abs(mu_t - mu_p),
                width,
                label=r"$|\mu^*-\hat{\mu}|$",
                color="#7C3AED",
                alpha=0.8,
            )
            ax.bar(
                dims + width / 2,
                np.abs(mu_t - mu_b),
                width,
                label=r"$|\mu^*-\mu_{\rm base}|$",
                color="#16A34A",
                alpha=0.8,
            )
            ax.legend(fontsize=8)
        else:
            ax.bar(dims, np.abs(mu_t - mu_p), color="#7C3AED", alpha=0.8)
        ax.set_xticks(dims)
        ax.set_xlabel("dim")
        ax.set_title(rf"$|\mu^* - \hat{{\mu}}|$ (inst {inst_idx})")

    plt.tight_layout()
    return fig


def energy_score(
    mu: torch.Tensor,
    D: torch.Tensor,
    V: torch.Tensor,
    y_ref: torch.Tensor,
    n_samples: int = 100,
) -> torch.Tensor:
    """Energy score of N(mu, diag(D) + V V^T) at y_ref: E||Y - y|| - 0.5 E||Y - Y'||.

    Args:
        mu: (d,).
        D: (d,).
        V: (d, r).
        y_ref: (d,).
        n_samples: Monte Carlo samples.

    Returns:
        Scalar energy score (lower is better).
    """
    d = mu.shape[-1]
    r = V.shape[-1]

    # Sample via reparameterisation: Y = mu + D^{1/2} eps_diag + V eps_low
    eps_diag = torch.randn(n_samples, d, device=mu.device)  # (M, d)
    eps_low = torch.randn(n_samples, r, device=mu.device)  # (M, r)

    # samples: (M, d)
    samples = mu + D.sqrt() * eps_diag + (eps_low @ V.T)

    # E[||Y - y_ref||]  — distance from each sample to the reference
    diff_ref = (samples - y_ref.unsqueeze(0)).norm(dim=-1)  # (M,)
    term1 = diff_ref.mean()

    # E[||Y - Y'||]  — mean pairwise distance over all M^2 pairs
    term2 = torch.cdist(samples, samples).mean()  # scalar

    return term1 - 0.5 * term2


def kl_gaussian(
    mu_q: torch.Tensor,
    D_q: torch.Tensor,
    V_q: torch.Tensor,
    mu_p: torch.Tensor,
    Sigma_p: torch.Tensor,
) -> torch.Tensor:
    """KL(Q || P) for Q = N(mu_q, diag(D_q) + V_q V_q^T) and dense P = N(mu_p, Sigma_p).

    Args:
        mu_q: (d,).
        D_q: (d,).
        V_q: (d, r).
        mu_p: (d,).
        Sigma_p: (d, d).

    Returns:
        Scalar KL in nats.
    """
    d = mu_q.shape[0]

    # Build Q covariance (dense) for Cholesky
    Sigma_q = torch.diag(D_q) + V_q @ V_q.T  # (d, d)

    # Cholesky factors
    L_p = _safe_cholesky(Sigma_p)  # (d, d)
    L_q = _safe_cholesky(Sigma_q)  # (d, d)

    # log|Sigma_p| = 2 * sum log diag(L_p)
    log_det_p = 2.0 * L_p.diagonal().log().sum()

    # log|Sigma_q| = 2 * sum log diag(L_q)
    log_det_q = 2.0 * L_q.diagonal().log().sum()

    # tr(Sigma_p^{-1} Sigma_q) = ||L_p^{-1} L_q||_F^2.
    A = torch.linalg.solve_triangular(L_p, L_q, upper=False)  # (d, d)
    trace_term = (A * A).sum()

    # (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
    diff = (mu_p - mu_q).unsqueeze(-1)  # (d, 1)
    v = torch.linalg.solve_triangular(L_p, diff, upper=False)  # (d, 1)
    quad_term = (v * v).sum()

    kl = 0.5 * (log_det_p - log_det_q - d + trace_term + quad_term)
    return kl


def copula_nll(
    W_tilde: torch.Tensor,
    z_test: torch.Tensor,
    test_mask: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Gaussian copula NLL for R = eps I + W_tilde W_tilde^T, via Woodbury.

        L = 0.5 (log|R| + z^T R^{-1} z - z^T z) / N

    Args:
        W_tilde: (B, N_max, r+1) unit-row-norm factor.
        z_test: (B, N_max), 0 on padding.
        test_mask: (B, N_max) bool.
        eps: diagonal jitter.

    Returns:
        Scalar mean loss.
    """
    B, device = W_tilde.shape[0], W_tilde.device
    r1 = W_tilde.shape[-1]  # r+1
    eye_r = torch.eye(r1, device=device)

    losses = []
    for b in range(B):
        N = int(test_mask[b].sum().item())
        if N == 0:
            continue

        W = W_tilde[b, :N]  # (N, r+1)
        z = z_test[b, :N]  # (N,)

        # Capacitance matrix M = I_{r+1} + (1/ε) W^T W     shape (r+1, r+1)
        M = eye_r + (W.T @ W) / eps
        L_M = _safe_cholesky(M)  # (r+1, r+1)

        # Matrix Determinant Lemma: log|R_ε| = N log(ε) + log|M|
        log_det_M = 2.0 * L_M.diagonal().log().sum()
        log_det = N * math.log(eps) + log_det_M

        # Woodbury: R^{-1} z = (z - W M^{-1} W^T z / eps) / eps (solve_triangular; see woodbury_nll).
        WTz = W.T @ z  # (r+1,)
        rhs_v = (WTz / eps).unsqueeze(-1)
        tmp_v = torch.linalg.solve_triangular(L_M, rhs_v, upper=False)
        v = torch.linalg.solve_triangular(L_M.T, tmp_v, upper=True).squeeze(-1)  # (r+1,)
        R_inv_z = (z - W @ v) / eps  # (N,)

        # Copula NLL: 0.5 * (log|R| + z^T R^{-1} z - z^T z) / N
        loss_b = 0.5 * (log_det + z @ R_inv_z - z @ z) / N
        losses.append(loss_b)

    if len(losses) == 0:
        return W_tilde.sum() * 0.0  # differentiable zero

    return torch.stack(losses).mean()


def oracle_copula_nll(
    R_star: torch.Tensor,
    z_test: torch.Tensor,
    test_mask: torch.Tensor,
) -> torch.Tensor:
    """Copula NLL under the oracle correlation R_star (dense Cholesky).

    Args:
        R_star: (B, N_max, N_max).
        z_test: (B, N_max).
        test_mask: (B, N_max).
    """
    B, N_max, _ = R_star.shape
    n_test = test_mask.sum(-1).float()  # (B,)

    mask_2d = test_mask.unsqueeze(-1) & test_mask.unsqueeze(-2)
    eye = torch.eye(N_max, device=R_star.device, dtype=R_star.dtype).unsqueeze(0)
    R_safe = torch.where(mask_2d, R_star, eye)

    L = _safe_cholesky(R_safe)

    # Padded L-diagonal = 1 → log(1) = 0, padded z = 0 → no contribution
    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)  # (B,)
    rhs = z_test.unsqueeze(-1)
    tmp = torch.linalg.solve_triangular(L, rhs, upper=False)
    R_inv_z = torch.linalg.solve_triangular(L.mT, tmp, upper=True).squeeze(-1)
    losses = 0.5 * (log_det + (z_test * R_inv_z).sum(-1) - (z_test**2).sum(-1)) / n_test.clamp(min=1)

    valid = n_test > 0
    if not valid.any():
        return z_test.sum() * 0.0
    return losses[valid].mean()


def y_space_nll(
    Sigma: torch.Tensor,
    z_test: torch.Tensor,
    log_pdf_test: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict:
    """Negative log-likelihood of Y by Sklar's theorem: copula NLL + marginal NLL.

        copula = 0.5 log|Sigma| + 0.5 z^T (Sigma^{-1} - I) z
        marginal = -sum_i log p(y_i | x_i, context)

    Args:
        Sigma: (B, N_max, N_max) dense correlation, or a LowRankCorrelationFactor
            (computed by _y_space_nll_lowrank; same result).
        z_test: (B, N_max), 0 on padding.
        log_pdf_test: (B, N_max) marginal log-densities at y_test.
        test_mask: (B, N_max) bool.

    Returns:
        dict with total, copula and marginal, per-instance means averaged over the batch.
    """
    if isinstance(Sigma, LowRankCorrelationFactor):
        return _y_space_nll_lowrank(Sigma, z_test, log_pdf_test, test_mask)

    B, N_max, _ = Sigma.shape
    n_test = test_mask.sum(-1).float()  # (B,)

    # Masked blocks become identity; with zero-padded z_test and log_pdf_test the
    # padded positions contribute nothing.
    mask_2d = test_mask.unsqueeze(-1) & test_mask.unsqueeze(-2)  # (B, N_max, N_max)
    eye = torch.eye(N_max, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
    S_safe = torch.where(mask_2d, Sigma, eye)

    L = _safe_cholesky(S_safe)  # (B, N_max, N_max)

    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)  # (B,)

    rhs = z_test.unsqueeze(-1)  # (B, N_max, 1)
    tmp = torch.linalg.solve_triangular(L, rhs, upper=False)
    S_inv_z = torch.linalg.solve_triangular(L.mT, tmp, upper=True).squeeze(-1)  # (B, N_max)

    n_safe = n_test.clamp(min=1)
    copula = 0.5 * (log_det + (z_test * S_inv_z).sum(-1) - (z_test**2).sum(-1)) / n_safe
    marginal = -log_pdf_test.sum(-1) / n_safe

    valid = n_test > 0
    if not valid.any():
        zero = Sigma.sum() * 0.0
        return {"total": zero, "copula": zero, "marginal": zero}

    copula_mean = copula[valid].mean()
    marginal_mean = marginal[valid].mean()
    return {
        "total": copula_mean + marginal_mean,
        "copula": copula_mean,
        "marginal": marginal_mean,
    }


def _y_space_nll_lowrank(
    factor: LowRankCorrelationFactor,
    z_test: torch.Tensor,
    log_pdf_test: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict:
    """y_space_nll for Sigma = U U^T + diag(D) without forming Sigma.

    With U~ = D^{-1/2} U, z~ = D^{-1/2} z and M = I_r + U~^T U~ = L L^T:

        log|Sigma| = sum_i log D_i + log|M|
        z^T Sigma^{-1} z = ||z~||^2 - ||L^{-1} U~^T z~||^2

    Padded rows have U=0, D=1, z=0. Computed in float64.
    """
    out_dtype = factor.U.dtype
    mask = test_mask.bool()
    n_test = mask.sum(-1).to(out_dtype)  # (B,)

    U = factor.U.double() * mask.unsqueeze(-1)  # (B, N, r)
    D = torch.where(mask, factor.D.double(), torch.ones_like(factor.D, dtype=torch.float64))
    z = z_test.double() * mask

    d_isqrt = D.rsqrt()
    U_t = U * d_isqrt.unsqueeze(-1)  # Ũ = D^{-1/2} U
    z_t = z * d_isqrt  # z̃ = D^{-1/2} z

    r = U.shape[-1]
    eye_r = torch.eye(r, dtype=torch.float64, device=U.device)
    M = eye_r + U_t.transpose(-1, -2) @ U_t  # (B, r, r)
    L_M = _safe_cholesky(M)

    log_det = D.log().sum(-1) + 2.0 * L_M.diagonal(dim1=-2, dim2=-1).clamp_min(1e-300).log().sum(-1)
    w = torch.linalg.solve_triangular(L_M, (U_t.transpose(-1, -2) @ z_t.unsqueeze(-1)), upper=False).squeeze(
        -1
    )  # L^{-1} Ũ^T z̃, (B, r)
    quad = (z_t * z_t).sum(-1) - (w * w).sum(-1)  # z^T Σ^{-1} z

    n_safe = n_test.clamp(min=1)
    copula = (0.5 * (log_det + quad - (z * z).sum(-1))).to(out_dtype) / n_safe
    marginal = -log_pdf_test.sum(-1) / n_safe

    valid = n_test > 0
    if not valid.any():
        zero = factor.U.sum() * 0.0
        return {"total": zero, "copula": zero, "marginal": zero}

    copula_mean = copula[valid].mean()
    marginal_mean = marginal[valid].mean()
    return {
        "total": copula_mean + marginal_mean,
        "copula": copula_mean,
        "marginal": marginal_mean,
    }


def gp_oracle_y_nll(
    Sigma_star: torch.Tensor,
    mu_star: torch.Tensor,
    y_test: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict:
    """Exact GP NLL -log N(y_test | mu_star, Sigma_star), per instance.

    marginal is the NLL under diag(Sigma_star); copula = total - marginal.

    Args:
        Sigma_star: (B, N_max, N_max).
        mu_star: (B, N_max).
        y_test: (B, N_max).
        test_mask: (B, N_max) bool.

    Returns:
        dict with total, copula, marginal: scalar means over the batch.
    """
    B, N_max, _ = Sigma_star.shape
    log_2pi = math.log(2.0 * math.pi)
    n_test = test_mask.sum(-1).float()  # (B,)

    mask_2d = test_mask.unsqueeze(-1) & test_mask.unsqueeze(-2)
    eye = torch.eye(N_max, device=Sigma_star.device, dtype=Sigma_star.dtype).unsqueeze(0)
    S_safe = torch.where(mask_2d, Sigma_star, eye)

    L = _safe_cholesky(S_safe)

    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)  # (B,)

    # Zero-padded residuals: y_test and mu_star are 0-padded by collate_fn
    r = (y_test - mu_star) * test_mask  # (B, N_max)
    rhs = r.unsqueeze(-1)
    tmp = torch.linalg.solve_triangular(L, rhs, upper=False)
    S_inv_r = torch.linalg.solve_triangular(L.mT, tmp, upper=True).squeeze(-1)
    quad = (r * S_inv_r).sum(-1)  # (B,)

    # Diagonal marginal — mask padded positions to 1 to avoid log(0)
    diag = Sigma_star.diagonal(dim1=-2, dim2=-1).masked_fill(~test_mask, 1.0)  # (B, N_max)
    log_det_diag = diag.log().sum(-1)  # (B,), padded entries: log(1) = 0
    quad_diag = ((y_test - mu_star).pow(2) / diag * test_mask).sum(-1)  # (B,)

    n_safe = n_test.clamp(min=1)
    total_nll = 0.5 * (log_det + quad + n_test * log_2pi) / n_safe
    mar_nll = 0.5 * (log_det_diag + quad_diag + n_test * log_2pi) / n_safe
    cop_nll = total_nll - mar_nll

    valid = n_test > 0
    if not valid.any():
        zero = Sigma_star.sum() * 0.0
        return {"total": zero, "copula": zero, "marginal": zero}
    return {
        "total": total_nll[valid].mean(),
        "copula": cop_nll[valid].mean(),
        "marginal": mar_nll[valid].mean(),
    }
