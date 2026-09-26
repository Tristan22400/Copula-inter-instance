"""correlation_factory.py — low-rank correlation-matrix parametrizations.

Every builder below maps raw, unconstrained instance-encoder outputs to a
valid low-rank-plus-diagonal correlation matrix

    R = U U^T + diag(D),   U in R^{B x d x r},   D in R^{B x d},   D_ii > 0

and returns a ``LowRankCorrelationFactor`` wrapping ``(U, D)`` rather than
the dense ``R`` itself, so downstream code can choose between:

  - ``.dense()``            — materialize R (B, d, d); what the existing
    training pipeline consumes today (it runs dense Cholesky on Sigma,
    which is fine for d <= ~100 — see loss.py's N<=100 comment).
  - ``.log_det()``           — O(d r^2), Matrix Determinant Lemma.
  - ``.solve(z)`` / ``.quad_form(z)`` — O(d r^2), Woodbury identity; the
    actual operation an NLL needs and the only one that avoids ever
    forming a dense (d, d) tensor.
  - ``.inverse()``           — dense (B, d, d) precision matrix, built via
    Woodbury. NOTE: writing out a dense d x d result is inherently O(d^2 r)
    (d^2 output entries), not O(d r^2) — the O(d r^2) bound only holds for
    log_det()/solve(), which never materialize the dense inverse. Kept here
    because it is the natural thing to unit-test against
    ``torch.linalg.inv(dense())``.

Symmetry and strict positive-definiteness are structural: R is symmetric by
construction (U U^T is symmetric, D is diagonal) and D_ii > 0 is enforced by
every builder below (softplus, or an analytic 1 - ||u_i||^2 with ||u_i|| < 1
strictly), so R = U U^T + D is always strictly PD (sum of a PSD and a
strictly-PD diagonal matrix).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-6


@dataclass
class LowRankCorrelationFactor:
    """R = U @ U^T + diag(D), exposed via Woodbury-identity operations.

    U: (B, d, r) low-rank interaction factor.
    D: (B, d)    strictly-positive diagonal ("idiosyncratic noise"), stored
       as a vector — never materialized as a (B, d, d) diagonal matrix
       except where the caller explicitly asks for the dense matrix.
    """

    U: Tensor
    D: Tensor

    def dense(self) -> Tensor:
        """Materialize R = U U^T + diag(D) as a dense (B, d, d) tensor."""
        R = torch.matmul(self.U, self.U.transpose(-1, -2))
        d = self.D.shape[-1]
        idx = torch.arange(d, device=self.D.device)
        R = R.clone()
        R[..., idx, idx] = R[..., idx, idx] + self.D
        return R

    def _capacitance_cholesky(self) -> Tensor:
        """Cholesky factor L of the r x r capacitance matrix
        M = I_r + U^T D^{-1} U, used by both log_det() and the Woodbury ops.
        """
        B, d, r = self.U.shape
        D_inv = self.D.reciprocal()  # (B, d)
        Ut_Dinv = self.U.transpose(-1, -2) * D_inv.unsqueeze(-2)  # (B, r, d)
        M = torch.matmul(Ut_Dinv, self.U)  # (B, r, r)
        eye_r = torch.eye(r, device=self.U.device, dtype=self.U.dtype)
        M = M + eye_r
        return torch.linalg.cholesky(M)  # (B, r, r)

    def log_det(self) -> Tensor:
        """log det(R) via the Matrix Determinant Lemma. O(d r^2)."""
        L_M = self._capacitance_cholesky()  # (B, r, r)
        log_det_M = 2.0 * L_M.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)
        log_det_D = self.D.clamp_min(1e-12).log().sum(-1)
        return log_det_M + log_det_D

    def solve(self, z: Tensor) -> Tensor:
        """Apply R^{-1} to z via the Woodbury identity. O(d r^2).

        R^{-1} z = D^{-1} z - D^{-1} U (I_r + U^T D^{-1} U)^{-1} U^T D^{-1} z

        z: (B, d) — batch dims must match self.U/self.D.
        """
        D_inv = self.D.reciprocal()
        D_inv_z = D_inv * z  # (B, d)
        L_M = self._capacitance_cholesky()  # (B, r, r)
        Ut_Dinv_z = torch.matmul(self.U.transpose(-1, -2), D_inv_z.unsqueeze(-1))  # (B, r, 1)
        tmp = torch.linalg.solve_triangular(L_M, Ut_Dinv_z, upper=False)
        v = torch.linalg.solve_triangular(L_M.transpose(-1, -2), tmp, upper=True)  # (B, r, 1)
        correction = torch.matmul(self.U, v).squeeze(-1)  # (B, d)
        correction = D_inv * correction
        return D_inv_z - correction

    def quad_form(self, z: Tensor) -> Tensor:
        """z^T R^{-1} z, per batch element. O(d r^2)."""
        return (z * self.solve(z)).sum(-1)

    def inverse(self) -> Tensor:
        """Dense (B, d, d) precision matrix R^{-1} via the Woodbury identity.

        R^{-1} = D^{-1} - D^{-1} U (I_r + U^T D^{-1} U)^{-1} U^T D^{-1}

        See module docstring: materializing this dense (d, d) tensor costs
        O(d^2 r), not O(d r^2) — the low-rank complexity bound applies to
        solve()/log_det(), which never form this dense matrix.
        """
        B, d, r = self.U.shape
        D_inv = self.D.reciprocal()  # (B, d)
        L_M = self._capacitance_cholesky()  # (B, r, r)
        M_inv = torch.cholesky_inverse(L_M)  # (B, r, r)

        Dinv_U = self.U * D_inv.unsqueeze(-1)  # (B, d, r) == D^{-1} U
        correction = torch.matmul(Dinv_U, torch.matmul(M_inv, Dinv_U.transpose(-1, -2)))  # (B, d, d)

        idx = torch.arange(d, device=self.U.device)
        R_inv = -correction
        R_inv[..., idx, idx] = R_inv[..., idx, idx] + D_inv
        return R_inv


# ---------------------------------------------------------------------------
# Parametrization 1: Covariance Normalization (CovNorm)
# ---------------------------------------------------------------------------


def covnorm_correlation(W: Tensor, v: Tensor, eps: float = _EPS) -> LowRankCorrelationFactor:
    """CovNorm: raw W W^T + softplus diagonal, normalized to unit diagonal.

    W: (B, d, r) unconstrained.
    v: (B, d)    unconstrained; softplus(v) + eps sits on the diagonal.

    D_diag = softplus(v) + eps
    C = W W^T + diag(D_diag)
    S = diag(C)^{-1/2}
    R = S C S  =>  effective U = S W,  effective D = S^2 * D_diag
    """
    D_diag = F.softplus(v) + eps  # (B, d) > 0
    C_diag = (W * W).sum(-1) + D_diag  # diag(W W^T + diag(D_diag)), (B, d)
    S = C_diag.clamp_min(eps).rsqrt()  # (B, d)

    U = W * S.unsqueeze(-1)  # S W  (row-scaling, since S is diagonal)
    D = S * S * D_diag  # S^2 * D_diag  (diagonal-diagonal product commutes)
    return LowRankCorrelationFactor(U=U, D=D)


# ---------------------------------------------------------------------------
# Parametrization 2: L2-Normalized Cosine Similarity (CosSim)
# ---------------------------------------------------------------------------


def cossim_correlation(V: Tensor, g: Tensor, eps: float = _EPS) -> LowRankCorrelationFactor:
    """CosSim: unit-sphere row directions, scaled by a sigmoid gate.

    V: (B, d, r) unconstrained.
    g: (B, d)    unconstrained gate; s_i = sigmoid(g_i) * sqrt(1 - eps).

    U_i = s_i * V_i / (||V_i|| + eps)
    D_ii = 1 - s_i^2   (analytic; strictly positive since s_i < sqrt(1-eps))
    """
    V_norm = V.norm(dim=-1, keepdim=True)  # (B, d, 1)
    V_hat = V / (V_norm + eps)  # (B, d, r), unit rows (up to eps)

    s = torch.sigmoid(g) * (1.0 - eps) ** 0.5  # (B, d), strictly < 1
    U = V_hat * s.unsqueeze(-1)
    D = 1.0 - s * s  # (B, d), strictly > 0
    return LowRankCorrelationFactor(U=U, D=D)


# ---------------------------------------------------------------------------
# Parametrization 3: Tanh Row-Norm Projection (TanhNorm)
# ---------------------------------------------------------------------------


def tanhnorm_correlation(W: Tensor, eps: float = _EPS) -> LowRankCorrelationFactor:
    """TanhNorm: clamp each row's magnitude through tanh, keep its direction.

    W: (B, d, r) unconstrained.

    n_i = ||W_i||
    U_i = (tanh(n_i) * sqrt(1 - eps)) * W_i / (n_i + eps)
    D_ii = 1 - ||U_i||^2   (analytic; strictly positive since ||U_i|| < 1)
    """
    n = W.norm(dim=-1, keepdim=True)  # (B, d, 1)
    scale = torch.tanh(n) * (1.0 - eps) ** 0.5  # (B, d, 1), in [0, sqrt(1-eps))
    U = scale * W / (n + eps)  # (B, d, r)
    D = 1.0 - (U * U).sum(-1)  # (B, d), strictly > 0
    return LowRankCorrelationFactor(U=U, D=D)


# ---------------------------------------------------------------------------
# Parametrization 4: Sparse Covariance Normalization (SparseCovNorm)
# ---------------------------------------------------------------------------


def sparse_covnorm_correlation(
    W: Tensor, v: Tensor, lam_raw: Tensor, eps: float = _EPS
) -> LowRankCorrelationFactor:
    """SparseCovNorm: CovNorm with a soft-thresholded (sparsified) W.

    W: (B, d, r) unconstrained.
    v: (B, d)    unconstrained diagonal-variance logits.
    lam_raw: (B,) or (1,) unconstrained threshold; softplus(lam_raw) >= 0
             is broadcast against W before soft-thresholding.

    W_tilde = sign(W) * relu(|W| - softplus(lam_raw))
    D_diag  = softplus(v) + eps
    C = W_tilde W_tilde^T + diag(D_diag)
    S = diag(C)^{-1/2}  =>  U = S W_tilde,  D = S^2 * D_diag
    """
    lam = F.softplus(lam_raw)  # (B,) or (1,), >= 0
    while lam.dim() < W.dim() - 1:
        lam = lam.unsqueeze(-1)
    lam = lam.unsqueeze(-1)  # broadcastable against (B, d, r)
    W_tilde = torch.sign(W) * F.relu(W.abs() - lam)

    D_diag = F.softplus(v) + eps
    C_diag = (W_tilde * W_tilde).sum(-1) + D_diag
    S = C_diag.clamp_min(eps).rsqrt()

    U = W_tilde * S.unsqueeze(-1)
    D = S * S * D_diag
    return LowRankCorrelationFactor(U=U, D=D)
