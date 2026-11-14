"""The Bayes predictive of a GP episode over the generator's hyperparameter prior.

A context-only model never sees the episode's hyperparameters theta (kernel parameters,
noise, mean function, sign modulation). Its Bayes-optimal predictive at the queries is

    p(y_Q | y_C) = int p(y_Q | y_C, theta) p(theta | y_C) dtheta
                 ~ sum_k w_k N(y_Q; mu_k, Sigma_k),   theta_k ~ p(theta),  w_k ∝ p(y_C | theta_k),

with theta_k drawn by the generator's own sampler (_sample_episode_priors) for the episode's
kernel family (or chain) and active dims, and the kernel inputs x_kernel held at the episode's.
So "Bayes" here marginalises the hyperparameters only: the kernel family, the feature warps and
the hidden kernel warp stay known, which a context-only model does not have. Two fairer variants
were tried on table2_rbf (2026-11-14, 8 episodes, 262144 draws) and dropped: drawing the family,
active dims and hidden warp from the prior too collapses the importance ESS to ~1-7 (NLL 0.2-2.0
nats/point, worse than the warp-known prior); evaluating the kernel on the visible x_norm with
every column active gives NLL -0.1-1.7, worse than the TabICL marginal alone. So this reference
estimates the gain with the warp known, which (more information never hurts the Bayes-optimal
predictor in expectation) roughly bounds what a context-only copula can reach; it is not a
target a context-only model is known to attain.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from copula_inter.data.gp_episodes import _CallShape, _evaluate_kernel_dense, _sample_episode_priors


@dataclass
class BayesPredictive:
    """A weighted Gaussian mixture over the retained prior draws (raw y scale, float64)."""

    log_weights: Tensor  # (K,) normalised over the retained draws
    mu: Tensor  # (K, N)
    chol: Tensor  # (K, N, N) Cholesky of each Sigma_k
    ess: float  # Kish effective sample size over every draw
    n_draws: int  # prior draws with a finite context likelihood
    dropped_mass: float  # importance weight of the draws beyond the retained ones

    def log_prob(self, y: Tensor) -> Tensor:
        """log p(y) for y (S, N): logsumexp_k [log w_k + log N(y; mu_k, Sigma_k)]."""
        y = y.to(self.mu)
        S, N = y.shape
        out = torch.empty(S, dtype=self.mu.dtype, device=self.mu.device)
        half_log_2pi = 0.5 * N * math.log(2.0 * math.pi)
        log_det = self.chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (K,) half log det
        for s0 in range(0, S, 64):
            diff = y[s0 : s0 + 64, None, :] - self.mu[None]  # (s, K, N)
            sol = torch.linalg.solve_triangular(self.chol[None], diff.unsqueeze(-1), upper=False).squeeze(-1)
            comp = -0.5 * (sol**2).sum(-1) - log_det[None] - half_log_2pi  # (s, K)
            out[s0 : s0 + 64] = torch.logsumexp(comp + self.log_weights[None], dim=-1)
        return out

    def marginal_log_prob(self, y: Tensor) -> Tensor:
        """sum_j log p(y_j) for y (S, N), each coordinate's marginal a 1-D mixture."""
        y = y.to(self.mu)
        sd = self.chol.pow(2).sum(-1).sqrt()  # (K, N): sqrt(diag Sigma_k)
        comp = torch.distributions.Normal(self.mu[None], sd[None]).log_prob(y[:, None, :])  # (S, K, N)
        return torch.logsumexp(comp + self.log_weights[None, :, None], dim=1).sum(-1)

    def sample(self, n: int, generator: torch.Generator) -> Tensor:
        """n draws (n, N) from the mixture; the generator lives on the CPU."""
        probs = self.log_weights.exp().cpu()
        idx = torch.multinomial(probs, n, replacement=True, generator=generator).to(self.mu.device)
        eps = torch.randn(n, self.mu.shape[1], generator=generator, dtype=self.mu.dtype).to(self.mu.device)
        return self.mu[idx] + (self.chol[idx] @ eps.unsqueeze(-1)).squeeze(-1)


def episode_call_shape(ep: dict, B: int) -> _CallShape:
    """The generator's call shape for B fresh draws of ep's kernel family (or chain) and active dims."""
    P = int(ep["x_kernel_train"].shape[0])
    N = int(ep["x_kernel_test"].shape[0])
    d = int(ep["x_kernel_train"].shape[1])
    cols = [int(c) for c in ep["kernel_feature_indices"].tolist()]
    systematic = "kernel_components" in ep
    return _CallShape(
        d=d,
        kernel_name=str(ep["kernel"]),
        systematic=systematic,
        chain_names=list(ep["kernel_components"]) if systematic else [],
        chain_ops=list(ep["kernel_ops"]) if systematic else [],
        P=P,
        N=N,
        P_C=0,
        B=B,
        kernel_cols=None if cols == list(range(d)) and str(ep["kernel"]) == "dot_product" else cols,
    )


def _draw_conditionals(cfg: DictConfig, ep: dict, B: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """B prior draws' context log-likelihoods (B,) and query conditionals mu (B, N), Sigma (B, N, N)."""
    shape = episode_call_shape(ep, B)
    P = shape.P
    priors = _sample_episode_priors(cfg, shape, device)
    x = torch.cat([ep["x_kernel_train"], ep["x_kernel_test"]]).to(device=device, dtype=torch.float32)
    x = x.unsqueeze(0).expand(B, -1, -1).contiguous()
    with torch.no_grad():
        K = _evaluate_kernel_dense(priors.kernel_obj, x).double()  # (B, T, T)
        mean = priors.mean_module(x).double()  # (B, T)
        noise = priors.likelihood.noise.reshape(B).double()
    T = K.shape[-1]
    K = 0.5 * (K + K.mT) + noise[:, None, None] * torch.eye(T, dtype=torch.float64, device=device)
    y_C = ep["y_train"].to(device=device, dtype=torch.float64)
    r = (y_C - mean[:, :P]).unsqueeze(-1)  # (B, P, 1)
    L, info = torch.linalg.cholesky_ex(K[:, :P, :P])
    ok = info == 0
    L = torch.where(ok[:, None, None], L, torch.eye(P, dtype=torch.float64, device=device))
    alpha = torch.cholesky_solve(r, L)
    log_lik = (
        -0.5 * (r * alpha).sum((1, 2)) - L.diagonal(dim1=-2, dim2=-1).log().sum(-1) - 0.5 * P * math.log(2 * math.pi)
    )
    K_sf = K[:, P:, :P]
    V = torch.linalg.solve_triangular(L, K_sf.mT, upper=False)  # (B, P, N)
    mu = mean[:, P:] + (K_sf @ alpha).squeeze(-1)
    Sigma = K[:, P:, P:] - V.mT @ V
    Sigma = 0.5 * (Sigma + Sigma.mT)
    log_lik = torch.where(ok & log_lik.isfinite(), log_lik, torch.full_like(log_lik, -math.inf))
    return log_lik, mu, Sigma


def bayes_predictive(
    data_cfg: dict | DictConfig,
    ep: dict,
    n_draws: int,
    device: torch.device,
    seed: int,
    max_keep: int = 4096,
    chunk: int | None = None,
) -> BayesPredictive:
    """Importance-sample the hyperparameter posterior from the generator's prior (see the module doc).

    The mixture keeps max_keep components by priority sampling (Duffield, Lund & Thorup 2007):
    priority q_i = w_i / u_i with u_i ~ U(0, 1), the max_keep largest kept, tau the next one, and
    each kept weight estimated by max(w_i, tau) -- unbiased for every subset sum of the weights,
    in one streaming pass. With at most max_keep finite draws every draw is kept at its own
    weight. ess (Kish) and dropped_mass (the true weight of the draws not kept) are over every
    draw. Python, numpy and torch RNG states are restored afterwards.
    """
    data = OmegaConf.to_container(data_cfg, resolve=False) if OmegaConf.is_config(data_cfg) else dict(data_cfg)
    cfg = OmegaConf.create({"data": data})
    P = int(ep["x_kernel_train"].shape[0])
    N = int(ep["x_kernel_test"].shape[0])
    if chunk is None:
        chunk = max(16, min(4096, int(1.5e9 // (8 * 4 * (P + N) ** 2))))
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [device] if device.type == "cuda" else []
    all_ll: list[Tensor] = []
    keep_q = torch.empty(0, dtype=torch.float64, device=device)  # log priority
    keep_ll = torch.empty(0, dtype=torch.float64, device=device)
    keep_mu = torch.empty(0, N, dtype=torch.float64, device=device)
    keep_S = torch.empty(0, N, N, dtype=torch.float64, device=device)
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            for start in range(0, n_draws, chunk):
                ll, mu, Sigma = _draw_conditionals(cfg, ep, min(chunk, n_draws - start), device)
                all_ll.append(ll)
                u = torch.rand(ll.shape, dtype=torch.float64, device=device).clamp(min=1e-300)
                keep_q = torch.cat([keep_q, ll - u.log()])
                keep_ll = torch.cat([keep_ll, ll])
                keep_mu = torch.cat([keep_mu, mu])
                keep_S = torch.cat([keep_S, Sigma])
                if keep_q.shape[0] > max_keep + 1:
                    top = keep_q.topk(max_keep + 1).indices
                    keep_q, keep_ll, keep_mu, keep_S = keep_q[top], keep_ll[top], keep_mu[top], keep_S[top]
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    ll = torch.cat(all_ll)
    finite = ll.isfinite()
    if not bool(finite.any()):
        raise RuntimeError("bayes_predictive: no prior draw has a finite context likelihood")
    log_norm = torch.logsumexp(ll[finite], 0)
    ess = float(1.0 / (ll[finite] - log_norm).exp().pow(2).sum())
    kept = keep_ll.isfinite()
    keep_q, keep_ll, keep_mu, keep_S = keep_q[kept], keep_ll[kept], keep_mu[kept], keep_S[kept]
    log_w = keep_ll
    if keep_q.shape[0] > max_keep:
        order = keep_q.argsort(descending=True)
        tau = keep_q[order[max_keep]]
        sel = order[:max_keep]
        keep_ll, keep_mu, keep_S = keep_ll[sel], keep_mu[sel], keep_S[sel]
        log_w = torch.maximum(keep_ll, tau)
    dropped = float((1.0 - (keep_ll - log_norm).exp().sum()).clamp(min=0.0))
    eye = torch.eye(N, dtype=torch.float64, device=device)
    # Each Sigma_k already carries the noise on its diagonal; a tiny jitter guards round-off.
    chol, info = torch.linalg.cholesky_ex(keep_S + 1e-9 * eye)
    good = info == 0
    if not bool(good.all()):
        evals, evecs = torch.linalg.eigh(keep_S[~good])
        repaired = evecs @ torch.diag_embed(evals.clamp(min=1e-9)) @ evecs.mT
        chol[~good] = torch.linalg.cholesky(0.5 * (repaired + repaired.mT))
    return BayesPredictive(
        log_weights=log_w - torch.logsumexp(log_w, 0),
        mu=keep_mu,
        chol=chol,
        ess=ess,
        n_draws=int(finite.sum()),
        dropped_mass=dropped,
    )
