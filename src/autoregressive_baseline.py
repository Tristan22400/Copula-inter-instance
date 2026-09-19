"""Autoregressive marginal-chain baseline.

Builds correlated joint predictions from a frozen TabICL marginal ALONE (no
copula head): for a query set of size N, chain N single-point forward
passes, growing the context by one row after each query point is processed
(the "Autoregressive Conditional Neural Process" construction of Bruinsma
et al., ICLR 2023). Correlation among query points is captured implicitly
by re-conditioning on the growing context, rather than by an explicit
copula correlation matrix.

Two distinct uses, not to be confused:
  * ``run_autoregressive_chain_nll`` -- TEACHER-FORCED: the true target is
    appended to the context at every step. This is the exact chain-rule
    decomposition log p(y_1..N) = sum_i log p(y_i | context, y_{<i}) of the
    true joint density under the model, and is what should be compared
    against a copula model's total NLL.
  * ``run_autoregressive_chain_sample`` -- ANCESTRAL SAMPLING: a value is
    drawn from each step's predictive distribution and THAT sampled value
    (not the true one) is appended to the context. This draws one joint
    sample path from the model's implied distribution, for visualization --
    a different quantity from the teacher-forced NLL above, not a cheaper
    way to compute it.

All tensors are expected already scaled via ``pit.normalize_targets`` (same
convention ``eval/runners/eval_checkpoint.py::_marginal_pit`` uses) -- the
caller is responsible for any raw-nats Jacobian correction
(``log p_raw(y) = log p_scaled(y_scaled) - log(std)``, see
``pit.normalize_targets``'s docstring).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pit import tabicl_forward


@torch.no_grad()
def _marginal_forward_dist(
    tabicl: nn.Module,
    X_context: torch.Tensor,
    Y_context: torch.Tensor,
    X_query: torch.Tensor,
):
    """One forward pass (no K-fold): the QuantileDistribution over
    ``X_query``, conditioned on ``(X_context, Y_context)`` as TabICL's
    context.

    Mirrors ``run_pit``'s test-side computation (``src/pit.py:290-306``),
    generalized to an arbitrary context/query split -- the autoregressive
    chain never needs a train-side K-fold PIT, only the query-side PPD at
    each step.

    Args:
        X_context : (P, p_x)
        Y_context : (P, d)
        X_query   : (M, p_x)

    Returns:
        A ``QuantileDistribution`` with batch_shape ``(d * M,)``, laid out
        in the same (d, M) row-major order as ``run_pit`` uses for its own
        ``.cdf()``/``.log_prob()`` calls (permute-then-flatten).
    """
    device = X_context.device
    d = Y_context.shape[1]
    M = X_query.shape[0]

    X_concat = torch.cat([X_context, X_query], dim=0)                # (P+M, p_x)
    X_batch = X_concat.unsqueeze(0).expand(d, -1, -1).contiguous()   # (d, P+M, p_x)
    y_context_batch = Y_context.permute(1, 0).contiguous()           # (d, P)

    logits = tabicl_forward(tabicl, X_batch, y_context_batch)        # (d, M, Q)
    logits = logits.to(device)
    Q = logits.shape[-1]
    return tabicl.quantile_dist(logits.reshape(d * M, Q))


@torch.no_grad()
def run_autoregressive_chain_nll(
    tabicl: nn.Module,
    X_context: torch.Tensor,
    Y_context: torch.Tensor,
    X_query: torch.Tensor,
    Y_query: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced per-step log-density of ``Y_query`` (d=1 targets).

    Args:
        X_context : (P, p_x)
        Y_context : (P, 1)
        X_query   : (N, p_x)
        Y_query   : (N, 1)

    Returns:
        (N,) log-density at each true query target, in the query's given
        order. Sum for the total teacher-forced joint log-density.
    """
    N = X_query.shape[0]
    log_probs = torch.empty(N, dtype=Y_query.dtype, device=Y_query.device)
    ctx_x, ctx_y = X_context, Y_context
    for i in range(N):
        x_i = X_query[i : i + 1]
        y_i = Y_query[i : i + 1]
        dist = _marginal_forward_dist(tabicl, ctx_x, ctx_y, x_i)
        log_probs[i] = dist.log_prob(y_i.reshape(-1))[0]
        ctx_x = torch.cat([ctx_x, x_i], dim=0)
        ctx_y = torch.cat([ctx_y, y_i], dim=0)
    return log_probs


@torch.no_grad()
def run_autoregressive_chain_sample(
    tabicl: nn.Module,
    X_context: torch.Tensor,
    Y_context: torch.Tensor,
    X_query: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Ancestral joint sample: draw ``u ~ Uniform(0,1)`` and set
    ``y_i = dist_i.icdf(u)`` at each step, appending the SAMPLED row (not
    the true one) to the context before the next step.

    Args:
        X_context : (P, p_x)
        Y_context : (P, 1)
        X_query   : (N, p_x)
        generator : optional CPU ``torch.Generator`` for reproducibility.

    Returns:
        (N,) sampled targets (scaled space), in the query's given order.
    """
    N = X_query.shape[0]
    samples = torch.empty(N, dtype=X_context.dtype, device=X_context.device)
    ctx_x, ctx_y = X_context, Y_context
    for i in range(N):
        x_i = X_query[i : i + 1]
        dist = _marginal_forward_dist(tabicl, ctx_x, ctx_y, x_i)
        u = torch.rand(1, generator=generator).to(device=x_i.device, dtype=samples.dtype)
        # icdf treats a plain (n,) alpha as n shared quantile levels broadcast
        # across the whole batch, not one alpha per distribution -- an
        # explicit trailing size-1 axis (matching src/train.py's
        # _era5_viz_field:855-857 usage) is what selects one quantile per
        # distribution instead of an (batch, n) grid.
        y_i = dist.icdf(u.unsqueeze(-1)).squeeze(-1)
        samples[i] = y_i[0]
        ctx_x = torch.cat([ctx_x, x_i], dim=0)
        ctx_y = torch.cat([ctx_y, y_i.reshape(1, 1)], dim=0)
    return samples
