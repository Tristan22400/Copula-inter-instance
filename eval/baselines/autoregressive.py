"""Joint predictive density from a marginal model alone, by the chain rule.

Reveal the test points one at a time, each prediction conditioned on the
context and the points already revealed:

    log p(y_1..y_N | ctx) = sum_i log p(y_s(i) | ctx, y_s(1)..y_s(i-1))

The model always sees the same P + N rows; only the context/query boundary
moves, so step 0 equals the one-shot PIT's log_pdf_test. Split as
marginal = one-shot NLL, total = chain NLL, copula = total - marginal.

conditioning="teacher_forcing" (default) appends the true y, giving an exact
joint log-density; "sample" appends a draw (ancestral sampling), which is not
a density of y_test. The total depends on the visit order (default: a seeded
random permutation per episode).
"""

from __future__ import annotations

import zlib
from typing import Optional

import torch

from copula_inter.pit import tabicl_forward

__all__ = [
    "autoregressive_log_pdf",
    "ar_parts_from_log_pdf",
    "AR_ORDERS",
    "AR_CONDITIONINGS",
]

AR_ORDERS = ("random", "natural")
AR_CONDITIONINGS = ("teacher_forcing", "sample")


def _orderings(
    B: int, N: int, order: str, seed: int, episode_indices: Optional[list[int]],
) -> torch.Tensor:
    """(B, N) visit orders, seeded per episode from zlib.crc32 of (seed, global index); "natural" is 0..N-1."""
    if order not in AR_ORDERS:
        raise ValueError(f"order must be one of {AR_ORDERS}, got {order!r}")
    if order == "natural":
        return torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    idxs = episode_indices if episode_indices is not None else list(range(B))
    rows = []
    for b in range(B):
        g = torch.Generator()
        g.manual_seed(zlib.crc32(f"ar-order:{seed}:{int(idxs[b])}".encode()) & 0x7FFFFFFF)
        rows.append(torch.randperm(N, generator=g))
    return torch.stack(rows)


@torch.no_grad()
def autoregressive_log_pdf(
    tabicl,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    order: str = "random",
    conditioning: str = "teacher_forcing",
    max_context: Optional[int] = None,
    seed: int = 0,
    episode_indices: Optional[list[int]] = None,
    progress_every: int = 0,
) -> dict:
    """Chain-rule log-densities at every test point of B episodes sharing P and N.

    Args:
        tabicl: TabICL marginal module.
        x_train: (B, P, d_x).
        y_train: (B, P) raw targets.
        x_test: (B, N, d_x).
        y_test: (B, N) raw targets.
        order: "random" (seeded per episode) or "natural".
        conditioning: "teacher_forcing" or "sample".
        max_context: cap on context rows (the P context rows are always kept;
            the oldest revealed points are dropped first).
        seed: run seed, mixed with each episode's global index.
        episode_indices: global episode indices (default 0..B-1).
        progress_every: print progress every this many steps (0 = silent).

    Returns:
        dict with log_pdf (B, N) in raw nats in the episodes' own test order,
        order (B, N) the visit order, and appended (B, N) the values appended at
        each step (visit order, raw units).
    """
    if conditioning not in AR_CONDITIONINGS:
        raise ValueError(
            f"conditioning must be one of {AR_CONDITIONINGS}, got {conditioning!r}"
        )
    B, P, d_x = x_train.shape
    N = x_test.shape[1]
    device, dtype = x_train.device, x_train.dtype

    # Scale y by the context's mean/std, fixed for the whole chain (as normalize_targets).
    mean = y_train.mean(dim=-1, keepdim=True)                  # (B, 1)
    std = y_train.std(dim=-1, keepdim=True).clamp(min=1e-8)    # (B, 1)
    y_train_s = (y_train - mean) / std
    y_test_s = (y_test - mean) / std

    visit = _orderings(B, N, order, seed, episode_indices).to(device)   # (B, N)

    # Context buffer: the P context rows, then room for every revealed point.
    ctx_x = torch.empty(B, P + N, d_x, device=device, dtype=dtype)
    ctx_y = torch.empty(B, P + N, device=device, dtype=y_train_s.dtype)
    ctx_x[:, :P] = x_train
    ctx_y[:, :P] = y_train_s

    log_pdf_s = torch.empty(B, N, device=device, dtype=y_train_s.dtype)
    appended = torch.empty(B, N, device=device, dtype=y_train_s.dtype)
    bidx = torch.arange(B, device=device)

    for i in range(N):
        rem = visit[:, i:]                                               # (B, N-i)
        x_rem = x_test.gather(1, rem.unsqueeze(-1).expand(-1, -1, d_x))  # (B, N-i, d_x)

        lo = 0 if max_context is None else max(0, (P + i) - int(max_context))
        if lo > 0:
            # Keep the episode's own context, drop the oldest revealed points.
            ctx_keep_x = torch.cat([ctx_x[:, :P], ctx_x[:, P + lo:P + i]], dim=1)
            ctx_keep_y = torch.cat([ctx_y[:, :P], ctx_y[:, P + lo:P + i]], dim=1)
        else:
            ctx_keep_x, ctx_keep_y = ctx_x[:, :P + i], ctx_y[:, :P + i]

        X = torch.cat([ctx_keep_x, x_rem], dim=1)                        # (B, n_ctx+N-i, d_x)
        logits = tabicl_forward(tabicl, X, ctx_keep_y)                   # (B, N-i, Q)
        # TabICL may return its output on CPU; move it back.
        logits = logits.to(device)
        # Only the point revealed at this step is scored.
        dist = tabicl.quantile_dist(logits[:, 0, :])                     # batch_shape (B,)

        tgt = visit[:, i]                                                # (B,)
        y_true = y_test_s[bidx, tgt]                                     # (B,)
        log_pdf_s[bidx, tgt] = dist.log_prob(y_true).to(log_pdf_s.dtype)

        if conditioning == "teacher_forcing":
            y_next = y_true
        else:
            # CPU generator per step, moved to the device.
            g = torch.Generator()
            g.manual_seed(zlib.crc32(f"ar-sample:{seed}:{i}".encode()) & 0x7FFFFFFF)
            u = torch.rand(B, generator=g, dtype=torch.float32).to(device)
            y_next = dist.icdf(u).to(ctx_y.dtype)

        ctx_x[:, P + i] = x_rem[:, 0, :]
        ctx_y[:, P + i] = y_next
        appended[:, i] = y_next

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [ar] step {i + 1}/{N}", flush=True)

    # Back to raw nats: log p_raw = log p_scaled - log(std).
    return {
        "log_pdf": log_pdf_s - std.log(),
        "order": visit,
        "appended": appended * std + mean,
    }


def ar_parts_from_log_pdf(
    ar_log_pdf: torch.Tensor, marginal_log_pdf: torch.Tensor,
) -> dict[str, float]:
    """{total, marginal, copula} per point for one episode: marginal from the one-shot log_pdf_test, total from the chain, copula = total - marginal."""
    total = -float(ar_log_pdf.mean())
    marginal = -float(marginal_log_pdf.mean())
    return {"total": total, "marginal": marginal, "copula": total - marginal}
