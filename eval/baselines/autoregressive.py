"""autoregressive.py — a joint predictive density built from a *marginal*
model alone, by the chain rule.

The marginal branch of this repo's Sklar decomposition (a frozen/fine-tuned
TabICL, see src/pit.py) predicts one test point at a time, conditioned only on
the episode's context. Scored that way it is an INDEPENDENCE model over the
test set: its joint log-density is the sum of per-point terms and its copula
term is exactly 0. The copula head is what is supposed to supply the missing
dependence.

There is a second, copula-free way to get dependence out of the very same
marginal: reveal the test points one at a time and let each prediction
condition on the ones already revealed.

    log p(y_1..y_N | ctx) = sum_i log p(y_{s(i)} | ctx, y_{s(1)}..y_{s(i-1)})

for any fixed ordering s. That is an exact factorization of a joint density —
no approximation — so the number it produces is directly comparable, in the
same nats-per-point units, to every other row of eval_checkpoint.py's total
Y-space NLL table. It is the natural "how much dependence can the marginal
capture on its own?" reference for the copula head, and on real ERA5 (where
the copula head currently scores WORSE than independence) it is the obvious
thing to measure.

Two deliberate design choices, both of which matter for the number to mean
what it says:

CONSTANT TABLE, MOVING SPLIT. At every step the model sees the same P + N
rows it saw in the one-shot pass; only the boundary between "context" (y
known) and "query" (y unknown) moves. TabICL's column embedding computes its
feature statistics over the whole table, test rows included, so a step that
dropped the not-yet-revealed queries would change the model's input
distribution as the chain progressed and would NOT reproduce the one-shot
marginal at step 0. With the full table kept, step 0 is exactly the one-shot
PIT's log_pdf_test at that point (tests/test_autoregressive.py pins this),
which makes the AR total decompose cleanly against it:

    marginal = one-shot (independence) NLL   <- step 0's conditioning, all i
    total    = the chain-rule NLL
    copula   = total - marginal              <- what the sequencing bought

the same three columns, with the same meanings, that every other row of that
table reports, and copula == 0 recovers independence exactly.

TEACHER FORCING IS THE DEFAULT. ``conditioning="teacher_forcing"`` appends the
TRUE y_i before moving on, which is what makes the sum above an exact joint
log-density and a proper scoring rule. ``conditioning="sample"`` appends a DRAW
from the step's predictive instead: that is ancestral sampling from the model's
implied joint — useful for generating fields and for diagnostics — but the
accompanying log-density sum is then evaluated along a sampled conditioning
path and is NOT a joint density of y_test. Do not put it in a table next to
the other NLLs; the caller is warned in eval_checkpoint.py's --ar_conditioning
help and in the printed header.

A caveat that no ordering fixes: an in-context learner is not a coherent joint
distribution, so the chain-rule total DOES depend on the ordering s even
though a true joint's would not. The default ordering is a per-episode random
permutation (seeded, reproducible) rather than the grid's row-major order,
which would hand every step a neighbour it had just revealed and read as a
best case rather than a typical one.
"""

from __future__ import annotations

import os
import sys
import zlib
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pit import tabicl_forward

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
    """(B, N) long tensor of per-episode test-point visit orders.

    Seeded from the episode's GLOBAL index, not from its position in the batch,
    so an episode's ordering is the same whether it was scored in one long run
    or in a shard — the determinism contract every other per-episode quantity
    here already honours (see era5_episodes._episode_rng). crc32, never Python's
    hash(): hash() is PYTHONHASHSEED-salted and would make this silently
    irreproducible across processes.
    """
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
    """Chain-rule log-densities at every test point of B episodes.

    Every episode in the batch must share P and N (the fixed-shape ERA5
    geometry does by construction) — the B axis is folded into TabICL's own
    batch axis, exactly as pit.run_pit_batched folds it, so the whole group
    advances one chain step per forward pass instead of B of them.

    Args:
        tabicl    : the marginal model (a TabICL module, as pit.load_tabicl
                    returns — including a Phase-A fine-tuned one).
        x_train   : (B, P, d_x) context inputs
        y_train   : (B, P)      context targets, RAW units
        x_test    : (B, N, d_x) test inputs
        y_test    : (B, N)      test targets, RAW units
        order     : "random" (default, seeded per episode) or "natural".
        conditioning : "teacher_forcing" (default) appends the true y and the
                    result is an exact joint log-density; "sample" appends a
                    draw instead (ancestral sampling — see the module
                    docstring; the returned log-densities are then NOT a joint
                    density of y_test).
        max_context : cap on the number of context rows. The episode's own P
                    context points are always kept; beyond that only the most
                    recently revealed ``max_context - P`` are, oldest dropped
                    first. None (default) keeps everything.
        seed      : run seed, mixed with each episode's global index.
        episode_indices : per-episode GLOBAL indices, for that mixing. Defaults
                    to 0..B-1.
        progress_every : print a progress line every this many chain steps
                    (0 = silent). A 546-step chain is minutes of wall time.

    Returns dict of:
        log_pdf  : (B, N) chain-rule log-density at each test point, in RAW
                   target units and laid out in the episode's OWN test-point
                   order (not the visit order) — so it lines up index-for-index
                   with y_test and with the one-shot PIT's log_pdf_test.
        order    : (B, N) the visit order actually used.
        appended : (B, N) the value appended to the context at each visit step,
                   in VISIT order and raw units — the revealed truth under
                   teacher forcing, the ancestral sample otherwise.
    """
    if conditioning not in AR_CONDITIONINGS:
        raise ValueError(
            f"conditioning must be one of {AR_CONDITIONINGS}, got {conditioning!r}"
        )
    B, P, d_x = x_train.shape
    N = x_test.shape[1]
    device, dtype = x_train.device, x_train.dtype

    # y is z-scored by the context's OWN mean/std and held FIXED for the whole
    # chain -- the same statistics (and the same clamp) pit.normalize_targets
    # applies before any TabICL call, without which absolute-scale targets
    # saturate the frozen quantile head. Fixed rather than re-estimated as the
    # context grows: re-estimating would still be a legal chain rule, but step
    # 0 would no longer reproduce the one-shot marginal, which is the identity
    # the marginal/copula split rests on.
    mean = y_train.mean(dim=-1, keepdim=True)                  # (B, 1)
    std = y_train.std(dim=-1, keepdim=True).clamp(min=1e-8)    # (B, 1)
    y_train_s = (y_train - mean) / std
    y_test_s = (y_test - mean) / std

    visit = _orderings(B, N, order, seed, episode_indices).to(device)   # (B, N)

    # Context buffer: the P real context rows, then room for every revealed
    # test point. Written in place so no step reallocates the whole thing.
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
        # TabICL's InferenceManager may offload its output to CPU under low
        # free VRAM regardless of input device -- re-sync, same as run_pit.
        logits = logits.to(device)
        # Only the point being revealed this step is scored; the rest of the
        # query block is present for the table's sake (see module docstring).
        dist = tabicl.quantile_dist(logits[:, 0, :])                     # batch_shape (B,)

        tgt = visit[:, i]                                                # (B,)
        y_true = y_test_s[bidx, tgt]                                     # (B,)
        log_pdf_s[bidx, tgt] = dist.log_prob(y_true).to(log_pdf_s.dtype)

        if conditioning == "teacher_forcing":
            y_next = y_true
        else:
            # CPU generator + .to(device): a torch.Generator is device-typed,
            # and seeding a CUDA one per step would be both slower and a
            # different stream than a CPU rerun of the same config would see.
            g = torch.Generator()
            g.manual_seed(zlib.crc32(f"ar-sample:{seed}:{i}".encode()) & 0x7FFFFFFF)
            u = torch.rand(B, generator=g, dtype=torch.float32).to(device)
            y_next = dist.icdf(u).to(ctx_y.dtype)

        ctx_x[:, P + i] = x_rem[:, 0, :]
        ctx_y[:, P + i] = y_next
        appended[:, i] = y_next

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [ar] step {i + 1}/{N}", flush=True)

    # Back to raw target units: log p_raw(y) = log p_scaled(y_scaled) - log(std)
    # (pit.normalize_targets' own convention), so these sit in the same nats as
    # the PIT's log_pdf_test and the classical baselines' mvn_nll.
    return {
        "log_pdf": log_pdf_s - std.log(),
        "order": visit,
        "appended": appended * std + mean,
    }


def ar_parts_from_log_pdf(
    ar_log_pdf: torch.Tensor, marginal_log_pdf: torch.Tensor,
) -> dict[str, float]:
    """The {"total", "marginal", "copula"} triple eval_checkpoint.py's total
    Y-space table wants, for ONE episode, from that episode's two (N,) raw-nats
    log-density vectors.

    `marginal` is the one-shot (independence) marginal the AR chain starts
    from — literally the PIT's log_pdf_test — so it matches the icl row's
    marginal column exactly, and `copula` isolates what the sequencing bought:
    negative means conditioning on revealed neighbours helped, 0 means it was
    worth nothing, positive means it actively hurt. Same sign convention, and
    the same per-point (nats/point) normalization, as loss.y_space_nll.
    """
    total = -float(ar_log_pdf.mean())
    marginal = -float(marginal_log_pdf.mean())
    return {"total": total, "marginal": marginal, "copula": total - marginal}
