"""copula_inference.py — reusable inference API for the copula inter-instance model.

Thin wrappers around existing repo internals (``src/pit.py``, ``src/model.py``,
``src/data_gen.py``) plus new code for sampling correlated trajectories and for
loading/querying the real PFN4BO baseline. No hardcoded paths or experiment
logic lives here — every path is an argument.

Callers must standardize their features with ``normalize_features`` (zero
mean/unit std, jointly over train+test — matching ``data_gen.py``'s own
convention) before calling anything else here; none of the functions below
do this internally, since they generally don't see the full train+test set
at once.

Two marginal backends are supported, with genuinely different conventions:

  TabICL   — ``get_marginal_quantiles`` — emits its own fixed 999-point quantile
             grid (``p_j = j/1001``); custom probability levels are obtained via
             ``QuantileDistribution.icdf`` (spline + tail extrapolation), not
             re-interpolation of the 999-grid.
  PFN4BO   — ``get_marginal_quantiles_pfn4bo`` — a bucketed (bar) distribution
             over a fixed y-support; quantiles at arbitrary probability levels
             come from inverting the bucket CDF. PFN4BO expects x in [0,1]^d
             (mapped here via the Gaussian CDF) and y power-transformed
             (Yeo-Johnson, fit on context y only);
             the returned quantile grid is inverse-transformed back to
             original y-units before returning.

Both marginal-query functions return ``(quantile_grid, probs)`` with
``quantile_grid.shape == (n_query, len(probs))`` and
``quantile_grid[i, j] = F_i^{-1}(probs[j])`` — the shared convention every
other function in this module (and both experiment scripts) build on.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
_PFNS4BO_ROOT = os.path.join(_REPO_ROOT, "pfns4bo_upstream")
for _p in (_SRC, _PFNS4BO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import CopulaTabICL, build_copula_transformer, low_rank_correlation  # noqa: E402
from pit import load_tabicl, run_pit  # noqa: E402

__all__ = [
    "normalize_features",
    "load_tabicl_marginal",
    "get_marginal_quantiles",
    "loo_pit",
    "load_copula_model",
    "get_test_correlation",
    "load_pfn4bo",
    "get_marginal_quantiles_pfn4bo",
    "sample_trajectories",
]


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score features to match the training-time convention.

    ``src/data_gen.py::generate_gp_batch`` standardizes ``x`` to zero-mean /
    unit-std **jointly over the full train+test sequence within one episode**
    (``x_norm = (x_raw - x_raw.mean(1)) / x_raw.std(1)``, computed before the
    train/test split) — neither TabICL's forward pass nor ``CopulaTabICL``
    do any further normalization internally (``src/pit.py::run_pit`` and
    ``src/model.py`` both take ``x`` as-is). Every other function in this
    module therefore expects its ``X_train``/``X_test`` arguments to already
    be on this scale; call this first if your raw features aren't already
    zero-mean/unit-std (e.g. a ``[0, 1]`` grid, or real-world units).

    Note: this does NOT replicate ``tabiclv2_warp_features`` (the random
    per-column marginal-shape warp applied before z-scoring at training
    time) — that's a training-time diversity augmentation over many
    possible feature *shapes*, not a canonical transform real/query features
    need to undergo. Only the mean/std standardization is a hard requirement
    the model relies on.

    Args:
        X_train : (P, d) training features, raw scale.
        X_test  : (N, d) test features, raw scale.

    Returns:
        (X_train_norm, X_test_norm), z-scored per column using the combined
        train+test mean/std (matching ``data_gen.py``'s joint-episode
        statistics — NOT train-only statistics, which would diverge from
        training whenever the train and test marginal distributions of x
        differ, e.g. a sparse train subset of a denser test grid).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    X_all = np.concatenate([X_train, X_test], axis=0)
    mean = X_all.mean(axis=0, keepdims=True)
    std = np.clip(X_all.std(axis=0, ddof=1, keepdims=True), 1e-8, None)
    return (X_train - mean) / std, (X_test - mean) / std


# ---------------------------------------------------------------------------
# TabICL marginal backend
# ---------------------------------------------------------------------------


def load_tabicl_marginal(ckpt_name: str, device: str) -> torch.nn.Module:
    """Load a frozen TabICL regressor for marginal quantile queries.

    Thin re-export of ``pit.load_tabicl`` — kept here so callers only need to
    import from this module.
    """
    return load_tabicl(ckpt_name, device)


@torch.no_grad()
def get_marginal_quantiles(
    tabicl: torch.nn.Module,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Query TabICL's predictive quantile grid at ``X_query``.

    Args:
        tabicl    : frozen TabICL regressor (e.g. from ``load_tabicl_marginal``).
        X_context : (n_ctx, d) context features.
        y_context : (n_ctx,) context targets.
        X_query   : (n_q, d) query features.
        probs     : optional (Q,) probability levels. If None, uses TabICL's
                    own 999-level grid (``p_j = j/1001``) directly (no extra
                    interpolation). If given, values are obtained via
                    ``QuantileDistribution.icdf`` (spline/tail extrapolation)
                    at exactly those levels — NOT by re-interpolating the
                    999-grid, which would silently degrade tail accuracy.

    Returns:
        quantile_grid : (n_q, len(probs)) — quantile_grid[i, j] = F_i^{-1}(probs[j])
        probs         : (len(probs),) probability levels actually used.
    """
    device = next(tabicl.parameters()).device
    dtype = next(tabicl.parameters()).dtype

    X_ctx_t = torch.as_tensor(np.asarray(X_context), dtype=dtype, device=device)
    y_ctx_t = torch.as_tensor(np.asarray(y_context), dtype=dtype, device=device)
    X_qry_t = torch.as_tensor(np.asarray(X_query), dtype=dtype, device=device)

    X_full = torch.cat([X_ctx_t, X_qry_t], dim=0).unsqueeze(0)  # (1, T, d_x)
    y_ctx_b = y_ctx_t.unsqueeze(0)  # (1, P)

    raw_quantiles = tabicl(X_full, y_ctx_b)  # (1, n_q, 999)
    dist = tabicl.quantile_dist(raw_quantiles)

    if probs is None:
        quantile_grid = dist.quantiles[0]
        probs_out = dist.alpha_levels
    else:
        probs_t = torch.as_tensor(np.asarray(probs), dtype=raw_quantiles.dtype, device=device)
        quantile_grid = dist.icdf(probs_t)[0]  # (n_q, len(probs))
        probs_out = probs_t

    return quantile_grid.cpu().numpy(), probs_out.cpu().numpy()


def loo_pit(
    tabicl: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
) -> np.ndarray:
    """Gaussianized PIT residuals for the training set, via TabICL's marginal CDF.

    Despite the name (kept for parity with the "leave-one-out" framing this
    is usually described with), the default is **K-fold** partitioning
    (``k_folds=10``), not true LOO — this matches the rest of the repo's
    dataset-generation convention (``src/pit.py::DEFAULT_K_FOLDS``): true LOO
    (``k_folds=len(X_train)``) is more accurate but ~K_loo/k_folds times
    slower. Pass ``k_folds=len(X_train)`` for true LOO.

    Internally calls ``pit.run_pit`` with a single dummy "test" row (discarded)
    since ``run_pit`` also computes a test-set PIT in the same call; the
    training-set K-fold PIT this function returns does not depend on that
    dummy row's value.

    Args:
        tabicl  : frozen TabICL regressor.
        X_train : (P, d) training features.
        y_train : (P,) training targets.
        k_folds : number of disjoint folds (see above).
        eps     : clamp before the probit transform.

    Returns:
        Z_train : (P,) Gaussianized residuals.
    """
    device = next(tabicl.parameters()).device
    dtype = next(tabicl.parameters()).dtype

    X_t = torch.as_tensor(np.asarray(X_train), dtype=dtype, device=device)
    y_t = torch.as_tensor(np.asarray(y_train), dtype=dtype, device=device).unsqueeze(-1)  # (P, 1)

    out = run_pit(tabicl, X_t, y_t, X_t[:1], y_t[:1], k_folds=k_folds, eps=eps)
    return out["z_train"].squeeze(-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Copula model
# ---------------------------------------------------------------------------


def load_copula_model(
    ckpt_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> tuple[CopulaTabICL, DictConfig]:
    """Load a trained CopulaTabICL checkpoint.

    Mirrors ``src/evaluate_baselines.py``'s loading pattern: reads the training
    config saved inside the checkpoint (falling back to ``config_path`` if
    given, or raising if neither is available), builds the model architecture
    from it, then loads the state dict.

    Args:
        ckpt_path   : path to a checkpoint saved by ``train.py``'s
                      ``save_checkpoint`` (has ``state_dict``/``model_state``
                      and, normally, ``cfg``).
        config_path : Hydra config to use if the checkpoint has no saved
                      ``cfg`` (older checkpoints). Ignored if the checkpoint
                      does have one.
        device      : torch device string.

    Returns:
        (model, cfg) — the loaded ``CopulaTabICL`` in eval mode, and the
        ``DictConfig`` it was built from (useful for reading e.g.
        ``cfg.data.oracle_mode``).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get("cfg")
    if cfg is None:
        if config_path is None:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has no saved 'cfg' and no config_path was given."
            )
        cfg = OmegaConf.load(config_path)
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    model = build_copula_transformer(cfg).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict"))
    raw = getattr(model, "_orig_mod", model)
    raw.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def get_test_correlation(
    copula_model: CopulaTabICL,
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Query the copula model's predicted test-test correlation matrix.

    Args:
        copula_model : loaded ``CopulaTabICL`` (e.g. from ``load_copula_model``).
        X_train : (P, d) training features.
        Z_train : (P,) Gaussianized training residuals (e.g. from ``loo_pit``).
        X_test  : (N, d) test features.

    Returns:
        R_test : (N, N) — symmetrized, with the diagonal forced to exactly 1.0
        (a neural-net output won't satisfy either exactly).
    """
    device = next(copula_model.parameters()).device
    dtype = next(copula_model.parameters()).dtype

    x_train_t = torch.as_tensor(np.asarray(X_train), dtype=dtype, device=device).unsqueeze(0)
    x_test_t = torch.as_tensor(np.asarray(X_test), dtype=dtype, device=device).unsqueeze(0)
    z_train_t = torch.as_tensor(np.asarray(Z_train), dtype=dtype, device=device).unsqueeze(0)

    batch = {"x_train": x_train_t, "x_test": x_test_t, "z_train": z_train_t}
    out = copula_model(batch)
    Sigma = low_rank_correlation(out["W"], out["s"])  # (1, N, N)

    R = Sigma[0].cpu().numpy()
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# PFN4BO marginal backend
# ---------------------------------------------------------------------------


def _patch_pfns4bo_torch_compat() -> None:
    """``pfns4bo_upstream/pfns4bo/layer.py`` does
    ``from torch.nn.modules.transformer import ..., Optional, ...`` — a name
    older torch versions re-exported from that module but current torch no
    longer does. Patched here (not in ``pfns4bo_upstream``, which stays
    pristine) before importing anything from ``pfns4bo``.
    """
    import typing

    import torch.nn.modules.transformer as _t

    if not hasattr(_t, "Optional"):
        _t.Optional = typing.Optional


def load_pfn4bo(model_name: str = "hebo_plus_model", device: str = "cpu") -> torch.nn.Module:
    """Load a pretrained PFN4BO checkpoint from ``pfns4bo_upstream``.

    ``model_name`` is an attribute of the vendored ``pfns4bo`` package
    (``hebo_plus_model``, ``hebo_plus_userprior_model``, or ``bnn_model``);
    accessing it auto-downloads/unzips the weights on first use. The
    checkpoint is a whole pickled ``TransformerModel`` (not a state dict),
    with a ``.criterion`` attribute (a ``BarDistribution``) already attached.
    """
    _patch_pfns4bo_torch_compat()
    import pfns4bo  # type: ignore[import]

    model_path = getattr(pfns4bo, model_name)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def _vectorized_bar_icdf(criterion, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Vectorized equivalent of ``BarDistribution.icdf`` over many probability
    levels at once (upstream's own ``icdf`` only accepts a scalar
    ``left_prob``; looping it 999 times is needlessly slow). Reimplements the
    exact same searchsorted/linear-interpolation math against
    ``criterion.borders`` without modifying ``pfns4bo_upstream``.

    Args:
        criterion : a ``BarDistribution`` (or subclass) — only ``.borders``
                    and ``.num_bars`` are read.
        logits    : (..., num_bars).
        probs     : (Q,) probability levels.

    Returns:
        (..., Q) quantile values.
    """
    p = torch.softmax(logits, dim=-1)  # (..., B)
    cumprobs = torch.cumsum(p, dim=-1)  # (..., B)
    prefix_shape = cumprobs.shape[:-1]

    probs_b = probs.view(*([1] * len(prefix_shape)), -1).expand(*prefix_shape, -1).contiguous()
    idx = torch.searchsorted(cumprobs.contiguous(), probs_b).clamp(0, criterion.num_bars - 1)

    zeros = torch.zeros(*prefix_shape, 1, device=logits.device, dtype=logits.dtype)
    cumprobs_padded = torch.cat([zeros, cumprobs], dim=-1)
    left_cum = cumprobs_padded.gather(-1, idx)
    rest_prob = probs_b - left_cum

    left_border = criterion.borders[idx]
    right_border = criterion.borders[idx + 1]
    bucket_p = p.gather(-1, idx).clamp(min=1e-12)

    return left_border + (right_border - left_border) * rest_prob / bucket_p


def _yeo_johnson_valid_domain(lam: float) -> tuple[float, float]:
    """Valid domain of the transformed value `t` for Yeo-Johnson's INVERSE
    transform at parameter ``lam`` (sklearn's ``PowerTransformer`` silently
    returns NaN outside it — `(negative base)**(non-integer exponent)`).

    Inverting `t = ((y+1)**lam - 1)/lam` for `y >= 0` requires `lam*t+1 >= 0`,
    which only bounds `t` from above when `lam < 0` (t <= -1/lam).
    Inverting `t = -((-y+1)**(2-lam) - 1)/(2-lam)` for `y < 0` requires
    `(lam-2)*t + 1 >= 0`, which only bounds `t` from below when `lam > 2`
    (t >= -1/(lam-2)).
    """
    t_min = -1.0 / (lam - 2.0) if lam > 2.0 else -np.inf
    t_max = -1.0 / lam if lam < 0.0 else np.inf
    return t_min, t_max


@torch.no_grad()
def get_marginal_quantiles_pfn4bo(
    pfn4bo_model: torch.nn.Module,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Query PFN4BO's predictive quantile grid at ``X_query``.

    Handles PFN4BO's own input/output conventions, none of which match
    TabICL's:
      - x must lie in [0,1]^d — mapped here via the Gaussian CDF
        (``torch.special.ndtr``). Feature padding up to the model's fixed
        input width is handled automatically by its own
        ``VariableNumFeaturesEncoder`` — no manual padding needed.
      - y is power-transformed (Yeo-Johnson, fit on ``y_context`` only) before
        being fed to the model; the returned quantile *values* are inverse-
        transformed back to original y-units before returning (valid since
        monotonic transforms commute with quantile functions).
      - the model itself outputs bucketed ("bar distribution") logits, not
        raw quantiles — inverted via ``_vectorized_bar_icdf``.

    Args:
        pfn4bo_model : loaded PFN4BO model (e.g. from ``load_pfn4bo``).
        X_context    : (n_ctx, d) context features.
        y_context    : (n_ctx,) context targets.
        X_query      : (n_q, d) query features.
        probs        : optional (Q,) probability levels; defaults to the
                       same 999-level convention as ``get_marginal_quantiles``
                       (``linspace(0, 1, 1001)[1:-1]``) for easy side-by-side use.

    Returns:
        quantile_grid : (n_q, len(probs)), in original y-units.
        probs         : (len(probs),) probability levels used.
    """
    from sklearn.preprocessing import PowerTransformer

    if probs is None:
        probs = np.linspace(0.0, 1.0, 999 + 2)[1:-1]
    probs = np.asarray(probs)

    device = next(pfn4bo_model.parameters()).device

    y_ctx = np.asarray(y_context, dtype=np.float64).reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    y_ctx_transformed = pt.fit_transform(y_ctx).reshape(-1)

    X_ctx = np.asarray(X_context, dtype=np.float64)
    X_qry = np.asarray(X_query, dtype=np.float64)
    x_ctx_t = torch.special.ndtr(torch.as_tensor(X_ctx, dtype=torch.float32))
    x_qry_t = torch.special.ndtr(torch.as_tensor(X_qry, dtype=torch.float32))

    n_ctx = x_ctx_t.shape[0]
    x_full = torch.cat([x_ctx_t, x_qry_t], dim=0).to(device).unsqueeze(1)  # (T, 1, d_x)
    y_full = torch.as_tensor(y_ctx_transformed, dtype=torch.float32, device=device).unsqueeze(1)  # (P, 1)

    # TransformerModel's decoder only runs on positions >= single_eval_pos
    # (see transformer.py::_forward's out_range_start) — the returned tensor
    # is already query-only, shape (n_q, 1, num_bars), NOT the full sequence.
    logits = pfn4bo_model((None, x_full, y_full), single_eval_pos=n_ctx)  # (n_q, 1, num_bars)
    logits_query = logits[:, 0, :]  # (n_q, num_bars)

    criterion = pfn4bo_model.criterion
    probs_t = torch.as_tensor(probs, dtype=logits_query.dtype, device=device)
    quantile_grid_transformed = _vectorized_bar_icdf(criterion, logits_query, probs_t)  # (n_q, Q)

    # Clip to Yeo-Johnson's invertible domain: BarDistribution's tail
    # extrapolation can push extreme-probability quantiles (e.g. p=0.001)
    # past the range any real y could have produced under the fitted
    # transform, which sklearn's inverse_transform would otherwise silently
    # turn into NaN (a fractional power of a negative base).
    t_min, t_max = _yeo_johnson_valid_domain(float(pt.lambdas_[0]))
    margin = 1e-6
    flat = quantile_grid_transformed.cpu().numpy().reshape(-1, 1).astype(np.float64)
    flat = np.clip(flat, t_min + margin, t_max - margin)

    n_q = quantile_grid_transformed.shape[0]
    quantile_grid = pt.inverse_transform(flat).reshape(n_q, -1)

    # Even inside the invertible domain, the inverse Yeo-Johnson transform's
    # derivative blows up near the domain boundary computed above, so
    # extreme-tail probability levels (e.g. p=0.001) that BarDistribution's
    # tail extrapolated far from the observed data can still invert to
    # absurdly large finite y-values (dominating any downstream mean/std
    # summary). Clip to a generous but finite multiple of the observed
    # context range — same spirit as sample_trajectories' probability
    # clipping: bound extrapolation rather than let it explode.
    y_ctx_min, y_ctx_max = float(y_ctx.min()), float(y_ctx.max())
    y_range = max(y_ctx_max - y_ctx_min, 1e-6)
    quantile_grid = np.clip(quantile_grid, y_ctx_min - 10 * y_range, y_ctx_max + 10 * y_range)

    return quantile_grid, probs


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_trajectories(
    quantile_grid: np.ndarray,
    probs: np.ndarray,
    R: np.ndarray,
    n_samples: int,
    eps_reg: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, int]:
    """Sample correlated trajectories via shared Gaussian noise + quantile inversion.

    For each sample k: draw ``eps_k ~ N(0, I)`` (dimension = n_test, one
    shared draw per trajectory), ``z = L @ eps_k`` with
    ``L = cholesky(R + eps_reg*I)``, ``p = Phi(z)`` (clipped to
    ``[probs[0], probs[-1]]`` to avoid extrapolating past the quantile grid),
    ``y_i = interp(p_i, probs, quantile_grid[i])``.

    Args:
        quantile_grid : (n_test, Q) — quantile_grid[i, j] = F_i^{-1}(probs[j]).
        probs         : (Q,) probability levels, shared across test points.
        R             : (n_test, n_test) correlation matrix (pass
                        ``np.eye(n_test)`` for the independent baseline).
        n_samples     : number of trajectories to draw.
        eps_reg       : jitter added to R's diagonal before Cholesky.
        rng           : optional ``np.random.Generator`` for reproducibility.

    Returns:
        samples   : (n_samples, n_test).
        n_clipped : total count of (sample, test-point) pairs whose implied
                    probability fell outside ``[probs[0], probs[-1]]`` and had
                    to be clipped — a diagnostic for how often the sampler
                    extrapolated past the quantile grid's support.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_test = R.shape[0]
    L = np.linalg.cholesky(R + eps_reg * np.eye(n_test))

    eps = rng.standard_normal((n_samples, n_test))
    z = eps @ L.T  # (n_samples, n_test)
    p = norm.cdf(z)

    lo, hi = probs[0], probs[-1]
    n_clipped = int(np.sum((p < lo) | (p > hi)))
    p_clipped = np.clip(p, lo, hi)

    samples = np.empty_like(p_clipped)
    for i in range(n_test):
        samples[:, i] = np.interp(p_clipped[:, i], probs, quantile_grid[i])

    return samples, n_clipped
