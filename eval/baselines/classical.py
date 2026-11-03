"""Non-ICL baselines scored on a copula episode, all under the same conventions.

Methods:
    independence: R = I (copula NLL 0).
    gp_prior_rbf: RBF prior correlation with a median bandwidth.
    gp_mle_{rbf, matern32, periodic, rq, dot_product, polynomial} and ARD
        variants (gp_mle_ard_*): GP posterior with MAP-fitted hyperparameters,
        fitted on raw y; polynomial searches every degree in
        [poly_power_min, poly_power_max].
    dkl_{rbf, matern32, rq, dot_product}: deep kernel learning, an MLP
        (d_x -> 32 -> 16) feeding a GP, trained on the marginal likelihood.
    per_ep_transformer: small set transformer trained on the episode's
        z-scored y_train.

eval_baselines_episode is the entry point. baseline_fingerprint,
episode_cache_key, save_baseline_entry and load_baseline_cache cache results
per episode (one file each), keyed so that any change to the generating
config or fitting settings invalidates them.
"""

from __future__ import annotations

import copy
import hashlib
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Adam

if TYPE_CHECKING:
    from omegaconf import DictConfig


from copula_inter.artifacts import atomic_torch_save
from copula_inter.baselines.gp_mle import ARD_ELIGIBLE, fit_and_eval_gpytorch
from copula_inter.copula.loss import gp_oracle_y_nll, oracle_copula_nll
from copula_inter.copula.model import low_rank_correlation
from copula_inter.data.manifest import dataset_identity
from eval.configs.constants import GP_VAL_SELECT_MODES
from eval.results import NAN_PARTS as _NAN_PARTS

__all__ = [
    "corr_nll_single",
    "gp_prior_corr_rbf",
    "eval_baselines_episode",
    "baseline_fingerprint",
    "episode_cache_key",
    "load_baseline_cache",
    "save_baseline_cache",
    "save_baseline_entry",
    "GP_VAL_SELECT_MODES",
    "EXPECTED_BASELINE_KEYS",
    "assert_shared_z_test",
    "fit_zero_mean_gp_on_marginal",
    "zero_mean_gp_prior_cfg",
]


def assert_shared_z_test(z_test: Tensor, ep: dict) -> None:
    """Raise if the z_test being scored is not ep["z_test"] (the shared-marginal table needs one z_test for every method)."""
    ep_z_test = ep["z_test"].to(z_test.device, z_test.dtype)
    assert torch.equal(z_test, ep_z_test), (
        "z_test used for a shared-marginal copula-NLL score has diverged "
        "from ep['z_test'] — this breaks _print_table's cross-method "
        "ranking (see assert_shared_z_test's docstring)."
    )


# Keys eval_baselines_episode returns; cached entries missing any are refitted.
EXPECTED_BASELINE_KEYS = frozenset(
    {"independence", "gp_prior_rbf", "per_ep_transformer"}
    | {
        "gp_mle_rbf",
        "gp_mle_ard_rbf",
        "gp_mle_matern32",
        "gp_mle_ard_matern32",
        "gp_mle_periodic",
        "gp_mle_ard_periodic",
        "gp_mle_rq",
        "gp_mle_ard_rq",
        "gp_mle_dot_product",
        "gp_mle_polynomial",
    }
    | {"dkl_rbf", "dkl_matern32", "dkl_rq", "dkl_dot_product"}
)

# Bump when a fitting algorithm changes in a way the fingerprint cannot detect.
_BASELINE_ALGO_VERSION = 4


def _resolve_val_select(mode: str, ard: bool) -> bool:
    """Whether this kernel selects its fit on a held-out split.

    "ard" (default): only ARD kernels; "always"; "never" (lowest training loss).
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "ard":
        return ard
    raise ValueError(f"gp_val_select={mode!r} is not one of {GP_VAL_SELECT_MODES}")


def corr_nll_single(R: Tensor, z: Tensor) -> float:
    """Copula NLL of an (N,) z under an (N, N) correlation matrix, with the shared z_test."""
    N = z.shape[0]
    mask = torch.ones(1, N, dtype=torch.bool, device=z.device)
    return oracle_copula_nll(R.unsqueeze(0), z.unsqueeze(0), mask).item()


# GP baselines (gpytorch).


# Zero-mean GP fitted on a real marginal's z_train (not cached with the other
# baselines, since it depends on the marginal).


def zero_mean_gp_prior_cfg(prior_cfg: dict | None = None) -> dict:
    """Hyperprior overrides for fitting on a unit-variance z_train: outputscale prior Gamma(2, 2)."""
    cfg = dict(prior_cfg or {})
    cfg["alpha2_gamma_concentration"] = 2.0
    cfg["alpha2_gamma_rate"] = 2.0
    return cfg


def fit_zero_mean_gp_on_marginal(
    X_train: Tensor,
    z_train: Tensor,
    X_test: Tensor,
    kernel_name: str,
    n_steps: int,
    lr: float,
    n_restarts: int,
    oracle_mode: str = "prior",
    prior_cfg: dict | None = None,
    jitter: float = 1e-6,
) -> dict[str, Tensor]:
    """Zero-mean GP-MLE on (X_train, z_train), z_train being a real marginal's PIT; returns {"R", "mean", "Sigma"} in z space."""
    return fit_and_eval_gpytorch(
        X_train,
        z_train,
        X_test,
        kernel_name,
        n_steps=n_steps,
        lr=lr,
        ard=False,
        jitter=jitter,
        oracle_mode=oracle_mode,
        prior_cfg=zero_mean_gp_prior_cfg(prior_cfg),
        n_restarts=n_restarts,
        val_select=False,
    )


def gp_prior_corr_rbf(X_test: Tensor) -> Tensor:
    """RBF prior correlation at the test points with a median bandwidth."""
    from copula_inter.data.gp_kernels import _sq_dist

    sq = _sq_dist(X_test, X_test)
    h2 = torch.pdist(X_test).pow(2).median().clamp(min=1e-6)
    R = torch.exp(-sq / (2.0 * h2))
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-1)
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-2)
    return R


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# Feature extractor for DKL: Linear(d_x, 32) -> SiLU -> Linear(32, 16).
DKLFeatureExtractor = _MLP


class _SelfAttn(nn.Module):
    def __init__(self, m: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(m)
        self.norm2 = nn.LayerNorm(m)
        self.attn = nn.MultiheadAttention(m, n_heads, dropout=dropout, batch_first=True)
        d_ff = max(round(8 / 3 * m / 32) * 32, 32)
        self.ff = nn.Sequential(nn.Linear(m, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, m))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class _CrossAttn(nn.Module):
    def __init__(self, m: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(m)
        self.norm_kv = nn.LayerNorm(m)
        self.attn = nn.MultiheadAttention(m, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        return q + self.drop(h)


class PerEpisodeTransformer(nn.Module):
    """Small set transformer trained from scratch on one episode.

    forward(X_ctx (n_sup, d_x), z_ctx (n_sup,), X_qry (n_qry, d_x)) returns
    W (n_qry, r) and s (n_qry,), fed to model.low_rank_correlation.
    """

    def __init__(
        self,
        d_x: int,
        m: int = 32,
        r: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.r = r

        self.x_enc = _MLP(d_x, m, m, dropout)
        self.row_enc = _MLP(m + 1, m, m, dropout)
        self.self_attn = nn.ModuleList([_SelfAttn(m, n_heads, dropout) for _ in range(n_layers)])
        self.W_q = nn.Linear(m, m)
        self.cross_attn = _CrossAttn(m, n_heads, dropout)
        self.head = nn.Linear(m, r + 1)

        # Small random head init (W = 0 is a zero-gradient saddle at Sigma = I).
        nn.init.normal_(self.head.weight, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, X_ctx: Tensor, z_ctx: Tensor, X_qry: Tensor) -> tuple[Tensor, Tensor]:
        ex = self.x_enc(X_ctx)  # (n_sup, m)
        row = self.row_enc(torch.cat([ex, z_ctx.unsqueeze(-1)], dim=-1))  # (n_sup, m)
        row = row.unsqueeze(0)  # (1, n_sup, m)
        for block in self.self_attn:
            row = block(row)

        eq = self.x_enc(X_qry).unsqueeze(0)  # (1, n_qry, m)
        q_emb = self.W_q(eq)
        h = self.cross_attn(q_emb, row).squeeze(0)  # (n_qry, m)

        out = self.head(h)  # (n_qry, r+1)
        W = out[:, : self.r]  # (n_qry, r)
        s = out[:, self.r]  # (n_qry,)
        return W, s


def _standardize_y(y: Tensor) -> Tensor:
    """Z-score y with its own sample mean and std."""
    return (y - y.mean()) / y.std(unbiased=True).clamp(min=1e-6)


def train_per_episode(
    X_train: Tensor,
    z_train: Tensor,
    r: int,
    n_steps: int = 500,
    lr: float = 1e-3,
    patience: int = 100,
    val_every: int = 10,
    device: torch.device = torch.device("cpu"),
) -> PerEpisodeTransformer:
    """Train a PerEpisodeTransformer on one episode.

    A fixed 20% split is used for early stopping; the rest is split 80/20 into
    support/query at every step. The rank is capped relative to P.
    """
    d_x = X_train.shape[1]
    P = X_train.shape[0]

    # Cap the rank relative to P (large ranks overfit a single episode).
    r = max(2, min(r, P // 4))

    n_val = max(2, int(round(0.2 * P)))
    perm = torch.randperm(P, device=device)
    val_idx, pool_idx = perm[:n_val], perm[n_val:]

    X_val, z_val = X_train[val_idx], z_train[val_idx]
    X_pool, z_pool = X_train[pool_idx], z_train[pool_idx]
    n_pool = X_pool.shape[0]

    model = PerEpisodeTransformer(d_x, r=r).to(device)
    opt = Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    model.train()
    for step in range(n_steps):
        n_sup = max(1, int(round(0.8 * n_pool)))
        perm_p = torch.randperm(n_pool, device=device)
        X_s, z_s = X_pool[perm_p[:n_sup]], z_pool[perm_p[:n_sup]]
        X_q, z_q = X_pool[perm_p[n_sup:]], z_pool[perm_p[n_sup:]]

        if X_q.shape[0] < 2:
            continue

        W, s = model(X_s, z_s, X_q)
        Sigma = low_rank_correlation(W.unsqueeze(0), s.unsqueeze(0)).squeeze(0)
        mask = torch.ones(1, X_q.shape[0], dtype=torch.bool, device=device)
        loss = oracle_copula_nll(Sigma.unsqueeze(0), z_q.unsqueeze(0), mask)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                W_v, s_v = model(X_pool, z_pool, X_val)
                Sv = low_rank_correlation(W_v.unsqueeze(0), s_v.unsqueeze(0)).squeeze(0)
                val_nll = corr_nll_single(Sv, z_val)
            model.train()

            if val_nll < best_val - 1e-4:
                best_val = val_nll
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += val_every

            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def eval_baselines_episode(
    ep: dict,
    icl_rank: int,
    n_steps_mle: int,
    lr_mle: float,
    n_steps_dkl: int,
    lr_dkl: float,
    n_steps_per_ep: int,
    patience_per_ep: int,
    device: torch.device,
    oracle_mode: str = "prior",
    prior_cfg: dict | None = None,
    n_restarts_mle: int = 1,
    n_restarts_dkl: int = 1,
    fit_seed: int | None = None,
    gp_val_select: str = "ard",
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, dict[str, float]]]:
    """Evaluate every baseline (all but the ICL model and the oracle) on one episode.

    fit_seed, when given, reseeds torch first so the fit depends only on the
    episode and settings (derive it from the episode's global index).

    Returns:
        nlls: {method: copula NLL} against the shared z_test.
        R_dict: {method: (N, N) correlation}.
        y_space_nlls: {method: {total, marginal, copula}} in raw y units under
            each fitted method's own marginal (not the unfitted references).
    """
    if fit_seed is not None:
        torch.manual_seed(fit_seed)

    X_train = ep["x_norm_train"].to(device)  # (P, d_x)
    y_train = ep["y_train"].to(device)  # (P,)  raw target, used to fit the GP-MLE/DKL baselines
    z_train_self = _standardize_y(y_train)  # (P,) z-scored y_train, used to train per_ep_transformer
    X_test = ep["x_norm_test"].to(device)  # (N, d_x)
    z_test = ep["z_test"].to(device)  # (N,)
    assert_shared_z_test(z_test, ep)
    y_test = ep["y_test"].to(device)  # (N,)  raw target, for total Y-space NLL

    N = X_test.shape[0]
    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}
    y_space_nlls: dict[str, dict[str, float]] = {}
    test_mask = torch.ones(1, N, dtype=torch.bool, device=device)

    def _nll_parts(mean: Tensor, Sigma: Tensor) -> dict[str, float]:
        parts = gp_oracle_y_nll(
            Sigma.unsqueeze(0),
            mean.unsqueeze(0),
            y_test.unsqueeze(0),
            test_mask,
        )
        return {k: v.item() for k, v in parts.items()}

    # --- independence ---
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)
    nlls["independence"] = corr_nll_single(R_I, z_test)
    R_dict["independence"] = R_I

    # --- GP prior RBF ---
    R_prior = gp_prior_corr_rbf(X_test)
    nlls["gp_prior_rbf"] = corr_nll_single(R_prior, z_test)
    R_dict["gp_prior_rbf"] = R_prior

    # --- GP MLE baselines (plain + ARD for lengthscale kernels) ---
    _GP_KERNELS = ["rbf", "matern32", "periodic", "rational_quadratic", "dot_product", "polynomial"]
    _LABEL_MAP = {
        ("rbf", False): "gp_mle_rbf",
        ("rbf", True): "gp_mle_ard_rbf",
        ("matern32", False): "gp_mle_matern32",
        ("matern32", True): "gp_mle_ard_matern32",
        ("periodic", False): "gp_mle_periodic",
        ("periodic", True): "gp_mle_ard_periodic",
        ("rational_quadratic", False): "gp_mle_rq",
        ("rational_quadratic", True): "gp_mle_ard_rq",
        ("dot_product", False): "gp_mle_dot_product",
        ("polynomial", False): "gp_mle_polynomial",
    }
    for kname in _GP_KERNELS:
        for ard in [False, True] if ARD_ELIGIBLE[kname] else [False]:
            label = _LABEL_MAP[(kname, ard)]
            try:
                # Held-out selection per _resolve_val_select.
                fit = fit_and_eval_gpytorch(
                    X_train,
                    y_train,
                    X_test,
                    kname,
                    n_steps=n_steps_mle,
                    lr=lr_mle,
                    ard=ard,
                    oracle_mode=oracle_mode,
                    prior_cfg=prior_cfg,
                    n_restarts=n_restarts_mle,
                    val_select=_resolve_val_select(gp_val_select, ard),
                )
                nlls[label] = corr_nll_single(fit["R"], z_test)
                R_dict[label] = fit["R"]
                y_space_nlls[label] = _nll_parts(fit["mean"], fit["Sigma"])
            except Exception as exc:
                print(f"  [{label}] failed: {exc}")
                nlls[label] = float("nan")
                R_dict[label] = R_I.clone()
                y_space_nlls[label] = _NAN_PARTS.copy()

    # DKL kernels (periodic excluded).
    _DKL_KERNELS = ["rbf", "matern32", "rational_quadratic", "dot_product"]
    _DKL_LABEL_MAP = {
        "rbf": "dkl_rbf",
        "matern32": "dkl_matern32",
        "rational_quadratic": "dkl_rq",
        "dot_product": "dkl_dot_product",
    }

    def _make_dkl_mlp(d_x: int = X_train.shape[1], dev: torch.device = device) -> nn.Module:
        return DKLFeatureExtractor(d_x, hidden=32, out_dim=16, dropout=0.0).to(dev)

    for kname in _DKL_KERNELS:
        label = _DKL_LABEL_MAP[kname]
        try:
            fit = fit_and_eval_gpytorch(
                X_train,
                y_train,
                X_test,
                kname,
                n_steps=n_steps_dkl,
                lr=lr_dkl,
                ard=False,
                feature_extractor_factory=_make_dkl_mlp,
                oracle_mode=oracle_mode,
                prior_cfg=prior_cfg,
                n_restarts=n_restarts_dkl,
            )
            nlls[label] = corr_nll_single(fit["R"], z_test)
            R_dict[label] = fit["R"]
            y_space_nlls[label] = _nll_parts(fit["mean"], fit["Sigma"])
        except Exception as exc:
            print(f"  [{label}] failed: {exc}")
            nlls[label] = float("nan")
            R_dict[label] = R_I.clone()
            # Failed fits get _NAN_PARTS (a dict) so the cache entry stays valid.
            y_space_nlls[label] = _NAN_PARTS.copy()

    # per_ep_transformer: fitted on z-scored y_train, scored against the shared z_test;
    # its Y-space NLL uses the empirical Gaussian marginal.
    try:
        per_ep_model = train_per_episode(
            X_train,
            z_train_self,
            r=icl_rank,
            n_steps=n_steps_per_ep,
            patience=patience_per_ep,
            device=device,
        )
        with torch.no_grad():
            W_te, s_te = per_ep_model(X_train, z_train_self, X_test)
            Sigma_te = low_rank_correlation(W_te.unsqueeze(0), s_te.unsqueeze(0)).squeeze(0)
        nlls["per_ep_transformer"] = corr_nll_single(Sigma_te, z_test)
        R_dict["per_ep_transformer"] = Sigma_te
        mean_tr = y_train.mean()
        std_tr = y_train.std(unbiased=True).clamp(min=1e-6)
        y_space_nlls["per_ep_transformer"] = _nll_parts(
            mean_tr.expand(N),
            (std_tr**2) * Sigma_te,
        )
    except Exception as exc:
        print(f"  [per_ep_transformer] failed: {exc}")
        nlls["per_ep_transformer"] = float("nan")
        R_dict["per_ep_transformer"] = R_I.clone()
        y_space_nlls["per_ep_transformer"] = _NAN_PARTS.copy()

    return nlls, R_dict, y_space_nlls


# Baseline cache.


def baseline_fingerprint(
    gen_cfg: DictConfig,
    live_generate: bool,
    dataset_dir: str | None,
    seed: int,
    icl_rank: int,
    oracle_mode: str,
    n_steps_mle: int,
    lr_mle: float,
    n_restarts_mle: int,
    n_steps_dkl: int,
    lr_dkl: float,
    n_steps_per_ep: int,
    patience_per_ep: int,
    gp_val_select: str = "ard",
    n_restarts_dkl: int = 1,
) -> dict:
    """Digest of everything, besides the episode itself, that determines the baseline fits.

    Includes gen_cfg.data (the generating config, not the checkpoint's own),
    icl_rank, oracle_mode and the fitting settings.
    """
    data_cfg = OmegaConf.select(gen_cfg, "data", default=None)
    fp = {
        "algo_version": _BASELINE_ALGO_VERSION,
        "data_cfg": OmegaConf.to_container(data_cfg) if data_cfg is not None else {},
        "icl_rank": icl_rank,
        "live_generate": live_generate,
        "dataset_dir": os.path.abspath(dataset_dir) if (dataset_dir and not live_generate) else None,
        "seed": seed,
        "oracle_mode": oracle_mode,
        "n_steps_mle": n_steps_mle,
        "lr_mle": lr_mle,
        "n_restarts_mle": n_restarts_mle,
        "n_steps_dkl": n_steps_dkl,
        "lr_dkl": lr_dkl,
        "n_restarts_dkl": n_restarts_dkl,
        "n_steps_per_ep": n_steps_per_ep,
        "patience_per_ep": patience_per_ep,
        "gp_val_select": gp_val_select,
    }
    # Only on-disk datasets get a content identity (same-path replacement). Live / ERA5 runs keep
    # 42c0402's exact key set, so baseline caches written before the refactor stay valid.
    if dataset_dir and not live_generate and os.path.isdir(dataset_dir):
        fp["dataset_identity"] = dataset_identity(dataset_dir)
    return fp


def episode_cache_key(
    live_generate: bool,
    dataset_dir: str | None,
    seed: int,
    ep_i: int,
    *,
    source: str | None = None,
) -> str:
    """Cache key of one episode.

        "{source}:seed{seed}:idx{ep_i}"   when source is given (e.g. "era5")
        "live:seed{seed}:idx{ep_i}"       for live generation
        "dataset:{dataset_dir}:idx{ep_i}" for an on-disk dataset

    ep_i is the episode's global index.
    """
    if source is not None:
        return f"{source}:seed{seed}:idx{ep_i}"
    if live_generate:
        return f"live:seed{seed}:idx{ep_i}"
    if dataset_dir is None:
        raise ValueError("an on-disk episode needs dataset_dir")
    return f"dataset:{os.path.abspath(dataset_dir)}:idx{ep_i}"


def _shard_dir(path: str) -> str:
    """Directory holding this cache's per-episode shards (<path>.d)."""
    return f"{path}.d"


def _shard_name(cache_key: str) -> str:
    """Filename of one episode's shard: a hash of its cache key (the key is also stored inside)."""
    return hashlib.sha1(cache_key.encode()).hexdigest() + ".pt"


def save_baseline_entry(path: str, fingerprint: dict, cache_key: str, entry: dict) -> None:
    """Write one episode's fitted baselines to its own shard under <path>.d/."""
    d = _shard_dir(path)
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, _shard_name(cache_key))
    atomic_torch_save({"fingerprint": fingerprint, "cache_key": cache_key, "entry": entry}, dest)


def load_baseline_cache(path: str, fingerprint: dict) -> dict[str, dict]:
    """Load {episode_key: entry} from the per-episode shards and the legacy single file, keeping fingerprint matches.

    Shards win over the legacy file; unreadable shards are skipped.
    """
    entries: dict[str, dict] = {}
    n_stale = 0

    if os.path.exists(path):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if blob.get("fingerprint") == fingerprint:
                entries.update(blob.get("entries", {}))
                print(f"  [baseline_cache] loaded {len(entries)} episode(s) from {path}")
            else:
                n_stale += len(blob.get("entries", {}))
        except Exception as exc:
            print(f"  [baseline_cache] failed to load {path}: {exc} — ignoring it")

    d = _shard_dir(path)
    if os.path.isdir(d):
        n_shard = 0
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".pt"):
                continue
            try:
                blob = torch.load(os.path.join(d, fn), map_location="cpu", weights_only=False)
            except Exception as exc:
                print(f"  [baseline_cache] shard {fn} unreadable ({exc}) — will refit it")
                continue
            if blob.get("fingerprint") != fingerprint:
                n_stale += 1
                continue
            entries[blob["cache_key"]] = blob["entry"]
            n_shard += 1
        if n_shard:
            print(f"  [baseline_cache] loaded {n_shard} episode(s) from {d}")

    if n_stale:
        print(
            f"  [baseline_cache] ignored {n_stale} entr(ies) built under different "
            "generation/fitting settings — those episodes will be refitted"
        )
    return entries


def save_baseline_cache(path: str, fingerprint: dict, entries: dict[str, dict]) -> None:
    """Write the whole cache to one file atomically (legacy layout, still readable)."""
    atomic_torch_save({"fingerprint": fingerprint, "entries": entries}, path)
    print(f"  [baseline_cache] saved {len(entries)} episode(s) to {path}", flush=True)
