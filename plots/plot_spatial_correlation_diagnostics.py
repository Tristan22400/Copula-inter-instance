"""
plot_spatial_correlation_diagnostics.py — Spatial correlogram sanity check
for TabICLv2 (CopulaTabICL): does the model's learned inter-instance
dependence structure reproduce the physical spatial correlation decay of the
real ERA5 field?

Four curves, all binned by great-circle distance and plotted together:

  1. Ground truth: empirical correlation of 24h persistence residuals
     E_t = Z_true_t - Z_true_{t-24} across the spatial grid.
  2. Independent TabICLv2: a model with no inter-instance copula assumes
     conditional independence across the grid, so its implied correlation
     matrix IS the identity by construction — no forward pass needed.
  3. TabICLv2 learned PRIOR copula: an (as-near-as-architecturally-possible)
     unconditional forward pass of the trained CopulaTabICL checkpoint,
     matching how it's trained (cfg.data.oracle_mode="prior", see
     src/data_gen.py) to output R_star ignoring in-context conditioning.
  4. TabICLv2 + Copula POSTERIOR: a real joint forward pass over all D grid
     points at once, conditioned on a historical in-context sample of the
     SAME 24h persistence residual field as (1) — not the raw absolute
     temperatures — so the posterior is conditioned on the same physical
     quantity whose spatial decay it's being compared against. Context
     labels z_train are NOT a naive (y - mean) / std standardization: they
     are the K-fold leave-one-out Probability Integral Transform of each
     context point's true residual under TabICLv2's own learned marginal
     (see src/pit.py::run_pit) — i.e. u_i = F_hat(y_i | other context
     points), z_i = Phi^-1(u_i) — matching how z_train is actually defined
     during training (data_gen.py's GP-oracle LOO PIT) instead of assuming
     a Gaussian marginal by fiat.

Reuses:
  - plots/generate_plots.py: load_era5_data (+ synthetic-GP fallback) and
    haversine_distance_km, so ERA5 I/O and great-circle distance math are
    never reimplemented here.
  - src/model.py: build_copula_transformer + low_rank_correlation, the same
    (W, s) -> Sigma projection used by generate_plots.py's
    build_copula_correlation_fn — this script loads the checkpoint once and
    reuses it for both the prior (3) and posterior (4) extractions instead
    of loading it twice.
  - src/pit.py: load_tabicl + run_pit, the same frozen-TabICL-quantile-head
    K-fold LOO PIT used to build z_train for real (non-GP-oracle) data —
    reused as-is rather than reimplemented with a naive standardization.

Usage:
    python plots/plot_spatial_correlation_diagnostics.py --ckpt ./checkpoints/systematic-composition-8/step_0180000.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PLOTS_DIR)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _PLOTS_DIR not in sys.path:
    sys.path.insert(0, _PLOTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# src/pit.py imports its src/-local siblings (e.g. data_gen.py) with bare
# names ("from data_gen import ..."), so src/ itself must be on sys.path too
# -- not just the repo root -- for `from src.pit import run_pit` to work.
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from generate_plots import haversine_distance_km, load_era5_data  # noqa: E402

OUT_PATH = os.path.join(_PLOTS_DIR, "spatial_correlation_diagnostics.png")


# ---------------------------------------------------------------------------
# Ground truth: empirical spatial correlation from 24h persistence residuals
# ---------------------------------------------------------------------------
def compute_persistence_residuals(field_all: np.ndarray) -> np.ndarray:
    """24h persistence residuals E_t = Z_true_t - Z_true_{t-24}.

    `field_all` (n_snapshots, H, W) is one grid snapshot per time index, at a
    fixed cadence of 24h apart (era5_temperature.nc, real or synthetic, is
    one daily snapshot at a fixed hour — see generate_plots.load_era5_data),
    so a lag of 1 index IS a 24h lag; consecutive-day differencing is exactly
    E_t = Z_true_t - Z_true_{t-24} rather than an approximation of it.

    Returns (n_snapshots - 1, H * W).
    """
    n = field_all.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 time snapshots to form 24h persistence residuals, got {n}.")
    flat = field_all.reshape(n, -1)
    return flat[1:] - flat[:-1]


def empirical_spatial_correlation(data: dict) -> np.ndarray:
    """Pearson correlation matrix R_emp (D x D) of the 24h persistence residuals."""
    residuals = compute_persistence_residuals(data["t2m"])
    return np.corrcoef(residuals.T)


# ---------------------------------------------------------------------------
# TabICLv2 / CopulaTabICL: shared checkpoint loading + prior/posterior extraction
# ---------------------------------------------------------------------------
def load_copula_model(ckpt_path: str, device: "str | None" = None):
    """Load a CopulaTabICL checkpoint, mirroring
    generate_plots.build_copula_correlation_fn's loading path exactly (same
    build_copula_transformer factory + state_dict load) so this script stays
    consistent with every other real-checkpoint entry point in this repo.
    """
    import torch
    from omegaconf import OmegaConf

    from src.model import build_copula_transformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_copula_transformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded CopulaTabICL checkpoint '{ckpt_path}' (step {ckpt.get('step')}) on {device}.")
    return model, cfg, device


def load_marginal_tabicl(cfg, device: str):
    """Load the frozen, pretrained TabICL quantile regressor used ONLY as a
    marginal-CDF oracle for the PIT transform in
    extract_model_posterior_correlation — NOT the same object as the
    CopulaTabICL backbone in load_copula_model, whose quantile decoder has
    been stripped and replaced by the copula head (see src/model.py). This
    is the same checkpoint (cfg.tabicl.ckpt) CopulaTabICL's backbone was
    initialized from, loaded a second time with its native quantile head
    intact (see src/pit.py::load_tabicl / run_pit).

    Returns None (with a warning) if the checkpoint's backbone was trained
    from scratch (cfg.tabicl.pretrained=False), since there is then no
    quantile-calibrated marginal model available to PIT against.
    """
    if not bool(cfg.tabicl.get("pretrained", True)):
        print("Warning: cfg.tabicl.pretrained=False — no pretrained quantile "
              "head available for PIT; posterior z_train will fall back to "
              "naive standardization.")
        return None

    from src.pit import load_tabicl

    return load_tabicl(cfg.tabicl.ckpt, device)


def _forward_correlation(model, device, x_train_norm: np.ndarray, z_train: np.ndarray, x_test_norm: np.ndarray) -> np.ndarray:
    """Shared (x_train, z_train, x_test) -> Sigma forward pass, used by both
    the prior and posterior extractions below (see src/model.py:CopulaTabICL
    and low_rank_correlation)."""
    import torch

    from src.model import low_rank_correlation

    x_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device).unsqueeze(0)
    x_test_t = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device).unsqueeze(0)
    z_train_t = torch.as_tensor(z_train, dtype=torch.float32, device=device).unsqueeze(0)
    batch = {"x_train": x_train_t, "x_test": x_test_t, "z_train": z_train_t}

    with torch.no_grad():
        out = model(batch)
        Sigma = low_rank_correlation(out["W"], out["s"], jitter=1e-4)
    return Sigma[0].cpu().numpy()


def extract_model_prior_correlation(model, device, coords_test: np.ndarray) -> np.ndarray:
    """Extract the model's PRIOR correlation matrix via an unconditional
    forward pass — i.e. with no informative historical in-context examples,
    so the output cannot depend on any specific test-time context, only on
    the learned prior.

    CopulaTabICL has no separate closed-form "prior head": (W, s) are always
    produced from a forward pass over (x_train, z_train, x_test). A literal
    zero-row x_train/z_train (P=0) is not supported by the underlying TabICL
    backbone — its target-aware column embedding unconditionally computes
    `y_train.max()` (see tabicl_upstream/src/tabicl/_model/embedding.py),
    which raises on an empty tensor regardless of oracle mode. The closest
    architecturally-valid stand-in for "no historical context" is therefore
    a single dummy context row at x_train=0, z_train=0 (P=1) — a constant,
    content-free input carrying no information about any real historical
    series. Under cfg.data.oracle_mode="prior" training (the current
    default, see conf/data/gp_tasks.yaml), R_star is defined to ignore
    training-context conditioning entirely, so a well-trained model's output
    here should not be sensitive to which dummy value is fed in.
    """
    coords_test = np.asarray(coords_test, dtype=np.float64)
    x_mean = coords_test.mean(axis=0, keepdims=True)
    x_std = coords_test.std(axis=0, keepdims=True).clip(min=1e-8)
    x_test_norm = (coords_test - x_mean) / x_std

    x_train_norm = np.zeros((1, coords_test.shape[1]), dtype=np.float64)
    z_train = np.zeros(1, dtype=np.float64)
    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


def extract_model_posterior_correlation(
    model, device, tabicl_marginal, context_coords: np.ndarray, context_values: np.ndarray,
    coords_test: np.ndarray, k_folds: int = 10,
) -> np.ndarray:
    """Extract the model's POSTERIOR correlation matrix via a single joint
    forward pass over all of `coords_test` at once, conditioned on a real
    historical in-context sample (context_coords, context_values) — the
    actual model-in-the-loop copula posterior.

    z_train is the K-fold leave-one-out PIT of each context point's true
    value under `tabicl_marginal`'s own predicted marginal distribution
    (src/pit.py::run_pit, reused as-is): each context point is held out in
    one of k_folds disjoint folds, its marginal CDF F_hat is predicted from
    the OTHER context points via the frozen pretrained TabICL quantile
    head, u_i = F_hat(y_i) is its resulting quantile, and z_i = Phi^-1(u_i)
    Gaussianizes it. This is the real-data analogue of how z_train is
    defined during training (data_gen.py's GP-oracle LOO PIT, R&W Eq.
    5.12) — same PIT definition, model marginal instead of GP closed form
    — replacing the previous naive (y - mean) / std standardization, which
    assumed a Gaussian marginal instead of estimating one.

    If `tabicl_marginal` is None (scratch-trained backbone, no quantile
    head available), falls back to the naive standardization.
    """
    x_mean = context_coords.mean(axis=0, keepdims=True)
    x_std = context_coords.std(axis=0, keepdims=True).clip(min=1e-8)
    x_train_norm = (context_coords - x_mean) / x_std
    x_test_norm = (coords_test - x_mean) / x_std

    if tabicl_marginal is None:
        y_std = max(context_values.std(), 1e-8)
        z_train = (context_values - context_values.mean()) / y_std
    else:
        import torch

        from src.pit import run_pit

        X_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device)
        Y_train_t = torch.as_tensor(context_values, dtype=torch.float32, device=device).unsqueeze(-1)  # (P, 1)
        pit_out = run_pit(
            tabicl_marginal, X_train_t, Y_train_t, X_train_t[:1], Y_train_t[:1], k_folds=k_folds,
        )
        z_train = pit_out["z_train"].squeeze(-1).cpu().numpy()  # (P,)

    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


# ---------------------------------------------------------------------------
# Distance binning shared by every curve
# ---------------------------------------------------------------------------
def bin_correlation_by_distance(R: np.ndarray, dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Mean correlation per distance bin, over the upper-triangle pairwise entries."""
    iu = np.triu_indices_from(R, k=1)
    corr, d = R[iu], dist[iu]
    n_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(d, bin_edges) - 1, 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            means[b] = corr[mask].mean()
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a trained CopulaTabICL / TabICLv2 checkpoint "
        "(e.g. ./checkpoints/systematic-composition-8/step_0180000.pt).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=None,
        help="Day indices t providing the historical in-context sample: for each t, the posterior "
        "conditions on the 24h persistence residual field[t] - field[t-1] (same quantity as the "
        "ground-truth curve), sampled at --n-context grid points. Each must be >= 1. One posterior "
        "curve is computed per day and shown faint, plus their mean shown bold, so you can see "
        "whether the posterior's shape is a systematic model behavior or single-day noise. "
        "Default: an evenly spaced spread of up to 8 days across the whole dataset.",
    )
    parser.add_argument("--n-context", type=int, default=50, help="Number of historical context points sampled per day.")
    parser.add_argument(
        "--pit-k-folds", type=int, default=10,
        help="Number of disjoint folds for the K-fold leave-one-out PIT that turns real context "
        "values into z_train (src/pit.py::run_pit). Fixed, small K rather than true LOO -- see "
        "project convention.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15, help="Number of spatial-distance bins.")
    parser.add_argument("--output", type=str, default=OUT_PATH)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = load_era5_data()
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])  # (D, 2) = (lon, lat)
    D = coords.shape[0]

    n_days = data["t2m"].shape[0]
    if args.days is None:
        n_pick = min(8, n_days - 1)
        args.days = sorted(set(np.linspace(1, n_days - 1, n_pick).round().astype(int).tolist()))
    for d in args.days:
        if not (1 <= d < n_days):
            parser.error(f"each --days value must be in [1, {n_days - 1}] (need day-1 to form a 24h residual), got {d}")

    print("Computing empirical spatial correlation from 24h persistence residuals...")
    R_emp = empirical_spatial_correlation(data)

    R_indep = np.eye(D)

    print(f"Loading TabICLv2 checkpoint '{args.ckpt}'...")
    model, cfg, device = load_copula_model(args.ckpt, device=args.device)

    print("Loading frozen pretrained TabICL quantile head for posterior context PIT...")
    tabicl_marginal = load_marginal_tabicl(cfg, device)

    print("Extracting the model's unconditional PRIOR correlation matrix (no real context)...")
    R_prior = extract_model_prior_correlation(model, device, coords)

    dist = haversine_distance_km(coords)
    bin_edges = np.linspace(0.0, dist.max(), args.n_bins + 1)
    dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_context = min(args.n_context, D)
    context_idx = rng.choice(D, size=n_context, replace=False)  # same context locations every day, only values change
    context_coords = coords[context_idx]

    rho_posterior_per_day = []
    for d in args.days:
        print(f"Extracting the joint copula POSTERIOR correlation matrix (context day={d}, "
              f"conditioned on the 24h persistence residual field[{d}] - field[{d - 1}])...")
        residual_day = data["t2m"][d].ravel() - data["t2m"][d - 1].ravel()
        context_values = residual_day[context_idx]
        R_posterior = extract_model_posterior_correlation(
            model, device, tabicl_marginal, context_coords, context_values, coords, k_folds=args.pit_k_folds,
        )
        rho_posterior_per_day.append(bin_correlation_by_distance(R_posterior, dist, bin_edges))
    rho_posterior_per_day = np.array(rho_posterior_per_day)  # (n_days, n_bins)
    rho_posterior_mean = np.nanmean(rho_posterior_per_day, axis=0)

    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)
    rho_indep = bin_correlation_by_distance(R_indep, dist, bin_edges)
    rho_prior = bin_correlation_by_distance(R_prior, dist, bin_edges)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(dist_centers, rho_emp, "--", color="black", marker="o",
            label="Ground Truth: empirical corr. of real 24h residuals\n"
                  "$E_t = Z_t - Z_{t-24}$, averaged over all days")
    ax.plot(dist_centers, rho_indep, "-", color="red", marker="^",
            label="Independent TabICLv2: no copula, so $\\rho \\equiv 0$ by construction")
    ax.plot(dist_centers, rho_prior, "-", color="tab:orange", marker="D",
            label="Copula model with dummy context")
    for i, (d, rho_d) in enumerate(zip(args.days, rho_posterior_per_day)):
        ax.plot(dist_centers, rho_d, "-", color="blue", alpha=0.18, linewidth=1,
                label=f"Copula model with {n_context} context points: individual days\n({len(args.days)} days shown faint)" if i == 0 else None)
    ax.plot(dist_centers, rho_posterior_mean, "-", color="blue", marker="s", linewidth=2.2,
            label=f"Copula model with {n_context} context points: mean over {len(args.days)} days")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Spatial distance (km)")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Spatial Correlation Decay: Ground Truth vs. Copula Model (Independent / Dummy Context / With Context)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
