"""
plot_spatial_correlation_diagnostics.py — Spatial correlogram sanity check
for TabICLv2 (CopulaTabICL): does the model's learned inter-instance
dependence structure reproduce the physical spatial correlation decay of the
real ERA5 field?

Four empirical/model curves plus theoretical-law overlays, all binned (or,
for the theory curves, evaluated) by great-circle distance and plotted
together:

  1. Ground truth: empirical correlation of 24h persistence residuals
     E_t = Z_true_t - Z_true_{t-24} across the spatial grid.
  2. Independent TabICLv2: a model with no inter-instance copula assumes
     conditional independence across the grid, so its implied correlation
     matrix IS the identity by construction — no forward pass needed.
  3. Copula model with dummy context: an (as-near-as-architecturally-possible)
     unconditional forward pass of the trained CopulaTabICL checkpoint. This
     is NOT a Bayesian posterior extraction, just a forward pass with a
     content-free dummy context row, matching how the model is trained
     (cfg.data.oracle_mode="prior", see src/data_gen.py) to output R_star
     ignoring in-context conditioning.
  4. Copula model with N context points: a real joint forward pass over all
     D grid points at once, conditioned on a historical in-context sample of
     the SAME 24h persistence residual field as (1) — not the raw absolute
     temperatures — so it's conditioned on the same physical quantity whose
     spatial decay it's being compared against. This is likewise NOT a
     Bayesian posterior — it's the same forward pass as (3), just with real
     context instead of a dummy one. Context labels z_train are NOT a naive
     (y - mean) / std standardization: they are the K-fold leave-one-out
     Probability Integral Transform of each context point's true residual
     under TabICLv2's own learned marginal (see src/pit.py::run_pit) — i.e.
     u_i = F_hat(y_i | other context points), z_i = Phi^-1(u_i) — matching
     how z_train is actually defined during training (data_gen.py's
     GP-oracle LOO PIT) instead of assuming a Gaussian marginal by fiat.
  5. Theoretical decay laws (--theory-models, default: ALL FOUR at once):
     every isotropic correlation law in the reference literature --
     exponential (Hansen & Lebedeff 1987, nu=1/2), Gaussian (nu->infinity),
     Whittle (Matern nu=1, the closed-form omega->0 limit of North, Wang &
     Genton 2011's energy-balance/damped-diffusion model), and the general
     Matern (free smoothness nu, fit directly rather than assumed) -- each
     independently fit by weighted nonlinear least squares to curve (1)
     ONLY (never to the model curves, and never to the synthetic-GP
     fallback's own generating kernel), so the fit is an independent
     physical sanity check: if the ground truth doesn't track a known decay
     law reasonably well (see the printed weighted R^2 per law), something
     is off with the empirical curve itself before comparing it to the
     model at all. Plotting all four together also shows which shape family
     the data actually prefers (e.g. Matern's fitted nu should land close
     to whichever of exponential/Gaussian/Whittle fits best).

Reuses:
  - plots/generate_plots.py: load_era5_data (+ synthetic-GP fallback) and
    haversine_distance_km, so ERA5 I/O and great-circle distance math are
    never reimplemented here.
  - src/model.py: build_copula_transformer + low_rank_correlation, the same
    (W, s) -> Sigma projection used by generate_plots.py's
    build_copula_correlation_fn — this script loads the checkpoint once and
    reuses it for both the dummy-context (3) and real-context (4)
    extractions instead of loading it twice.
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
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_fn, kv as bessel_k

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
# TabICLv2 / CopulaTabICL: shared checkpoint loading + dummy/real-context extraction
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
    extract_model_context_correlation — NOT the same object as the
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
              "head available for PIT; context z_train will fall back to "
              "naive standardization.")
        return None

    from src.pit import load_tabicl

    return load_tabicl(cfg.tabicl.ckpt, device)


def _forward_correlation(model, device, x_train_norm: np.ndarray, z_train: np.ndarray, x_test_norm: np.ndarray) -> np.ndarray:
    """Shared (x_train, z_train, x_test) -> Sigma forward pass, used by both
    the dummy-context and real-context extractions below (see
    src/model.py:CopulaTabICL and low_rank_correlation)."""
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


def extract_model_dummy_context_correlation(model, device, coords_test: np.ndarray) -> np.ndarray:
    """Extract the model's correlation matrix via an unconditional forward
    pass — i.e. with no informative historical in-context examples, so the
    output cannot depend on any specific test-time context, only on what
    the model learned as its context-free behavior. This is NOT a Bayesian
    prior extraction in any closed-form sense — just a forward pass fed a
    dummy context (see below).

    CopulaTabICL has no separate closed-form "no-context head": (W, s) are
    always produced from a forward pass over (x_train, z_train, x_test). A
    literal zero-row x_train/z_train (P=0) is not supported by the
    underlying TabICL backbone — its target-aware column embedding
    unconditionally computes `y_train.max()` (see
    tabicl_upstream/src/tabicl/_model/embedding.py), which raises on an
    empty tensor regardless of oracle mode. The closest architecturally-valid
    stand-in for "no historical context" is therefore a single dummy context
    row at x_train=0, z_train=0 (P=1) — a constant, content-free input
    carrying no information about any real historical series. Under
    cfg.data.oracle_mode="prior" training (the current default, see
    conf/data/gp_tasks.yaml), R_star is defined to ignore training-context
    conditioning entirely, so a well-trained model's output here should not
    be sensitive to which dummy value is fed in.
    """
    coords_test = np.asarray(coords_test, dtype=np.float64)
    x_mean = coords_test.mean(axis=0, keepdims=True)
    x_std = coords_test.std(axis=0, keepdims=True).clip(min=1e-8)
    x_test_norm = (coords_test - x_mean) / x_std

    x_train_norm = np.zeros((1, coords_test.shape[1]), dtype=np.float64)
    z_train = np.zeros(1, dtype=np.float64)
    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


def extract_model_context_correlation(
    model, device, tabicl_marginal, context_coords: np.ndarray, context_values: np.ndarray,
    coords_test: np.ndarray, k_folds: int = 10,
) -> np.ndarray:
    """Extract the model's correlation matrix via a single joint forward
    pass over all of `coords_test` at once, conditioned on a real historical
    in-context sample (context_coords, context_values). This is NOT a
    Bayesian posterior extraction — it's the same forward pass as
    extract_model_dummy_context_correlation, just with real context points
    instead of a dummy one.

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
def _bin_indices(d: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin index per distance, or -1 for out-of-[bin_edges[0], bin_edges[-1]] values.

    Pairs beyond bin_edges[-1] are DROPPED (index -1), not clipped into the last
    bin: bin_edges may be capped below dist.max() (see --max-dist-percentile) to
    exclude the corner-only, high-variance tail of a bounded non-periodic
    lat/lon domain, and clipping would silently pull that excluded tail back
    into the last visible bin under a misleadingly low distance label.
    """
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(d, bin_edges) - 1
    bin_idx[(d < bin_edges[0]) | (d > bin_edges[-1])] = -1
    bin_idx[bin_idx == n_bins] = n_bins - 1  # d == bin_edges[-1] exactly
    return bin_idx


def bin_correlation_by_distance(R: np.ndarray, dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Mean correlation per distance bin, over the upper-triangle pairwise entries."""
    iu = np.triu_indices_from(R, k=1)
    corr, d = R[iu], dist[iu]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(d, bin_edges)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            means[b] = corr[mask].mean()
    return means


def pair_counts_by_distance(dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Number of upper-triangle (i, j) pairs falling in each distance bin.

    On a bounded, non-periodic lat/lon rectangle (see AREA in generate_plots.py),
    the pair population thins out sharply near the max distance -- only the
    handful of pairs straddling opposite corners of the box reach it -- so a
    bin's mean correlation is only as trustworthy as its count here. This is a
    diagnostic for exactly that: low counts in the tail flag bins whose mean is
    a small-sample, corner-biased estimate rather than a real isotropic decay.
    """
    iu_dist = dist[np.triu_indices_from(dist, k=1)]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(iu_dist, bin_edges)
    counts = np.bincount(bin_idx[bin_idx >= 0], minlength=n_bins)
    return counts[:n_bins]


# ---------------------------------------------------------------------------
# Theoretical spatial-correlation decay laws (North, Wang & Genton 2011;
# Whittle 1954; Matern 1960) -- an independent physical sanity check, fit
# ONLY to the ground-truth empirical curve and NOT derived from the model
# or (for the synthetic fallback) the GP's own generating kernel, so a good
# fit is informative rather than circular.
# ---------------------------------------------------------------------------
def exponential_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r / L) -- Hansen & Lebedeff (1987); Matern nu=1/2. Cusped
    (non-smooth) at r=0; the classic empirical baseline, fit this first."""
    return np.exp(-np.asarray(r, dtype=np.float64) / L)


def gaussian_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r^2 / (2 L^2)) -- Matern nu -> infinity, i.e. an
    infinitely smooth field. Usually too smooth for real temperature data."""
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-(r ** 2) / (2.0 * L ** 2))


def matern_law(r: np.ndarray, L: float, nu: float) -> np.ndarray:
    """General Matern correlation: rho(r) = 2^(1-nu)/Gamma(nu) (r/L)^nu K_nu(r/L),
    rho(0)=1 by the x K_nu(x) -> ... limit (handled explicitly below since
    K_nu(0) itself diverges for nu>0). nu=1/2 reduces to exponential_law,
    nu=1 is whittle_law, nu->inf approaches gaussian_law. Free-nu Matern is
    usually the best unconstrained empirical fit (nu~1 typical for real
    temperature fields)."""
    r = np.asarray(r, dtype=np.float64)
    out = np.ones_like(r)
    nz = r > 0
    x = r[nz] / L
    out[nz] = (2.0 ** (1.0 - nu) / gamma_fn(nu)) * (x ** nu) * bessel_k(nu, x)
    return out


def whittle_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = (r/L) K_1(r/L) -- Matern nu=1, AND the omega->0 (long-time-
    averaging) closed-form limit of the North, Wang & Genton (2011)
    energy-balance / damped-diffusion model (their Eq. 6). This is the one
    law here with an actual physical derivation behind its shape, not just
    a good empirical fit -- see the reference doc section 2.3."""
    return matern_law(r, L, nu=1.0)


# name -> (callable(r, *params), ordered param names); param order must match
# each callable's positional signature for curve_fit's p0/bounds below.
THEORY_LAWS = {
    "exponential": (exponential_law, ["L"]),
    "gaussian": (gaussian_law, ["L"]),
    "whittle": (whittle_law, ["L"]),
    "matern": (matern_law, ["L", "nu"]),
}

# name -> (color, linestyle, linewidth, display label) for the overlay plot.
# matern is drawn bolder/opaque since it's the general (free-nu) parent model
# every other law here is a special case of (see THEORY_LAWS docstrings).
THEORY_STYLE = {
    "exponential": ("purple", "-.", 1.6, "Exponential (Hansen & Lebedeff 1987, $\\nu$=1/2)"),
    "gaussian": ("saddlebrown", "--", 1.6, "Gaussian ($\\nu\\to\\infty$)"),
    "whittle": ("darkgreen", ":", 1.8, "Whittle / EBCM $\\omega\\to0$ limit (North et al. 2011, $\\nu$=1)"),
    "matern": ("magenta", "-", 2.4, "Matérn (free $\\nu$)"),
}


def _correlation_length_guess(dist_centers: np.ndarray, rho: np.ndarray) -> float:
    """Initial L guess for curve_fit: distance at which the empirical curve
    crosses 1/e (the correlation-length definition used throughout the
    reference doc), by linear interpolation between the bracketing bin
    centers. Falls back to half the plotted distance range if the curve
    never crosses 1/e (e.g. too noisy or too short a distance range)."""
    valid = np.isfinite(rho)
    d, r = dist_centers[valid], rho[valid]
    below = np.where(r <= 1.0 / np.e)[0]
    if len(below) == 0 or below[0] == 0:
        return float(d[-1] / 2.0) if len(d) else 1000.0
    i = below[0]
    d0, d1, r0, r1 = d[i - 1], d[i], r[i - 1], r[i]
    if r0 == r1:
        return float(d0)
    frac = (1.0 / np.e - r0) / (r1 - r0)
    return float(d0 + frac * (d1 - d0))


def fit_theoretical_law(
    dist_centers: np.ndarray, rho_emp: np.ndarray, pair_counts: np.ndarray, model: str,
) -> "dict | None":
    """Nonlinear least-squares fit of `model` (see THEORY_LAWS) to the binned
    ground-truth empirical curve, weighted by sqrt(pair_counts) per bin: a
    bin's mean-correlation estimate is lower-variance the more pairs back it
    (see pair_counts_by_distance's docstring on the corner-biased tail), so
    high-count bins should pull the fit harder than sparse tail bins.

    Returns None (with a printed warning) instead of raising if curve_fit
    fails to converge or too few bins are populated, so the caller can just
    skip drawing the theory curve rather than crash the whole diagnostic
    plot over an unfittable curve.
    """
    if model not in THEORY_LAWS:
        raise ValueError(f"Unknown --theory-model '{model}', choose from {sorted(THEORY_LAWS)}.")
    law_fn, param_names = THEORY_LAWS[model]

    mask = np.isfinite(rho_emp) & (pair_counts > 0)
    if mask.sum() < len(param_names) + 1:
        print(f"Warning: too few valid distance bins ({mask.sum()}) to fit '{model}'; skipping theory curve.")
        return None
    d, r, n = dist_centers[mask], rho_emp[mask], pair_counts[mask]
    sigma = 1.0 / np.sqrt(n)  # SEM of a Pearson-r bin mean scales ~ 1/sqrt(n pairs)

    L0 = _correlation_length_guess(dist_centers, rho_emp)
    L_hi = max(50.0 * L0, 10.0 * float(dist_centers[-1]))
    if param_names == ["L"]:
        p0, bounds = [L0], ([1.0], [L_hi])
    else:  # ["L", "nu"]
        p0, bounds = [L0, 1.0], ([1.0, 0.05], [L_hi, 8.0])

    try:
        popt, _ = curve_fit(law_fn, d, r, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
    except RuntimeError as exc:
        print(f"Warning: curve_fit failed to converge for '{model}' ({exc}); skipping theory curve.")
        return None

    pred = law_fn(d, *popt)
    resid = r - pred
    weighted_ss_res = float(np.sum((resid / sigma) ** 2))
    r_bar = np.average(r, weights=1.0 / sigma ** 2)
    weighted_ss_tot = float(np.sum(((r - r_bar) / sigma) ** 2))
    r_squared = 1.0 - weighted_ss_res / weighted_ss_tot if weighted_ss_tot > 0 else float("nan")

    return {"model": model, "law_fn": law_fn, "params": dict(zip(param_names, popt)), "r_squared": r_squared}


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
        help="Day indices t providing the historical in-context sample: for each t, the model "
        "conditions on the 24h persistence residual field[t] - field[t-1] (same quantity as the "
        "ground-truth curve), sampled at --n-context grid points. Each must be >= 1. One "
        "context-conditioned curve is computed per day and shown faint, plus their mean shown "
        "bold, so you can see whether the curve's shape is a systematic model behavior or "
        "single-day noise. Default: an evenly spaced spread of up to 8 days across the whole "
        "dataset.",
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
    parser.add_argument(
        "--max-dist-percentile", type=float, default=90.0,
        help="Cap the binned distance range at this percentile of all pairwise distances instead "
        "of the raw max. On a bounded, non-periodic lat/lon rectangle the pairs near the raw max "
        "distance are almost exclusively opposite-corner pairs -- a tiny, geometrically special, "
        "non-isotropic subset -- so their bin mean is a high-variance, corner-biased estimate, not "
        "a real feature of the spatial decay. Set to 100 to restore the old raw-max behavior.",
    )
    parser.add_argument(
        "--theory-models", type=str, nargs="+", default=["exponential", "gaussian", "whittle", "matern"],
        choices=["exponential", "gaussian", "whittle", "matern", "none"],
        help="Theoretical spatial-correlation decay law(s) to fit (weighted nonlinear least squares, each "
        "independently) to the ground-truth empirical curve and overlay, as a physical sanity check "
        "independent of the model (North, Wang & Genton 2011 / Whittle 1954 / Matern 1960). Default plots "
        "every law from the reference literature at once so their shapes can be compared directly: "
        "'exponential' (nu=1/2, Hansen & Lebedeff 1987 baseline), 'gaussian' (nu->inf, infinitely-smooth "
        "upper bound), 'whittle' (nu=1, the physically-derived long-averaging-limit EBCM law), 'matern' "
        "(free smoothness nu, the most flexible fit -- nu is estimated directly by the fit, not assumed). "
        "Pass 'none' alone to disable the overlay entirely.",
    )
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

    print("Loading frozen pretrained TabICL quantile head for context PIT...")
    tabicl_marginal = load_marginal_tabicl(cfg, device)

    print("Extracting the model's unconditional correlation matrix (dummy context)...")
    R_dummy_context = extract_model_dummy_context_correlation(model, device, coords)

    dist = haversine_distance_km(coords)
    dist_iu = dist[np.triu_indices_from(dist, k=1)]
    max_dist = np.percentile(dist_iu, args.max_dist_percentile)
    bin_edges = np.linspace(0.0, max_dist, args.n_bins + 1)
    dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pair_counts = pair_counts_by_distance(dist, bin_edges)
    print(f"Pairwise distances range [0, {dist_iu.max():.0f}] km; binning out to the "
          f"{args.max_dist_percentile:.0f}th percentile ({max_dist:.0f} km) to avoid the "
          f"corner-only, high-variance tail of the raw max. Pairs per bin:")
    for lo, hi, n in zip(bin_edges[:-1], bin_edges[1:], pair_counts):
        print(f"  [{lo:7.0f}, {hi:7.0f}) km: {n:8d} pairs")

    n_context = min(args.n_context, D)
    context_idx = rng.choice(D, size=n_context, replace=False)  # same context locations every day, only values change
    context_coords = coords[context_idx]

    rho_context_per_day = []
    for d in args.days:
        print(f"Extracting the joint copula correlation matrix with real context (context day={d}, "
              f"conditioned on the 24h persistence residual field[{d}] - field[{d - 1}])...")
        residual_day = data["t2m"][d].ravel() - data["t2m"][d - 1].ravel()
        context_values = residual_day[context_idx]
        R_context = extract_model_context_correlation(
            model, device, tabicl_marginal, context_coords, context_values, coords, k_folds=args.pit_k_folds,
        )
        rho_context_per_day.append(bin_correlation_by_distance(R_context, dist, bin_edges))
    rho_context_per_day = np.array(rho_context_per_day)  # (n_days, n_bins)
    rho_context_mean = np.nanmean(rho_context_per_day, axis=0)

    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)
    rho_indep = bin_correlation_by_distance(R_indep, dist, bin_edges)
    rho_dummy_context = bin_correlation_by_distance(R_dummy_context, dist, bin_edges)

    theory_models = [] if args.theory_models == ["none"] else args.theory_models
    theory_fits = []
    for theory_model in theory_models:
        print(f"Fitting theoretical decay law '{theory_model}' to the ground-truth empirical curve...")
        fit = fit_theoretical_law(dist_centers, rho_emp, pair_counts, theory_model)
        if fit is not None:
            param_str = ", ".join(f"{k}={v:.0f} km" if k == "L" else f"{k}={v:.2f}"
                                   for k, v in fit["params"].items())
            print(f"  Fitted {theory_model}: {param_str}, weighted R^2={fit['r_squared']:.3f}")
            theory_fits.append(fit)

    fig, (ax, ax_count) = plt.subplots(
        2, 1, figsize=(10.5, 7.3), sharex=True, height_ratios=[3.2, 1],
        gridspec_kw={"hspace": 0.08},
    )
    ax.plot(dist_centers, rho_emp, "--", color="black", marker="o",
            label="Ground Truth: empirical corr. of real 24h residuals\n"
                  "$E_t = Z_t - Z_{t-24}$, averaged over all days")
    r_dense = np.linspace(0.0, bin_edges[-1], 300)
    for fit in theory_fits:
        color, linestyle, linewidth, display_name = THEORY_STYLE[fit["model"]]
        rho_theory = fit["law_fn"](r_dense, *fit["params"].values())
        L_fit = fit["params"]["L"]
        nu_str = f", $\\nu$={fit['params']['nu']:.2f}" if "nu" in fit["params"] else ""
        ax.plot(r_dense, rho_theory, linestyle, color=color, linewidth=linewidth,
                label=f"{display_name} fit to ground truth:\n"
                      f"$L$={L_fit:.0f} km{nu_str}, weighted $R^2$={fit['r_squared']:.3f}")
        ax.axvline(L_fit, color=color, linewidth=0.8, linestyle=linestyle, alpha=0.4)
    ax.plot(dist_centers, rho_indep, "-", color="red", marker="^",
            label="Independent TabICLv2: no copula, so $\\rho \\equiv 0$ by construction")
    ax.plot(dist_centers, rho_dummy_context, "-", color="tab:orange", marker="D",
            label="Copula model with dummy context")
    for i, (d, rho_d) in enumerate(zip(args.days, rho_context_per_day)):
        ax.plot(dist_centers, rho_d, "-", color="blue", alpha=0.18, linewidth=1,
                label=f"Copula model with {n_context} context points: individual days\n({len(args.days)} days shown faint)" if i == 0 else None)
    ax.plot(dist_centers, rho_context_mean, "-", color="blue", marker="s", linewidth=2.2,
            label=f"Copula model with {n_context} context points: mean over {len(args.days)} days")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Spatial Correlation Decay: Ground Truth vs. Copula Model (Independent / Dummy Context / With Context)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

    bin_width = bin_edges[1] - bin_edges[0]
    ax_count.bar(dist_centers, pair_counts, width=bin_width * 0.9, color="steelblue", alpha=0.8)
    ax_count.set_yscale("log")
    ax_count.set_xlabel("Spatial distance (km)")
    ax_count.set_ylabel("Pairs per bin\n(log scale)", fontsize=8)
    ax_count.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
