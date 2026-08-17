"""calibration.py — TabICL calibration diagnostics: per-quantile-level
Expected Calibration Error (marginal calibration — is any single grid
cell's quantile forecast correct on its own?) plus independence-copula
multivariate calibration (given only per-cell MARGINAL quantile
predictions, no correlation/copula model — is the INDEPENDENCE JOINT
implied by those marginals correct?). Promoted from plots/generate_plots.py
(compute_quantile_ece, generate_era5_reliability_diagram,
plot_era5_quantile_reliability, calc_kendall_pit,
calc_mahalanobis_distances, calc_exceedance_probs, calc_spatial_coverage and
their plot_* counterparts), used by eval/runners/era5_calibration_eval.py.
`plot_era5_quantile_reliability` uses the real eval.tabicl_utils
TabICLRegressor wrapper instead of generate_plots.py's local
TabICLv2_Regressor (a thin, non-mock wrapper around the same
tabicl.TabICLRegressor — same swap as era5_calibration_eval.py's
MockTabICLv2 replacement).
"""

from __future__ import annotations

import os

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2

from eval.tabicl_utils import make_tabicl_regressor

__all__ = [
    "compute_quantile_ece",
    "generate_era5_reliability_diagram",
    "plot_era5_quantile_reliability",
    "calc_kendall_pit",
    "plot_kendall_pit",
    "calc_mahalanobis_distances",
    "plot_mahalanobis_pp",
    "calc_exceedance_probs",
    "plot_spatial_reliability",
    "calc_spatial_coverage",
    "plot_spatial_coverage_curve",
]


def compute_quantile_ece(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: "list | np.ndarray",
) -> "tuple[float, np.ndarray]":
    """
    Compute the quantile-regression Expected Calibration Error (ECE) — per-
    cell MARGINAL calibration (unlike calc_kendall_pit et al. below, which
    probe the JOINT/independence-copula assumption on top of the marginals).

    For each nominal quantile level, the empirical coverage is the fraction
    of instances where the true value falls at or below the predicted
    quantile value. The ECE is the mean absolute gap between the nominal
    levels and their empirical coverages (0 for perfect calibration).

    Args:
        y_true: 1D array of shape (n_samples,) with observed targets.
        y_pred_quantiles: 2D array of shape (n_samples, n_quantiles); column
            k holds the predicted value for quantiles[k].
        quantiles: Nominal quantile levels in (0, 1), e.g. [0.1, ..., 0.9].

    Returns:
        Tuple (ece, empirical_coverage), where empirical_coverage has shape
        (n_quantiles,) and aligns positionally with `quantiles`.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles, dtype=float)

    if y_pred_quantiles.ndim != 2:
        raise ValueError("y_pred_quantiles must be 2D (n_samples, n_quantiles).")
    if y_pred_quantiles.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} samples but y_pred_quantiles has {y_pred_quantiles.shape[0]}."
        )
    if y_pred_quantiles.shape[1] != quantiles.shape[0]:
        raise ValueError(
            f"y_pred_quantiles has {y_pred_quantiles.shape[1]} quantile columns "
            f"but {quantiles.shape[0]} nominal quantiles were given."
        )

    empirical_coverage = np.mean(y_true[:, None] <= y_pred_quantiles, axis=0)
    ece = float(np.mean(np.abs(quantiles - empirical_coverage)))
    return ece, empirical_coverage


def generate_era5_reliability_diagram(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: "list | np.ndarray", output_path: str,
) -> float:
    """
    Build and save a reliability diagram for TabICL's ERA5 quantile
    predictions, with nominal quantile on the x-axis and empirical coverage
    on the y-axis.

    Returns:
        The scalar ECE score (also annotated on the figure).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ece, empirical_coverage = compute_quantile_ece(y_true, y_pred_quantiles, quantiles)
    quantiles = np.asarray(quantiles, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(quantiles, empirical_coverage, "o-", color="tab:blue", label="TabICL")
    ax.fill_between(quantiles, quantiles, empirical_coverage, color="tab:blue", alpha=0.2, label="Calibration gap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability Diagram: TabICL on ERA5 Dataset")
    ax.text(
        0.05, 0.95, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path} (ECE={ece:.4f})")
    return ece


def plot_era5_quantile_reliability(
    data: dict,
    context_idx: np.ndarray,
    quantiles: np.ndarray = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
    output_path: "str | None" = None,
    tabicl_ckpt: "str | None" = None,
    device: "str | None" = None,
) -> float:
    """
    Real-data driver for the reliability diagram: context locations
    (`context_idx`) are fixed once, but the context LABELS are re-drawn from
    that day's field and TabICL is re-queried per day, so we iterate over
    every timestamp in the dataset to build up (y_true, y_pred_quantiles)
    across all days x grid points before scoring calibration.

    Evaluation is restricted to grid points OUTSIDE `context_idx`: querying
    the model at its own context locations would hand it the true label as
    a training example and then score it against that same label, inflating
    the apparent coverage. All quantile levels for a given day are requested
    from a single fit()+predict() call — TabICL's quantile spline comes from
    one backbone forward pass regardless of how many `quantiles` are
    requested.
    """
    field_all = data["t2m"]
    n_days = field_all.shape[0]
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    context_coords = coords[context_idx]

    target_idx = np.setdiff1d(np.arange(coords.shape[0]), context_idx)
    target_coords = coords[target_idx]

    quantiles = np.asarray(quantiles, dtype=float)
    regressor = make_tabicl_regressor(checkpoint=tabicl_ckpt, device=device)
    y_true_chunks, y_pred_chunks = [], []
    for d in range(n_days):
        context_values = field_all[d].ravel()[context_idx]
        regressor.fit(context_coords, context_values)
        preds = regressor.predict(target_coords, output_type="quantiles", alphas=list(quantiles))  # (n_target, n_quantiles)
        y_true_chunks.append(field_all[d].ravel()[target_idx])
        y_pred_chunks.append(preds)

    y_true = np.concatenate(y_true_chunks)
    y_pred_quantiles = np.concatenate(y_pred_chunks, axis=0)

    return generate_era5_reliability_diagram(y_true, y_pred_quantiles, quantiles, output_path)


def calc_kendall_pit(cdf_values: np.ndarray) -> np.ndarray:
    """
    Copula PIT (Kendall's transform) for the independence copula.

    Under H_i(y_i) = prod_d F_{i,d}(y_{i,d}), the copula-level "witness"
    w_i = prod_d F_{i,d}(y_{i,d}) is Uniform(0, 1)-distributed at every
    dimension only if the joint model is exactly correct; W = prod of D iid
    U(0,1) is NOT itself uniform (it concentrates near 0), so w_i must be
    passed through W's own CDF -- the Kendall distribution function -- to
    recover a Uniform(0, 1) PIT value:

        z_i = w_i * sum_{k=0}^{D-1} (-ln w_i)^k / k!

    Args:
        cdf_values: (n_samples, D) array of F_{i,d}(y_{i,d}) marginal CDF
            values, one row per instance and one column per spatial
            dimension.

    Returns:
        (n_samples,) array of Kendall PIT values z_i in [0, 1].
    """
    cdf_values = np.clip(np.asarray(cdf_values, dtype=np.float64), 1e-300, 1.0)
    n, D = cdf_values.shape

    log_w = np.log(cdf_values).sum(axis=1)  # log(w_i) = sum_d log F_{i,d}(y_{i,d})
    x = np.clip(-log_w, 0.0, None)  # x_i = -ln(w_i) >= 0

    k = np.arange(D, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_terms = k[None, :] * np.log(x)[:, None] - gammaln(k + 1.0)[None, :]
    zero_mask = x == 0.0
    log_terms[zero_mask, :] = -np.inf
    log_terms[zero_mask, 0] = 0.0

    log_series = logsumexp(log_terms, axis=1)  # log sum_k x^k / k!
    z = np.exp(log_w + log_series)  # z_i = w_i * series
    return np.clip(z, 0.0, 1.0)


def plot_kendall_pit(z_values: np.ndarray, ax, n_bins: int = 20):
    """Histogram of Kendall PIT values against the theoretical Uniform(0, 1) density."""
    z_values = np.asarray(z_values, dtype=np.float64)
    ax.hist(
        z_values, bins=n_bins, range=(0.0, 1.0), density=True,
        color="tab:blue", alpha=0.75, edgecolor="white", label="Empirical",
    )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Uniform(0, 1)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Kendall PIT value $z$")
    ax.set_ylabel("Density")
    ax.set_title("Kendall PIT Histogram (Independence Copula)")
    ax.legend()
    return ax


def calc_mahalanobis_distances(
    y_true: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """
    Mahalanobis distance under a diagonal covariance (Gaussian marginals,
    independence copula): d_i^2 = sum_d (y_{i,d} - mu_{i,d})^2 / sigma^2_{i,d}.
    Under perfect calibration, d_i^2 ~ chi^2_D.

    Args:
        y_true, means, variances: (n_samples, D) arrays.

    Returns:
        (n_samples,) array of squared Mahalanobis distances.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if not (y_true.shape == means.shape == variances.shape):
        raise ValueError(
            f"y_true, means, variances must share shape (n_samples, D); got "
            f"{y_true.shape}, {means.shape}, {variances.shape}."
        )
    return np.sum((y_true - means) ** 2 / variances, axis=1)


def plot_mahalanobis_pp(distances: np.ndarray, dim_d: int, ax):
    """
    PP-plot (probability-probability plot) of squared Mahalanobis distances
    against the theoretical chi^2_D CDF. Points on the y = x diagonal
    indicate correct calibration.
    """
    distances = np.sort(np.asarray(distances, dtype=np.float64))
    n = distances.shape[0]
    empirical_p = (np.arange(1, n + 1) - 0.5) / n
    theoretical_p = chi2.cdf(distances, df=dim_d)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal calibration")
    ax.plot(theoretical_p, empirical_p, color="tab:blue", linewidth=1.5, label=f"$\\chi^2_{{{dim_d}}}$ PP-plot")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Theoretical $\chi^2_D$ CDF")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Mahalanobis $\\chi^2$ PP-Plot ($D={dim_d}$)")
    ax.legend()
    return ax


def calc_exceedance_probs(y_true: np.ndarray, cdf_func, thresholds: np.ndarray):
    """
    Spatial exceedance events and their independence-copula predicted
    probabilities, for a set of thresholds tau_m:

        e_{i,m} = 1[max_d y_{i,d} > tau_m]                (true event)
        pi_{i,m} = 1 - prod_d F_{i,d}(tau_m)              (predicted prob)

    Args:
        y_true: (n_samples, D) array of observed targets.
        cdf_func: callable, cdf_func(tau) -> (n_samples, D) array of marginal
            CDF values F_{i,d}(tau) at threshold tau, for every instance i
            and dimension d.
        thresholds: (n_thresholds,) array of thresholds tau_m.

    Returns:
        Tuple (predicted_probs, true_events), both (n_samples, n_thresholds).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    max_y = y_true.max(axis=1)  # (n_samples,)
    true_events = (max_y[:, None] > thresholds[None, :]).astype(np.float64)

    predicted_probs = np.empty((y_true.shape[0], thresholds.shape[0]), dtype=np.float64)
    for m, tau in enumerate(thresholds):
        F = np.clip(np.asarray(cdf_func(tau), dtype=np.float64), 0.0, 1.0)
        predicted_probs[:, m] = 1.0 - np.prod(F, axis=1)
    return predicted_probs, true_events


def plot_spatial_reliability(predicted_probs: np.ndarray, true_events: np.ndarray, num_bins: int, ax):
    """Binned reliability diagram of predicted vs. empirical spatial exceedance probability."""
    pred_flat = np.asarray(predicted_probs, dtype=np.float64).ravel()
    true_flat = np.asarray(true_events, dtype=np.float64).ravel()

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idx = np.clip(np.digitize(pred_flat, bins) - 1, 0, num_bins - 1)
    bin_pred, bin_true = [], []
    for b in range(num_bins):
        mask = bin_idx == b
        if mask.any():
            bin_pred.append(pred_flat[mask].mean())
            bin_true.append(true_flat[mask].mean())

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal calibration")
    ax.plot(bin_pred, bin_true, "o-", color="tab:green", label="Spatial exceedance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Predicted exceedance probability $\pi_{i,m}$")
    ax.set_ylabel("Empirical exceedance frequency")
    ax.set_title("Spatial Exceedance Reliability Diagram")
    ax.legend()
    return ax


def calc_spatial_coverage(y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """
    Empirical spatial (joint) central coverage: the fraction of instances
    where EVERY spatial dimension falls within its own central interval,
    P_hat_c = (1/N) sum_i 1[forall d, y_{i,d} in [q_lower_{i,d}, q_upper_{i,d}]].

    Args:
        y_true, q_lower, q_upper: (n_samples, D) arrays.

    Returns:
        Scalar empirical coverage in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    q_lower = np.asarray(q_lower, dtype=np.float64)
    q_upper = np.asarray(q_upper, dtype=np.float64)
    inside = (y_true >= q_lower) & (y_true <= q_upper)  # (n_samples, D)
    return float(inside.all(axis=1).mean())


def plot_spatial_coverage_curve(y_true: np.ndarray, quantile_func, nominal_coverages: np.ndarray, ax):
    """
    Spatial central coverage curve: for each nominal coverage c = 1 - alpha,
    query `quantile_func(alpha)` for the per-dimension central interval
    [Q(alpha/2), Q(1 - alpha/2)] and plot the empirical joint coverage
    (`calc_spatial_coverage`) against c. Points below the y = x diagonal mean
    the joint intervals are too narrow (overconfident independence copula).

    Args:
        y_true: (n_samples, D) array of observed targets.
        quantile_func: callable, quantile_func(alpha) -> (q_lower, q_upper),
            each (n_samples, D), the per-dimension alpha/2 and 1 - alpha/2
            predictive quantiles.
        nominal_coverages: (n_levels,) array of nominal coverages c in (0, 1).
        ax: matplotlib axis to draw on.
    """
    nominal_coverages = np.asarray(nominal_coverages, dtype=np.float64)
    empirical = np.empty_like(nominal_coverages)
    for i, c in enumerate(nominal_coverages):
        alpha = 1.0 - c
        q_lower, q_upper = quantile_func(alpha)
        empirical[i] = calc_spatial_coverage(y_true, q_lower, q_upper)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal calibration")
    ax.plot(nominal_coverages, empirical, "o-", color="tab:purple", label="Spatial coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Nominal joint coverage $c = 1 - \alpha$")
    ax.set_ylabel("Empirical joint coverage")
    ax.set_title("Spatial Coverage Curve")
    ax.legend()
    return ax
