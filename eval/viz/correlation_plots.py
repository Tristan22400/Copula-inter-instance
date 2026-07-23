"""correlation_plots.py — correlation-matrix diagnostics shared by every
benchmark runner: pairwise distance/value extraction plus the two plots
built from it (heatmaps, binned correlation-vs-distance)."""

from __future__ import annotations

import os

import numpy as np

__all__ = [
    "collect_pair_distances_and_values",
    "plot_correlation_vs_distance",
    "plot_correlation_heatmaps",
    "plot_corr_grid",
]


def collect_pair_distances_and_values(X_norm: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every i<j pair of test points, return the Euclidean distance
    between them in X_norm (the normalized-feature space every method's
    quantile/correlation query already operates in) and the matching entry
    M[i, j] — used both for correlation matrices (R) and for
    ground-truth-proxy matrices like outer(z, z).
    """
    n = X_norm.shape[0]
    iu = np.triu_indices(n, k=1)
    dists = np.linalg.norm(X_norm[iu[0]] - X_norm[iu[1]], axis=1)
    vals = M[iu]
    return dists, vals


def plot_correlation_vs_distance(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str,
    n_bins: int = 15,
    scatter_series: str | None = None,
) -> None:
    """Binned-mean correlation vs. pairwise distance, one line per series,
    pooled across every episode of a benchmark (a single episode rarely has
    enough pairs per distance bin to be meaningful on its own).

    Args:
        series: {series_name: (distances, values)} — both arrays already
            concatenated across all episodes of one benchmark. One series is
            typically "ground_truth" (analytical, when known) or
            "empirical_ground_truth" (PIT z_i*z_j proxy, for real datasets
            with no known generative kernel); the rest are method names.
        scatter_series: if given, also draw a faint raw-pair scatter for
            that one series (usually the ground-truth one) for visual
            context — omitted by default since 3+ overlapping scatters are
            unreadable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_dists = np.concatenate([d for d, _ in series.values()])
    bins = np.linspace(0.0, all_dists.max() + 1e-9, n_bins + 1)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    if scatter_series is not None and scatter_series in series:
        d, v = series[scatter_series]
        ax.scatter(d, v, s=2, alpha=0.05, color="gray", label=f"{scatter_series} (raw pairs)")

    line_styles = ["o-", "s-", "^-", "d--", "v-."]
    for (name, (d, v)), style in zip(series.items(), line_styles):
        bin_idx = np.clip(np.digitize(d, bins) - 1, 0, n_bins - 1)
        centers, means = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            centers.append(0.5 * (bins[b] + bins[b + 1]))
            means.append(v[mask].mean())
        ax.plot(centers, means, style, label=f"{name} (binned mean)", markersize=4)

    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("pairwise distance (normalized feature space)")
    ax.set_ylabel("correlation")
    ax.set_title("Correlation vs. distance")
    ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmaps(R_by_method: dict[str, np.ndarray], out_path: str) -> None:
    """Side-by-side correlation-matrix heatmaps, one subplot per method, shared colorbar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(R_by_method.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 4), squeeze=False)
    axes = axes[0]
    im = None
    for ax, method in zip(axes, methods):
        im = ax.imshow(R_by_method[method], vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_corr_grid(
    estimators: dict[str, "torch.Tensor"],
    oracle_R: "torch.Tensor",
    title: str = "",
    max_show: int = 40,
) -> "plt.Figure":
    """Side-by-side heatmaps of oracle R_star vs each estimator's predicted R,
    for a single episode — every fitted baseline plus the ICL model in one
    row, oracle first with a highlighted border.

    Unlike plot_correlation_heatmaps (numpy arrays, no designated "ground
    truth" panel, used to compare a handful of methods pooled/averaged across
    a whole benchmark), this takes torch tensors straight from
    eval_baselines_episode / the ICL forward pass for one episode, always
    puts oracle first with a red border, and subsamples down to max_show
    points so a >40-point episode's heatmap stays legible.

    Args:
        estimators : {label: (N, N) tensor}
        oracle_R   : (N, N) tensor — ground-truth correlation
        title      : overall figure title
        max_show   : max N to display (subsampled if larger)
    Returns:
        matplotlib Figure — caller is responsible for savefig/close.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ["oracle"] + list(estimators.keys())
    mats = [oracle_R.cpu().float()] + [v.cpu().float() for v in estimators.values()]

    N = oracle_R.shape[0]
    if N > max_show:
        import torch

        idx = torch.linspace(0, N - 1, max_show).long()
        mats = [m[idx][:, idx] for m in mats]

    n_cols = len(labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, lbl, R in zip(axes, labels, mats):
        R_np = R.numpy()
        sns.heatmap(
            R_np,
            ax=ax,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=lbl == labels[-1],
        )
        color = "red" if lbl == "oracle" else "black"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2 if lbl == "oracle" else 1)
        ax.set_title(lbl, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    return fig
