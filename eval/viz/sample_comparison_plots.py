"""sample_comparison_plots.py — one-dimensional sample-comparison plot for
the Copula Model vs. the autoregressive marginal-chain baseline
(src/autoregressive_baseline.py). Kept separate from correlation_plots.py,
which draws ERA5 lat/lon field grids (pcolormesh) — a different shape of
plot from this generic per-point line/scatter chart over an arbitrary GP
episode's covariates."""

from __future__ import annotations

import os

import numpy as np

__all__ = ["plot_sample_comparison"]


def plot_sample_comparison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test_true: np.ndarray,
    y_copula_sample: np.ndarray,
    y_ar_chain_sample: np.ndarray,
    output_path: str,
    title: str | None = None,
) -> None:
    """Plot one joint sample from each method against the true query
    targets, sorted by ``x_test``'s first covariate dimension.

    For ``d_x > 1`` this only shows one projection of the input space (the
    first column) — the other covariates are not represented on the x-axis.

    Args:
        x_train : (P, d_x) context inputs, raw scale.
        y_train : (P,) context targets, raw scale.
        x_test  : (N, d_x) query inputs, raw scale.
        y_test_true : (N,) true query targets, raw scale.
        y_copula_sample : (N,) one joint sample from the Copula Model,
            raw scale, same order as x_test.
        y_ar_chain_sample : (N,) one joint sample from the autoregressive
            marginal-chain baseline, raw scale, same order as x_test.
        output_path : PNG path to write.
        title : optional figure title (e.g. the episode's kernel label).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0_test = np.asarray(x_test)[:, 0]
    order = np.argsort(x0_test)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if x_train.shape[0] > 0:
        x0_train = np.asarray(x_train)[:, 0]
        ax.scatter(x0_train, y_train, s=18, color="gray", alpha=0.7, label="context (train)")
    ax.plot(x0_test[order], np.asarray(y_test_true)[order], "k.-", linewidth=1, markersize=4, label="true (test)")
    ax.plot(x0_test[order], np.asarray(y_copula_sample)[order], "o-", color="#4c72b0",
            linewidth=1, markersize=3, alpha=0.85, label="Copula Model sample")
    ax.plot(x0_test[order], np.asarray(y_ar_chain_sample)[order], "s-", color="#c44e52",
            linewidth=1, markersize=3, alpha=0.85, label="Autoregressive marginal-chain sample")
    ax.set_xlabel("x (first covariate)")
    ax.set_ylabel("y")
    ax.set_title(title or "Sample comparison: Copula Model vs. autoregressive marginal-chain")
    ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
