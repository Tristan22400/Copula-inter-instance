"""Correlation-matrix plots: pair distances, correlation-vs-distance curves, heatmaps and field grids."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import torch

__all__ = [
    "collect_pair_distances_and_values",
    "plot_correlation_vs_distance",
    "plot_correlation_heatmaps",
    "plot_corr_grid",
    "plot_residual_grid",
    "plot_synthetic_residual_grid",
    "plot_z_predictor_samples",
    "plot_marginal_variance_grid",
]


def collect_pair_distances_and_values(X_norm: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every pair i < j, the distance between X_norm rows and M[i, j]."""
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
    """Binned mean correlation vs pairwise distance, one line per series.

    Args:
        series: {name: (distances, values)}, pooled across episodes.
        scatter_series: optionally also scatter the raw pairs of one series.
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
    """Correlation heatmaps side by side, one per method, with a shared colorbar."""
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
    """Heatmaps of the oracle R_star (first, red border) and each estimator's R for one episode.

    Args:
        estimators: {label: (N, N) tensor}.
        oracle_R: (N, N) tensor.
        title: figure title.
        max_show: subsample to at most this many points.

    Returns:
        matplotlib Figure.
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


def _plot_field_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_shape: tuple,
    true_fields: list,
    col_titles: list,
    output_path: "str | None",
    row0_label: str,
    suptitle: str,
    predicted_fields: "list[np.ndarray] | None" = None,
    predicted_fields_2: "list[np.ndarray] | None" = None,
    independent_fields: "list[np.ndarray] | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
    context_coords: "np.ndarray | None" = None,
    pred_row_label: str = "Copula model\n(predicted)\n +marginal\nLatitude",
    pred2_row_label: str = "Copula model\n(2nd variant)\nLatitude",
    indep_row_label: str = "Independent\n(no copula)\nLatitude",
    oracle_row_label: str = "Oracle correlation\n+ marginal\nLatitude",
    xlabel: str = "Longitude",
    cbar_label: str = "Residual (deg C)",
):
    """Render rows of fields on grid_shape with one color scale and Moran's I per panel.

    Rows: true_fields, then optional oracle, predicted, predicted_2 and
    independent rows (flat (D,) arrays). output_path=None returns the figure;
    otherwise saves, closes and returns None.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eval.spatial.diagnostics import morans_i

    has_pred = predicted_fields is not None
    has_pred2 = predicted_fields_2 is not None
    has_indep = independent_fields is not None
    has_oracle = oracle_fields is not None
    # Reshape true_fields to grid_shape (they may arrive flat).
    true_fields = [np.asarray(f).reshape(grid_shape) for f in true_fields]
    pred_grids = [f.reshape(grid_shape) for f in predicted_fields] if has_pred else []
    pred2_grids = [f.reshape(grid_shape) for f in predicted_fields_2] if has_pred2 else []
    indep_grids = [f.reshape(grid_shape) for f in independent_fields] if has_indep else []
    oracle_grids = [f.reshape(grid_shape) for f in oracle_fields] if has_oracle else []

    vmax = float(np.max(np.abs(true_fields + pred_grids + pred2_grids + indep_grids + oracle_grids)))

    def _annotate_morans_i(ax, field) -> None:
        ax.text(
            0.97,
            0.95,
            f"$I$={morans_i(field):.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    n_cols = len(true_fields)
    n_rows = 1 + int(has_oracle) + int(has_pred) + int(has_pred2) + int(has_indep)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    mesh = None
    for j, (title, field) in enumerate(zip(col_titles, true_fields)):
        mesh = axes[0][j].pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        axes[0][j].set_title(title, fontsize=9)
        _annotate_morans_i(axes[0][j], field)
    axes[0][0].set_ylabel(row0_label)

    def _plot_row(row_idx, grids, ylabel):
        for j, field in enumerate(grids):
            mesh_local = axes[row_idx][j].pcolormesh(
                lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
            )
            _annotate_morans_i(axes[row_idx][j], field)
            if context_coords is not None:
                axes[row_idx][j].scatter(
                    context_coords[:, 0],
                    context_coords[:, 1],
                    c="black",
                    s=8,
                    marker="o",
                    linewidths=0.4,
                    edgecolors="white",
                    label="Context points" if j == 0 else None,
                )
        axes[row_idx][0].set_ylabel(ylabel)
        if context_coords is not None:
            axes[row_idx][0].legend(loc="upper left", fontsize=6, framealpha=0.7)
        return mesh_local

    row = 1
    if has_oracle:
        mesh = _plot_row(row, oracle_grids, oracle_row_label)
        row += 1
    if has_pred:
        mesh = _plot_row(row, pred_grids, pred_row_label)
        row += 1
    if has_pred2:
        mesh = _plot_row(row, pred2_grids, pred2_row_label)
        row += 1
    if has_indep:
        mesh = _plot_row(row, indep_grids, indep_row_label)

    for j in range(n_cols):
        axes[-1][j].set_xlabel(xlabel)
    fig.suptitle(suptitle)
    plt.tight_layout(rect=(0.0, 0.0, 0.93, 0.96))
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.85, label=cbar_label)
    if output_path is None:
        return fig
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return None


def plot_residual_grid(
    data: dict,
    days: list,
    predicted_fields: "list[np.ndarray] | None",
    output_path: "str | None",
    context_coords: "np.ndarray | None" = None,
    predicted_fields_2: "list[np.ndarray] | None" = None,
    independent_fields: "list[np.ndarray] | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
    oracle_row_label: str = "Fitted GP posterior\n(fitted kernel)\nsample\nLatitude",
    pred2_row_label: str = "Second model\n(predicted)\nLatitude",
    target: str = "raw",
):
    """Grid of ERA5 fields, one column per day: ground truth (raw or 24 h residual), then optional reference (oracle_fields), model, second model and independent (R = I) rows.

    context_coords are overlaid on the model rows. data["t2m"] is an
    (n_time, H, W) array or a {day: (H, W)} dict. output_path=None returns the
    figure.
    """
    lat, lon = data["latitude"], data["longitude"]
    grid_shape = data["t2m"][days[0]].shape
    if target == "residual":
        true_fields = [data["t2m"][d] - data["t2m"][d - 1] for d in days]
        col_titles = [f"day {d}: $E_t = Z_{{{d}}} - Z_{{{d - 1}}}$" for d in days]
        suptitle = (
            "24h Persistence Residual Fields: Ground Truth vs. Copula Model Prediction"
            if predicted_fields is not None
            else "24h Persistence Residual Fields (ground-truth input to $R_{emp}$ and real-context conditioning)"
        )
        cbar_label = "Residual (deg C)"
    else:
        true_fields = [data["t2m"][d] for d in days]
        col_titles = [f"day {d}: $Z_{{{d}}}$ (raw)" for d in days]
        suptitle = (
            "Raw Temperature Fields: Ground Truth vs. Copula Model Prediction"
            if predicted_fields is not None
            else "Raw Temperature Fields (ground-truth input to $R_{emp}$ and real-context conditioning)"
        )
        cbar_label = "Temperature (deg C)"
    if oracle_fields is not None and predicted_fields is not None:
        suptitle += "\n(prediction rows are posterior SAMPLES on the same context and latent noise)"
    return _plot_field_grid(
        lat,
        lon,
        grid_shape,
        true_fields,
        col_titles,
        output_path,
        row0_label="Ground truth\nLatitude",
        suptitle=suptitle,
        predicted_fields=predicted_fields,
        predicted_fields_2=predicted_fields_2,
        independent_fields=independent_fields,
        oracle_fields=oracle_fields,
        oracle_row_label=oracle_row_label,
        pred2_row_label=pred2_row_label,
        context_coords=context_coords,
        cbar_label=cbar_label,
    )


def plot_mean_removed_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_shape: tuple,
    days: list,
    true_resid_fields: list[np.ndarray],
    output_path: "str | None",
    predicted_fields: "list[np.ndarray] | None" = None,
    predicted_fields_2: "list[np.ndarray] | None" = None,
    independent_fields: "list[np.ndarray] | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
    context_coords: "np.ndarray | None" = None,
    oracle_row_label: str = "Fitted GP posterior\nsample minus\nGP mean\nLatitude",
    pred2_row_label: str = "Fitted GP correlation\n+ TabICLv2 marginal\nsample minus\nmarginal mean\nLatitude",
):
    """Like plot_residual_grid with each row's own predictive mean subtracted at every location.

    output_path=None returns the figure.
    """
    col_titles = [f"day {d}" for d in days]
    return _plot_field_grid(
        lat,
        lon,
        grid_shape,
        true_resid_fields,
        col_titles,
        output_path,
        row0_label="Ground truth\nminus marginal mean\nLatitude",
        suptitle=(
            "Mean-Removed Residual Fields: Ground Truth vs. Copula Model Prediction\n"
            "(each row's own predictive mean subtracted at every location)"
        ),
        predicted_fields=predicted_fields,
        predicted_fields_2=predicted_fields_2,
        independent_fields=independent_fields,
        oracle_fields=oracle_fields,
        oracle_row_label=oracle_row_label,
        pred_row_label="Copula model\n(predicted) minus\nmarginal mean\nLatitude",
        pred2_row_label=pred2_row_label,
        indep_row_label="Independent\n(no copula) minus\nmarginal mean\nLatitude",
        context_coords=context_coords,
        cbar_label="Residual from predictive mean (deg C)",
    )


def plot_synthetic_residual_grid(
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    grid_shape: tuple,
    true_fields: list,
    predicted_fields_true_z: list,
    predicted_fields_tabicl_z: list,
    independent_fields: list,
    output_path: str,
    context_coords: "np.ndarray | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
) -> None:
    """Synthetic analogue of plot_residual_grid: one column per GP draw.

    Rows: ground truth, oracle correlation with the TabICL marginal, the model
    with the exact z_train, the model with TabICL's z_train, and independent.
    """
    col_titles = [f"draw {i + 1}" for i in range(len(true_fields))]
    _plot_field_grid(
        grid_y,
        grid_x,
        grid_shape,
        true_fields,
        col_titles,
        output_path,
        row0_label="Ground truth\n(synthetic kernel)\ny",
        suptitle="Synthetic GP Draws: Ground Truth vs. Copula Model Prediction",
        predicted_fields=predicted_fields_true_z,
        predicted_fields_2=predicted_fields_tabicl_z,
        independent_fields=independent_fields,
        context_coords=context_coords,
        oracle_fields=oracle_fields,
        pred_row_label="Copula model\n(true z_train)\ny",
        pred2_row_label="Copula model\n(TabICLv2 z_train)\ny",
        indep_row_label="Independent\n(no copula)\ny",
        oracle_row_label="Oracle correlation\n+ TabICLv2 marginal\ny",
        xlabel="x",
        cbar_label="Field value",
    )


def plot_z_predictor_samples(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_shape: tuple,
    day: int,
    independent_fields: list,
    predicted_fields: list,
    gp_fields: list,
    output_path: "str | None" = None,
    context_coords: "np.ndarray | None" = None,
):
    """Samples of the latent z for one day: rows are independent, copula-model and GP correlation, columns share white noise.

    independent_fields must already be (H, W).
    """
    col_titles = [f"day {day}: sample {i + 1}" for i in range(len(independent_fields))]
    suptitle = f"Copula Latent z-Samples (day {day}): Independent vs. Copula Model vs. GP Baseline"
    independent_grids = [f.reshape(grid_shape) for f in independent_fields]
    return _plot_field_grid(
        lat,
        lon,
        grid_shape,
        independent_grids,
        col_titles,
        output_path,
        row0_label="Independent\n(no copula)\nLatitude",
        suptitle=suptitle,
        predicted_fields=predicted_fields,
        oracle_fields=gp_fields,
        pred_row_label="Copula model\nLatitude",
        oracle_row_label="GP baseline\n(fitted correlation)\nLatitude",
        context_coords=context_coords,
        xlabel="Longitude",
        cbar_label="z (copula latent, std normal)",
    )


def plot_marginal_variance_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_shape: tuple,
    days: list,
    var_fields: list,
    gp_var_fields: "list | None" = None,
    output_path: "str | None" = None,
    context_coords: "np.ndarray | None" = None,
    gp_row_label: str = "Fitted GP\nposterior",
):
    """Per-location predictive variance (sequential colormap from 0) per day: the TabICL marginal and, optionally, the fitted GP; context overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_gp = gp_var_fields is not None
    grids = [f.reshape(grid_shape) for f in var_fields]
    gp_grids = [f.reshape(grid_shape) for f in gp_var_fields] if has_gp else []
    vmax = float(np.max(grids + gp_grids)) if (grids or gp_grids) else 1.0
    n_cols = len(grids)
    n_rows = 1 + int(has_gp)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    mesh = None

    def _plot_row(row_idx, row_grids, ylabel, show_col_titles) -> None:
        nonlocal mesh
        for j, field in enumerate(row_grids):
            mesh = axes[row_idx][j].pcolormesh(lon, lat, field, cmap="viridis", vmin=0.0, vmax=vmax, shading="auto")
            if show_col_titles:
                axes[row_idx][j].set_title(f"day {days[j]}", fontsize=9)
            if context_coords is not None:
                axes[row_idx][j].scatter(
                    context_coords[:, 0],
                    context_coords[:, 1],
                    c="red",
                    s=8,
                    marker="o",
                    linewidths=0.4,
                    edgecolors="white",
                    label="Context points" if j == 0 else None,
                )
        axes[row_idx][0].set_ylabel(ylabel)
        if context_coords is not None:
            axes[row_idx][0].legend(loc="upper left", fontsize=6, framealpha=0.7)

    _plot_row(0, grids, "TabICL marginal\nLatitude", show_col_titles=True)
    if has_gp:
        _plot_row(1, gp_grids, f"{gp_row_label}\nLatitude", show_col_titles=False)
    for j in range(n_cols):
        axes[-1][j].set_xlabel("Longitude")
    fig.suptitle("Predictive Variance vs. Distance from Context")
    plt.tight_layout(rect=(0.0, 0.0, 0.93, 0.94))
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.85, label="Var[y | x] (deg C²)")
    if output_path is None:
        return fig
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return None
