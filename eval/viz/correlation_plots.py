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
    "plot_residual_grid",
    "plot_synthetic_residual_grid",
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
    axes = [axes] if n_cols == 1 else list(axes)

    # cbar=False on EVERY panel, with one shared colorbar attached to all of
    # them afterwards (the same construction plot_correlation_heatmaps above
    # uses). Drawing the colorbar inside the last heatmap call -- the obvious
    # thing, and what this did originally -- takes the colorbar's width out of
    # that ONE axes, and square=True then shrinks its height to match, so the
    # last matrix renders visibly smaller than the others: measured 2.92in vs
    # 3.56in (18% smaller) for two identical 64x64 inputs. That reads as a
    # shape mismatch between the oracle and the estimator when both matrices
    # are the same size by construction, which is exactly how it was reported.
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
            cbar=False,
        )
        color = "red" if lbl == "oracle" else "black"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2 if lbl == "oracle" else 1)
        ax.set_title(lbl, fontsize=9)

    # fraction/pad are taken off every axes equally, so the panels stay the
    # same size as each other. No tight_layout(): fig.colorbar(ax=axes) has
    # already re-laid the axes out, and calling it afterwards re-introduces
    # per-axes width differences.
    fig.colorbar(axes[-1].collections[0], ax=axes, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    return fig


def _plot_field_grid(
    lat: np.ndarray, lon: np.ndarray, grid_shape: tuple, true_fields: list, col_titles: list,
    output_path: "str | None", row0_label: str, suptitle: str,
    predicted_fields: "list[np.ndarray] | None" = None,
    predicted_fields_2: "list[np.ndarray] | None" = None,
    independent_fields: "list[np.ndarray] | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
    context_coords: "np.ndarray | None" = None,
    pred_row_label: str = "Copula model\n(predicted)\n +marginal\nLatitude",
    pred2_row_label: str = "Copula model\n(2nd variant)\nLatitude",
    indep_row_label: str = "Independent\n(no copula)\nLatitude",
    oracle_row_label: str = "Oracle correlation\n+ marginal\nLatitude",
    xlabel: str = "Longitude", cbar_label: str = "Residual (deg C)",
):
    """Shared small-multiples renderer behind both plot_residual_grid (real
    ERA5 days) and plot_synthetic_residual_grid (synthetic GP draws): top row
    `true_fields`, then an optional oracle-correlation row, then optional
    predicted/predicted_2/independent rows below (flat (D,) arrays, reshaped
    to grid_shape here). `oracle_fields`/`predicted_fields_2` are only ever
    passed by plot_synthetic_residual_grid. All rows share one color scale
    and a Moran's I annotation (see eval.spatial.diagnostics.morans_i).

    `output_path=None` skips the save-to-disk step and returns the open
    Figure instead (e.g. for src/train.py's live wandb.Image logging, which
    has no use for an on-disk copy) -- otherwise behaves exactly as before:
    saves, closes, prints, returns None."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eval.spatial.diagnostics import morans_i

    has_pred = predicted_fields is not None
    has_pred2 = predicted_fields_2 is not None
    has_indep = independent_fields is not None
    has_oracle = oracle_fields is not None
    pred_grids = [f.reshape(grid_shape) for f in predicted_fields] if has_pred else []
    pred2_grids = [f.reshape(grid_shape) for f in predicted_fields_2] if has_pred2 else []
    indep_grids = [f.reshape(grid_shape) for f in independent_fields] if has_indep else []
    oracle_grids = [f.reshape(grid_shape) for f in oracle_fields] if has_oracle else []

    vmax = float(np.max(np.abs(true_fields + pred_grids + pred2_grids + indep_grids + oracle_grids)))

    def _annotate_morans_i(ax, field):
        ax.text(
            0.97, 0.95, f"$I$={morans_i(field):.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    n_cols = len(true_fields)
    n_rows = 1 + int(has_oracle) + int(has_pred) + int(has_pred2) + int(has_indep)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), sharex=True, sharey=True, squeeze=False)
    mesh = None
    for j, (title, field) in enumerate(zip(col_titles, true_fields)):
        mesh = axes[0][j].pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        axes[0][j].set_title(title, fontsize=9)
        _annotate_morans_i(axes[0][j], field)
    axes[0][0].set_ylabel(row0_label)

    def _plot_row(row_idx, grids, ylabel):
        for j, field in enumerate(grids):
            mesh_local = axes[row_idx][j].pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            _annotate_morans_i(axes[row_idx][j], field)
            if context_coords is not None:
                axes[row_idx][j].scatter(
                    context_coords[:, 0], context_coords[:, 1],
                    c="black", s=8, marker="o", linewidths=0.4, edgecolors="white",
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
    data: dict, days: list, predicted_fields: "list[np.ndarray] | None", output_path: "str | None",
    context_coords: "np.ndarray | None" = None,
    independent_fields: "list[np.ndarray] | None" = None,
    target: str = "raw",
):
    """Small-multiples panel of the `target` field (raw temperature Z_t by
    default, or the 24h persistence residual E_t = Z_t - Z_{t-24}) on the
    lat/lon grid, one column per day in `days`: top row is the ground-truth
    field; if `predicted_fields` is given, the next row is the copula
    model's predicted field for that SAME day (see
    eval.spatial.diagnostics.predict_copula_residual_field); if
    `independent_fields` is also given, a third row shows the SAME
    marginal-per-point prediction with the copula's cross-location
    correlation switched off (R replaced by the identity), isolating what
    the learned correlation structure itself adds on top of the per-point
    marginal. If `context_coords` is given, the real-context locations that
    condition both predicted rows are overlaid as black markers on those
    rows only.

    `data["t2m"]` may be either the full (n_time, H, W) array `load_era5_data`
    returns (`days` then indexes positionally into it) or a plain
    ``{day: (H, W) frame}`` dict covering just `days` (src/train.py's
    validate() builds one of these instead of retaining every fetched day) --
    both support the same `data["t2m"][d]` lookup this function relies on.

    `output_path=None` returns the open Figure instead of saving it to disk
    (see `_plot_field_grid`); otherwise returns None as before.
    """
    lat, lon = data["latitude"], data["longitude"]
    grid_shape = data["t2m"][days[0]].shape
    if target == "residual":
        true_fields = [data["t2m"][d] - data["t2m"][d - 1] for d in days]
        col_titles = [f"day {d}: $E_t = Z_{{{d}}} - Z_{{{d - 1}}}$" for d in days]
        suptitle = (
            "24h Persistence Residual Fields: Ground Truth vs. Copula Model Prediction" if predicted_fields is not None
            else "24h Persistence Residual Fields (ground-truth input to $R_{emp}$ and real-context conditioning)"
        )
        cbar_label = "Residual (deg C)"
    else:
        true_fields = [data["t2m"][d] for d in days]
        col_titles = [f"day {d}: $Z_{{{d}}}$ (raw)" for d in days]
        suptitle = (
            "Raw Temperature Fields: Ground Truth vs. Copula Model Prediction" if predicted_fields is not None
            else "Raw Temperature Fields (ground-truth input to $R_{emp}$ and real-context conditioning)"
        )
        cbar_label = "Temperature (deg C)"
    return _plot_field_grid(
        lat, lon, grid_shape, true_fields, col_titles, output_path,
        row0_label="Ground truth\nLatitude", suptitle=suptitle,
        predicted_fields=predicted_fields, independent_fields=independent_fields, context_coords=context_coords,
        cbar_label=cbar_label,
    )


def plot_synthetic_residual_grid(
    grid_y: np.ndarray, grid_x: np.ndarray, grid_shape: tuple,
    true_fields: list, predicted_fields_true_z: list, predicted_fields_tabicl_z: list,
    independent_fields: list,
    output_path: str, context_coords: "np.ndarray | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
    pred2_row_label: str = "Copula model\n(TabICLv2 z_train)\ny",
    oracle_row_label: str = "Oracle correlation\n+ TabICLv2 marginal\ny",
    suptitle: str = "Synthetic GP Draws: Ground Truth vs. Copula Model Prediction",
) -> "plt.Figure | None":
    """Synthetic-mode analogue of plot_residual_grid: one column per
    independent GP draw from the sampled ground-truth kernel instead of one
    column per real ERA5 day, with the SAME five rows (ground truth /
    oracle-correlation+marginal / copula-predicted-with-true-z_train /
    copula-predicted-with-TabICLv2-estimated-z_train / independent) and
    shared color scale / Moran's I annotation, via the same _plot_field_grid
    renderer plot_residual_grid uses. `oracle_fields` reuses the TabICLv2
    marginal from the predicted rows but with the TRUE kernel correlation
    matrix instead of the model's R_pred, isolating correlation-estimation
    error from marginal-estimation error. `predicted_fields_true_z` and
    `predicted_fields_tabicl_z` are the SAME copula-model forward pass,
    differing only in whether z_train is the exact GP leave-one-out value or
    TabICLv2's practical K-fold PIT estimate of it.

    `pred2_row_label`/`oracle_row_label`/`suptitle` exist so the ANALYTIC-only
    regime can reuse this exact renderer with different row semantics. There,
    no TabICL PIT exists at all, so the row-2 slot carries the best-performing
    fitted kernel baseline instead of a TabICLv2-z_train variant, and the
    oracle row carries the exact analytic marginal rather than TabICLv2's --
    same five rows, same shared color scale and Moran's I annotation, only the
    labels differ (see eval/runners/analytic_copula_report.py). Defaults
    reproduce the original TabICL-era labels exactly, so existing callers are
    unaffected.

    Returns whatever _plot_field_grid returns: None when `output_path` is a
    path (saved and closed), or the open Figure when it is None."""
    col_titles = [f"draw {i + 1}" for i in range(len(true_fields))]
    return _plot_field_grid(
        grid_y, grid_x, grid_shape, true_fields, col_titles, output_path,
        row0_label="Ground truth\n(synthetic kernel)\ny", suptitle=suptitle,
        predicted_fields=predicted_fields_true_z, predicted_fields_2=predicted_fields_tabicl_z,
        independent_fields=independent_fields, context_coords=context_coords,
        oracle_fields=oracle_fields,
        pred_row_label="Copula model\n(true z_train)\ny",
        pred2_row_label=pred2_row_label,
        indep_row_label="Independent\n(no copula)\ny",
        oracle_row_label=oracle_row_label,
        xlabel="x", cbar_label="Field value",
    )
