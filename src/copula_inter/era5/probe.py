"""Frozen real-ERA5 probe for one region: empirical correlogram, a fixed PIT'd context sample and held-out points.

Built once and reused, by the training loop's era5_fit/<region> validation
(training.era5_probes), by Phase-A validation (marginal.finetune_data) and by
the spatial eval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from copula_inter.era5.fetch import fetch as fetch_era5
from copula_inter.era5.io import haversine_distance_km, load_era5_data
from copula_inter.era5.regions import REGIONS
from copula_inter.era5.spatial_stats import (
    bin_correlation_by_distance,
    empirical_spatial_correlation,
    pair_counts_by_distance,
)
from copula_inter.marginal.pit import compute_context_z_train, normalize_features

if TYPE_CHECKING:
    from copula_inter.marginal.pit import TabICLLike

__all__ = ["build_era5_probe", "MAX_DIST_PERCENTILE", "N_NLL_TEST", "PIT_K_FOLDS", "PROBE_SEED"]

PROBE_SEED = 42
MAX_DIST_PERCENTILE = 90.0  # cap the binned distance range at this percentile (excludes the
# corner-only, high-variance tail of a bounded lat/lon rectangle)
N_NLL_TEST = 30  # held-out (never-in-context) points scored per task/day
PIT_K_FOLDS = 10  # K-fold leave-one-out PIT folds for real-context z_train estimation


def build_era5_probe(
    region_name: str,
    grid_size: int,
    n_days_fetch: int,
    n_days_probe: int,
    n_context: int,
    n_bins: int,
    tabicl_marginal: TabICLLike | None,
    device: str,
    seed: int = PROBE_SEED,
) -> dict:
    """Frozen real-ERA5 probe for region_name, computed once for the training loop.

    Holds the empirical correlogram (rho_emp), a fixed context sample with its
    PIT z_train (when tabicl_marginal is given), and held-out points
    (nll_test_idx) with their values on the same days, for validate()'s
    Y-space NLL.
    """
    rng = np.random.default_rng(seed)
    lat_bounds, lon_bounds = REGIONS[region_name]
    nc_path = fetch_era5(region_name, lat_bounds, lon_bounds, grid_size, n_days_fetch)
    data = load_era5_data(nc_path)
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    D = coords.shape[0]

    static_cols = []
    for v in (
        "geopotential_at_surface",
        "land_sea_mask",
        "standard_deviation_of_orography",
        "slope_of_sub_gridscale_orography",
    ):
        if v in data:
            static_cols.append(data[v].ravel())
    if static_cols:
        features = np.column_stack([lon_grid.ravel(), lat_grid.ravel()] + static_cols)
    else:
        features = coords

    R_emp = empirical_spatial_correlation(data, target="raw")
    dist = haversine_distance_km(coords)
    dist_iu = dist[np.triu_indices_from(dist, k=1)]
    max_dist = np.percentile(dist_iu, MAX_DIST_PERCENTILE)
    bin_edges = np.linspace(0.0, max_dist, n_bins + 1)
    dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pair_counts = pair_counts_by_distance(dist, bin_edges).astype(float)
    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)

    n_time = data["t2m"].shape[0]
    n_pick = min(n_days_probe, n_time)
    days = sorted(set(np.linspace(0, n_time - 1, n_pick).round().astype(int).tolist()))

    n_context_eff = max(1, min(n_context, D - 1))
    context_idx = rng.choice(D, size=n_context_eff, replace=False)
    context_features = features[context_idx]

    x_train_norm, x_test_norm = normalize_features(context_features, features)

    # Held-out points for the Y-space NLL (same construction as run_real_config).
    remaining_idx = np.setdiff1d(np.arange(D), context_idx)
    nll_test_idx = rng.choice(remaining_idx, size=min(N_NLL_TEST, len(remaining_idx)), replace=False)
    x_nll_test_norm = x_test_norm[nll_test_idx]

    z_train_per_day = []
    context_values_per_day = []
    nll_test_values_per_day = []
    for d in days:
        day_values = data["t2m"][d].ravel()
        context_values = day_values[context_idx]
        z_train = compute_context_z_train(
            x_train_norm,
            context_values,
            tabicl_marginal,
            device,
            k_folds=PIT_K_FOLDS,
        )
        z_train_per_day.append(z_train)
        context_values_per_day.append(context_values)
        nll_test_values_per_day.append(day_values[nll_test_idx])

    return {
        "region": region_name,
        "x_train_norm": x_train_norm,  # (P, 2)
        "x_test_norm": x_test_norm,  # (D, 2)
        "z_train_per_day": np.stack(z_train_per_day, axis=0),  # (n_days_probe, P)
        "x_nll_test_norm": x_nll_test_norm,  # (n_nll_test, 2)
        "nll_test_idx": nll_test_idx,  # (n_nll_test,) indices into x_test_norm/coords
        "context_values_per_day": np.stack(context_values_per_day, axis=0),  # (n_days_probe, P) raw t2m
        "nll_test_values_per_day": np.stack(nll_test_values_per_day, axis=0),  # (n_days_probe, n_nll_test) raw t2m
        "dist": dist,  # (D, D)
        "bin_edges": bin_edges,
        "dist_centers": dist_centers,
        "pair_counts": pair_counts,
        "rho_emp": rho_emp,
        "D": int(D),
        "n_context": int(n_context_eff),
    }
