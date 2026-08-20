"""constants.py — numeric constants and law/kernel name lists shared across
the spatial-correlation diagnostic/sweep/baseline/report tooling, single-
sourced here instead of duplicated across plots/*.py."""

from __future__ import annotations

import numpy as np

N_CONTEXT = 30              # in-context sample size for --profile sweeps (see regions.SWEEP_PROFILES)
N_BINS = 15                 # distance bins for correlation-vs-distance binning
SEED = 42
N_DAYS = 60                 # daily ERA5 snapshots fetched per (region, grid_size)
MAX_DIST_PERCENTILE = 90.0  # cap the binned distance range at this percentile (excludes the
                             # corner-only, high-variance tail of a bounded lat/lon rectangle)
PIT_K_FOLDS = 10            # K-fold leave-one-out PIT folds for real-context z_train estimation
N_SYNTHETIC_DRAWS = 20      # independent GP draws averaged per synthetic-mode config
EARTH_RADIUS_KM = 6371.0

# Total (marginal+copula) joint-NLL diagnostic (eval/metrics/joint_nll.py::
# compute_joint_nll), shared by compare_marginal_backbones.py and
# sweep_core.py::run_real_config -- neither the per-episode NLL tables in
# eval_checkpoint.py/run_benchmarks.py nor spatial_model_r2 (a binned
# correlation-curve-shape diagnostic, not a proper scoring rule) cover this
# real-ERA5 / cross-backend setting.
N_NLL_TEST = 30             # held-out (never-in-context) points scored per task/day
NLL_PROBS = np.linspace(0.02, 0.98, 49)  # quantile-grid probability levels for compute_joint_nll

# Direct-curve-fit law names (eval.spatial.diagnostics.fit_theoretical_law's
# THEORY_LAWS keys, as fit by the `baseline` subcommand) -- a DIFFERENT
# concept from SYNTHETIC_SWEEP_KERNELS below (data_gen.py *generating*
# kernel names, not fit shapes).
CURVE_FIT_LAWS = ["gaussian", "matern", "rational_quadratic"]

# Kernel families sampled as synthetic-mode ground truth in `sweep --mode
# synthetic` / `diagnose --mode synthetic` (src/data_gen.py's registry).
SYNTHETIC_SWEEP_KERNELS = ["rbf", "matern12", "matern32", "periodic", "rational_quadratic"]

# Synthetic-mode analogue of regions.SWEEP_PROFILES["low_context_7config"]:
# 3 grid resolutions (kernel fixed = rbf) + 4 kernel families (grid fixed =
# 24) -- the profile plots/run_synthetic_checkpoint_comparison.py hardcoded.
SYNTHETIC_SWEEP_PROFILES = {
    "low_context_7config": [
        ("grid_08x08_rbf", "rbf", 8),
        ("grid_16x16_rbf", "rbf", 16),
        ("grid_24x24_rbf", "rbf", 24),
        ("kernel_matern12_g24", "matern12", 24),
        ("kernel_matern32_g24", "matern32", 24),
        ("kernel_periodic_g24", "periodic", 24),
        ("kernel_rational_quadratic_g24", "rational_quadratic", 24),
    ],
}
