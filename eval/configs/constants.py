"""Constants and name lists for the spatial-correlation tooling."""

from __future__ import annotations

import numpy as np

N_CONTEXT = 30  # in-context sample size for --profile sweeps (see regions.SWEEP_PROFILES)
N_BINS = 15  # distance bins for correlation-vs-distance binning
SEED = 42
N_DAYS = 60  # daily ERA5 snapshots fetched per (region, grid_size)
MAX_DIST_PERCENTILE = 90.0  # cap the binned distance range at this percentile (excludes the
# corner-only, high-variance tail of a bounded lat/lon rectangle)
PIT_K_FOLDS = 10  # K-fold leave-one-out PIT folds for real-context z_train estimation
N_SYNTHETIC_DRAWS = 20  # independent GP draws averaged per synthetic-mode config
EARTH_RADIUS_KM = 6371.0

# Joint y-space samples per probe day for the real-ERA5 model correlation curve (pooled across days).
N_YSPACE_MC_SAMPLES = 20

# Held-out points for the real-ERA5 joint-NLL diagnostic.
N_NLL_TEST = 30  # held-out (never-in-context) points scored per task/day
NLL_PROBS = np.linspace(0.02, 0.98, 49)  # quantile-grid probability levels for compute_joint_nll

# GP-MLE baseline settings for the real-ERA5 held-out NLL (eval_checkpoint's defaults).
GP_BASELINE_KERNELS = ["rbf", "matern12", "matern32", "matern52", "rational_quadratic"]
GP_N_STEPS_MLE = 1000
GP_LR_MLE = 0.05
GP_N_RESTARTS_MLE = 5

# Theoretical-law names for the `baseline` curve fits.
CURVE_FIT_LAWS = ["gaussian", "matern", "rational_quadratic"]

# data_gen kernel families used as synthetic ground truth.
SYNTHETIC_SWEEP_KERNELS = ["rbf", "matern12", "matern32", "periodic", "rational_quadratic"]

# Synthetic sweep profile: 3 grid sizes (rbf) and 4 kernel families (grid 24).
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
