"""Constants and name lists for the evaluation runners and the spatial-correlation tooling.

Values that library code also needs live with their owner in copula_inter and
are re-exported here, so each number has a single definition.
"""

from __future__ import annotations

import numpy as np

from copula_inter.baselines.gp_mle import GP_LR_MLE as GP_LR_MLE
from copula_inter.era5.io import EARTH_RADIUS_KM as EARTH_RADIUS_KM
from copula_inter.era5.probe import MAX_DIST_PERCENTILE as MAX_DIST_PERCENTILE
from copula_inter.era5.probe import N_NLL_TEST as N_NLL_TEST
from copula_inter.era5.probe import PIT_K_FOLDS as PIT_K_FOLDS
from copula_inter.paths import ERA5_GLOBAL_VAL_CACHE_DIR

# eval_checkpoint choice lists, kept here so the config schema imports no model code.
AR_ORDERS = ("random", "natural")
AR_CONDITIONINGS = ("teacher_forcing", "sample")
GP_VAL_SELECT_MODES = ("ard", "always", "never")
# Held-out ERA5 year, disjoint from the training corpus.
ERA5_DEFAULT_CORPUS_DIR = str(ERA5_GLOBAL_VAL_CACHE_DIR)

N_CONTEXT = 30  # in-context sample size for --profile sweeps (see regions.SWEEP_PROFILES)
N_BINS = 15  # distance bins for correlation-vs-distance binning
SEED = 42
N_DAYS = 60  # daily ERA5 snapshots fetched per (region, grid_size)
N_SYNTHETIC_DRAWS = 20  # independent GP draws averaged per synthetic-mode config

# Joint y-space samples per probe day for the real-ERA5 model correlation curve (pooled across days).
N_YSPACE_MC_SAMPLES = 20

# Held-out points for the real-ERA5 joint-NLL diagnostic.
NLL_PROBS = np.linspace(0.02, 0.98, 49)  # quantile-grid probability levels for compute_joint_nll

# GP-MLE baseline settings for the real-ERA5 held-out NLL (eval_checkpoint's defaults).
GP_BASELINE_KERNELS = ["rbf", "matern12", "matern32", "matern52", "rational_quadratic"]
GP_N_STEPS_MLE = 1000
GP_N_RESTARTS_MLE = 5

# Theoretical-law names for the `baseline` curve fits.
CURVE_FIT_LAWS = ["gaussian", "matern", "rational_quadratic"]

# gp_episodes kernel families used as synthetic ground truth.
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
