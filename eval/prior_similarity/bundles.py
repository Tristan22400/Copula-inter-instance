"""bundles.py — the common representation every prior-similarity indicator
consumes, plus one sampler per episode source.

A `FieldBundle` is "R realizations of a scalar field observed on D points":

    coords : (D, d)   the design the field lives on
    fields : (R, D)   R independent realizations of the field on that design

The correspondence R <-> "realization" is deliberately the *episode-level*
one, because that is what the copula head's prior is over:

  * real ERA5   : one realization = one calendar day over a fixed lat/lon box.
                  The copula's target R_post = Corr(y_test | context) is the
                  conditional of this across-day law, exactly the convention
                  eval/spatial/diagnostics.py::empirical_spatial_correlation
                  already uses.
  * synthetic   : one realization = one GP draw from the episode's own kernel.
                  Marginal-shape indicators instead use the episode's real
                  `y_test` (mean function included), one realization each, since
                  those are per-episode spatial statistics, not cross-episode.

Everything downstream is therefore source-agnostic: an indicator sees only
(coords, fields, grid_shape) and cannot special-case ERA5 vs. GP.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field as _field
from typing import Optional, Sequence

import numpy as np
from scipy.io.netcdf import netcdf_file

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

__all__ = ["FieldBundle", "ERA5Pool", "sample_synthetic_bundles", "sample_lattice_matern_bundles"]


@dataclass
class FieldBundle:
    """R realizations of a field on a shared D-point design. See module docstring."""

    coords: np.ndarray                      # (D, d) float64
    fields: np.ndarray                      # (R, D) float64
    source: str                             # "era5" | "synthetic_current" | ...
    grid_shape: Optional[tuple] = None      # (n_rows, n_cols) iff lattice-structured
    dist: Optional[np.ndarray] = None       # (D, D) precomputed pairwise distance
    R_prior_analytic: Optional[np.ndarray] = None  # (D, D), when the source knows it exactly
    # Realizations to use for PER-REALIZATION spatial statistics (Tier 3:
    # marginal shape, increment kurtosis, spectrum). Separate from `fields`
    # because a source may legitimately supply many *re-draws* for estimating
    # the cross-realization covariance while having only the genuine episodes
    # to offer for marginal shape -- see sample_synthetic_bundles, where the
    # re-draws are zero-mean and would silently wash out the episode's mean
    # function and feature warps. None means "same as fields".
    marginal_fields: Optional[np.ndarray] = None
    meta: dict = _field(default_factory=dict)

    def __post_init__(self):
        if self.marginal_fields is None:
            self.marginal_fields = self.fields

    @property
    def D(self) -> int:
        return int(self.coords.shape[0])

    @property
    def d(self) -> int:
        return int(self.coords.shape[1])

    @property
    def R(self) -> int:
        return int(self.fields.shape[0])


# ---------------------------------------------------------------------------
# Source 1: real ERA5
# ---------------------------------------------------------------------------


class ERA5Pool:
    """A RAM-resident pool of whole-globe daily 2m-temperature snapshots, so
    that many random (region, resolution) bundles can be cut from one disk
    read instead of re-reading the corpus per bundle.

    `n_months` whole months are drawn at random from the corpus (whole months,
    not scattered days, so each read is one contiguous file) and every day in
    them is kept. Drawing months at random across the full multi-decade
    archive keeps the seasonal mix representative, which matters: a large part
    of ERA5's raw across-day correlation IS the seasonal cycle.
    """

    def __init__(self, cache_dir: str, n_months: int = 40, seed: int = 0, verbose: bool = True):
        paths = sorted(glob.glob(os.path.join(cache_dir, "era5_global_t2m_*.nc")))
        if not paths:
            raise FileNotFoundError(
                f"No era5_global_t2m_*.nc under {cache_dir!r} — run "
                "`python eval/data/fetch_era5_global.py` first."
            )
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(paths), size=min(n_months, len(paths)), replace=False)
        chunks = []
        for j, i in enumerate(sorted(pick.tolist())):
            f = netcdf_file(paths[i], "r", mmap=True)
            chunks.append(f.variables["t2m"][:].astype(np.float32))
            if not hasattr(self, "lat"):
                self.lat = f.variables["latitude"][:].astype(np.float64).copy()
                self.lon = f.variables["longitude"][:].astype(np.float64).copy()
            f.close()
            if verbose and (j + 1) % 10 == 0:
                print(f"[ERA5Pool] read {j + 1}/{len(pick)} months")
        self.days = np.concatenate(chunks, axis=0)  # (R_total, H, W) float32
        self.month_paths = [paths[i] for i in sorted(pick.tolist())]
        # Static covariates (the 4 extra x-columns real episodes carry).
        from eval.data.fetch_era5_static import STATIC_VARS, load_static

        st = load_static()
        self.static_names = list(STATIC_VARS)
        self.static = np.stack([st[v].astype(np.float32) for v in STATIC_VARS], axis=0)  # (4, H, W)
        if verbose:
            print(f"[ERA5Pool] pool ready: {self.days.shape[0]} days, grid {self.days.shape[1:]}")

    def sample_bundle(
        self,
        rng: np.random.Generator,
        grid_size_range: tuple = (8, 28),
        box_deg_range: tuple = (5.0, 25.0),
        n_realizations: int = 1200,
        max_tries: int = 50,
    ) -> Optional[FieldBundle]:
        """One (region, resolution) bundle, drawn exactly the way
        era5_global_corpus.sample_episode draws its region/resolution:
        area-uniform box centre, uniform box width, evenly-spaced index
        decimation onto a grid_size x grid_size lattice.
        """
        H, W = self.days.shape[1], self.days.shape[2]
        for _ in range(max_tries):
            grid_size = int(rng.integers(grid_size_range[0], grid_size_range[1] + 1))
            box_deg = float(rng.uniform(*box_deg_range))
            half = box_deg / 2.0
            # Area-uniform latitude (arcsin of uniform), same as the corpus's
            # own justification for not over-representing the poles.
            lat_c = float(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0))))
            lat_c = float(np.clip(lat_c, -90.0 + half, 90.0 - half))
            lon_c = float(rng.uniform(0.0, 360.0))

            lat_mask = np.abs(self.lat - lat_c) <= half
            dlon = ((self.lon - lon_c + 180.0) % 360.0) - 180.0
            lon_mask = np.abs(dlon) <= half
            row_idx = np.nonzero(lat_mask)[0]
            col_idx = np.nonzero(lon_mask)[0]
            if len(row_idx) < grid_size or len(col_idx) < grid_size:
                continue
            row_pick = row_idx[np.linspace(0, len(row_idx) - 1, grid_size).round().astype(int)]
            col_pick = col_idx[np.linspace(0, len(col_idx) - 1, grid_size).round().astype(int)]

            R_total = self.days.shape[0]
            day_pick = rng.choice(R_total, size=min(n_realizations, R_total), replace=False)
            sub = self.days[np.ix_(day_pick, row_pick, col_pick)]  # (R, gs, gs)
            fields = sub.reshape(len(day_pick), -1).astype(np.float64)

            lon_grid, lat_grid = np.meshgrid(self.lon[col_pick], self.lat[row_pick])
            static_cols = np.stack(
                [self.static[k][np.ix_(row_pick, col_pick)].ravel() for k in range(self.static.shape[0])],
                axis=1,
            ).astype(np.float64)
            # d = 2 + len(STATIC_VARS) = 6, matching what a real episode actually
            # carries (era5_global_corpus.sample_episode_fixed_shape column-stacks
            # the static covariates onto lon/lat). Reporting d_x = 2 here -- as an
            # earlier version did by stashing the statics in `meta` -- understates
            # ERA5's dimensionality and makes the d comparison against the
            # synthetic prior wrong. No other indicator is affected: every
            # distance comes from `dist` below, which is haversine on lon/lat, and
            # the detrend/anisotropy/blocking helpers all read coords[:, :2].
            coords = np.column_stack(
                [lon_grid.ravel(), lat_grid.ravel(), static_cols]
            ).astype(np.float64)

            from eval.data.era5_io import haversine_distance_km

            return FieldBundle(
                coords=coords,
                fields=fields,
                source="era5",
                grid_shape=(grid_size, grid_size),
                dist=haversine_distance_km(coords),
                meta={
                    "grid_size": grid_size,
                    "box_deg": box_deg,
                    "lat_c": lat_c,
                    "lon_c": lon_c,
                    "land_frac": float(np.mean(static_cols[:, self.static_names.index("land_sea_mask")] > 0.5)),
                    "static": static_cols,
                    "n_days": int(len(day_pick)),
                },
            )
        return None


# ---------------------------------------------------------------------------
# Source 2: the synthetic prior as it exists today (src/data_gen.py)
# ---------------------------------------------------------------------------


def sample_synthetic_bundles(
    cfg,
    n_bundles: int,
    n_realizations: int = 1200,
    device: str = "cpu",
    seed: int = 0,
) -> list:
    """Bundles from the CURRENT training prior, via `data_gen.generate_gp_batch`.

    Each episode's test block is the design; `R_prior` (the episode's exact
    prior correlation over those points) is kept as `R_prior_analytic`, and
    `n_realizations` field draws are simulated from it so that the *estimation
    noise* in every cross-realization indicator matches the ERA5 side's (which
    only ever has a finite number of days). The episode's own `y_test` is kept
    as realization 0 so per-realization spatial indicators (marginal shape,
    increment kurtosis, Moran's I) see the real thing — mean function, warps
    and all — rather than a zero-mean re-draw.
    """
    import sys

    if os.path.join(_REPO_ROOT, "src") not in sys.path:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
    import torch
    from data_gen import generate_gp_batch

    bundles = []
    rng = np.random.default_rng(seed)
    b_per_call = 8
    call = 0
    while len(bundles) < n_bundles:
        cfg.seed = int(seed * 100003 + call)
        call += 1
        eps = generate_gp_batch(cfg, b_per_call, device=device)
        for ep in eps:
            if len(bundles) >= n_bundles:
                break
            x = ep["x_norm_test"].double().numpy()          # (N, d)
            y = ep["y_test"].double().numpy()               # (N,)
            R_prior = ep["R_prior"].double().numpy()        # (N, N)
            fields = _simulate_from_correlation(R_prior, n_realizations, rng)
            bundles.append(
                FieldBundle(
                    coords=x,
                    fields=fields,
                    # Only the episode's own y_test is a genuine realization of
                    # this prior *including* its mean function and feature
                    # warps; the re-draws above are zero-mean and would dilute
                    # exactly the marginal shape Tier 3 is asking about.
                    marginal_fields=y[None, :],
                    source="synthetic_current",
                    grid_shape=None,
                    dist=_euclidean(x),
                    R_prior_analytic=R_prior,
                    meta={"P": int(ep["x_norm_train"].shape[0]), "N": int(x.shape[0]), "d": int(x.shape[1])},
                )
            )
        del eps
        if device == "cuda":
            torch.cuda.empty_cache()
    return bundles


# ---------------------------------------------------------------------------
# Source 3: reference PROBE for the proposed ERA5-like prior.
#
# NOT wired into training and deliberately not in src/data_gen.py -- it exists
# only so the plan's claims ("a 2-D lattice design plus a two-scale
# anisotropic Matern moves indicators X, Y, Z") are measured rather than
# asserted. The production version, if approved, goes into data_gen.py as a
# geometry/kernel branch.
# ---------------------------------------------------------------------------


def sample_lattice_matern_bundles(
    n_bundles: int,
    n_realizations: int = 1200,
    seed: int = 0,
    grid_size_range: tuple = (8, 28),
    range_grid_units: tuple = (2.0, 30.0),
    nu_choices: Sequence[float] = (0.5, 1.0, 1.5, 2.5),
    aniso_ratio_range: tuple = (1.0, 3.0),
    large_scale_frac_range: tuple = (0.3, 0.9),
    nonstat_strength_range: tuple = (0.0, 1.2),
) -> list:
    """Probe generator: anisotropic, non-stationary, two-scale Matern field on
    a regular 2-D lattice.

    Three ingredients, each targeting one measured ERA5/prior gap:
      1. lattice design (fixes the geometry gap: nearest-neighbour regularity,
         and the range-to-spacing ratio that governs posterior rank);
      2. a `large_scale_frac` share of the variance in a near-constant
         synoptic/seasonal mode plus the rest in a shorter Matern (fixes the
         "one correlation scale" gap that makes the synthetic posterior
         low-rank);
      3. a smooth log-variance / log-range modulation field (fixes the
         stationarity gap: real boxes mix land/sea/orography regimes).
    """
    rng = np.random.default_rng(seed)
    bundles = []
    for _ in range(n_bundles):
        gs = int(rng.integers(grid_size_range[0], grid_size_range[1] + 1))
        ax = np.arange(gs, dtype=np.float64)
        gx, gy = np.meshgrid(ax, ax)
        coords = np.column_stack([gx.ravel(), gy.ravel()])
        D = coords.shape[0]

        # (2) two scales, in units of the lattice spacing (=1).
        L = float(np.exp(rng.uniform(np.log(range_grid_units[0]), np.log(range_grid_units[1]))))
        nu = float(rng.choice(np.asarray(nu_choices)))
        w_large = float(rng.uniform(*large_scale_frac_range))

        # (1)+(3) anisotropy + smooth deformation of the coordinates.
        ratio = float(np.exp(rng.uniform(0.0, np.log(aniso_ratio_range[1]))))
        theta = float(rng.uniform(0.0, np.pi))
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        A = rot @ np.diag([1.0, 1.0 / ratio]) @ rot.T
        cw = coords @ A.T

        dist = np.sqrt(np.maximum(((cw[:, None, :] - cw[None, :, :]) ** 2).sum(-1), 0.0))
        K_short = _matern_corr(dist, L, nu)
        # Large-scale mode: a very long-range component (range >> box), which
        # is what a seasonal/synoptic anomaly looks like inside one box.
        K_large = _matern_corr(dist, L * float(rng.uniform(5.0, 40.0)), 1.5)
        C = w_large * K_large + (1.0 - w_large) * K_short

        # (3) non-stationary marginal variance: smooth random log-sd field.
        s = float(rng.uniform(*nonstat_strength_range))
        if s > 0:
            g = rng.normal(size=D)
            smooth = _matern_corr(dist, max(gs / 3.0, 1.0), 1.5)
            w, V = np.linalg.eigh(smooth + 1e-8 * np.eye(D))
            logsd = V @ (np.sqrt(np.clip(w, 0, None)) * (V.T @ g))
            logsd = s * (logsd - logsd.mean()) / (logsd.std() + 1e-9)
            sd = np.exp(logsd)
            C = C * np.outer(sd, sd)

        # Simulate from the COVARIANCE, not from its correlation: normalizing
        # first would erase the non-stationary marginal variance that is the
        # entire point of the `nonstat_strength` knob (and would make
        # `nonstat_var_cv` structurally blind to it).
        corr = _cov_to_corr(C)
        fields = _simulate_from_covariance(C, n_realizations, rng)
        bundles.append(
            FieldBundle(
                coords=coords,
                fields=fields,
                source="lattice_matern_probe",
                grid_shape=(gs, gs),
                dist=_euclidean(coords),
                R_prior_analytic=corr,
                meta={"grid_size": gs, "L_grid_units": L, "nu": nu, "w_large": w_large,
                      "aniso_ratio": ratio, "nonstat_strength": s},
            )
        )
    return bundles


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _euclidean(x: np.ndarray) -> np.ndarray:
    sq = (x ** 2).sum(-1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _standardize(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def _cov_to_corr(C: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(s, s)


def _matern_corr(r: np.ndarray, L: float, nu: float) -> np.ndarray:
    from scipy.special import gamma as gamma_fn, kv as bessel_k

    out = np.ones_like(r)
    nz = r > 0
    x = np.sqrt(2.0 * nu) * r[nz] / L
    out[nz] = (2.0 ** (1.0 - nu) / gamma_fn(nu)) * (x ** nu) * bessel_k(nu, x)
    return np.nan_to_num(out, nan=1.0)


def _simulate_from_covariance(C: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n zero-mean draws with covariance C, via an eigen factor with negative
    eigenvalues clipped (matrices coming out of data_gen are PSD by
    construction but can be a hair outside it after a float32 round-trip)."""
    if n <= 0:
        return np.zeros((0, C.shape[0]))
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    Lf = V * np.sqrt(np.clip(w, 0.0, None))[None, :]
    return rng.normal(size=(n, C.shape[0])) @ Lf.T


# A correlation matrix is just a covariance whose diagonal is 1; kept as a
# separate name so call sites read as what they mean.
_simulate_from_correlation = _simulate_from_covariance
