"""indicators.py — the similarity indicator suite.

Every indicator is a function of a `FieldBundle` alone, so real ERA5 and any
synthetic prior go through identical code. Indicators are grouped in tiers by
what they are diagnostic OF; each returns plain floats so the whole suite
reduces to one row per bundle and the real-vs-synthetic gap becomes a table.

  Tier 0  geometry        the design, before any field values are looked at
  Tier 1  second order    the prior correlation structure (shape, range, law)
  Tier 2  posterior       what the copula head actually has to output
  Tier 3  marginal/HO     Gaussian-copula adequacy and marginal shape
  Tier 4  aggregate       (in the runner) two-sample separability of the above

DISTANCE UNITS. Distances are always reported in units of the design's own
MEDIAN NEAREST-NEIGHBOUR SPACING. That is the only unit in which a scattered
10-d Gaussian point cloud and a 24x24 lat/lon lattice are comparable at all,
and it is also the physically meaningful one here: `range_over_nn` (how many
sampled points fit inside one correlation length) is what decides whether the
conditional correlation the model must emit is low-rank and smooth or
near-full-rank and rough. See [[project_era5_copula_rank_ceiling]].
"""

from __future__ import annotations

import numpy as np

from eval.spatial.diagnostics import (
    bin_correlation_by_distance,
    fit_theoretical_law,
    morans_i,
    pair_counts_by_distance,
)

__all__ = ["compute_all", "INDICATOR_TIERS", "correlogram"]

N_BINS = 16
MAX_DIST_PERCENTILE = 90.0


# ---------------------------------------------------------------------------
# shared reductions
# ---------------------------------------------------------------------------


def _nn_spacing(dist: np.ndarray) -> np.ndarray:
    """Per-point nearest-neighbour distance."""
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _empirical_corr(fields: np.ndarray, shrink: bool = True) -> np.ndarray:
    """Correlation across realizations (fields are (R, D)), Ledoit-Wolf
    shrunk toward the identity.

    Shrinkage is not optional here. Raw across-day ERA5 correlation is ~0.99
    between every pair of points inside a box (that is the seasonal cycle, a
    near-rank-1 mode), so the sample correlation is desperately
    ill-conditioned and a raw Schur complement against it produces posterior
    variances at the 1e-9 level whose "correlations" then blow up to 1e7.
    Both sources go through the same estimator with the same number of
    realizations, so the comparison stays fair.
    """
    X = fields - fields.mean(axis=0, keepdims=True)
    sd = X.std(axis=0) + 1e-12
    Xs = X / sd
    if not shrink:
        C = (Xs.T @ Xs) / max(X.shape[0] - 1, 1)
        C[np.diag_indices_from(C)] = 1.0
        return C
    from sklearn.covariance import ledoit_wolf

    C, _ = ledoit_wolf(Xs, assume_centered=True)
    s = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    C = C / np.outer(s, s)
    C[np.diag_indices_from(C)] = 1.0
    return C


def _lw_shrinkage(fields: np.ndarray) -> float:
    """The shrinkage intensity itself, as an indicator: how far the field's
    covariance is from being estimable at this realization count -- i.e. how
    concentrated its spectrum is."""
    from sklearn.covariance import ledoit_wolf_shrinkage

    X = fields - fields.mean(axis=0, keepdims=True)
    return float(ledoit_wolf_shrinkage(X / (X.std(axis=0) + 1e-12), assume_centered=True))


def _psd_corr(C: np.ndarray, floor_frac: float = 1e-6) -> np.ndarray:
    """Correlation matrix of a covariance that conditioning may have pushed
    marginally indefinite: symmetrize, clip eigenvalues at 0, floor the
    diagonal relative to its own mean before normalizing."""
    C = (C + C.T) / 2.0
    w, V = np.linalg.eigh(C)
    if w.min() < 0:
        C = (V * np.clip(w, 0.0, None)) @ V.T
    dg = np.diag(C).copy()
    dg = np.clip(dg, floor_frac * max(dg.mean(), 1e-12), None)
    s = np.sqrt(dg)
    R = C / np.outer(s, s)
    return np.clip(R, -1.0, 1.0)


def _detrend_plane(bundle) -> np.ndarray:
    """Per-realization removal of the best-fit constant+linear plane in the
    first two coordinate columns. Cheap stand-in for deseasonalization that is
    defined identically for both sources: it strips the single dominant
    large-scale mode and leaves the mesoscale structure the model has to get
    right once context has pinned the large scale down."""
    X = np.column_stack([np.ones(bundle.D), bundle.coords[:, :2]])
    Q, _ = np.linalg.qr(X)
    F = bundle.fields
    return F - (F @ Q) @ Q.T


def correlogram(R: np.ndarray, dist: np.ndarray, nn: float, n_bins: int = N_BINS):
    """(bin centres in NN units, mean correlation per bin, pair counts)."""
    iu = np.triu_indices_from(dist, k=1)
    dmax = np.percentile(dist[iu], MAX_DIST_PERCENTILE)
    edges = np.linspace(0.0, dmax, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rho = bin_correlation_by_distance(R, dist, edges)
    counts = pair_counts_by_distance(dist, edges)
    return centres / nn, rho, counts


def _crossing(centres: np.ndarray, rho: np.ndarray, level: float) -> float:
    """First distance at which the binned curve drops through `level`, by
    linear interpolation; NaN if it never does inside the measured range."""
    ok = np.isfinite(rho)
    c, r = centres[ok], rho[ok]
    below = np.where(r <= level)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(c[0])
    r0, r1, c0, c1 = r[i - 1], r[i], c[i - 1], c[i]
    if r0 == r1:
        return float(c0)
    return float(c0 + (level - r0) / (r1 - r0) * (c1 - c0))


def _eigen_stats(C: np.ndarray, prefix: str) -> dict:
    """Cumulative eigenvalue mass at the ranks the copula head can actually
    represent. `model.rank` is 32 today, so evr32 is the share of the target
    correlation a rank-32 covnorm head could capture at best; 1 - evr32 is the
    independent white-noise floor it is forced to leave behind."""
    w = np.linalg.eigvalsh((C + C.T) / 2.0)[::-1]
    w = np.clip(w, 0.0, None)
    tot = w.sum() + 1e-12
    cum = np.cumsum(w) / tot
    D = len(w)
    p = w / tot
    ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
    out = {f"{prefix}_eff_rank": float(np.exp(ent)), f"{prefix}_eff_rank_frac": float(np.exp(ent) / D)}
    for k in (8, 32, 64, 128):
        out[f"{prefix}_evr{k}"] = float(cum[min(k, D) - 1])
    return out


def _offdiag(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(C, k=1)
    return C[iu]


# ---------------------------------------------------------------------------
# Tier 0 — design geometry
# ---------------------------------------------------------------------------


def tier0_geometry(bundle) -> dict:
    nn = _nn_spacing(bundle.dist)
    iu = np.triu_indices_from(bundle.dist, k=1)
    pair = bundle.dist[iu]
    med_nn = float(np.median(nn))
    return {
        "D": float(bundle.D),
        "d_x": float(bundle.d),
        "R_realizations": float(bundle.R),
        # How REGULAR the design is: 0 for a perfect lattice, ~0.52 for a
        # Poisson process, larger for a Gaussian cloud whose density varies.
        "nn_cv": float(np.std(nn) / (med_nn + 1e-12)),
        # How SPREAD the design is relative to its own spacing: sqrt(D) for a
        # 2-D lattice, much smaller for a high-d cloud where every point is
        # roughly equidistant from every other (the curse-of-dimensionality
        # signature).
        "spread_over_nn": float(np.median(pair) / (med_nn + 1e-12)),
        "maxdist_over_nn": float(np.percentile(pair, 99) / (med_nn + 1e-12)),
        "_med_nn": med_nn,
    }


# ---------------------------------------------------------------------------
# Tier 1 — prior second-order structure
# ---------------------------------------------------------------------------


def tier1_second_order(bundle, med_nn: float, detrended: bool = False) -> dict:
    fields = _detrend_plane(bundle) if detrended else bundle.fields
    C = _empirical_corr(fields)
    centres, rho, counts = correlogram(C, bundle.dist, med_nn)
    tag = "det_" if detrended else ""

    out = {
        f"{tag}rho_at_1nn": float(np.interp(1.0, centres[np.isfinite(rho)], rho[np.isfinite(rho)])),
        f"{tag}range_over_nn": _crossing(centres, rho, 0.5),
        f"{tag}range_e_over_nn": _crossing(centres, rho, 1.0 / np.e),
        f"{tag}rho_mean": float(np.nanmean(rho)),
        # Correlation at the far edge of the box. `range_over_nn` is NaN
        # whenever the field never decorrelates inside the sampled window --
        # a real and common ERA5 outcome, not a failure -- so this keeps the
        # far-field level measurable in those cases.
        f"{tag}rho_far": float(rho[np.isfinite(rho)][-1]) if np.isfinite(rho).any() else np.nan,
        f"{tag}decorrelates_in_box": float(np.isfinite(_crossing(centres, rho, 0.5))),
        f"{tag}od_mu": float(np.mean(np.abs(_offdiag(C)))),
        f"{tag}frac_pairs_gt_03": float(np.mean(np.abs(_offdiag(C)) > 0.3)),
    }
    # Law shape. fit_theoretical_law's lower bound on L is 1.0 in whatever
    # unit it is handed; our distances are in NN units where L can legitimately
    # be < 1 (a prior whose correlation dies inside one sample spacing), so fit
    # on a x1000 scale and convert back.
    fit = fit_theoretical_law(centres * 1000.0, rho, counts, "matern")
    if fit is not None:
        out[f"{tag}matern_L_over_nn"] = float(fit["params"]["L"] / 1000.0)
        out[f"{tag}matern_nu"] = float(fit["params"]["nu"])
        out[f"{tag}matern_r2"] = float(fit["r_squared"])
    else:
        out.update({f"{tag}matern_L_over_nn": np.nan, f"{tag}matern_nu": np.nan, f"{tag}matern_r2": np.nan})
    out[f"{tag}_curve"] = (centres, rho, counts)
    return out


def tier1_anisotropy(bundle, med_nn: float, range_nn: float) -> dict:
    """Orientation dependence at matched distance. Real ERA5 has a genuine
    zonal (E-W) excess; a separable/ARD kernel bank produces a much larger,
    lattice-locked one (see [[project_kernel_sweep_tabicl_retrain_spatial_diag]]
    findings 6 and 8). Only defined for a 2-D design."""
    if bundle.d < 2:
        return {"aniso_ratio": np.nan, "orient_spread": np.nan}
    # Measured on the detrended field: a box whose raw correlation is ~0.99
    # everywhere (ERA5's seasonal mode) has no measurable orientation
    # structure left to compare against a synthetic prior.
    C = _empirical_corr(_detrend_plane(bundle))
    dx = bundle.coords[:, 0][:, None] - bundle.coords[:, 0][None, :]
    dy = bundle.coords[:, 1][:, None] - bundle.coords[:, 1][None, :]
    iu = np.triu_indices_from(C, k=1)
    r = bundle.dist[iu] / med_nn
    ax = np.abs(dx[iu]) / np.maximum(np.sqrt(dx[iu] ** 2 + dy[iu] ** 2), 1e-12)
    c = C[iu]
    # Fixed short-range shell (1.5-3 sample spacings) rather than a shell
    # keyed to the fitted range: that is where correlation is still large, so
    # an orientation CONTRAST there is well determined. A ratio would be
    # unstable (the denominator passes through zero at longer range), so this
    # reports the signed difference instead.
    shell = (r > 1.5) & (r < 3.0)
    if shell.sum() < 50:
        return {"aniso_diff": np.nan, "orient_spread": np.nan}
    ew = shell & (ax > 0.85)
    ns = shell & (ax < 0.15)
    groups = [c[shell & (ax >= lo) & (ax < hi)] for lo, hi in ((0.0, 0.33), (0.33, 0.66), (0.66, 1.01))]
    means = [g.mean() for g in groups if len(g) > 10]
    return {
        "aniso_diff": float(c[ew].mean() - c[ns].mean()) if ew.sum() > 10 and ns.sum() > 10 else np.nan,
        "orient_spread": float(np.std(means)) if len(means) >= 2 else np.nan,
    }


def tier1_nonstationarity(bundle, med_nn: float, n_blocks: int = 4) -> dict:
    """Do different parts of ONE episode have different marginal variance and
    different correlation range? A stationary GP says no (up to estimation
    noise, which the synthetic side measures for us since both sides use the
    same number of realizations). Real boxes mix land/sea/orography and say
    yes, loudly.

    The two halves deliberately use different fields. Variance is measured on
    the RAW field: projecting out a plane removes a position-dependent amount
    of variance (most at the edges of the coordinate spread, least at its
    centre), which manufactures block-to-block variance differences out of a
    perfectly stationary field -- measured at ~0.14 for the stationary
    synthetic prior, i.e. the same magnitude as ERA5's real signal, which made
    the detrended version of this indicator useless. Range is measured on the
    detrended field, where removing the dominant large-scale mode is what
    makes a within-block 0.5-crossing exist at all.
    """
    from scipy.cluster.vq import kmeans2

    if bundle.D < 4 * n_blocks:
        return {"nonstat_var_cv": np.nan, "nonstat_range_cv": np.nan}
    xy = bundle.coords[:, :2]
    _, lab = kmeans2(xy, n_blocks, minit="++", seed=0)
    det = _detrend_plane(bundle)
    var = bundle.fields.var(axis=0)
    block_var, block_rng = [], []
    for b in range(n_blocks):
        m = lab == b
        if m.sum() < 8:
            continue
        block_var.append(var[m].mean())
        Cb = _empirical_corr(det[:, m])
        cb, rb, _ = correlogram(Cb, bundle.dist[np.ix_(m, m)], med_nn, n_bins=10)
        block_rng.append(_crossing(cb, rb, 0.5))
    block_var = np.asarray(block_var, dtype=float)
    block_rng = np.asarray(block_rng, dtype=float)
    ok = np.isfinite(block_rng)
    return {
        "nonstat_var_cv": float(np.std(block_var) / (np.mean(block_var) + 1e-12)) if len(block_var) >= 2 else np.nan,
        "nonstat_range_cv": float(np.std(block_rng[ok]) / (np.mean(block_rng[ok]) + 1e-12)) if ok.sum() >= 2 else np.nan,
    }


# ---------------------------------------------------------------------------
# Tier 2 — the posterior, i.e. what the copula head must emit
# ---------------------------------------------------------------------------


def tier2_posterior(bundle, med_nn: float, context_fracs=(0.05, 0.20), seed: int = 0) -> dict:
    """Condition the estimated prior on a random context subset (Schur
    complement) and describe the resulting correlation the model is asked to
    predict. This is the tier that ties directly to the known failure mode:
    on ERA5 the posterior is near-full-rank while a rank-32 head can only
    represent `post_evr32` of it."""
    rng = np.random.default_rng(seed)
    C = _empirical_corr(bundle.fields)
    out = _eigen_stats(C, "prior")
    out["prior_od_mu"] = float(np.mean(np.abs(_offdiag(C))))
    out["lw_shrinkage"] = _lw_shrinkage(bundle.fields)
    # The same spectrum after the single dominant large-scale mode is removed:
    # for ERA5 that mode is the seasonal cycle, which context conditioning
    # removes anyway, so this is the part of the structure the copula head is
    # really responsible for.
    out.update(_eigen_stats(_empirical_corr(_detrend_plane(bundle)), "prior_det"))
    for f in context_fracs:
        P = int(np.clip(round(f * bundle.D), 2, bundle.D - 4))
        perm = rng.permutation(bundle.D)
        ctx, tst = perm[:P], perm[P:]
        Cff = C[np.ix_(ctx, ctx)]
        Cfs = C[np.ix_(ctx, tst)]
        Css = C[np.ix_(tst, tst)]
        sol = np.linalg.solve(Cff + 1e-8 * np.eye(P), Cfs)
        Post = Css - Cfs.T @ sol
        var_ratio = float(np.mean(np.clip(np.diag(Post), 0, None)))
        Rp = _psd_corr(Post)
        tag = f"post{int(f * 100):02d}"
        out.update(_eigen_stats(Rp, tag))
        out[f"{tag}_var_ratio"] = var_ratio           # posterior/prior variance: how much context removes
        out[f"{tag}_od_mu"] = float(np.mean(np.abs(_offdiag(Rp))))
        cb, rb, _ = correlogram(Rp, bundle.dist[np.ix_(tst, tst)], med_nn)
        out[f"{tag}_range_over_nn"] = _crossing(cb, rb, 0.5)
        # How different is the posterior from the prior at all? If a prior is
        # so diffuse that conditioning barely changes it (today's synthetic
        # regime: P=32 points scattered in ~10 dimensions), the model is never
        # taught to condition.
        Rprior_sub = C[np.ix_(tst, tst)]
        out[f"{tag}_rel_change"] = float(
            np.linalg.norm(Rp - Rprior_sub) / (np.linalg.norm(Rprior_sub) + 1e-12)
        )
    return out


# ---------------------------------------------------------------------------
# Tier 3 — marginal shape and Gaussian-copula adequacy
# ---------------------------------------------------------------------------


def tier3_marginal(bundle, max_realizations: int = 200) -> dict:
    """Per-realization SPATIAL statistics: the model sees one field at a time,
    so this is the marginal TabICL is asked to fit and the non-Gaussianity the
    Gaussian copula has to absorb."""
    from scipy.stats import kstest

    F = bundle.marginal_fields[: min(max_realizations, len(bundle.marginal_fields))]
    Z = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + 1e-12)
    skew = np.mean(Z ** 3, axis=1)
    kurt = np.mean(Z ** 4, axis=1) - 3.0
    ks = np.mean([kstest(z, "norm").statistic for z in Z[: min(40, len(Z))]])

    # Increment (gradient) kurtosis at one nearest-neighbour lag: the front /
    # intermittency signature. Any GP draw is exactly 0 here by construction,
    # so a large value means the field is NOT a Gaussian process of the
    # coordinates and no stationary-kernel prior can reproduce it.
    d = bundle.dist.copy()
    np.fill_diagonal(d, np.inf)
    nb = d.argmin(axis=1)
    inc = Z - Z[:, nb]
    inc = (inc - inc.mean(axis=1, keepdims=True)) / (inc.std(axis=1, keepdims=True) + 1e-12)
    return {
        "spatial_skew_abs": float(np.mean(np.abs(skew))),
        "spatial_exkurt": float(np.mean(kurt)),
        "spatial_gauss_ks": float(ks),
        "increment_exkurt": float(np.mean(np.mean(inc ** 4, axis=1) - 3.0)),
    }


def tier3_tail_dependence(bundle, u: float = 0.95, max_pairs: int = 2000, seed: int = 0) -> dict:
    """Empirical upper-tail dependence chi(u) for nearest-neighbour pairs,
    computed ACROSS realizations. A Gaussian copula has chi(u) -> 0; if the
    real field has chi(u) materially above the Gaussian reference at the same
    correlation, the Gaussian-copula model is misspecified and that gap is a
    floor on achievable NLL, not something a better prior can fix."""
    rng = np.random.default_rng(seed)
    d = bundle.dist.copy()
    np.fill_diagonal(d, np.inf)
    nb = d.argmin(axis=1)
    idx = rng.choice(bundle.D, size=min(max_pairs, bundle.D), replace=False)
    # Detrended: on the raw field every neighbour pair has rho ~ 0.99 (the
    # seasonal mode) and chi(u) saturates near 1 for real and synthetic alike,
    # which measures nothing.
    F = _detrend_plane(bundle)
    U = np.argsort(np.argsort(F, axis=0), axis=0) / (bundle.R - 1)
    a, b = U[:, idx], U[:, nb[idx]]
    ex_a = a > u
    joint = (ex_a & (b > u)).sum(axis=0)
    denom = ex_a.sum(axis=0)
    ok = denom > 5
    chi = float(np.mean(joint[ok] / denom[ok])) if ok.any() else np.nan
    # Gaussian reference at each pair's own correlation, so the number is a
    # *deviation from Gaussian*, not just "neighbours are correlated".
    C = _empirical_corr(F)
    rho = np.clip(C[idx, nb[idx]], -0.999, 0.999)
    chi_gauss = float(np.mean(_gaussian_chi(rho, u)))
    return {"chi_095": chi, "chi_095_gauss_ref": chi_gauss, "chi_095_excess": chi - chi_gauss}


def _gaussian_chi(rho: np.ndarray, u: float) -> np.ndarray:
    """P(U2>u | U1>u) under a bivariate Gaussian copula, by quadrature."""
    from scipy.stats import multivariate_normal, norm

    z = norm.ppf(u)
    out = np.empty_like(rho)
    for i, r in enumerate(np.atleast_1d(rho)):
        joint = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([z, z])
        joint_upper = 1.0 - 2.0 * u + joint
        out[i] = joint_upper / (1.0 - u)
    return out


def tier3_lattice_spectrum(bundle, max_realizations: int = 60) -> dict:
    """Radially averaged 2-D power spectrum slope (lattice designs only). ERA5
    t2m has a well-known mesoscale power law; a mismatch in slope means the
    synthetic field has the wrong balance of large- to small-scale variance
    even when its correlogram looks right. Also picks up the lattice-locked
    ripple artifact as excess power at the grid Nyquist."""
    if bundle.grid_shape is None:
        return {"spec_slope": np.nan, "spec_nyquist_excess": np.nan, "morans_i": np.nan}
    nr, nc = bundle.grid_shape
    F = bundle.marginal_fields[: min(max_realizations, len(bundle.marginal_fields))].reshape(-1, nr, nc)
    F = F - F.mean(axis=(1, 2), keepdims=True)
    win = np.hanning(nr)[:, None] * np.hanning(nc)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(F * win), axes=(1, 2))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(nr))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(nc))[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    nb = max(6, min(nr, nc) // 2)
    edges = np.linspace(1.0 / max(nr, nc), 0.5, nb + 1)
    ctr, pw = [], []
    Pm = P.mean(axis=0)
    for i in range(nb):
        m = (k >= edges[i]) & (k < edges[i + 1])
        if m.sum() > 3:
            ctr.append(0.5 * (edges[i] + edges[i + 1]))
            pw.append(Pm[m].mean())
    if len(ctr) < 4:
        return {"spec_slope": np.nan, "spec_nyquist_excess": np.nan, "morans_i": float(np.mean([morans_i(f) for f in F]))}
    lk, lp = np.log(np.asarray(ctr)), np.log(np.asarray(pw) + 1e-30)
    slope = float(np.polyfit(lk, lp, 1)[0])
    pred = np.polyval(np.polyfit(lk[:-1], lp[:-1], 1), lk[-1])
    return {
        "spec_slope": slope,
        "spec_nyquist_excess": float(lp[-1] - pred),
        "morans_i": float(np.mean([morans_i(f) for f in F])),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

INDICATOR_TIERS = {
    "tier0_geometry": ["D", "d_x", "nn_cv", "spread_over_nn", "maxdist_over_nn"],
    "tier1_second_order": [
        "rho_at_1nn", "range_over_nn", "rho_far", "decorrelates_in_box", "od_mu", "frac_pairs_gt_03",
        "matern_L_over_nn", "matern_nu", "matern_r2",
        "det_rho_at_1nn", "det_range_over_nn", "det_rho_far", "det_od_mu",
        "det_matern_L_over_nn", "det_matern_nu", "det_matern_r2",
        "aniso_diff", "orient_spread", "nonstat_var_cv", "nonstat_range_cv",
    ],
    "tier2_posterior": [
        "prior_evr32", "prior_eff_rank_frac", "prior_od_mu", "lw_shrinkage",
        "prior_det_evr32", "prior_det_evr64", "prior_det_evr128", "prior_det_eff_rank_frac",
        "post05_evr32", "post05_evr64", "post05_evr128", "post05_eff_rank_frac",
        "post05_var_ratio", "post05_od_mu", "post05_range_over_nn", "post05_rel_change",
        "post20_evr32", "post20_eff_rank_frac", "post20_var_ratio", "post20_od_mu", "post20_rel_change",
    ],
    "tier3_marginal": [
        "spatial_skew_abs", "spatial_exkurt", "spatial_gauss_ks", "increment_exkurt",
        "chi_095", "chi_095_gauss_ref", "chi_095_excess",
        "spec_slope", "spec_nyquist_excess", "morans_i",
    ],
}


def compute_all(bundle, seed: int = 0) -> dict:
    """Every indicator for one bundle, as a flat dict of floats (plus the
    `_curve` entries the runner uses for the correlogram figure)."""
    g = tier0_geometry(bundle)
    med_nn = g.pop("_med_nn")
    out = dict(g)
    t1 = tier1_second_order(bundle, med_nn, detrended=False)
    out.update(t1)
    out.update(tier1_second_order(bundle, med_nn, detrended=True))
    out.update(tier1_anisotropy(bundle, med_nn, t1.get("range_over_nn", np.nan)))
    out.update(tier1_nonstationarity(bundle, med_nn))
    out.update(tier2_posterior(bundle, med_nn, seed=seed))
    out.update(tier3_marginal(bundle))
    out.update(tier3_tail_dependence(bundle, seed=seed))
    out.update(tier3_lattice_spectrum(bundle))
    out["source"] = bundle.source
    out["med_nn"] = med_nn
    return out
