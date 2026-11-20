"""Random-intercept linear mixed model on the grouped BeyondArena splits (joint-track baseline; tabarena venv).

Usage:
    python fit_lmm.py                                # every grouped dataset exported to data_dir
    python fit_lmm.py 'datasets=[<name>]'

Model: y = a + X beta + b_g + e, with b_g ~ N(0, sigma_b2) and e ~ N(0, sigma2), fitted by REML
(statsmodels MixedLM). X: the exported features without the group column, mean-imputed with
training means, standardized, and projected on the principal axes with a singular value of at
least 1e-6 of the largest (a rank-revealing basis: same fitted values, no collinearity failures).
The fit uses at most p_marg training rows (seeded). If REML fails, OLS plus the one-way ANOVA
moment estimate of sigma_b2 on the residuals is used, and fit="moments" is recorded.

For a test group unseen in training, the joint predictive of its rows is
N(a + X beta, sigma_b2 11^T + sigma2 I). Writes <out_dir>/<dataset>/<split>.npz with mu_te (all
test rows, raw units), sigma_b2, sigma2, and fit (reml | moments); beyondarena_joint_eval reads
it through lmm_dir=<out_dir>.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "data_dir": str(REPO / "data/beyondarena"),
    "out_dir": str(REPO / "results/beyondarena/lmm"),
    "datasets": [],
    "p_marg": 50_000,
    "seed": 42,
}


def _design(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        mu = np.nanmean(X_tr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X_tr, X_te = np.where(np.isnan(X_tr), mu, X_tr), np.where(np.isnan(X_te), mu, X_te)
    m, s = X_tr.mean(0), X_tr.std(0)
    keep = s > 1e-8
    A, B = (X_tr[:, keep] - m[keep]) / s[keep], (X_te[:, keep] - m[keep]) / s[keep]
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    k = int((sv >= 1e-6 * sv[0]).sum()) if sv.size else 0
    return A @ Vt[:k].T, B @ Vt[:k].T


def _moments(y: np.ndarray, F: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, float, float]:
    D = np.column_stack([np.ones(len(y)), F])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ beta
    _, inv = np.unique(g, return_inverse=True)
    n_k = np.bincount(inv).astype(float)
    K, N = len(n_k), float(len(r))
    means = np.bincount(inv, weights=r) / n_k
    msb = (n_k * (means - r.mean()) ** 2).sum() / max(K - 1, 1)
    msw = ((r - means[inv]) ** 2).sum() / max(N - K, 1)
    n0 = (N - (n_k**2).sum() / N) / max(K - 1, 1)
    return beta, float(max((msb - msw) / n0, 0.0)), float(msw)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli())
    import statsmodels.api as sm

    for ddir in sorted(Path(cfg.data_dir).iterdir()):
        meta_p = ddir / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        if meta["task_type"] != "grouped" or (cfg.datasets and meta["dataset"] not in cfg.datasets):
            continue
        for rep, fold in meta["splits"]:
            key = f"r{rep}_f{fold}"
            out = Path(cfg.out_dir) / meta["dataset"] / f"{key}.npz"
            if out.exists() or not (ddir / f"{key}.npz").exists():
                continue
            d = dict(np.load(ddir / f"{key}.npz"))
            X_tr, X_te = d["X_tr"].astype(np.float64), d["X_te"].astype(np.float64)
            gf = int(d["group_feature"])
            if gf >= 0:
                X_tr, X_te = np.delete(X_tr, gf, 1), np.delete(X_te, gf, 1)
            idx = np.arange(len(d["y_tr"]))
            rng = np.random.default_rng([int(cfg.seed), rep, fold])
            if len(idx) > cfg.p_marg:
                idx = np.sort(rng.choice(idx, int(cfg.p_marg), replace=False))
            F_tr, F_te = _design(X_tr[idx], X_te)
            y, g = d["y_tr"][idx], d["g_tr"][idx]
            fit = "reml"
            try:
                res = sm.MixedLM(y, sm.add_constant(F_tr, has_constant="add"), groups=g).fit(
                    reml=True, method=["lbfgs"]
                )
                if not res.converged:
                    raise RuntimeError("not converged")
                beta = np.asarray(res.fe_params)
                sigma_b2, sigma2 = float(np.asarray(res.cov_re).ravel()[0]), float(res.scale)
            except Exception as exc:  # noqa: BLE001 -- any REML failure falls back to the moment fit
                print(f"{meta['dataset']} {key}: REML failed ({exc}); moment fit", flush=True)
                beta, sigma_b2, sigma2 = _moments(y, F_tr, g)
                fit = "moments"
            mu_te = np.column_stack([np.ones(len(F_te)), F_te]) @ beta
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, mu_te=mu_te, sigma_b2=sigma_b2, sigma2=sigma2, fit=fit)
            print(
                f"{meta['dataset']} {key}: {fit} sigma_b2 {sigma_b2:.4g} sigma2 {sigma2:.4g} ICC {sigma_b2 / (sigma_b2 + sigma2):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
