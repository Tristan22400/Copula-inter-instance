"""TabArena / AutoGluon model classes with a joint predictive over the test rows (beyondarena-joint venv).

CopulaJointModel: one of the six marginals (TabICLv2 / TabPFN-3.5 Fast / Kumo small, pretrained or
ERA5-fine-tuned), with or without the copula trained on it.
- fit: keeps the (preprocessed) training rows as the in-context set. Nothing else is learned.
- predict (the point prediction TabArena scores, RMSE): the mean of the marginal's quantiles of each
  row given the whole training set as context.
- predict_joint(X, y): for the scoring runner only (y is used to evaluate the predictive at the
  observed targets, never to form it). It returns
  - the marginal quantiles, PIT z and log-density of every test row given the whole training set;
  - z_ctx: out-of-fold PITs of the training rows (k_folds random folds: the model knows nothing of the
    task's groups or time order);
  - for a copula: its factor rows (U, D), with the whole training set and z_ctx as copula context.
The bag protocol is TabArena's foundation-model one (refit_folds): the 8 fold models give the
out-of-fold validation predictions, and the test fold is predicted by one model refit on all the
training rows. So the joint predictive has the whole training fold as context.

MixedLMJointModel: random-intercept linear mixed model (grouped tasks), y = a + X b + u_g + e. Its joint
predictive for an unseen group is N(a + X b, s_u^2 11^T + s_e^2 I). It is a group-aware reference
baseline: by definition it needs the groups of the training rows, which the runner passes to it.

Features: AutoGluon's preprocessed frame (TabArena's official pipeline, which drops the group column),
categoricals as their codes (missing -> NaN), mean-imputed and z-scored with training statistics.
CopulaJointModel never sees group labels or times: its input is exactly BeyondArena's.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from autogluon.core.models import AbstractModel

from eval.beyondarena.joint_metrics import anova_icc, keep_columns

_BACKENDS: dict[tuple[str, str | None, str], Any] = {}
_COPULAS: dict[tuple[str, str], Any] = {}


def _frame_to_array(X: pd.DataFrame) -> np.ndarray:
    """Float matrix: category columns as their codes (missing -> NaN), booleans as 0/1."""
    cols = []
    for c in X.columns:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            v = s.cat.codes.to_numpy(np.float64)
            v[v < 0] = np.nan
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
        cols.append(v)
    return np.column_stack(cols) if cols else np.zeros((len(X), 0))


class _JointBase(AbstractModel):
    """Shared preprocessing (training-set imputation / z-scoring) and GPU defaults."""

    _supported_problem_types: ClassVar[list[str]] = ["regression"]
    default_num_gpus = 1

    def _set_default_params(self) -> None:
        for k, v in {"seed": 0}.items():
            self._set_default_param_value(k, v)

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["regression"]

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs: Any) -> dict:
        # TabArena's foundation-model protocol: the fold models predict their validation rows, then one model
        # refit on all the training rows predicts the test fold.
        return super()._get_default_ag_args_ensemble(**kwargs) | {
            "refit_folds": True,
            "fold_fitting_strategy": "sequential_local",
        }

    def _get_default_resources(self) -> tuple[int, float]:
        num_cpus, _ = super()._get_default_resources()
        return num_cpus, 1 if torch.cuda.is_available() else 0

    def _features(self, X: pd.DataFrame, is_train: bool) -> np.ndarray:
        A = _frame_to_array(X)
        if is_train:
            with np.errstate(all="ignore"):
                mu = np.nanmean(A, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            A0 = np.where(np.isnan(A), mu, A)
            sd = A0.std(axis=0)
            self._mu, self._sd = mu, np.where(sd < 1e-8, 1.0, sd)
            self._const = sd < 1e-8
        A = np.where(np.isnan(A), self._mu, A)
        A = (A - self._mu) / self._sd
        A[:, self._const] = 0.0
        return A.astype(np.float32)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


class CopulaJointModel(_JointBase):
    """A marginal backend (copula_inter.marginal) with or without its Gaussian copula (copula_inter.copula)."""

    ag_key = "CJ"
    ag_name = "CopulaJoint"
    ag_priority = 60

    def _set_default_params(self) -> None:
        super()._set_default_params()
        defaults = {
            "marginal": "tabicl",
            "tabicl_ckpt": "pretrained",
            "marginal_ckpt": None,
            "copula_ckpt": None,
            "k_folds": 10,
            "max_features": None,
            "query_batch": 2048,
        }
        for k, v in defaults.items():
            self._set_default_param_value(k, v)

    # ------------------------------------------------------------- fit / predict

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_gpus: float = 0, **kwargs: Any) -> None:
        p = self._get_model_params()
        X = self.preprocess(X, is_train=True)
        self._device = "cuda" if num_gpus and torch.cuda.is_available() else "cpu"
        A = self._features(X, is_train=True)
        self._cols = np.arange(A.shape[1])
        if p["max_features"] and A.shape[1] > p["max_features"]:  # TabPFN-3.5 takes at most 500 columns
            self._cols = keep_columns(A.shape[1], int(p["max_features"]), -1, int(p["seed"]))
        self._x_ctx = A[:, self._cols]
        self._y_ctx = y.to_numpy(np.float64)
        self.model = "in-context"  # AutoGluon checks that a fitted model is set

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        x = self._features(X, is_train=False)[:, self._cols]
        q, levels = self._quantiles(x)
        return q.mean(axis=1)

    def predict_joint(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        preprocess_nonadaptive: bool = False,
    ) -> dict[str, Any]:
        """The joint predictive of the rows of X, evaluated at their targets y (see the module docstring)."""
        p = self._get_model_params()
        X = self.preprocess(X, preprocess_nonadaptive=preprocess_nonadaptive)
        x = self._features(X, is_train=False)[:, self._cols]
        t0 = time.time()
        m = self._marginal_at(x, np.asarray(y, dtype=np.float64))
        out: dict[str, Any] = m | {"scale": float(np.std(self._y_ctx)) or 1.0, "seconds_marginal": time.time() - t0}
        t0 = time.time()
        z_ctx = self._context_z()
        out["seconds_crossfit"] = time.time() - t0
        out["z_ctx"] = z_ctx
        out["P_ctx"] = len(self._y_ctx)
        if p["copula_ckpt"]:
            from eval.runners.ctr23_full_eval import _copula_rows

            cop = self._copula()
            dev = torch.device(self._device)
            t0 = time.time()
            with torch.no_grad():
                U, D = _copula_rows(
                    cop,
                    torch.from_numpy(self._x_ctx).to(dev),
                    torch.from_numpy(z_ctx).float().to(dev),
                    torch.from_numpy(x).to(dev),
                    int(p["query_batch"]),
                )
            if dev.type == "cuda":
                torch.cuda.synchronize()
            out["seconds_copula"] = time.time() - t0
            out["U"], out["D"] = U.double().cpu().numpy(), D.double().cpu().numpy()
        return out

    # ------------------------------------------------------------- backends

    def _backend(self) -> Any:
        p = self._get_model_params()
        key = (p["marginal"], p["tabicl_ckpt"] if p["marginal"] == "tabicl" else p["marginal_ckpt"], self._device)
        if key not in _BACKENDS:
            if p["marginal"] == "tabicl":
                from copula_inter.checkpoint_registry import resolve_marginal_checkpoint
                from copula_inter.marginal.pit import load_tabicl

                _BACKENDS[key] = load_tabicl(resolve_marginal_checkpoint(p["tabicl_ckpt"]), self._device)
            else:
                from copula_inter.marginal.backends import make_regressor

                _BACKENDS[key] = make_regressor(p["marginal"], device=self._device, ckpt=p["marginal_ckpt"])
        return _BACKENDS[key]

    def _copula(self) -> Any:
        p = self._get_model_params()
        key = (str(p["copula_ckpt"]), self._device)
        if key not in _COPULAS:
            from eval.runners.ctr23_full_eval import check_marginal_pairing
            from inference.copula_inference import load_copula_model

            cop, cfg = load_copula_model(str(p["copula_ckpt"]), device=self._device)

            class _Pair:  # the fields check_marginal_pairing reads
                ckpt = p["copula_ckpt"]
                marginal = p["marginal"]
                tabicl_ckpt = p["tabicl_ckpt"] if p["marginal"] == "tabicl" else None
                marginal_ckpt = p["marginal_ckpt"]

            check_marginal_pairing(cfg, _Pair())
            _COPULAS[key] = cop
        return _COPULAS[key]

    def _tensors(
        self, x: np.ndarray, y: np.ndarray | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = torch.device(self._device)
        yq = np.zeros(len(x)) if y is None else y
        return (
            torch.from_numpy(self._x_ctx).to(dev),
            torch.from_numpy(self._y_ctx).float().to(dev),
            torch.from_numpy(x).to(dev),
            torch.from_numpy(yq).float().to(dev),
        )

    def _quantiles(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, Q) raw-unit quantiles and their levels, given the whole training set."""
        p = self._get_model_params()
        xc, yc, xq, yq = self._tensors(x, None)
        with torch.no_grad():
            if p["marginal"] == "tabicl":
                from eval.runners.ctr23_full_eval import _tabicl_test_only

                m = _tabicl_test_only(self._backend(), xc, yc, xq, yq)
                return m["q_test"].double().cpu().numpy(), m["levels"].double().cpu().numpy()
            from eval.runners.ctr23_full_eval import _BACKEND_LEVELS, _backend_test_quantiles

            q = _backend_test_quantiles(p["marginal"], self._backend(), xc, yc, xq, int(p["query_batch"]))
            return q.double().cpu().numpy(), _BACKEND_LEVELS

    def _marginal_at(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Quantiles, z = Phi^{-1}(F(y)) and log f(y) of the rows (raw units), given the whole training set."""
        p = self._get_model_params()
        xc, yc, xq, yq = self._tensors(x, y)
        with torch.no_grad():
            if p["marginal"] == "tabicl":
                from eval.runners.ctr23_full_eval import _tabicl_test_only

                m = _tabicl_test_only(self._backend(), xc, yc, xq, yq)
            else:
                from eval.runners.ctr23_full_eval import _backend_test_only

                m = _backend_test_only(p["marginal"], self._backend(), xc, yc, xq, yq, int(p["query_batch"]))
        return {
            "q": m["q_test"].float().cpu().numpy(),
            "levels": m["levels"].double().cpu().numpy(),
            "z": m["z_test"].double().cpu().numpy(),
            "logf": m["log_pdf_test"].double().cpu().numpy(),
        }

    def _context_z(self) -> np.ndarray:
        """Out-of-fold z of every training row: k_folds random folds, fold k predicted from the others."""
        p = self._get_model_params()
        P, k = len(self._y_ctx), int(p["k_folds"])
        folds = np.random.default_rng([int(p["seed"]), P]).permutation(np.arange(P) % k)
        z = np.empty(P)
        x_all, y_all = self._x_ctx, self._y_ctx
        for f in range(k):
            q_idx, c_idx = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
            if len(q_idx) == 0:
                continue
            saved = self._x_ctx, self._y_ctx
            self._x_ctx, self._y_ctx = x_all[c_idx], y_all[c_idx]
            try:
                z[q_idx] = self._marginal_at(x_all[q_idx], y_all[q_idx])["z"]
            finally:
                self._x_ctx, self._y_ctx = saved
        return z


class MixedLMJointModel(_JointBase):
    """Random-intercept LMM (statsmodels REML) on rank-revealing principal axes of the features; grouped tasks only.

    The official preprocessing drops the group column, so the groups of the training rows reach the model
    only through predict_joint (from the task, like CopulaJointModel). The point predictions TabArena scores
    (validation and RMSE) use the OLS fixed effects fitted in _fit; predict_joint fits the REML model
    (GLS fixed effects + variance components) with the groups and uses it for the joint predictive.
    """

    ag_key = "CJ-LMM"
    ag_name = "MixedLMJoint"
    ag_priority = 10
    default_num_gpus = 0

    def _set_default_params(self) -> None:
        super()._set_default_params()
        self._set_default_param_value("max_fit_rows", 100_000)
        # Principal axes kept: singular value >= rcond * the largest (condition number <= 1 / rcond).
        self._set_default_param_value("rcond", 1e-3)

    def _get_default_resources(self) -> tuple[int, float]:
        num_cpus, _ = super()._get_default_resources()
        return num_cpus, 0

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        X = self.preprocess(X, is_train=True)
        A = self._features(X, is_train=True)
        self._y_ctx = y.to_numpy(np.float64)
        rows = self._fit_rows()
        _, sv, Vt = np.linalg.svd(A[rows], full_matrices=False)
        rcond = float(self._get_model_params()["rcond"])
        self._V = Vt[: int((sv >= rcond * sv[0]).sum())].T if sv.size else np.zeros((A.shape[1], 0))
        S = A @ self._V
        # Test rows are clipped to the training range of each axis: a linear model must not extrapolate
        # along low-variance axes (group splits put unseen feature combinations in the test fold).
        self._s_lo, self._s_hi = S[rows].min(axis=0), S[rows].max(axis=0)
        self._F = np.column_stack([np.ones(len(A)), S])
        self._beta, *_ = np.linalg.lstsq(self._F, self._y_ctx, rcond=None)
        self._scale = float(np.std(self._y_ctx)) or 1.0
        self.model = "lmm"

    def _fit_rows(self) -> np.ndarray:
        p = self._get_model_params()
        idx = np.arange(len(self._y_ctx))
        if len(idx) > p["max_fit_rows"]:
            idx = np.sort(
                np.random.default_rng([int(p["seed"]), len(idx)]).choice(idx, int(p["max_fit_rows"]), replace=False)
            )
        return idx

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        A = self._features(X, is_train=False)
        return np.column_stack([np.ones(len(A)), np.clip(A @ self._V, self._s_lo, self._s_hi)])

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        return self._design(self.preprocess(X, **kwargs)) @ self._beta

    def predict_joint(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        regime: str,
        ctx_groups: np.ndarray | None,
        ctx_times: np.ndarray | None,
        preprocess_nonadaptive: bool = False,
    ) -> dict[str, Any]:
        import statsmodels.api as sm
        from scipy.stats import norm

        if regime != "grouped" or ctx_groups is None:
            raise ValueError("MixedLMJointModel needs the training groups (grouped tasks only)")
        idx = self._fit_rows()
        F, yv, g = self._F[idx], self._y_ctx[idx], np.asarray(ctx_groups)[idx]
        fit = "reml"
        v_ols = float((yv - F @ self._beta).var())
        try:
            res = sm.MixedLM(yv, F, groups=g).fit(reml=True, method=["lbfgs"])
            if not res.converged:
                raise RuntimeError("REML did not converge")
            beta = np.asarray(res.fe_params)
            s_u2, s_e2 = float(np.asarray(res.cov_re).ravel()[0]), float(res.scale)
            # A degenerate optimum (seen: s_u2 = 0 with s_e2 ~ 0 under lbfgs) gives a near-zero predictive sd;
            # the total variance must match the OLS residual variance up to the share the fixed effects move.
            if not (np.isfinite(s_u2) and np.isfinite(s_e2) and np.all(np.isfinite(beta))):
                raise RuntimeError("REML returned non-finite estimates")
            if s_u2 < 0 or not 0.1 * v_ols <= s_u2 + s_e2 <= 10.0 * v_ols or s_e2 < 1e-3 * v_ols:
                raise RuntimeError(f"REML variance components off scale: s_u2={s_u2:.3g} s_e2={s_e2:.3g}")
        except Exception:  # noqa: BLE001 -- any REML failure falls back to OLS + the ANOVA moment estimate
            fit = "moments"
            beta = self._beta
            r = yv - F @ beta
            icc = anova_icc(r / (r.std() or 1.0), g)
            icc = float(np.clip(icc, 0.0, 0.99)) if np.isfinite(icc) else 0.0
            s_u2, s_e2 = icc * r.var(), (1.0 - icc) * r.var()
        X = self.preprocess(X, preprocess_nonadaptive=preprocess_nonadaptive)
        D = self._design(X)
        # The predictive mean is the OLS one, i.e. the point prediction TabArena scores (RMSE): the test groups
        # are new, so the random intercept adds variance, not a shift. REML's GLS fixed effects on many
        # near-collinear axes were seen to diverge on test rows (mean off by ~1e6 sd, joint NLL ~1e12)
        # while the OLS fit stayed sane; REML is kept for the variance components only.
        mu = D @ self._beta
        sd = float(np.sqrt(s_u2 + s_e2)) or 1e-8
        mean_shift = float(np.median(np.abs(D @ beta - mu)) / sd)
        levels = np.linspace(0.001, 0.999, 999)
        r = (np.asarray(y, dtype=np.float64) - mu) / sd
        return {
            "kind": "gaussian",
            "q": (mu[:, None] + sd * norm.ppf(levels)[None]).astype(np.float32),
            "levels": levels,
            "z": r,
            "logf": norm.logpdf(r) - np.log(sd),
            "scale": self._scale,
            "rho_exch": s_u2 / (s_u2 + s_e2) if s_u2 + s_e2 > 0 else 0.0,
            "fit": fit,
            "s_u2": s_u2,
            "s_e2": s_e2,
            "var_resid_ols": v_ols,
            "reml_mean_shift_sd": mean_shift,
            "n_axes": int(self._V.shape[1]),
            "P_ctx": len(self._y_ctx),
        }


# One TabArena method per marginal: each needs its own name, since TabArena caches results by method name.
class CJTabICLPre(CopulaJointModel):
    ag_key, ag_name = "CJ-TABICLv2-PRE", "CJ-TabICLv2-pre"


class CJTabICLFt(CopulaJointModel):
    ag_key, ag_name = "CJ-TABICLv2-FT", "CJ-TabICLv2-ft"


class CJTabPFNPre(CopulaJointModel):
    ag_key, ag_name = "CJ-TABPFN35F-PRE", "CJ-TabPFN35F-pre"


class CJTabPFNFt(CopulaJointModel):
    ag_key, ag_name = "CJ-TABPFN35F-FT", "CJ-TabPFN35F-ft"


class CJKumoPre(CopulaJointModel):
    ag_key, ag_name = "CJ-KUMOS-PRE", "CJ-KumoS-pre"


class CJKumoFt(CopulaJointModel):
    ag_key, ag_name = "CJ-KUMOS-FT", "CJ-KumoS-ft"


MODEL_CLASSES: dict[str, type[CopulaJointModel]] = {
    "tabicl-pre": CJTabICLPre,
    "tabicl-ft": CJTabICLFt,
    "tabpfn-pre": CJTabPFNPre,
    "tabpfn-ft": CJTabPFNFt,
    "kumo-pre": CJKumoPre,
    "kumo-ft": CJKumoFt,
}
