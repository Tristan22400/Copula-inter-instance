"""TabArena experiment runner that also scores a joint predictive on the whole test fold.

JointExperimentRunner is TabArena's OOFExperimentRunner (fit, timed predict, RMSE as `metric_error`,
out-of-fold simulation artifacts) plus one step in `post_evaluate`: it loads the served bag's single
refit model (refit_folds), passes it the test rows through the same feature transform and bag
preprocessing as the timed predict, and stores eval.beyondarena.joint_metrics.joint_scores of its
`predict_joint` under `out["joint_metrics"]` ({variant: {score: value}}), with timings and context
facts under `out["joint_info"]`. The split's regime and its group / time columns come from the
BeyondArena task metadata. Use it with `Experiment(..., experiment_cls=JointExperimentRunner)`.
"""

from __future__ import annotations

import functools
import time
from typing import Any

import numpy as np
import pandas as pd
from tabarena.benchmark.experiment.experiment_runner import OOFExperimentRunner

from eval.beyondarena.joint_metrics import anova_icc, block_ids_for, joint_scores

REGIMES = {"random": "iid", "grouped": "grouped", "temporal": "temporal"}


@functools.cache
def _task_structure() -> dict[str, dict[str, Any]]:
    """{dataset name (with and without its hash): {regime, group_on, time_on}} from the BeyondArena metadata."""
    from tabarena.contexts import BeyondArenaContext

    df = BeyondArenaContext()._official_task_metadata_collection().per_dataset_frame()
    out: dict[str, dict[str, Any]] = {}
    for r in df.itertuples():
        v = {"regime": REGIMES.get(r.task_type, "iid"), "group_on": r.group_on, "time_on": r.time_on}
        out[str(r.dataset)] = v
        out[str(r.dataset_name)] = v
    return out


def _times(col: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.astype("int64").to_numpy(np.float64) / 1e9
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.99:
        return num.to_numpy(np.float64)
    return pd.to_datetime(col, errors="coerce").astype("int64").to_numpy(np.float64) / 1e9


class JointExperimentRunner(OOFExperimentRunner):
    """OOFExperimentRunner + joint scores of the refit model's joint predictive over the whole test fold."""

    def post_evaluate(self, out: dict) -> dict:
        out = super().post_evaluate(out)
        meta = _task_structure().get(str(self.task_name)) or _task_structure().get(str(out["task_metadata"]["name"]))
        if meta is None:
            raise KeyError(f"no BeyondArena metadata for task {self.task_name!r}")
        X_test, y_test = self._load_x_test(), self._load_y_test()
        regime = meta["regime"]
        g_col, t_col = meta["group_on"], meta["time_on"]
        groups = X_test[g_col].astype(str).to_numpy() if regime == "grouped" and g_col in X_test else None
        times = _times(X_test[t_col]) if regime == "temporal" and t_col in X_test else None

        bag = self.model._load_model()
        if len(bag.models) != 1:
            raise RuntimeError(f"joint scoring expects one refit model (refit_folds), the bag has {len(bag.models)}")
        child = bag.load_child(bag.models[0])
        X_inner = self.model.predictor._learner.transform_features(X_test)
        X_child = bag.preprocess(X_inner, model=child)
        # Group labels / times of the refit model's training rows (all the training rows, in the order it
        # was fit on: checked against the targets it stored).
        X_train, y_train = (self.X, self.y) if not self.task.lazy_load_data else self._train_test_split()[:2]
        y_ctx = getattr(child, "_y_ctx", None)
        if y_ctx is None or len(y_ctx) != len(y_train) or not np.allclose(y_ctx, y_train.to_numpy(np.float64)):
            raise RuntimeError("the refit model's training rows do not match the task's training split (order or rows)")
        # Used only by the scoring and the group-aware baselines (exch, lmm); never by the copula model.
        ctx_groups = X_train[g_col].astype(str).to_numpy() if groups is not None else None
        ctx_times = _times(X_train[t_col]) if times is not None else None
        t0 = time.time()
        if hasattr(child, "_beta"):  # the mixed model: a group-aware reference baseline, given the training groups
            pred = child.predict_joint(
                X_child,
                y_test.to_numpy(np.float64),
                regime=regime,
                ctx_groups=ctx_groups,
                ctx_times=ctx_times,
                preprocess_nonadaptive=False,
            )
        else:  # the marginal / copula model: BeyondArena's inputs only
            pred = child.predict_joint(X_child, y_test.to_numpy(np.float64), preprocess_nonadaptive=False)
            # The exchangeable baseline (group-aware): rho = ICC of the model's context PITs over the task's
            # context blocks (training groups, time windows of 32 rows, or random blocks of 32 rows).
            ids = block_ids_for(regime, ctx_groups, ctx_times, 32, seed=0, n=len(y_train))
            rho = anova_icc(pred.pop("z_ctx"), ids)
            pred["icc_ctx_z"] = float(rho)
            pred["rho_exch"] = float(np.clip(rho, 0.0, 0.99)) if np.isfinite(rho) else 0.0
        t_pred = time.time() - t0
        t0 = time.time()
        seed = int(self.fold) * 1000 + int(self.repeat)
        out["joint_metrics"] = joint_scores(pred, y_test.to_numpy(np.float64), regime, groups, times, seed=seed)
        out["joint_info"] = {
            "regime": regime,
            "seconds_predict_joint": t_pred,
            "seconds_scoring": time.time() - t0,
            **{k: v for k, v in pred.items() if np.isscalar(v) or isinstance(v, str)},
            "rank": int(pred["U"].shape[1]) if pred.get("U") is not None else 0,
        }
        return out
