"""Context-only MAP-GP baseline of eval.runners.marginal_calibration_eval."""

from __future__ import annotations

import math

import torch
from omegaconf import DictConfig, OmegaConf

from copula_inter.data.gp_episodes import generate_gp_batch
from eval.runners.marginal_calibration_eval import MarginalCalibrationSpec, _gp_baseline_metrics


def _episode(small_cfg: DictConfig) -> dict:
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.P_min = cfg.data.P_max = 24
    cfg.data.N_min = cfg.data.N_max = 8
    cfg.data.d_features = 2
    cfg.data.kernel = "rbf"
    torch.manual_seed(3)
    return generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0]


def test_gp_baseline_reports_every_candidate_and_selects_one_by_bic(small_cfg: DictConfig) -> None:
    ep = _episode(small_cfg)
    args = MarginalCalibrationSpec(
        gp_baseline_kernels=["rbf", "matern52"],
        gp_baseline_means=["constant", "linear"],
        gp_baseline_steps=30,
        gp_baseline_restarts=1,
    )
    out = _gp_baseline_metrics(ep, args, {}, "cpu")

    candidates = [out[f"nll_gp_{k}_{m}"] for k in args.gp_baseline_kernels for m in args.gp_baseline_means]
    assert all(math.isfinite(v) for v in candidates), out
    assert math.isfinite(out["nll_gp"]) and math.isfinite(out["nll_gp_mix"])
    # The BIC-best NLL is one of the candidates' (selection uses the context only, not these values).
    assert min(abs(out["nll_gp"] - v) for v in candidates) < 1e-9
