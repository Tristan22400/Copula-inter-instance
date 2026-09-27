"""Golden values pinning the numerics of the engineering-foundation refactor to pre-refactor main.

`compute(mods)` runs one fixed, seeded computation through every numerically load-bearing path --
GP kernels and posterior, seeded episode generation (kernel sampling, feature warps, analytic
PIT), the copula/Y-space losses, the correlation parametrizations, the model forward + training
loss + one Muon step, and the classical GP baselines. `mods` abstracts the two source layouts
(pre-refactor flat `src/*.py` vs the `copula_inter` package), whose public signatures are
identical, so the same code runs against both.

tests/data/refactor_golden.pt was written by running this file against main @ 42c0402:

    git archive 42c0402 | tar -x -C /tmp/main_src
    python tests/refactor_golden.py --layout main --root /tmp/main_src --out tests/data/refactor_golden.pt

tests/test_refactor_golden.py recomputes the same values with this checkout and compares.
Regenerate only for an intentional numerical change, and say so in the commit.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

GOLDEN_REF = "42c0402"

# Mirrors tests/conftest.py::small_model_cfg (identical on main and the branch).
SMALL_MODEL_CFG: dict[str, Any] = {
    "model": {"rank": 2, "unfreeze_backbone": True},
    "tabicl": {
        "pretrained": False,
        "arch": {
            "embed_dim": 16,
            "col_num_blocks": 1,
            "col_nhead": 2,
            "col_num_inds": 8,
            "row_num_blocks": 1,
            "row_nhead": 2,
            "row_num_cls": 2,
            "icl_num_blocks": 1,
            "icl_nhead": 2,
            "ff_factor": 1,
            "dropout": 0.0,
        },
    },
}

# conf/config.yaml training defaults used by train.py's Muon construction.
TRAIN_CFG: dict[str, Any] = {
    "muon_lr": 2.0e-4,
    "muon_momentum": 0.95,
    "muon_weight_decay": 0.01,
    "muon_matched_adamw_rms": 0.2,
    "muon_ns_steps": 5,
    "muon_nesterov": True,
    "muon_adamw_betas": (0.95, 0.95),
    "muon_adamw_eps": 1.0e-8,
}


def load_modules(layout: str, root: Path) -> SimpleNamespace:
    """Import the numerics modules from a pre-refactor (`main`) or package (`branch`) checkout."""
    if layout == "main":
        for p in (root / "tabicl_upstream" / "src", root, root / "src"):
            sys.path.insert(0, str(p))
        pkg = ""
        kernels = importlib.import_module("data_gen")
        train_core = importlib.import_module("train")
    elif layout == "branch":
        pkg = "copula_inter."
        kernels = importlib.import_module("copula_inter.gp_kernels")
        train_core = importlib.import_module("copula_inter.training_core")
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return SimpleNamespace(
        root=root,
        kernels=kernels,
        data_gen=importlib.import_module(pkg + "data_gen"),
        loss=importlib.import_module(pkg + "loss"),
        model=importlib.import_module(pkg + "model"),
        muon=importlib.import_module(pkg + "muon"),
        train_core=train_core,
        classical=importlib.import_module("eval.baselines.classical"),
    )


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _flatten(prefix: str, obj: Any, out: dict[str, torch.Tensor]) -> None:
    """Collect every tensor / number in a nested result under a dotted key."""
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj.detach().cpu().clone()
    elif isinstance(obj, bool):
        out[prefix] = torch.tensor(obj)
    elif isinstance(obj, (int, float, np.floating, np.integer)):
        out[prefix] = torch.tensor(float(obj), dtype=torch.float64)
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in "fiub":
        out[prefix] = torch.from_numpy(obj.copy())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}.{i}", v, out)
    elif hasattr(obj, "dense") and callable(obj.dense):  # LowRankCorrelationFactor
        _flatten(f"{prefix}.dense", obj.dense(), out)


def _data_cfg(root: Path, P: int, N: int) -> DictConfig:
    cfg = OmegaConf.create({"data": OmegaConf.load(root / "conf" / "data" / "gp_tasks.yaml")})
    cfg.data.P_min = cfg.data.P_max = P
    cfg.data.N_min = cfg.data.N_max = N
    return cfg


def _kernels(m: SimpleNamespace, out: dict[str, torch.Tensor]) -> None:
    _seed(0)
    X1, X2 = torch.randn(7, 3), torch.randn(5, 3)
    k = m.kernels
    for name, fn in {
        "rbf": lambda: k.rbf_kernel(X1, X2, l=0.7, alpha2=1.3),
        "matern12": lambda: k.matern12_kernel(X1, X2, l=0.7, alpha2=1.3),
        "matern32": lambda: k.matern32_kernel(X1, X2, l=0.7, alpha2=1.3),
        "matern52": lambda: k.matern52_kernel(X1, X2, l=0.7, alpha2=1.3),
        "cosine": lambda: k.cosine_kernel(X1, X2, l=0.7, alpha2=1.3),
        "periodic": lambda: k.periodic_kernel(X1, X2, l=0.7, alpha2=1.3, period=0.9),
        "rational_quadratic": lambda: k.rational_quadratic_kernel(X1, X2, l=0.7, alpha2=1.3, rq_alpha=2.0),
        "dot_product": lambda: k.dot_product_kernel(X1, X2, alpha2=1.3),
        "polynomial": lambda: k.polynomial_kernel(X1, X2, l=0.7, alpha2=1.3, power=3.0),
    }.items():
        out[f"kernel.{name}"] = fn().detach().clone()

    def kern(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return k.rbf_kernel(a, b, l=0.7, alpha2=1.3)

    y1 = torch.sin(X1.sum(-1))
    _flatten("gp_posterior", m.data_gen.gp_posterior(X1, y1, X2, kern, 0.1), out)


def _episodes(m: SimpleNamespace, out: dict[str, torch.Tensor]) -> None:
    cfg = _data_cfg(m.root, P=12, N=10)
    _seed(1234)
    eps = m.data_gen.generate_gp_batch(cfg, 3, "cpu")
    _flatten("episodes", eps, out)


def _losses(m: SimpleNamespace, out: dict[str, torch.Tensor]) -> None:
    L, mdl = m.loss, m.model
    _seed(7)
    B, N, r, d = 2, 6, 3, 2
    y, mu = torch.randn(B, N, d), torch.randn(B, N, d)
    D = torch.rand(B, N, d) + 0.5
    V = 0.3 * torch.randn(B, N, d, r)
    out["loss.woodbury_nll"] = L.woodbury_nll(y, mu, D, V)
    out["loss.marginal_nll"] = L.marginal_nll(y, mu, D)
    out["loss.indep_normal_nll"] = L.indep_normal_nll(torch.randn(B, N))

    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, -2:] = False  # exercise padding
    z = torch.randn(B, N) * mask
    log_pdf = -0.5 * torch.randn(B, N).abs() * mask
    W = torch.randn(B, N, r)
    s = torch.randn(B, N)
    W_tilde = torch.nn.functional.normalize(torch.randn(B, N, r + 1), dim=-1)
    # R = eps*I + W W^T is near-singular here (rank r+1 < N, eps=1e-4): in float32 the /eps after the
    # Woodbury cancellation leaves ~0.1-0.2 nats of CPU-dependent rounding error, so score in float64.
    out["loss.copula_nll"] = L.copula_nll(W_tilde.double(), z.double(), mask)

    for par in ("covnorm", "cossim", "tanhnorm", "sparse_covnorm"):
        lam = torch.tensor([0.1]) if par == "sparse_covnorm" else None
        s_par = None if par == "tanhnorm" else s
        R = mdl.low_rank_correlation(W, s_par, parametrization=par, lam=lam)
        out[f"corr.{par}.dense"] = R
        out[f"loss.oracle_copula_nll.{par}"] = L.oracle_copula_nll(R, z, mask)
        _flatten(f"loss.y_space_nll.dense.{par}", L.y_space_nll(R, z, log_pdf, mask), out)
        fac = mdl.low_rank_correlation_factor(W, s_par, parametrization=par, lam=lam)
        _flatten(f"corr.{par}.factor", fac, out)
        _flatten(f"loss.y_space_nll.factor.{par}", L.y_space_nll(fac, z, log_pdf, mask), out)

    A = torch.randn(B, N, N)
    Sigma_star = A @ A.mT / N + 0.5 * torch.eye(N)
    _flatten("loss.gp_oracle_y_nll", L.gp_oracle_y_nll(Sigma_star, torch.randn(B, N), torch.randn(B, N), mask), out)


def _train_step(m: SimpleNamespace, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Seeded model init -> forward -> training loss -> one Muon step -> loss again."""
    cfg = OmegaConf.create(SMALL_MODEL_CFG)
    _seed(0)
    model = m.model.build_copula_transformer(cfg)
    model.train()  # eval() routes through TabICL's inference manager (CUDA auto-select)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    _seed(1)
    B, P, N, d_x = 2, 10, 6, 2
    batch = {
        "x_train": torch.randn(B, P, d_x),
        "z_train": torch.randn(B, P),
        "x_test": torch.randn(B, N, d_x),
        "z_test": torch.randn(B, N),
        "log_pdf_test": -0.9 - 0.3 * torch.rand(B, N),
        "train_mask": torch.ones(B, P, dtype=torch.bool),
        "test_mask": torch.ones(B, N, dtype=torch.bool),
        "R_star": torch.eye(N).expand(B, -1, -1).clone(),
        "n_train": torch.full((B,), P, dtype=torch.long),
        "n_test": torch.full((B,), N, dtype=torch.long),
    }

    t = SimpleNamespace(**TRAIN_CFG)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = m.muon.Muon(
        [
            {
                "params": [p for p in trainable if p.ndim >= 2],
                "use_muon": True,
                "lr": t.muon_lr,
                "weight_decay": t.muon_weight_decay,
                "momentum": t.muon_momentum,
                "matched_adamw_rms": t.muon_matched_adamw_rms,
                "ns_steps": t.muon_ns_steps,
                "nesterov": t.muon_nesterov,
                "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
            {
                "params": [p for p in trainable if p.ndim < 2],
                "use_muon": False,
                "lr": t.muon_lr,
                "weight_decay": 0.0,
                "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
        ]
    )

    def step(tag: str) -> torch.Tensor:
        o, _Sigma, parts, loss, _aux = m.train_core._forward_and_loss(
            model=model,
            batch=batch,
            device="cpu",
            use_amp=False,
            amp_dtype=torch.float32,
            nll_weight=1.0,
            aux_mae_weight=0.0,
            jitter=1e-4,
            triu_cache={},
        )
        out[f"train.{tag}.W"] = o["W"].detach().clone()
        if o.get("s") is not None:
            out[f"train.{tag}.s"] = o["s"].detach().clone()
        _flatten(f"train.{tag}.parts", parts, out)
        out[f"train.{tag}.loss"] = loss.detach().clone()
        return loss

    loss0 = step("step0")
    optimizer.zero_grad()
    loss0.backward()
    grads = [p.grad.norm() for p in trainable if p.grad is not None]
    out["train.step0.grad_norm"] = torch.stack(grads).norm()
    optimizer.step()
    step("step1")
    out["train.param_delta_norm"] = torch.stack(
        [(v - init_state[k]).float().norm() for k, v in model.state_dict().items() if v.is_floating_point()]
    ).norm()
    return init_state


def _baselines(m: SimpleNamespace, out: dict[str, torch.Tensor]) -> None:
    c = m.classical
    _seed(3)
    X_train, X_test = torch.randn(14, 2), torch.randn(8, 2)
    y_train = torch.sin(2 * X_train[:, 0]) + 0.1 * torch.randn(14)
    for name, ard in (("rbf", False), ("matern32", True)):
        _seed(4)
        res = c.fit_and_eval_gpytorch(X_train, y_train, X_test, name, n_steps=15, lr=0.05, ard=ard)
        _flatten(f"baseline.gp_mle.{name}.ard{int(ard)}", res, out)
    _seed(5)
    z_train = torch.randn(14)
    res = c.fit_zero_mean_gp_on_marginal(X_train, z_train, X_test, "rbf", n_steps=15, lr=0.05, n_restarts=1)
    _flatten("baseline.zero_mean_gp.rbf", res, out)
    out["baseline.corr_nll_single"] = torch.tensor(c.corr_nll_single(res["R"], torch.randn(8)), dtype=torch.float64)


def compute(m: SimpleNamespace) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """All golden values, plus the seeded initial model state_dict (checkpoint-compat check)."""
    prev_det, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    prev_threads, prev_rng = torch.get_num_threads(), torch.get_rng_state()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    try:
        out: dict[str, torch.Tensor] = {}
        _kernels(m, out)
        _episodes(m, out)
        _losses(m, out)
        init_state = _train_step(m, out)
        _baselines(m, out)
    finally:  # don't leak global torch state into the rest of a pytest worker
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
        torch.set_num_threads(prev_threads)
        torch.set_rng_state(prev_rng)
    return out, init_state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=("main", "branch"), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    values, init_state = compute(load_modules(a.layout, a.root.resolve()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": GOLDEN_REF, "values": values, "init_state": init_state}, a.out)
    print(f"wrote {len(values)} values + {len(init_state)} state tensors to {a.out}")


if __name__ == "__main__":
    main()
