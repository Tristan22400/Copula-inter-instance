"""debug/common.py — shared helpers for every debug/ stage.

Deliberately thin: every function here delegates to the real implementation
in src/ or eval/ rather than re-deriving it (see debug/README.md's "reuse,
don't reimplement" list). What's added here is only the plumbing to make
those pieces composable across stages: paired same-seed episode generation,
checkpoint loading via the shared registry, and JSON/figure result IO.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf

from config import DebugConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint resolution / loading
# ---------------------------------------------------------------------------

def resolve_ckpt_path(ckpt: Optional[str]) -> Optional[str]:
    """Resolve a `--ckpt` token the same way eval/ tooling does (family name,
    "family:step", or a raw path) — see eval/configs/checkpoints.py::resolve_checkpoint.
    Returns None unchanged (fresh/untrained model)."""
    if ckpt is None:
        return None
    from eval.configs.checkpoints import resolve_checkpoint

    return resolve_checkpoint(ckpt)


def load_model(dcfg: DebugConfig):
    """Build CopulaTabICL from dcfg.cfg and, if dcfg.ckpt is set, load its
    trained weights (state_dict only — no optimizer/scheduler, this is for
    inference-only diagnostics). Falls back to a non-strict load on a
    key mismatch, same rationale as train.py::load_checkpoint."""
    from model import build_copula_transformer

    model = build_copula_transformer(dcfg.cfg).to(dcfg.device)
    ckpt_path = resolve_ckpt_path(dcfg.ckpt)
    if ckpt_path is not None:
        raw = torch.load(ckpt_path, map_location=dcfg.device, weights_only=False)
        state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


_TABICL_CACHE: dict[str, torch.nn.Module] = {}


def load_frozen_tabicl(dcfg: DebugConfig):
    """Frozen TabICL marginal for PIT, cached per (ckpt, device) — every
    stage that needs one (S2/S3/S5/S7) reuses the same loaded weights
    instead of re-downloading/re-loading per stage invocation."""
    from pit import load_tabicl, resolve_pit_ckpt

    ckpt = resolve_pit_ckpt(dcfg.cfg)
    if ckpt is None:
        raise ValueError(
            "This stage needs a resolvable TabICL checkpoint "
            "(tabicl.ckpt with tabicl.pretrained=true, or tabicl.pit_ckpt)."
        )
    key = f"{ckpt}@{dcfg.device}"
    if key not in _TABICL_CACHE:
        _TABICL_CACHE[key] = load_tabicl(ckpt, dcfg.device)
    return _TABICL_CACHE[key]


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episodes(
    dcfg: DebugConfig,
    n: int,
    *,
    tabicl_model=None,
    return_kernel_metadata: bool = True,
    seed_offset: int = 0,
    P_override: Optional[int] = None,
):
    """Thin wrapper over data_gen.generate_gp_batch with the debug config's
    own seed. `P_override` temporarily patches cfg.data.P_min/P_max (used by
    S0's P-sweep) without mutating the caller's dcfg.cfg in place."""
    from data_gen import generate_gp_batch

    cfg = dcfg.cfg
    if P_override is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"P_min": P_override, "P_max": P_override}}))
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"seed": dcfg.seed + seed_offset}))
    return generate_gp_batch(
        cfg, n, device=dcfg.device,
        tabicl_model=tabicl_model,
        tabicl_k_folds=int(cfg.data.get("z_train_tabicl_k_folds", 10)),
        return_kernel_metadata=return_kernel_metadata,
    )


def generate_paired_episodes(dcfg: DebugConfig, n: int, *, seed_offset: int = 0):
    """One analytic-z draw and one TabICL-PIT draw from the SAME seed (and
    hence the same kernel/context/test points — data_gen.py fully reseeds
    python/numpy/torch from cfg.seed per call, see its module docstring).
    Only z_train/z_test/log_pdf_test differ between the two lists; every
    other field (R_star, mu_star, x_norm_*, ...) is identical. This is the
    same paired-seed trick train.py::_compute_tabicl_z_train_gap uses.

    Returns (episodes_analytic, episodes_tabicl).
    """
    tabicl_model = load_frozen_tabicl(dcfg)
    analytic = generate_episodes(dcfg, n, tabicl_model=None, seed_offset=seed_offset)
    tabicl = generate_episodes(dcfg, n, tabicl_model=tabicl_model, seed_offset=seed_offset)
    return analytic, tabicl


# ---------------------------------------------------------------------------
# Posterior oracle (R_post) — per-episode, tolerant of unsupported kernels
# ---------------------------------------------------------------------------

def posterior_oracle(episode: dict):
    """gp_analytical_posterior(episode), or None for the rare unsupported
    kernel schema (whole-chain outer sign modulation) — see its docstring.
    Every stage that needs R_post/nll_post_* goes through this, not a
    direct call, so the "skip, don't crash" convention is enforced once."""
    from pit import gp_analytical_posterior

    try:
        return gp_analytical_posterior(episode)
    except (NotImplementedError, KeyError):
        return None


def collect_posteriors(dcfg: DebugConfig, n: int, *, P_override: Optional[int] = None,
                        seed_offset: int = 0, batch_size: int = 32):
    """Generate n analytic episodes (no TabICL — R_post/R_star/mu_star/
    Sigma_star are analytic regardless of z_train_source, see data_gen.py's
    override-block comment) and pair each with its gp_analytical_posterior,
    skipping the rare unsupported-kernel episode. Used by S0/S1/S3, which
    all walk the same (episode, post) pairs. Generated in chunks of
    `batch_size` (one generate_gp_batch call each, own RNG reseed per call)
    rather than one n-sized call, so a large n doesn't blow VRAM/RAM on
    return_kernel_metadata's cached Cholesky factors.
    """
    pairs = []
    remaining = n
    chunk_idx = 0
    while remaining > 0:
        b = min(batch_size, remaining)
        episodes = generate_episodes(
            dcfg, b, tabicl_model=None, P_override=P_override,
            seed_offset=seed_offset + chunk_idx * 104_729,
        )
        for ep in episodes:
            post = posterior_oracle(ep)
            if post is not None:
                pairs.append((ep, post))
        remaining -= b
        chunk_idx += 1
    return pairs


# ---------------------------------------------------------------------------
# Result IO
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_stage_result(dcfg: DebugConfig, stage: str, result: dict) -> str:
    """Write results/<run_id>/<stage>.json, stamped with git SHA + overrides
    (see report.py, which reads exactly this format)."""
    from config import _git_sha  # local import: avoid a cycle at module load

    path = os.path.join(dcfg.run_dir, f"{stage}.json")
    payload = {
        "stage": stage,
        "git_sha": _git_sha(_REPO_ROOT),
        "overrides": dcfg.overrides,
        "n_episodes": dcfg.n_episodes,
        "ckpt": dcfg.ckpt,
        "seed": dcfg.seed,
        "result": _to_jsonable(result),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def figure_path(dcfg: DebugConfig, stage: str, name: str) -> str:
    d = os.path.join(dcfg.run_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{stage}_{name}.png")


def load_stage_result(run_dir: str, stage: str) -> Optional[dict]:
    path = os.path.join(run_dir, f"{stage}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
