"""debug/config.py — shared config/CLI plumbing for the debug/ pipeline.

Every stage takes a `DebugConfig` built here, not its own bespoke argparse
setup, so `run_debug.py` can drive any subset of stages with one consistent
set of flags and every stage's output lands in the same
`debug/results/<run_id>/` tree (see report.py for the aggregation format).

Config is built the same way scripts/train_fast.py and
debug/stages/s4_overfit.py build it: load conf/config.yaml + a
conf/model/<name>.yaml preset + conf/data/gp_tasks.yaml via OmegaConf
directly (no Hydra decorator), then apply CLI-style "key=value" overrides.
This mirrors Hydra's own override syntax so a flag copied from a `train.py
data.z_train_source=tabicl ...` invocation works unchanged here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import subprocess
import sys
import time
from typing import Optional

from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_ROOT = os.path.join(_HERE, "results")


@dataclasses.dataclass
class DebugConfig:
    cfg: "OmegaConf"                       # merged Hydra-style config (cfg.data/model/tabicl/training)
    n_episodes: int = 200
    ckpt: Optional[str] = None             # checkpoint name/dir under ./checkpoints, or None (fresh model)
    out_dir: str = RESULTS_ROOT
    run_id: str = "adhoc"
    device: str = "cuda"
    seed: int = 20260826
    overrides: "list[str]" = dataclasses.field(default_factory=list)  # for provenance only

    @property
    def run_dir(self) -> str:
        d = os.path.join(self.out_dir, self.run_id)
        os.makedirs(d, exist_ok=True)
        return d


def _git_sha(repo_root: str) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _config_hash(overrides: "list[str]") -> str:
    h = hashlib.sha1("|".join(sorted(overrides)).encode("utf-8")).hexdigest()
    return h[:8]


def build_config(
    overrides: "list[str] | None" = None,
    model_preset: str = "copula_prod",
    n_episodes: int = 200,
    ckpt: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 20260826,
    run_id: Optional[str] = None,
) -> DebugConfig:
    """Build a DebugConfig the same way as scripts/train_fast.py /
    debug/stages/s4_overfit.py: load conf/config.yaml + conf/model/<preset>
    + conf/data/gp_tasks.yaml, merge, then apply `key.path=value` overrides
    (same dotted-path syntax Hydra uses on train.py's own CLI).
    """
    import torch

    overrides = list(overrides or [])
    base_cfg = OmegaConf.load(os.path.join(_REPO_ROOT, "conf", "config.yaml"))
    model_cfg = OmegaConf.load(os.path.join(_REPO_ROOT, "conf", "model", f"{model_preset}.yaml"))
    data_cfg = OmegaConf.load(os.path.join(_REPO_ROOT, "conf", "data", "gp_tasks.yaml"))
    OmegaConf.set_struct(base_cfg, False)
    cfg = OmegaConf.merge(base_cfg, model_cfg, OmegaConf.create({"data": data_cfg}))
    cfg.seed = seed
    for ov in overrides:
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"override {ov!r} must be key.path=value")
        OmegaConf.update(cfg, key.strip(), _coerce(val.strip()), merge=False)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if run_id is None:
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{_git_sha(_REPO_ROOT)}_{_config_hash(overrides)}"

    return DebugConfig(
        cfg=cfg, n_episodes=n_episodes, ckpt=ckpt, run_id=run_id,
        device=resolved_device, seed=seed, overrides=overrides,
    )


def _coerce(val: str):
    """Best-effort str -> int/float/bool/str, matching Hydra CLI override semantics closely
    enough for the debug pipeline's own use (no lists/dicts needed here)."""
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", default=None, help="Checkpoint name under ./checkpoints/, or a full path. Omit for a fresh (untrained) model where a stage doesn't need one.")
    p.add_argument("--model", default="copula_prod", help="conf/model/<name>.yaml preset (default: copula_prod)")
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--run-id", default=None, help="Results subdir name (default: timestamp_gitsha_confhash)")
    p.add_argument("--out-dir", default=RESULTS_ROOT)
    p.add_argument(
        "override", nargs="*",
        help="Hydra-style config overrides, e.g. data.P_max=64 model.rank=64",
    )
