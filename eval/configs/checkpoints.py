"""checkpoints.py — canonical checkpoint-family registry: paths, French
report labels, and plot colors, single-sourced here instead of the
CHECKPOINTS list duplicated across every plots/*.py sweep/report script."""

from __future__ import annotations

import os

_CHECKPOINTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints"
)

# name -> {dir, default_step, label, color}. `dir` is the run directory under
# checkpoints/; `default_step` picks which step_*.pt file `diagnose`/`sweep`
# use when a bare family name (not "family:step") is given. The two
# kernel-sweep-all-tabicl-retrain entries share one run dir but are kept as
# separate registry entries (distinct default_step/label/color) since the
# report treats "15k steps" and "60k steps" of that run as two comparison
# points, not one -- same convention plots/run_synthetic_checkpoint_comparison.py
# and plots/make_supervisor_report_figures.py already used (family keys
# "kernel-sweep-all-tabicl-retrain-15k" / "-60k").
CHECKPOINT_FAMILIES = {
    "kernel-sweep-all": {
        "dir": "kernel-sweep-all",
        "default_step": 500000,
        "label": "Entrainement normal (500k steps)",
        "color": "#888888",
    },
    "kernel-sweep-all-noisy-mae": {
        "dir": "kernel-sweep-all-noisy-mae",
        "default_step": 355000,
        "label": "Perte MAE + bruit leger (355k steps)",
        "color": "#4c72b0",
    },
    "kernel-sweep-classic-zcorrupt-noise-mild-bigN": {
        "dir": "kernel-sweep-classic-zcorrupt-noise-mild-bigN",
        "default_step": 285000,
        "label": "Bruit leger + Grand N (285k steps)",
        "color": "#55a868",
    },
    "kernel-sweep-all-tabicl-retrain-15k": {
        "dir": "kernel-sweep-all-tabicl-retrain",
        "default_step": 15000,
        "label": "Entrainement normal + 15k steps avec z_train TabICL",
        "color": "#c44e52",
    },
    "kernel-sweep-all-tabicl-retrain-60k": {
        "dir": "kernel-sweep-all-tabicl-retrain",
        "default_step": 60000,
        "label": "Entrainement normal + 60k steps avec z_train TabICL",
        "color": "#dd8452",
    },
    "kernel-sweep-classic-prod": {
        "dir": "kernel-sweep-classic-prod",
        "default_step": 40000,
        "label": "Classic prod (40k steps)",
        "color": "#8172b2",
    },
    "kernel-sweep-classic-zcorrupt-bigN-retrain": {
        "dir": "kernel-sweep-classic-zcorrupt-noise-mild-bigN-retrain",
        "default_step": 210000,
        "label": "zcorrupt bigN retrain (210k steps)",
        "color": "#937860",
    },
}


def resolve_checkpoint(name_or_path: str) -> str:
    """Resolve a `--ckpt`/`--checkpoints` token to a checkpoint file path.

    Accepts, in order:
      - a raw path that exists on disk (returned unchanged)
      - "family" -> CHECKPOINT_FAMILIES[family]'s dir + default_step
      - "family:step" -> CHECKPOINT_FAMILIES[family]'s dir + explicit step
    """
    if os.path.exists(name_or_path):
        return name_or_path
    family, _, step_str = name_or_path.partition(":")
    if family not in CHECKPOINT_FAMILIES:
        raise ValueError(
            f"Unknown checkpoint family '{family}' (not an existing path, not in "
            f"CHECKPOINT_FAMILIES: {sorted(CHECKPOINT_FAMILIES)})."
        )
    entry = CHECKPOINT_FAMILIES[family]
    step = int(step_str) if step_str else entry["default_step"]
    return os.path.join(_CHECKPOINTS_ROOT, entry["dir"], f"step_{step:07d}.pt")


def all_family_names() -> list[str]:
    """Every registered family name, in registry order -- what `--checkpoints
    all` (the default for `sweep`) auto-discovers."""
    return list(CHECKPOINT_FAMILIES)
