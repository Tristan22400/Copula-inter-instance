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
# use when a bare family name (not "family:step") is given.
#
# Kept to the best-performing checkpoint per training lineage (per the
# spatial-correlation model_r2 sweeps -- see project memory): dropped
# kernel-sweep-all@500k (worst overall, negative model_r2 almost everywhere
# despite the longest training), kernel-sweep-all-tabicl-retrain-60k
# (regresses vs. its own 15k-step sibling), and kernel-sweep-classic-prod@110k
# (flat, distance-blind correlation prediction, dominated by its own 5k-step
# TabICL finetune below).
CHECKPOINT_FAMILIES = {
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
    "kernel-sweep-classic-prod-tabicl-retrain": {
        "dir": "kernel-sweep-classic-prod-tabicl-retrain",
        "default_step": 5000,
        "label": "Classic-prod (40k) + 5k steps avec z_train TabICL",
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
