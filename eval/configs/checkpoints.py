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


# ---------------------------------------------------------------------------
# Marginal (Phase A) checkpoints — a SEPARATE registry, on purpose
# ---------------------------------------------------------------------------
#
# These are plain TabICL checkpoints ({"config", "state_dict"}) produced by
# src/finetune_marginal.py and consumed by pit.load_tabicl. CHECKPOINT_FAMILIES
# above holds COPULA checkpoints, consumed by builders that construct a
# CopulaTabICL and load a copula state dict into it. Putting a marginal entry in
# that dict would make `sweep --checkpoints all` (which iterates
# all_family_names()) try to load a TabICL state dict into a copula model and
# fail -- so the two kinds get two registries and two resolvers rather than one
# dict with a type tag nothing checks.
#
# `dir` is under checkpoints/; `default_step` picks the step_*.pt file for a
# bare family name. `tabicl.pit_ckpt` in a copula run takes the resolved path.
MARGINAL_FAMILIES: dict[str, dict] = {
    # The frozen pretrained marginal every copula run has used so far -- the
    # baseline row in eval/runners/marginal_calibration_eval.py, and the thing
    # a Phase-A run has to beat. Not a local path: pit.load_tabicl resolves a
    # bare name through the jingang/TabICL HF repo.
    "pretrained": {
        "hf_name": "tabicl-regressor-v2-20260212.ckpt",
        "label": "TabICL v2 pretrained (frozen baseline)",
    },
}


def resolve_marginal_checkpoint(name_or_path: str) -> str:
    """Resolve a marginal-checkpoint token to something ``pit.load_tabicl`` takes.

    Accepts, in order:
      - a raw path that exists on disk (returned unchanged)
      - a MARGINAL_FAMILIES name with an ``hf_name`` -> that HF filename
      - "family" / "family:step" -> that family's dir + step under checkpoints/
      - anything else -> returned unchanged, so a bare HF filename still works
        without needing a registry entry (load_tabicl treats a non-path as an
        HF filename anyway, and failing here would be a worse error than the
        one the HF hub gives).
    """
    if os.path.exists(name_or_path):
        return name_or_path
    family, _, step_str = name_or_path.partition(":")
    entry = MARGINAL_FAMILIES.get(family)
    if entry is None:
        return name_or_path
    if "hf_name" in entry:
        return entry["hf_name"]
    step = int(step_str) if step_str else entry["default_step"]
    return os.path.join(_CHECKPOINTS_ROOT, entry["dir"], f"step_{step:07d}.pt")
