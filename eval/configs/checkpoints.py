"""Checkpoint-family registry: paths, report labels and plot colours."""

from __future__ import annotations

import os
from typing import Any

_CHECKPOINTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints"
)

# name -> {dir, default_step, label, color}. dir is relative to checkpoints/;
# default_step is used when no "family:step" is given.
CHECKPOINT_FAMILIES: dict[str, dict[str, Any]] = {
    "kernel-sweep-all-noisy-mae": {
        "dir": "copula_prod/canonical/kernel-sweep-all-noisy-mae",
        "default_step": 355000,
        "label": "Perte MAE + bruit leger (355k steps)",
        "color": "#4c72b0",
    },
    "kernel-sweep-classic-zcorrupt-noise-mild-bigN": {
        "dir": "copula_prod/canonical/kernel-sweep-classic-zcorrupt-noise-mild-bigN",
        "default_step": 285000,
        "label": "Bruit leger + Grand N (285k steps)",
        "color": "#55a868",
    },
    "kernel-sweep-all-tabicl-retrain-15k": {
        "dir": "copula_prod/canonical/kernel-sweep-all-tabicl-retrain",
        "default_step": 15000,
        "label": "Entrainement normal + 15k steps avec z_train TabICL",
        "color": "#c44e52",
    },
    "kernel-sweep-classic-prod-tabicl-retrain": {
        "dir": "copula_prod/canonical/kernel-sweep-classic-prod-tabicl-retrain",
        "default_step": 5000,
        "label": "Classic-prod (40k) + 5k steps avec z_train TabICL",
        "color": "#937860",
    },
    "kernel-sweep-classic-prod": {
        "dir": "copula_prod/canonical/kernel-sweep-classic-prod",
        "default_step": 40000,
        "label": "Classic prod (40k steps)",
        "color": "#8172b2",
    },
    "kernel-sweep-classic-zcorrupt-bigN-retrain": {
        "dir": "copula_prod/canonical/kernel-sweep-classic-zcorrupt-noise-mild-bigN-retrain",
        "default_step": 210000,
        "label": "zcorrupt bigN retrain (210k steps)",
        "color": "#937860",
    },
    # Nano backbone, rank 512, trained with P=32, N=256, d_features=10 and the
    # era5-run1 marginal. Its rank changes the baseline fingerprint, so it needs its
    # own baselines.cache.
    "copula-nano-finetune-marginal-float32": {
        "dir": "copula_nano/copula-finetune-marginal-float32",
        "default_step": 630000,
        "label": "Nano + marginal fine-tune, float32 (630k steps, rank 512)",
        "color": "#da8bc3",
    },
}


def resolve_checkpoint(name_or_path: str) -> str:
    """Resolve a checkpoint token (a runner's ckpt / checkpoints setting) to a checkpoint file path.

    Accepts, in order:
      - a path (exists, or ends in .pt/.ckpt, or contains a separator) -> unchanged
      - "family" -> CHECKPOINT_FAMILIES[family]'s dir + default_step
      - "family:step" -> CHECKPOINT_FAMILIES[family]'s dir + explicit step
    """
    if os.path.exists(name_or_path) or name_or_path.endswith((".pt", ".ckpt")) or os.sep in name_or_path:
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
    """Every registered copula family name, in registry order."""
    return list(CHECKPOINT_FAMILIES)


# Marginal (Phase A) checkpoints: TabICL {"config", "state_dict"} files for
# pit.load_tabicl, kept separate from the copula families.
MARGINAL_FAMILIES: dict[str, dict] = {
    # Pretrained TabICL from the jingang/TabICL HF repo.
    "pretrained": {
        "hf_name": "tabicl-regressor-v2-20260212.ckpt",
        "label": "TabICL v2 pretrained (frozen baseline)",
    },
    "era5-33y": {
        "dir": "marginal/ablations/marginal_finetune_era5_33y",
        "filename": "step_0169600_final.pt",
        "default_step": 169600,
        "label": "TabICL v2 fine-tune ERA5 33y (step 169.6k final)",
    },
    "era5-12m": {
        "dir": "marginal/ablations/marginal_finetune",
        "filename": "step_0016200_final.pt",
        "default_step": 16200,
        "label": "TabICL v2 fine-tune ERA5 12m (step 16.2k final)",
    },
    # Default marginal (conf/model/copula_prod.yaml tabicl.ckpt, copula_nano tabicl.pit_ckpt).
    "era5-run1": {
        "dir": "marginal/ablations/marginal_finetune_era5_run1",
        "filename": "step_0177600_final.pt",
        "default_step": 177600,
        "label": "TabICL v2 fine-tune ERA5 run1 (step 177.6k final) — default",
    },
}

# Name of the default marginal (keep in sync with conf/model/*.yaml).
DEFAULT_MARGINAL_FAMILY = "era5-run1"


def resolve_marginal_checkpoint(name_or_path: str) -> str:
    """Resolve a marginal-checkpoint token for pit.load_tabicl.

    An existing path is returned as is; a MARGINAL_FAMILIES name maps to its HF
    filename or checkpoint file ("family:step" picks a step); anything else is
    returned unchanged (treated as an HF filename).
    """
    if os.path.exists(name_or_path):
        return name_or_path
    family, _, step_str = name_or_path.partition(":")
    entry = MARGINAL_FAMILIES.get(family)
    if entry is None:
        return name_or_path
    if "hf_name" in entry:
        return entry["hf_name"]
    if "filename" in entry and not step_str:
        return os.path.join(_CHECKPOINTS_ROOT, entry["dir"], entry["filename"])
    step = int(step_str) if step_str else entry["default_step"]
    return os.path.join(_CHECKPOINTS_ROOT, entry["dir"], f"step_{step:07d}.pt")
