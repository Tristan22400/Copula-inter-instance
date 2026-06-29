"""
dataset.py — CopulaDataset and collate_fn for inter-instance copula training.

Each episode file (one .pt per task) carries both the Y- and Z-space tensors
plus the per-instance marginal log-PDF emitted by the PIT pipeline.  The
collate_fn pads variable-length tasks to the batch maximum P and N.
"""

from __future__ import annotations

import os
from glob import glob
from typing import List, Optional

import torch
from torch.utils.data import Dataset


class CopulaDataset(Dataset):
    """Dataset of pre-computed PIT episodes (one .pt file per task)."""

    def __init__(
        self,
        episode_dir: Optional[str] = None,
        file_list: Optional[List[str]] = None,
    ):
        if file_list is not None:
            self.files = sorted(file_list)
        elif episode_dir is not None:
            self.files = sorted(glob(os.path.join(episode_dir, "*.pt")))
        else:
            raise ValueError("Provide either episode_dir or file_list.")

        # Filter out files that don't exist (NFS race / incomplete generation)
        existing = [f for f in self.files if os.path.isfile(f)]
        if len(existing) < len(self.files):
            import warnings
            warnings.warn(
                f"CopulaDataset: {len(self.files) - len(existing)} file(s) listed "
                f"but missing on disk — they will be skipped."
            )
        self.files = existing

        if not self.files:
            if episode_dir is not None:
                raise RuntimeError(f"No .pt files found in {episode_dir}")
            else:
                raise RuntimeError(
                    f"No .pt files available — all {len(file_list)} file(s) in file_list are missing on disk"
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        try:
            return torch.load(self.files[idx], map_location="cpu", weights_only=True)
        except FileNotFoundError:
            # File disappeared after init (NFS eviction); pick a random other slot.
            # (idx+1)%len would loop to itself when len==1, so exclude idx explicitly.
            import random
            candidates = [i for i in range(len(self.files)) if i != idx]
            if not candidates:
                raise  # only one file and it's gone — nothing to fall back to
            return torch.load(self.files[random.choice(candidates)], map_location="cpu", weights_only=True)


def collate_fn(samples: List[dict]) -> dict:
    """Pad a batch of variable-length tasks.

    Returns (all padded to batch-max P, N):
        x_train      : (B, P_max, d_x)
        x_test       : (B, N_max, d_x)
        y_train      : (B, P_max)
        y_test       : (B, N_max)
        z_train      : (B, P_max)
        z_test       : (B, N_max)
        log_pdf_test : (B, N_max)        (0 for padding → log(1)=0 contributes nothing)
        train_mask   : BoolTensor (B, P_max)
        test_mask    : BoolTensor (B, N_max)
        R_star       : (B, N_max, N_max)
        Sigma_star   : (B, N_max, N_max)
        mu_star      : (B, N_max)
        sigma_star   : (B, N_max)
        n_train      : LongTensor (B,)
        n_test       : LongTensor (B,)
    """
    B = len(samples)
    d_x = samples[0]["x_norm_train"].shape[-1]

    P_list = [int(s["n_train"].item()) for s in samples]
    N_list = [int(s["n_test"].item()) for s in samples]
    P_max = max(P_list)
    N_max = max(N_list)

    x_train = torch.zeros(B, P_max, d_x)
    x_test = torch.zeros(B, N_max, d_x)
    y_train = torch.zeros(B, P_max)
    y_test = torch.zeros(B, N_max)
    z_train = torch.zeros(B, P_max)
    z_test = torch.zeros(B, N_max)
    log_pdf_test = torch.zeros(B, N_max)
    train_mask = torch.zeros(B, P_max, dtype=torch.bool)
    test_mask = torch.zeros(B, N_max, dtype=torch.bool)
    R_star = torch.zeros(B, N_max, N_max)
    Sigma_star = torch.zeros(B, N_max, N_max)
    mu_star = torch.zeros(B, N_max)
    sigma_star = torch.zeros(B, N_max)

    for b, s in enumerate(samples):
        P = P_list[b]
        N = N_list[b]

        x_train[b, :P] = s["x_norm_train"]
        x_test[b, :N] = s["x_norm_test"]
        y_train[b, :P] = s["y_train"]
        y_test[b, :N] = s["y_test"]
        z_train[b, :P] = s["z_train"]
        z_test[b, :N] = s["z_test"]
        log_pdf_test[b, :N] = s["log_pdf_test"]
        train_mask[b, :P] = True
        test_mask[b, :N] = True
        R_star[b, :N, :N] = s["R_star"]
        Sigma_star[b, :N, :N] = s["Sigma_star"]
        mu_star[b, :N] = s["mu_star"]
        sigma_star[b, :N] = s["sigma_star"]

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "R_star": R_star,
        "Sigma_star": Sigma_star,
        "mu_star": mu_star,
        "sigma_star": sigma_star,
        "n_train": torch.tensor(P_list, dtype=torch.long),
        "n_test": torch.tensor(N_list, dtype=torch.long),
    }
