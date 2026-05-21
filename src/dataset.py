"""
dataset.py — CopulaDataset and collate_fn for inter-instance copula training.

Each episode file contains one GP task with latent z-scores from the PIT.
The collate_fn pads variable-length tasks to the batch maximum P and N and
produces boolean masks indicating valid (non-padded) rows.
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
        self, episode_dir: Optional[str] = None, file_list: Optional[List[str]] = None
    ):
        if file_list is not None:
            self.files = sorted(file_list)
        elif episode_dir is not None:
            self.files = sorted(glob(os.path.join(episode_dir, "*.pt")))
        else:
            raise ValueError("Provide either episode_dir or file_list.")

        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {episode_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], map_location="cpu", weights_only=True)


def collate_fn(samples: List[dict]) -> dict:
    """Pad a batch of variable-length tasks to (B, P_max, d_x) / (B, N_max).

    Returns:
        x_train    : (B, P_max, d_x)   — padded train features
        z_train    : (B, P_max)         — padded train z-scores (0 for padding)
        x_test     : (B, N_max, d_x)   — padded test features
        z_test     : (B, N_max)         — padded test z-scores
        train_mask : BoolTensor (B, P_max) — True for valid train rows
        test_mask  : BoolTensor (B, N_max) — True for valid test rows
        R_star     : (B, N_max, N_max)  — padded oracle correlation matrices
        mu_star    : (B, N_max)         — padded posterior means
        sigma_star : (B, N_max)         — padded posterior stds
        n_train    : LongTensor (B,)
        n_test     : LongTensor (B,)
    """
    B = len(samples)
    d_x = samples[0]["x_norm_train"].shape[-1]

    P_list = [s["n_train"].item() for s in samples]
    N_list = [s["n_test"].item() for s in samples]
    P_max = max(P_list)
    N_max = max(N_list)

    x_train = torch.zeros(B, P_max, d_x)
    z_train = torch.zeros(B, P_max)
    x_test = torch.zeros(B, N_max, d_x)
    z_test = torch.zeros(B, N_max)
    train_mask = torch.zeros(B, P_max, dtype=torch.bool)
    test_mask = torch.zeros(B, N_max, dtype=torch.bool)
    R_star = torch.zeros(B, N_max, N_max)
    mu_star = torch.zeros(B, N_max)
    sigma_star = torch.zeros(B, N_max)

    for b, s in enumerate(samples):
        P = P_list[b]
        N = N_list[b]

        x_train[b, :P] = s["x_norm_train"]
        z_train[b, :P] = s["z_train"]
        x_test[b, :N] = s["x_norm_test"]
        z_test[b, :N] = s["z_test"]
        train_mask[b, :P] = True
        test_mask[b, :N] = True
        R_star[b, :N, :N] = s["R_star"]
        mu_star[b, :N] = s["mu_star"]
        sigma_star[b, :N] = s["sigma_star"]

    return {
        "x_train": x_train,
        "z_train": z_train,
        "x_test": x_test,
        "z_test": z_test,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "R_star": R_star,
        "mu_star": mu_star,
        "sigma_star": sigma_star,
        "n_train": torch.tensor(P_list, dtype=torch.long),
        "n_test": torch.tensor(N_list, dtype=torch.long),
    }
