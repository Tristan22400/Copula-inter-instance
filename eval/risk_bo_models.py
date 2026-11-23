"""Posterior samplers for robust BO, in the original scenario loss units."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from copula_inter.baselines.gp_mle import fit_and_eval_gpytorch
from copula_inter.marginal.backends import make_regressor
from copula_inter.marginal.batched_pit import backend_pit_fn, run_context_pit_batched
from copula_inter.marginal.pit import normalize_features
from copula_inter.marginal.tabpfn_marginal import TabPFNMarginal, bar_icdf
from eval.runners.ctr23_full_eval import _copula_rows, check_marginal_pairing
from inference.copula_inference import load_copula_model

if TYPE_CHECKING:
    from copula_inter.copula.model import CopulaTabICL


@dataclass
class GPRiskSurrogate:
    log_output: bool
    n_steps: int
    kernel_name: str = "matern52"

    def draw(self, x_train: Tensor, y_train: Tensor, x_query: Tensor, base: Tensor, seed: int) -> tuple[Tensor, None]:
        target = y_train.log() if self.log_output else y_train
        mean, scale = target.mean(), target.std().clamp_min(1e-8)
        devices = [x_train.device] if x_train.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if x_train.is_cuda:
                with torch.cuda.device(x_train.device):
                    torch.cuda.manual_seed(seed)
            posterior = fit_and_eval_gpytorch(
                x_train,
                (target - mean) / scale,
                x_query,
                kernel_name=self.kernel_name,
                n_steps=self.n_steps,
                lr=0.1,
                ard=True,
                oracle_mode="posterior",
                observation_noise=False,
            )
        chol = torch.linalg.cholesky(posterior["Sigma"])
        sampled = (posterior["mean"] + base @ chol.T) * scale + mean
        return (sampled.exp() if self.log_output else sampled), None


@dataclass
class TabPFNPair:
    ckpt: str | None
    marginal_ckpt: str | None
    marginal: str = "tabpfn35_fast"
    tabicl_ckpt: str | None = None


def checkpoint_paths(root: Path, *, fine_tuned: bool) -> tuple[Path | None, Path]:
    marginal = root / "marginal/ablations/marginal_finetune_era5_tabpfn35_fast/step_0027500_final.pt"
    family = "size-robust-tabpfn35_fast" if fine_tuned else "pretrained-marginal-tabpfn35_fast"
    copula = root / "copula_nano" / family / "step_0100000.pt"
    return (marginal if fine_tuned else None), copula


class TabPFNRiskSurrogate:
    def __init__(
        self,
        root: Path,
        *,
        device: str,
        fine_tuned: bool,
        coupled: bool,
        query_batch: int,
        k_folds: int,
    ) -> None:
        marginal, copula = checkpoint_paths(root, fine_tuned=fine_tuned)
        pairing = TabPFNPair(str(copula) if coupled else None, str(marginal) if marginal is not None else None)
        self.copula: CopulaTabICL | None = None
        if pairing.ckpt:
            self.copula, cfg = load_copula_model(pairing.ckpt, device=device)
            check_marginal_pairing(cfg, pairing)
        self.marginal = cast(
            TabPFNMarginal, make_regressor(pairing.marginal, device=device, ckpt=pairing.marginal_ckpt)
        )
        self.marginal.module.eval()
        self.query_batch = query_batch
        self.k_folds = k_folds
        self.pit = backend_pit_fn(pairing.marginal, self.marginal)

    @torch.no_grad()
    def draw(
        self, x_train: Tensor, y_train: Tensor, x_query: Tensor, base: Tensor, seed: int
    ) -> tuple[Tensor, Callable[[], Tensor] | None]:
        x_t, x_q = normalize_features(x_train.cpu().numpy(), x_query.cpu().numpy())
        train = torch.as_tensor(x_t, dtype=torch.float32, device=x_train.device)
        query = torch.as_tensor(x_q, dtype=torch.float32, device=x_query.device)
        y = y_train.float()
        latent = base
        if self.copula is not None:
            y_z = ((y - y.mean()) / y.std().clamp_min(1e-8)).cpu().numpy()
            pit = run_context_pit_batched(self.pit, x_t[None], y_z[None], k_folds=self.k_folds, seed=seed)
            z_train = torch.as_tensor(pit[0], device=train.device)
            u, d = _copula_rows(self.copula, train, z_train, query, self.query_batch)
            latent = gaussian_paths(base, u, d, seed)

        output = []
        audit_chunks = []
        for start in range(0, len(query), self.query_batch):
            end = start + self.query_batch
            logits, mean, scale = self.marginal.batched_logits(train[None], y[None], query[None, start:end])
            bars = self.marginal.regressor.znorm_space_bardist_
            output.append(_inverse_cdf(bars, logits[0], latent[:, start:end]) * scale[0] + mean[0])
            if self.copula is not None:
                audit_chunks.append((bars, logits[0], base[:, start:end], scale[0], mean[0]))

        def independent() -> Tensor:
            return torch.cat(
                [_inverse_cdf(bars, logits, noise) * scale + mean for bars, logits, noise, scale, mean in audit_chunks],
                dim=1,
            )

        return torch.cat(output, dim=1), independent if audit_chunks else None


def gaussian_paths(base: Tensor, u: Tensor, d: Tensor, seed: int) -> Tensor:
    """Sample correlated normals with covariance U Uᵀ + diag(D) using shared point noise."""
    generator = torch.Generator(device=base.device).manual_seed(seed + 10_000_019)
    shared = torch.randn(base.shape[0], u.shape[1], device=base.device, dtype=base.dtype, generator=generator)
    return shared @ u.to(base).T + base * d.to(base).sqrt()


def _inverse_cdf(bars: object, logits: Tensor, latent: Tensor) -> Tensor:
    # bar_icdf broadcasts its probability input over the leading query axis.
    probs = torch.special.ndtr(latent.T).to(logits).clamp(1e-7, 1 - 1e-7)
    return bar_icdf(bars, logits, probs).T
