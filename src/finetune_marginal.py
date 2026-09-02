"""finetune_marginal.py — Phase A entry point.

Fine-tunes a STANDALONE TabICL (quantile decoder intact) so its marginal
posterior predictive is correct for the GP prior the copula is trained on, then
writes it in TabICL's own checkpoint schema so the copula run picks it up with:

    python src/train.py tabicl.pit_ckpt=<checkpoints/marginal_finetune/...pt>

Usage
-----
    python src/finetune_marginal.py
    python src/finetune_marginal.py marginal.tier=1 training.lr=2e-5
    python src/finetune_marginal.py wandb.mode=disabled training.steps=20   # smoke

A real Hydra application, not an argparse -> override translator that shells out
to train.py the way ``src/finetune_era5.py`` does: the Phase-A objective needs
its own model construction, its own loss and its own validation, so there is no
train.py invocation to translate INTO. Every knob is therefore a normal Hydra
override and the composed config is snapshotted into each checkpoint.

Why this is a separate loop rather than a ``training.objective: marginal`` branch
inside ``src/train.py``: that file's ~3000-line ``main`` is built end to end
around the copula path — live GP/ERA5 DataLoaders whose workers each hold their
own frozen TabICL, the copula head, z_train collation, Sigma diagnostics,
correlogram probes, Muon param groups. Phase A shares none of it: no DataLoader
at all (the trainable marginal must live in the MAIN process, since gradients do
not cross process boundaries and nothing can push updated weights into spawned
workers), no copula head, no z. Threading a second objective through that main
would mean touching model construction, data, loss, validation and
checkpointing, putting the working copula path at risk for no reuse. What IS
shared is shared by import — ``pit`` (the PIT forward), ``data_gen`` (the prior),
``lora`` (the freeze predicate), ``train.cosine_lr_lambda`` (the schedule) and
``eval.spatial.sweep_core`` (the ERA5 probe geometry) — so there is no duplicated
logic, only a duplicated ``for step in range(...)``.
"""

from __future__ import annotations

import os
import random
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT, os.path.join(_REPO_ROOT, "tabicl_upstream", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import generate_gp_batch  # noqa: E402
from marginal_finetune import (  # noqa: E402
    AnchorPenalty,
    MarginalLossWeights,
    apply_tier,
    build_era5_marginal_val_batches,
    phase_a_batch_loss,
    save_marginal_checkpoint,
    validate_era5_marginal,
    validate_synthetic_marginal,
)
from pit import load_tabicl  # noqa: E402
from train import cosine_lr_lambda  # noqa: E402


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ERA5EpisodeSampler:
    """Draws batches of real-ERA5 episodes that share P and N.

    ``run_pit_batched`` can only fold a batch into TabICL's own batch axis when
    every episode shares P/N, so this uses the corpus's
    ``sample_episode_fixed_shape`` (exact grid_size / n_context, redraw on a
    degenerate box) rather than ``sample_episode``'s ranges — the same reason
    ``src/era5_live_dataset.py`` groups its draws.

    Region, day and box width still vary per episode, so a batch is a genuine
    spread of real spatial fields at one shape, not one field repeated.
    """

    def __init__(self, corpus, *, grid_size: int, n_context: int,
                 box_deg_range: tuple[float, float], seed: int) -> None:
        self.corpus = corpus
        self.grid_size = int(grid_size)
        self.n_context = int(n_context)
        self.box_deg_range = box_deg_range
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int, max_tries: int = 200) -> list[dict]:
        out: list[dict] = []
        tries = 0
        while len(out) < B and tries < max_tries * B:
            tries += 1
            ep = self.corpus.sample_episode_fixed_shape(
                self.rng, self.grid_size, self.box_deg_range, self.n_context
            )
            if ep is None:
                continue
            out.append(
                {
                    "x_norm_train": torch.as_tensor(ep["x_norm_train"]),
                    "y_train": torch.as_tensor(ep["y_train"]),
                    "x_norm_test": torch.as_tensor(ep["x_norm_test"]),
                    "y_test": torch.as_tensor(ep["y_test"]),
                }
            )
        if len(out) < B:
            raise RuntimeError(
                f"ERA5 sampler produced {len(out)}/{B} episodes in {tries} draws at "
                f"grid_size={self.grid_size}, n_context={self.n_context}, "
                f"box_deg_range={self.box_deg_range}. Widen box_deg_max or lower "
                f"grid_size."
            )
        return out


def _gp_cfg(cfg: DictConfig) -> DictConfig:
    """The shape ``data_gen.generate_gp_batch`` expects: a top-level config with
    a ``data`` group and a ``seed``, not the ``data`` group on its own.

    ``generate_gp_batch`` reads ``cfg.data.*`` for the prior and
    ``getattr(cfg, "seed")`` for reproducibility, so handing it ``cfg.data``
    directly raises ``Missing key data``. Built once and re-seeded per call
    rather than reconstructed, since the prior block is large.
    """
    return OmegaConf.create(
        {"data": OmegaConf.to_container(cfg.data, resolve=True), "seed": int(cfg.seed)}
    )


def _build_gp_val_batches(cfg: DictConfig, device: str) -> list[list[dict]]:
    """A FIXED synthetic-GP validation set, drawn once with its own seed.

    Fixed rather than freshly sampled per call so that a change in
    ``val_marginal/gp/nll_gap_to_oracle`` between two validations is the model
    moving, not the episodes changing — the gap is a few nats on a quantity with
    real per-episode spread, so resampling would bury the signal in draw noise.
    """
    gp_cfg = _gp_cfg(cfg)
    batches = []
    for i in range(int(cfg.validation.gp_n_batches)):
        gp_cfg.seed = int(cfg.validation.gp_seed) + i
        batches.append(
            generate_gp_batch(
                gp_cfg, int(cfg.validation.gp_batch_size), device,
                return_kernel_metadata=True,
            )
        )
    return batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf", config_name="finetune_marginal", version_base=None)
def main(cfg: DictConfig) -> None:
    device = _resolve_device(str(cfg.training.device))
    _seed_everything(int(cfg.seed))
    print(OmegaConf.to_yaml(cfg))

    # ---- model + tier routing -------------------------------------------
    tabicl, tabicl_config = load_tabicl(
        str(cfg.marginal.ckpt), device, trainable=True, return_config=True
    )
    report = apply_tier(
        tabicl,
        int(cfg.marginal.tier),
        lora_rank=int(cfg.marginal.lora_rank),
        lora_alpha=float(cfg.marginal.lora_alpha),
        lora_target=str(cfg.marginal.lora_target),
    )
    tabicl.to(device)
    print(
        f"[tier {report['tier']}] {report['tier_desc']}: "
        f"{report['n_trainable_params']:,} / {report['n_total_params']:,} trainable "
        f"({100 * report['trainable_frac']:.2f}%), "
        f"{report['lora_modules_replaced']} LoRA module(s)"
    )

    weights = MarginalLossWeights(
        distill=float(cfg.marginal.loss.distill),
        nll=float(cfg.marginal.loss.nll),
        crps=float(cfg.marginal.loss.crps),
        anchor=float(cfg.marginal.loss.anchor),
        huber_delta=float(cfg.marginal.loss.huber_delta),
        tail_power=float(cfg.marginal.loss.tail_power),
    )
    # Real-data batches carry no analytic target, so distillation is off there.
    # The proper scores alone are still a valid (higher-variance) objective.
    era5_weights = MarginalLossWeights(
        distill=0.0, nll=weights.nll, crps=weights.crps, anchor=weights.anchor,
        huber_delta=weights.huber_delta, tail_power=weights.tail_power,
    )
    anchor = AnchorPenalty(tabicl) if weights.anchor > 0 else None

    # ---- optimizer -------------------------------------------------------
    # AdamW, one group, no ndim split. Two reasons this is not train.py's
    # Muon setup: src/muon.py's own header warns Muon "may not work well for
    # finetuning pretrained models", which is precisely this; and train.py's
    # positional optimizer-state restore (load_checkpoint matches Adam/Muon
    # moments to params by position in the flattened list) is a hazard the
    # moment param groups change, which a tier ladder does by construction.
    # A single group over `requires_grad` params has no positional ambiguity.
    params = [p for p in tabicl.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Tier routing left no trainable parameters.")
    opt = torch.optim.AdamW(
        params, lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    total_steps = int(cfg.training.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: cosine_lr_lambda(
            s, int(cfg.training.warmup_steps), total_steps,
            float(cfg.training.lr_min_frac),
        ),
    )

    # ---- data ------------------------------------------------------------
    gp_cfg = _gp_cfg(cfg)
    eps = float(cfg.marginal.pit_eps)
    k_folds = int(cfg.marginal.k_folds)
    folds_per_step = cfg.marginal.folds_per_step
    folds_per_step = None if folds_per_step is None else int(folds_per_step)
    mix_frac = float(cfg.marginal.era5.mix_frac)

    era5_sampler = None
    if mix_frac > 0:
        from eval.data.era5_global_corpus import GlobalERA5Corpus

        corpus = GlobalERA5Corpus(
            str(cfg.marginal.era5.corpus_dir),
            max_months=int(cfg.marginal.era5.max_months),
        )
        era5_sampler = ERA5EpisodeSampler(
            corpus,
            grid_size=int(cfg.marginal.era5.grid_size),
            n_context=int(cfg.marginal.era5.n_context),
            box_deg_range=(
                float(cfg.marginal.era5.box_deg_min),
                float(cfg.marginal.era5.box_deg_max),
            ),
            seed=int(cfg.seed) + 7717,
        )
        print(f"[era5] mixture on: {corpus.n_days_total} days loaded, mix_frac={mix_frac}")

    print("[val] building fixed validation sets (one-off ERA5 fetch/crop)...")
    era5_val = build_era5_marginal_val_batches(cfg.validation, device)
    gp_val = _build_gp_val_batches(cfg, device)
    print(f"[val] {len(era5_val)} ERA5 region(s), {len(gp_val)} synthetic GP batch(es)")

    # ---- wandb -----------------------------------------------------------
    run = None
    if str(cfg.wandb.mode) != "disabled":
        import wandb

        run = wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=str(cfg.wandb.mode),
        )
        wandb.watch(tabicl, log="gradients", log_freq=max(1, int(cfg.training.log_every)))
        wandb.log({f"model/{k}": v for k, v in report.items()
                   if isinstance(v, (int, float))}, step=0)

    def _log(payload: dict, step: int) -> None:
        if run is not None:
            run.log(payload, step=step)

    def _validate(step: int) -> None:
        t0 = time.time()
        metrics = validate_era5_marginal(tabicl, era5_val, eps=eps)
        metrics.update(
            validate_synthetic_marginal(
                tabicl, gp_val, k_folds=k_folds, eps=eps, device=device
            )
        )
        metrics["val_marginal/seconds"] = time.time() - t0
        _log(metrics, step)
        print(
            f"[val step {step}] "
            f"era5 nll={metrics.get('val_marginal/mean_nll', float('nan')):.4f} "
            f"ece={metrics.get('val_marginal/mean_ece', float('nan')):.4f} "
            f"ks={metrics.get('val_marginal/mean_ks', float('nan')):.4f} | "
            f"gp nll={metrics.get('val_marginal/gp/nll', float('nan')):.4f} "
            f"oracle={metrics.get('val_marginal/gp/nll_oracle', float('nan')):.4f} "
            f"gap={metrics.get('val_marginal/gp/nll_gap_to_oracle', float('nan')):.4f}"
        )

    def _save(step: int, tag: str = "") -> str | None:
        if cfg.training.ckpt_dir is None:
            return None
        name = f"step_{step:07d}{tag}.pt"
        path = os.path.join(str(cfg.training.ckpt_dir), name)
        save_marginal_checkpoint(
            path, tabicl, tabicl_config, step=step, cfg=cfg,
            extra={"tier_report": {k: v for k, v in report.items()
                                   if isinstance(v, (int, float, str))}},
        )
        print(f"[ckpt] {path}")
        return path

    # ---- train -----------------------------------------------------------
    _validate(0)
    rng = np.random.default_rng(int(cfg.seed) + 991)
    gen = torch.Generator().manual_seed(int(cfg.seed) + 13)
    B = int(cfg.training.batch_size)
    t_last = time.time()

    for step in range(1, total_steps + 1):
        use_era5 = era5_sampler is not None and rng.random() < mix_frac
        if use_era5:
            episodes = era5_sampler.batch(B)
            episodes = [
                {k: v.to(device) for k, v in ep.items()} for ep in episodes
            ]
            w = era5_weights
        else:
            gp_cfg.seed = int(cfg.seed) * 1_000_003 + step
            episodes = generate_gp_batch(
                gp_cfg, B, device, return_kernel_metadata=True
            )
            w = weights

        res = phase_a_batch_loss(
            tabicl, episodes, w,
            k_folds=k_folds, folds_per_step=folds_per_step,
            generator=gen, device=device, eps=eps,
        )
        loss = res["loss"]
        anchor_val = 0.0
        if anchor is not None:
            a = anchor(tabicl)
            loss = loss + weights.anchor * a
            anchor_val = float(a)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            params, float(cfg.training.clip_grad_norm)
        )
        opt.step()
        sched.step()

        if step % int(cfg.training.log_every) == 0:
            dt = (time.time() - t_last) / int(cfg.training.log_every)
            t_last = time.time()
            payload = {
                "train/loss": float(loss),
                "train/nll": float(res["nll"]),
                "train/crps": float(res["crps"]),
                "train/distill": float(res["distill"]),
                "train/anchor": anchor_val,
                "train/grad_norm": float(gnorm),
                "train/lr": sched.get_last_lr()[0],
                "train/sec_per_step": dt,
                "train/is_era5_batch": float(use_era5),
                "train/P": int(episodes[0]["x_norm_train"].shape[0]),
            }
            if not use_era5:
                payload["train/nll_oracle"] = res["oracle_nll"]
                payload["train/nll_gap_to_oracle"] = res["nll_gap_to_oracle"]
            _log(payload, step)
            print(
                f"step {step:>7} loss={float(loss):.4f} nll={float(res['nll']):.4f} "
                f"distill={float(res['distill']):.4f} "
                f"gap={res.get('nll_gap_to_oracle', float('nan')):.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} {dt:.2f}s/step"
                + ("  [era5]" if use_era5 else "")
            )

        if step % int(cfg.training.val_every) == 0:
            _validate(step)
        if step % int(cfg.training.save_every) == 0:
            _save(step)

    _validate(total_steps)
    final = _save(total_steps, tag="_final")
    if final:
        print(
            "\nPhase A done. Use it as the copula run's marginal with:\n"
            f"    python src/train.py tabicl.pit_ckpt={os.path.abspath(final)}\n"
            "and measure it first with:\n"
            f"    python eval/runners/marginal_calibration_eval.py --ckpt {os.path.abspath(final)}"
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
