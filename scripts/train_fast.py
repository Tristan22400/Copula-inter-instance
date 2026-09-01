#!/usr/bin/env python3
"""
train_fast.py — instant-startup debug trainer for the Copula Transformer.

train.py's startup (baselines.enabled's 8 synthetic-kernel probes + a TabICL
K-fold PIT pass over every one of them, a real ERA5 fetch + classical-GP-MLE
baseline fit, a second frozen-TabICL "sim-to-real diagnostic" load, a
500-episode fixed validation set built through TabICL's own K-fold PIT, live
DataLoader worker spawn, wandb.init's network round-trip) takes ~7 minutes
before the first training step runs. That's dead time when what you actually
need is "is this model/config training at all, and does the loss move" —
e.g. debugging a run that looks stuck.

This script builds the exact same Hydra config train.py would (same
`model=`/`data=` groups and CLI overrides), the exact same model, optimizer
(Muon), LR schedule, AMP setup, and per-step forward/loss/backward/clip/step
logic — imported directly from src/train.py, not reimplemented, so a step
here behaves identically to a step in the real run. It only diverges from
train.py in what it skips: no baselines/era5 probes, no wandb, no persistent
DataLoader workers, and a small in-process-generated validation set instead
of the fixed 500-episode one. First step happens in seconds, most of that
being CUDA context init + (if data.z_train_source=tabicl, the default) one
frozen-TabICL load.

The debug val set's z_test comes from real TabICL K-fold PIT by default,
regardless of what data.z_train_source/z_train_tabicl_mix_* the TRAINING steps
use (matches eval_checkpoint.py's own default of scoring against the real
deployment signal) -- so it needs a resolvable TabICL checkpoint even when
training itself is pure data.z_train_source=analytic. Pass
`+experiment=analytic_only` to score against the EXACT analytic GP marginal
instead: no TabICL is loaded at all, and no checkpoint is required.

The val line reports the gap on the TOTAL NLL, not the copula term alone --
the model's copula term and the oracle's are Sklar-split under different
marginals (prior vs. posterior sigma), so only the totals are comparable.
See _build_debug_val_batch's docstring.

Not a replacement for train.py — no wandb logging, no baselines/era5
validation metrics. Checkpointing (training.ckpt_dir/training.resume_ckpt)
uses train.py's own save_checkpoint/load_checkpoint, so a checkpoint saved
here is a normal checkpoint train.sh can resume, and training.resume_ckpt
here can load the actual stuck run's checkpoint to debug from where it left
off. For a full production run, use train.sh.

Usage (same Hydra override syntax as train.py):
    python scripts/train_fast.py
    python scripts/train_fast.py training.resume_ckpt=./checkpoints/copula_transformer/step_0029999.pt
    python scripts/train_fast.py model=copula_nano training.steps=200
    python scripts/train_fast.py data.z_train_source=analytic   # train on the analytic oracle -- val z_test still uses real TabICL
    python scripts/train_fast.py +experiment=analytic_only      # analytic on BOTH sides; loads no TabICL, needs no checkpoint
    python scripts/train_fast.py training.batch_size=8 data.N_max=64  # shrink episodes for an even faster loop
    # Alternate per-episode between the analytic z_train and real-TabICL PIT
    # z_train at a fixed 50/50 rate (every kernel family, no adaptive gap
    # measurement -- see the z_train_tabicl_mix_enabled block below):
    python scripts/train_fast.py data.z_train_tabicl_mix_enabled=true \
        data.z_train_tabicl_mix_floor_frac=0.5 data.z_train_tabicl_mix_max_frac=0.5
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

# Force line-buffered stdout even when piped/redirected (e.g. `| tee log.txt`)
# -- this script's entire point is watching output live to tell a genuinely
# stuck run apart from one that's just quiet because Python block-buffers
# non-tty stdout. Without this, output can sit in the buffer indefinitely.
sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from data_gen import _COMPOSABLE_KERNELS, generate_gp_batch
from dataset import collate_fn
from model import build_copula_transformer
from muon import Muon
from live_dataset import validate_analytic_only
from pit import episode_posterior_ceiling, load_tabicl, resolve_pit_ckpt
from train import (
    _forward_and_loss,
    _run_train_step,
    _sigma_stats,
    cosine_lr_lambda,
    load_checkpoint,
    save_checkpoint,
)

# Debug-loop cadence -- deliberately NOT tied to training.log_every/val_every
# (those default to 200/1000, tuned for multi-day production runs, not a
# "watch it start" debug session). Override by editing these constants
# directly if you want a different cadence.
DEBUG_LOG_EVERY = 1
DEBUG_VAL_EVERY = 20
# How many of live_dataset.py::build_fixed_live_val_batches' fixed val
# batches to reproduce here (see _build_debug_val_batch below) -- 2 keeps
# startup near-instant; raise for a lower-variance (but slower to build)
# val estimate.
DEBUG_VAL_N_BATCHES = 2


def _build_debug_val_batch(cfg: DictConfig, t: DictConfig, device: str, gen_device: str,
                            tabicl_model, tabicl_k_folds: int, tabicl_split_calib_frac: float):
    """The first `DEBUG_VAL_N_BATCHES` batches of train.py's own fixed
    live-generation validation set (see live_dataset.py::
    build_fixed_live_val_batches) -- NOT an independent sample.

    `tabicl_model` here is applied unconditionally when it is not None (this
    function never receives a tabicl_mix_weights -- see the call site in
    main(), which passes a val-only TabICL load decoupled from whatever
    data.z_train_source/z_train_tabicl_mix_* the training loop itself uses).
    So by default z_test (and hence z_train) in every val batch comes from
    real TabICL K-fold PIT, regardless of what the model trained on --
    scoring against the same approximate marginal real deployment data would
    produce.

    Under training.val_analytic_only (e.g. `+experiment=analytic_only`) main()
    passes tabicl_model=None instead, and the val batches are generated with
    the exact analytic GP-LOO residual and exact Gaussian log-density. That is
    the whole point of that mode -- isolating the copula head's error from the
    frozen marginal's approximation error -- and it also means this script
    needs no TabICL checkpoint at all there.

    generate_gp_batch fully reseeds python/numpy/torch RNGs from cfg.seed on
    every call (see data_gen.py's module docstring), so batch i there is a
    deterministic function of (cfg, val_seed + i*104_729, batch_size,
    tabicl checkpoint weights) alone. Reusing that exact seed formula here
    with the same training.live_val_seed/training.batch_size/data config/
    tabicl checkpoint as the run being debugged reproduces those episodes
    byte-for-byte -- so this val loss is a genuine (if smaller/higher-
    variance) subsample of whatever train.sh logs as val/y_nll_total, not a
    different validation distribution. That equivalence breaks the moment
    training.batch_size, training.live_val_seed, the data.* config, or the
    resolved TabICL checkpoint differ from the run you're comparing against.

    Also computes the gap's fixed operand here, once, via
    pit.episode_posterior_ceiling: the exact Schur-complement GP posterior per
    raw episode (return_kernel_metadata=True gives it the kernel metadata it
    needs), per-point-normalized. This is a property of the fixed episodes
    alone, independent of the model being trained, so it's computed once here
    rather than every validation call -- the same caching train.py's builders
    now do.

    The gap reported below is on the TOTAL (marginal + copula), not the copula
    term alone. That's deliberate, and it is a fix rather than a preference:
    the model's z_test is standardized under the GP PRIOR's (mu_star,
    sigma_star), while gp_analytical_posterior's nll_post_copula is the Sklar
    split of the POSTERIOR (see pit.py's "Deliberately NOT scored via
    corr_nll_single" comment). The two copula terms are therefore split under
    different marginals and are not like-for-like -- their difference is not a
    meaningful quantity. The totals are: both are ordinary Y-space log
    densities evaluated at the same y_test, so total - nll_post is a valid
    gap with a provable >= 0 expectation. nll_post_copula is still returned,
    labelled, for eyeballing where a gap sits, but it is not the headline.
    """
    # Each batch is kept SEPARATELY collated, exactly like
    # build_fixed_live_val_batches's own `batches: List[dict]` -- d_features
    # is sampled once per generate_gp_batch call (data_gen.py::
    # _sample_d_features) and can differ across the two calls below, so
    # concatenating their episodes into one collate_fn call would crash the
    # same way a cross-shard variable-d training batch would.
    val_seed = int(t.get("live_val_seed", 20260723))
    batch_size = int(t.batch_size)
    batches = []
    n_episodes = 0
    oracle_total_per_point: list[float] = []
    oracle_copula_per_point: list[float] = []
    for i in range(DEBUG_VAL_N_BATCHES):
        val_cfg = OmegaConf.merge(cfg, OmegaConf.create({"seed": val_seed + i * 104_729}))
        episodes = generate_gp_batch(
            val_cfg, batch_size, device=gen_device,
            tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac,
            return_kernel_metadata=True,
        )
        n_episodes += len(episodes)
        for ep in episodes:
            # Returns None on a kernel schema with no analytic posterior --
            # skip that one episode's ceiling rather than crash the run.
            ceil_ep = episode_posterior_ceiling(ep)
            if ceil_ep is None:
                continue
            oracle_total_per_point.append(ceil_ep["nll_post"])
            oracle_copula_per_point.append(ceil_ep["nll_post_copula"])
        batch = {k: v.to(device, non_blocking=True) for k, v in collate_fn(episodes).items()}
        batches.append(batch)
    oracle_total_nll = (
        sum(oracle_total_per_point) / len(oracle_total_per_point)
        if oracle_total_per_point else float("nan")
    )
    oracle_copula_nll = (
        sum(oracle_copula_per_point) / len(oracle_copula_per_point)
        if oracle_copula_per_point else float("nan")
    )
    return n_episodes, val_seed, batch_size, batches, oracle_total_nll, oracle_copula_nll


def _build_episode_batch(cfg: DictConfig, n: int, seed: int, device: str,
                          tabicl_model, tabicl_k_folds: int, tabicl_split_calib_frac: float,
                          gen_device: str, return_kernel_metadata: bool = False,
                          tabicl_mix_weights=None):
    call_cfg = OmegaConf.merge(cfg, OmegaConf.create({"seed": seed}))
    episodes = generate_gp_batch(
        call_cfg, n, device=gen_device,
        tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
        tabicl_split_calib_frac=tabicl_split_calib_frac,
        tabicl_mix_weights=tabicl_mix_weights,
        return_kernel_metadata=return_kernel_metadata,
    )
    batch = {k: v.to(device, non_blocking=True) for k, v in collate_fn(episodes).items()}
    return episodes, batch


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t_script0 = time.perf_counter()
    torch.manual_seed(cfg.seed)
    t = cfg.training
    device = (
        "cuda" if t.device == "auto" and torch.cuda.is_available()
        else ("cpu" if t.device == "auto" else t.device)
    )

    # conf/config.yaml's training.steps default (1_000_000) is sized for a
    # real production run, not a "watch it start" debug session -- if the
    # caller didn't lower it explicitly, cap it here instead of silently
    # looping for days. Explicit training.steps=N overrides always win since
    # they replace this value before this check ever runs.
    if int(t.steps) >= 100_000:
        print(f"[train_fast] training.steps={int(t.steps)} looks like the production default -- capping to 60 for this debug run (pass training.steps=N to override).")
        t.steps = 60

    z_train_source = str(cfg.data.get("z_train_source", "analytic"))
    tabicl_model = None
    gen_device = "cpu"
    tabicl_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 10))
    tabicl_split_calib_frac = (
        float(cfg.data.get("z_train_split_calib_frac", 1.0))
        if z_train_source == "tabicl_split" else 0.0
    )

    # Fixed-fraction z_train mixing (alternate per-episode between the
    # analytic residual and real-TabICL PIT instead of committing the whole
    # run to one source). Reuses train.py's data.z_train_tabicl_mix_enabled/
    # _floor_frac/_max_frac knobs (see conf/data/gp_tasks.yaml) rather than
    # inventing new ones, but train_fast.py only supports the FIXED-fraction
    # case (floor_frac == max_frac): train.py's floor != max path measures a
    # real per-kernel-family TabICL-vs-analytic gap via
    # train.py::_compute_tabicl_z_train_gap first, which runs its own
    # synthetic-kernel probes -- exactly the slow startup this script exists
    # to skip. Use train.py directly if you need that adaptive weighting.
    mix_enabled = bool(cfg.data.get("z_train_tabicl_mix_enabled", False))
    tabicl_mix_weights = None
    if mix_enabled:
        floor_frac = float(cfg.data.get("z_train_tabicl_mix_floor_frac", 0.05))
        max_frac = float(cfg.data.get("z_train_tabicl_mix_max_frac", 0.35))
        if floor_frac != max_frac:
            raise ValueError(
                f"data.z_train_tabicl_mix_enabled=true with floor_frac={floor_frac} != "
                f"max_frac={max_frac} needs the adaptive per-kernel-family gap "
                "measurement train_fast.py deliberately skips -- set both to the same "
                "fixed mixing fraction (e.g. 0.5 for a 50/50 alternation), or run "
                "src/train.py directly for the adaptive version."
            )
        tabicl_mix_weights = torch.full((len(_COMPOSABLE_KERNELS),), floor_frac, dtype=torch.float32)

    if z_train_source in ("tabicl", "tabicl_split") or mix_enabled:
        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                f"data.z_train_source={z_train_source} (or data.z_train_tabicl_mix_enabled=true) "
                "requires a resolvable TabICL checkpoint (tabicl.ckpt with tabicl.pretrained=true, "
                "or tabicl.pit_ckpt) -- or pass data.z_train_source=analytic and "
                "data.z_train_tabicl_mix_enabled=false to skip PIT entirely."
            )
        print(f"[train_fast] Loading frozen TabICL marginal for PIT: {ckpt}")
        t_pit0 = time.perf_counter()
        tabicl_model = load_tabicl(ckpt, device)
        gen_device = device
        print(f"[train_fast] TabICL marginal loaded in {time.perf_counter() - t_pit0:.1f}s")

    mix_desc = f" mix_frac={float(tabicl_mix_weights[0]):.2f}" if tabicl_mix_weights is not None else ""
    print(
        f"[train_fast] model={cfg.model.get('rank')}rank/"
        f"{cfg.model.get('correlation_parametrization', 'covnorm')} "
        f"data=P[{cfg.data.P_min}..{cfg.data.P_max}] N[{cfg.data.N_min}..{cfg.data.N_max}] "
        f"z_train_source={z_train_source}{mix_desc} device={device}"
    )

    # Debug val set's z_test (and hence z_train too -- data_gen.py couples
    # them, see generate_gp_batch's z_train-source-override comment) normally
    # comes from real-TabICL PIT, unconditionally -- decoupled from whatever
    # z_train_source/z_train_tabicl_mix_* the TRAINING steps above use. This
    # matches eval_checkpoint.py's own --z_train_source tabicl default ("the
    # real deployment" signal): you can train cheaply on the analytic oracle
    # (or a mix) while still validating against the actual approximate
    # TabICL marginal the model will see once deployed. Reuses tabicl_model
    # if training already loaded one (z_train_source=tabicl/tabicl_split or
    # mixing enabled); otherwise loads a second copy just for val.
    #
    # training.val_analytic_only (`+experiment=analytic_only`) is the explicit
    # opt-out: score against the exact analytic marginal instead, load no
    # TabICL at all, and consequently need no checkpoint to exist. Validated
    # against data.z_train_source/z_train_corruption_enabled by
    # validate_analytic_only, same as in train.py.
    analytic_only = validate_analytic_only(cfg)
    if analytic_only:
        val_tabicl_model, val_gen_device = None, "cpu"
        print(
            "[train_fast] training.val_analytic_only=true -- debug val set uses the "
            "EXACT analytic GP marginal (no TabICL PIT, no checkpoint needed)."
        )
    elif tabicl_model is not None:
        val_tabicl_model, val_gen_device = tabicl_model, gen_device
    else:
        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                "train_fast.py's debug val set scores z_test through real TabICL "
                "PIT, which requires a resolvable TabICL checkpoint (tabicl.ckpt with "
                "tabicl.pretrained=true, or tabicl.pit_ckpt) -- or pass "
                "+experiment=analytic_only to score against the exact analytic "
                "marginal instead, which needs no checkpoint."
            )
        print(f"[train_fast] Loading frozen TabICL marginal for val z_test: {ckpt}")
        t_pit0 = time.perf_counter()
        val_tabicl_model = load_tabicl(ckpt, device)
        val_gen_device = device
        print(f"[train_fast] TabICL marginal loaded in {time.perf_counter() - t_pit0:.1f}s")

    t_val0 = time.perf_counter()
    (n_val_debug, val_seed, val_batch_size, val_batches,
     oracle_total_nll, oracle_copula_nll) = _build_debug_val_batch(
        cfg, t, device, val_gen_device, val_tabicl_model, tabicl_k_folds, tabicl_split_calib_frac,
    )
    print(
        f"[train_fast] Built val set: first {n_val_debug} episodes "
        f"({DEBUG_VAL_N_BATCHES} batches of {val_batch_size}) of train.sh's own fixed "
        f"val set (live_val_seed={val_seed}) in {time.perf_counter() - t_val0:.1f}s -- "
        "byte-identical to (a prefix of) train.sh's val/y_nll_total ONLY if "
        "training.batch_size/training.live_val_seed/data.*/the resolved TabICL "
        "checkpoint match the run you're comparing against."
    )
    print(
        f"[train_fast] Oracle (exact GP posterior) TOTAL NLL on this val set: "
        f"{oracle_total_nll:.4f} nats/pt -- the gap below is the model's total NLL "
        f"minus this, and 0 is Bayes-optimal. (Its copula component is "
        f"{oracle_copula_nll:.4f}, shown for reference only: it is the Sklar split of "
        "the POSTERIOR while the model's copula term is split under the PRIOR's "
        "sigma_star, so the two copula numbers are not directly comparable -- only "
        "the totals are.)"
    )

    t_model0 = time.perf_counter()
    model = build_copula_transformer(cfg).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_fast] Model built in {time.perf_counter() - t_model0:.1f}s ({n_train_params:,} trainable params)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    muon_params = [p for p in trainable if p.ndim >= 2]
    adamw_params = [p for p in trainable if p.ndim < 2]
    optimizer = Muon(
        [
            {
                "params": muon_params, "use_muon": True, "lr": t.muon_lr,
                "weight_decay": t.muon_weight_decay, "momentum": t.muon_momentum,
                "matched_adamw_rms": t.muon_matched_adamw_rms, "ns_steps": t.muon_ns_steps,
                "nesterov": t.muon_nesterov, "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
            {
                "params": adamw_params, "use_muon": False, "lr": t.muon_lr,
                "weight_decay": 0.0, "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
        ]
    )
    lr_min_frac = t.muon_lr_min / t.muon_lr

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(device=device) if (use_amp and amp_dtype == torch.float16) else None

    start_step = 0
    resume_ckpt = t.get("resume_ckpt", None)
    if resume_ckpt:
        ckpt_step = load_checkpoint(resume_ckpt, model, device, optimizer=optimizer, scaler=scaler)
        if bool(t.get("resume_reset_schedule", False)):
            print(f"[train_fast] Resumed weights+optimizer from {resume_ckpt} (step {ckpt_step}) -- resetting to step 0")
        else:
            start_step = ckpt_step
            print(f"[train_fast] Resumed weights+optimizer from {resume_ckpt} -- continuing from step {start_step}")
    if start_step > 0:
        for group in optimizer.param_groups:
            group["initial_lr"] = t.muon_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(cfg.model.get("correlation_parametrization", "covnorm"))
    nll_weight = float(t.get("nll_weight", 1.0))
    aux_mae_weight = float(t.get("aux_mae_weight", 0.0))
    triu_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    print(f"[train_fast] Ready to train after {time.perf_counter() - t_script0:.1f}s (steps={int(t.steps)}, batch_size={int(t.batch_size)})")
    print("[train_fast] Training loop started (Ctrl-C to stop)\n")

    model.train()
    for step in range(start_step + 1, int(t.steps) + 1):
        step_t0 = time.perf_counter()
        _, batch = _build_episode_batch(
            cfg, int(t.batch_size), seed=int(cfg.seed) + step * 104_729, device=device,
            tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac, gen_device=gen_device,
            tabicl_mix_weights=tabicl_mix_weights,
        )
        data_ms = (time.perf_counter() - step_t0) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        out, Sigma, parts, loss, aux_mae, grad_norm = _run_train_step(
            model=model, optimizer=optimizer, scheduler=scheduler, trainable=trainable,
            batch=batch, device=device, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
            clip_grad_norm=float(t.clip_grad_norm), nll_weight=nll_weight, aux_mae_weight=aux_mae_weight,
            jitter=jitter, triu_cache=triu_cache, phase_start=lambda: None, phase_end=lambda name, s: None,
            parametrization=parametrization,
        )
        step_ms = (time.perf_counter() - step_t0) * 1000.0

        if step % DEBUG_LOG_EVERY == 0:
            stats = _sigma_stats(Sigma.detach(), batch["test_mask"])
            gn = float(grad_norm) if grad_norm is not None else float("nan")
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{step:6d}/{int(t.steps)}] loss={loss.item():.4f} "
                f"(cop={parts['copula'].item():.4f} mar={parts['marginal'].item():.4f} aux={aux_mae.item():.4f}) "
                f"offdiag_mean={stats['offdiag_mean']:+.4f} offdiag_std={stats['offdiag_std']:.4f} "
                f"grad_norm={gn:.3f} lr={lr_now:.2e} | data={data_ms:.0f}ms step={step_ms:.0f}ms"
            )

        if step % DEBUG_VAL_EVERY == 0 or step == int(t.steps):
            model.eval()
            totals, copulas, marginals = [], [], []
            with torch.no_grad():
                for vb in val_batches:
                    _, _, val_parts, _, _ = _forward_and_loss(
                        model=model, batch=vb, device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                        nll_weight=nll_weight, aux_mae_weight=0.0, jitter=jitter, triu_cache=triu_cache,
                        parametrization=parametrization,
                    )
                    totals.append(val_parts["total"].item())
                    copulas.append(val_parts["copula"].item())
                    marginals.append(val_parts["marginal"].item())
            model.train()
            model_total_nll = sum(totals) / len(totals)
            model_copula_nll = sum(copulas) / len(copulas)
            # TOTAL gap, not the copula gap this used to print. Both operands
            # here are ordinary Y-space log densities at the same y_test, so
            # this difference is a real quantity with a provable >= 0
            # expectation. The old copula_gap subtracted two copula terms that
            # are Sklar-split under DIFFERENT marginals (model: the prior's
            # sigma_star; oracle: the posterior) -- see _build_debug_val_batch.
            gap = model_total_nll - oracle_total_nll
            print(
                f"          -- val({n_val_debug}) total={model_total_nll:.4f} "
                f"copula={model_copula_nll:.4f} marginal={sum(marginals) / len(marginals):.4f} "
                f"| gap={gap:+.4f} (vs. oracle total {oracle_total_nll:.4f} -- lower is better, 0 = Bayes-optimal)"
            )

    if t.get("ckpt_dir", None):
        save_checkpoint(model, optimizer, scheduler, cfg, int(t.steps))
        print(f"[train_fast] Saved checkpoint to {t.ckpt_dir}")
    print("\n[train_fast] Done.")


if __name__ == "__main__":
    main()
