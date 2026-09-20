 Workflow to run:
  # 1. Generate PIT episodes. data.z_train_source selects the marginal:
  #    analytic (oracle) | tabicl | tabicl_split | exaone | tabpfn | tabldm.
  #    Non-TabICL backends need no GPU here (unlike live generation), just time.
  python src/generate_pit_dataset.py dataset.n_episodes=5000
  # 2. Train
  python src/train.py training.dataset_dir=./data/pit_episodes
  # 3. Evaluate vs. classical baselines (synthetic GP episodes). Defaults to
  #    --z_train_source tabicl (K-fold TabICL PIT context, matching real
  #    deployment) so the total marginal+copula NLL table is populated;
  #    pass --z_train_source oracle for the exact-GP-LOO idealized upper bound,
  #    or exaone/tabpfn/tabldm to score against the marginal a run trained with.
  python eval/runners/eval_checkpoint.py --ckpt ./checkpoints/copula_transformer/step_0029999_final.pt

  # 3a. SAME comparison, REAL data: the identical baseline table on real
  #     ARCO-ERA5 2m-temperature episodes instead of synthetic GP draws
  #     (eval/data/era5_episodes.py). Same classical baselines, same nested-CV
  #     best-of-baselines, same two tables, same resumable caches. Real data
  #     has no generating kernel, so the "Oracle (prior)" row and the analytic
  #     GP prior/posterior Y-space rows are nan, and the shared z_test every
  #     row is scored against is the frozen-TabICL K-fold PIT rather than a
  #     ground-truth marginal -- read the TOTAL Y-space NLL table, which is a
  #     proper scoring rule regardless. --z_train_source=oracle is rejected.
  #     Targets are z-scored per episode before the baselines are fitted
  #     (--era5_standardize_y, on by default) and converted back to raw Kelvin
  #     nats: ERA5 y is ~280 K while classical.py's GP hyperpriors assume
  #     data_gen.py's O(1) draws, so leaving them raw handicaps the baselines
  #     on units alone while our model normalizes internally.
  #     Needs the corpus cached once (defaults to the held-out val year):
  #       python eval/data/fetch_era5_global.py --start 2023-01 --n-months 12 \
  #           --cache-dir ./eval/data/cache/era5_global_val
  python eval/runners/eval_checkpoint.py --era5 --n_episodes 400 \
      --ckpt ./checkpoints/copula_transformer/step_0029999_final.pt
  oarsub -S "./scripts/eval_checkpoint_era5.sh --ckpt <ckpt>"   # on Grid5000
  #     Baseline fitting is ~98% of the runtime and is checkpoint-independent,
  #     so a SECOND checkpoint over the same episodes/geometry reuses the whole
  #     --baseline_cache and only redoes the ICL forward pass. Give each
  #     checkpoint its own --results_cache, share the --baseline_cache.

  # 3b. Evaluate on real-world datasets (UCI Beijing PM2.5, California Housing)
  python eval/runners/run_benchmarks.py

  # 4. Finetune an existing checkpoint on real, worldwide ARCO-ERA5 data
  #    (random geographic region + random grid resolution every episode,
  #    instead of synthetic GP kernels). One-time corpus fetch first, then:
  python eval/data/fetch_era5_global.py --start 2022-01 --n-months 24
  python src/finetune_era5.py --ckpt ./checkpoints/copula-tabicl/step_0029999_final.pt

Marginal fine-tuning (Phase A) — make the MARGINAL branch correct, separately from
the copula. The loss is copula + marginal (Sklar), but the marginal comes from a
FROZEN TabICL, so its term has zero trainable parameters and no copula run can
improve it. Phase A fine-tunes that standalone TabICL; the two phases meet only at
a checkpoint path. src/model.py and conf/config.yaml are untouched.
  # 1. Measure the defect first -- zero training, one table. The headline number is
  #    the marginal-NLL gap to the ANALYTIC GP oracle (y is a pure GP draw, so the
  #    correct marginal posterior is known in closed form).
  python eval/runners/marginal_calibration_eval.py --ckpt pretrained
  # 2. Fine-tune. Hydra-native (no argparse), own wandb project copula-inter-marginal.
  #    Tier 0 = label path + ICL norms + decoder (~1.6M/5.5%); escalate to tier 1
  #    (+ LoRA on icl_predictor) only if the oracle gap plateaus above zero.
  python src/finetune_marginal.py                      # or: marginal.tier=1
  oarsub -S ./scripts/finetune_marginal.sh             # on Grid5000
  #    marginal.lora_all_layers=true (the DEFAULT) puts LoRA at ONE shared rank
  #    on every 2-D weight matrix of whichever backbone is selected, so the
  #    tier ladder above only matters as an ablation (lora_all_layers=false).
  #    Measured coverage at rank 8: tabicl 153 matrices / 854K trainable
  #    (2.91%), tabldm 295 / 2.00M (2.73%), exaone 360 / 1.43M (6.33%).
  #    Other marginal backbones (src/marginal_backbones.py): tabpfn is wired
  #    but licence-gated and never executed here. Non-tabicl checkpoints are
  #    loaded back via marginal_backends.make_regressor(..., ckpt=<path>),
  #    not pit.load_tabicl.
  #    Phase A scores each model's NATIVE 999-level decoder grid by default
  #    (marginal.probs_n=null) -- TabICL, TabLDM and EXAONE all emit 999, so
  #    the objective is comparable across backbones. Set probs_n=<int> only to
  #    resample onto a coarser grid.
  python src/finetune_marginal.py marginal.backbone=tabldm
  python src/finetune_marginal.py marginal.backbone=exaone
  #    Stage-ladder ablation (only tabicl/tabldm can climb it -- exaone's
  #    attention has no swappable module, see marginal_backbones.MAX_TIER):
  python src/finetune_marginal.py marginal.lora_all_layers=false marginal.tier=1
  # 3. Re-measure, then gate on real data (must not regress -- the whole point of a
  #    TabICL marginal is non-Gaussian tabular transfer, which GP-only training can
  #    destroy), then hand the result to a normal copula run:
  python eval/runners/marginal_calibration_eval.py --ckpt <the _final.pt>
  python eval/runners/run_benchmarks.py
  python src/train.py tabicl.pit_ckpt=<the _final.pt>
Phase A checkpoints for backbone=tabicl are plain TabICL ({"config","state_dict"});
other backbones use the same shape plus a "backbone" tag. Both are registered in
eval/configs/checkpoints.py::MARGINAL_FAMILIES -- a SEPARATE registry from
CHECKPOINT_FAMILIES, which holds copula checkpoints that `sweep --checkpoints all`
iterates. Do not mix the two.

Spatial-correlation diagnostics (real ERA5 + synthetic-kernel ground truth), one CLI:
  # One-shot: sweep every registered checkpoint (real + synthetic) -> baseline curve
  # fits -> report figures. Auto-fetches/caches ERA5, zero required flags.
  # `sweep --mode real` (and hence `all`) also scores a real total (marginal+
  # copula) joint NLL per config on held-out real-ERA5 points, alongside the
  # correlation-curve-shape model_r2 -- model_r2 alone can't tell you how many
  # nats worse the actual predictive density is.
  python eval/runners/spatial_correlation_eval.py all

  # Individual subcommands (see --help on each for full flag list):
  python eval/runners/spatial_correlation_eval.py diagnose --ckpt kernel-sweep-all-tabicl-retrain-15k --mode real --region western_europe --grid-size 24
  python eval/runners/spatial_correlation_eval.py sweep --mode synthetic --checkpoints all
  python eval/runners/spatial_correlation_eval.py baseline --mode real
  python eval/runners/spatial_correlation_eval.py report

  # Real ERA5 marginal-quantile calibration (independence-copula + per-quantile ECE
  # diagnostics), real TabICL, auto-fetches a small ERA5 sample if --nc-path omitted:
  python eval/runners/era5_calibration_eval.py

Checkpoint families, named regions, and shared constants for the above live in
eval/configs/ (checkpoints.py, regions.py, constants.py) — add a new checkpoint family
there rather than hardcoding a path in a script.
