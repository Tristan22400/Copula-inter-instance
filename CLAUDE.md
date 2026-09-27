 Environment: `source scripts/_env.sh` (conda env + this checkout's src/, root and
tabicl_upstream/src first on PYTHONPATH -- works from any worktree), or
`uv sync --extra dev --extra gpu` and use `.venv/bin/python`. Code lives in the
`copula_inter` package (src/copula_inter/); no script edits sys.path.

Workflow to run:
  # 1. Generate PIT episodes. data.z_train_source selects the marginal:
  #    analytic (oracle) | tabicl | tabicl_split | exaone | tabpfn | tabldm.
  #    Non-TabICL backends need no GPU here (unlike live generation), just time.
  python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.dataset_dir=./data/pilot
  # 2. Train
  python -m copula_inter.train training.live_generation=false training.dataset_dir=./data/pilot/pit
  # 3. Evaluate vs. classical baselines (synthetic GP episodes). Defaults to
  #    marginal.z_train_source=tabicl (K-fold TabICL PIT context, matching real
  #    deployment) so the total marginal+copula NLL table is populated;
  #    pass marginal.z_train_source=oracle for the exact-GP-LOO idealized upper
  #    bound, or exaone/tabpfn/tabldm to score against the marginal a run
  #    trained with (then also autoregressive.enabled=false).
  #    Every eval runner takes Hydra key=value overrides (typed dataclass
  #    configs, conf/eval/<runner>.yaml); `--cfg job` lists every key.
  python -m eval.runners.eval_checkpoint ckpt=kernel-sweep-all-tabicl-retrain-15k

  # 3a. SAME comparison, REAL data: the identical baseline table on real
  #     ARCO-ERA5 2m-temperature episodes instead of synthetic GP draws
  #     (eval/data/era5_episodes.py). Same classical baselines, same nested-CV
  #     best-of-baselines, same two tables, same resumable caches. Real data
  #     has no generating kernel, so the "Oracle (prior)" row and the analytic
  #     GP prior/posterior Y-space rows are nan, and the shared z_test every
  #     row is scored against is the frozen-TabICL K-fold PIT rather than a
  #     ground-truth marginal -- read the TOTAL Y-space NLL table, which is a
  #     proper scoring rule regardless. marginal.z_train_source=oracle is rejected.
  #     Targets are z-scored per episode before the baselines are fitted
  #     (era5.standardize_y, on by default) and converted back to raw Kelvin
  #     nats: ERA5 y is ~280 K while classical.py's GP hyperpriors assume
  #     data_gen.py's O(1) draws, so leaving them raw handicaps the baselines
  #     on units alone while our model normalizes internally.
  #     Needs the corpus cached once (defaults to the held-out val year):
  #       python eval/data/fetch_era5_global.py --start 2023-01 --n-months 12 \
  #           --cache-dir ./eval/data/cache/era5_global_val
  python -m eval.runners.eval_checkpoint era5.enabled=true n_episodes=400 \
      ckpt=kernel-sweep-all-tabicl-retrain-15k
  oarsub -S "./scripts/eval_checkpoint_era5.sh ckpt=<ckpt>"   # on Grid5000
  #     Baseline fitting is ~98% of the runtime and is checkpoint-independent,
  #     so a SECOND checkpoint over the same episodes/geometry reuses the whole
  #     baselines.cache and only redoes the ICL forward pass. Give each
  #     checkpoint its own output.results_cache, share the baselines.cache.
  #
  #     AUTOREGRESSIVE row (eval/baselines/autoregressive.py), ON BY DEFAULT
  #     for the TabICL marginal. Same marginal, no copula head: reveal the test points one
  #     at a time so each prediction conditions on the ones already revealed.
  #     log p(y_1..y_N) = sum_i log p(y_s(i) | ctx, y_s(<i)) is an exact
  #     factorization of a joint density, so the row is directly comparable to
  #     every other row of the TOTAL table -- and it is the copula-free
  #     reference the copula head has to beat. Its Marginal column is the
  #     one-shot (independence) marginal, so its Copula column reads as exactly
  #     what the sequencing bought. Step 0 reproduces the one-shot PIT's
  #     log_pdf_test bit-for-bit (the chain keeps the full P+N table at every
  #     step and only moves the context/query split) -- tests/test_
  #     autoregressive.py pins that. Batched over era5.pit_batch episodes at
  #     once; GPU-bound, so its cost tracks the card: measured 1.5 s/episode
  #     on an RTX PRO 6000 Blackwell, 3.6 s/episode on an RTX A5000.
  #       autoregressive.enabled=false turn the row off
  #       autoregressive.order=natural reveal in grid order instead of a seeded
  #                                    per-episode permutation (an ICL model is
  #                                    not a coherent joint, so the total really
  #                                    does depend on the order; "natural" hands
  #                                    nearly every step a just-revealed
  #                                    neighbour and reads as a best case)
  #       autoregressive.conditioning=sample  append a DRAW instead of the true y
  #                                    (ancestral sampling). The printed number
  #                                    is then NOT a density of y_test and is
  #                                    NOT comparable to the other rows -- the
  #                                    table prints a warning saying so.
  #       autoregressive.max_context=K cap the chain's context (the episode's
  #                                    own P are always kept; oldest revealed
  #                                    dropped first)
  #       autoregressive.n_episodes=M  run the chain on the first M episodes
  #     TabICL marginal only -- the exaone/tabpfn/tabldm backends expose no
  #     one-query-at-a-time entry point and raise rather than drop the row.

  # 3b. Evaluate on real-world datasets (UCI Beijing PM2.5, California Housing)
  python -m eval.runners.run_benchmarks

  # 4. Finetune an existing checkpoint on real, worldwide ARCO-ERA5 data
  #    (random geographic region + random grid resolution every episode,
  #    instead of synthetic GP kernels). One-time corpus fetch first, then:
  python eval/data/fetch_era5_global.py --start 2022-01 --n-months 24
  python -m copula_inter.train experiment=finetune_era5 training.resume_ckpt=kernel-sweep-all-tabicl-retrain-15k model.rank=32
  #    model.* must match the checkpoint (that family is rank 32; the default
  #    preset is 128). The preset (conf/experiment/finetune_era5.yaml) lists the usual extras;
  #    oarsub -S "./scripts/finetune_era5.sh training.resume_ckpt=<ckpt> model.rank=32" on Grid5000 for this checkpoint family.

Marginal fine-tuning (Phase A) — make the MARGINAL branch correct, separately from
the copula. The loss is copula + marginal (Sklar), but the marginal comes from a
FROZEN TabICL, so its term has zero trainable parameters and no copula run can
improve it. Phase A fine-tunes that standalone TabICL; the two phases meet only at
a checkpoint path. src/copula_inter/model.py and conf/config.yaml are untouched.
  # 1. Measure the defect first -- zero training, one table. The headline number is
  #    the marginal-NLL gap to the ANALYTIC GP oracle (y is a pure GP draw, so the
  #    correct marginal posterior is known in closed form).
  python -m eval.runners.marginal_calibration_eval ckpt=pretrained
  # 2. Fine-tune. Hydra-native (no argparse), own wandb project copula-inter-marginal.
  #    Tier 0 = label path + ICL norms + decoder (~1.6M/5.5%); escalate to tier 1
  #    (+ LoRA on icl_predictor) only if the oracle gap plateaus above zero.
  python -m copula_inter.finetune_marginal                      # or: marginal.tier=1
  oarsub -S ./scripts/finetune_marginal.sh             # on Grid5000
  #    marginal.lora_all_layers=true (the DEFAULT) puts LoRA at ONE shared rank
  #    on every 2-D weight matrix of whichever backbone is selected, so the
  #    tier ladder above only matters as an ablation (lora_all_layers=false).
  #    Measured coverage at rank 8: tabicl 153 matrices / 854K trainable
  #    (2.91%), tabldm 295 / 2.00M (2.73%), exaone 360 / 1.43M (6.33%).
  #    Other marginal backbones (src/copula_inter/marginal_backbones.py): tabpfn is wired
  #    but licence-gated and never executed here. Non-tabicl checkpoints are
  #    loaded back via marginal_backends.make_regressor(..., ckpt=<path>),
  #    not pit.load_tabicl.
  #    Phase A scores each model's NATIVE 999-level decoder grid by default
  #    (marginal.probs_n=null) -- TabICL, TabLDM and EXAONE all emit 999, so
  #    the objective is comparable across backbones. Set probs_n=<int> only to
  #    resample onto a coarser grid.
  python -m copula_inter.finetune_marginal marginal.backbone=tabldm
  python -m copula_inter.finetune_marginal marginal.backbone=exaone
  #    Stage-ladder ablation (only tabicl/tabldm can climb it -- exaone's
  #    attention has no swappable module, see marginal_backbones.MAX_TIER):
  python -m copula_inter.finetune_marginal marginal.lora_all_layers=false marginal.tier=1
  # 3. Re-measure, then gate on real data (must not regress -- the whole point of a
  #    TabICL marginal is non-Gaussian tabular transfer, which GP-only training can
  #    destroy), then hand the result to a normal copula run:
  python -m eval.runners.marginal_calibration_eval ckpt=<the _final.pt>
  python -m eval.runners.run_benchmarks
  python -m copula_inter.train tabicl.pit_ckpt=<the _final.pt>
Phase A checkpoints for backbone=tabicl are plain TabICL ({"config","state_dict"});
other backbones use the same shape plus a "backbone" tag. Both are registered in
eval/configs/checkpoints.py::MARGINAL_FAMILIES -- a SEPARATE registry from
CHECKPOINT_FAMILIES, which holds copula checkpoints that `sweep --checkpoints all`
iterates. Do not mix the two.

Spatial-correlation diagnostics (real ERA5 + synthetic-kernel ground truth), one CLI:
  # One-shot: sweep every registered checkpoint (real + synthetic) -> baseline curve
  # fits -> report figures. Auto-fetches/caches ERA5, zero required flags.
  # command=sweep command.mode=real (and hence the default command=all) also scores a real total (marginal+
  # copula) joint NLL per config on held-out real-ERA5 points, alongside the
  # correlation-curve-shape model_r2 -- model_r2 alone can't tell you how many
  # nats worse the actual predictive density is.
  python -m eval.runners.spatial_correlation_eval

  # Individual subcommands (command=<name> --cfg job lists that command's keys):
  python -m eval.runners.spatial_correlation_eval command=diagnose command.ckpt=kernel-sweep-all-tabicl-retrain-15k command.mode=real command.region=western_europe command.grid_size=24
  python -m eval.runners.spatial_correlation_eval command=sweep command.mode=synthetic command.checkpoints=all
  python -m eval.runners.spatial_correlation_eval command=baseline command.mode=real
  python -m eval.runners.spatial_correlation_eval command=report

  # Real ERA5 marginal-quantile calibration (independence-copula + per-quantile ECE
  # diagnostics), real TabICL, auto-fetches a small ERA5 sample if nc_path is unset:
  python -m eval.runners.era5_calibration_eval

Checkpoint families, named regions, and shared constants for the above live in
eval/configs/ (checkpoints.py, regions.py, constants.py) — add a new checkpoint family
there rather than hardcoding a path in a script.
