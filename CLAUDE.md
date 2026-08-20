 Workflow to run:
  # 1. Generate PIT episodes
  python src/generate_pit_dataset.py dataset.n_episodes=5000
  # 2. Train
  python src/train.py training.dataset_dir=./data/pit_episodes
  # 3. Evaluate vs. classical baselines (synthetic GP episodes). Defaults to
  #    --z_train_source tabicl (K-fold TabICL PIT context, matching real
  #    deployment) so the total marginal+copula NLL table is populated;
  #    pass --z_train_source oracle for the exact-GP-LOO idealized upper bound.
  python eval/runners/eval_checkpoint.py --ckpt ./checkpoints/copula_transformer/step_0029999_final.pt

  # 3b. Evaluate on real-world datasets (UCI Beijing PM2.5, California Housing)
  python eval/runners/run_benchmarks.py

python src/train_on_datasets.py --config conf/config.yaml --ckpt ./checkpoints/copula-tabicl/

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
