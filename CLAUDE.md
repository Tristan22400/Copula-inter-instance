 Workflow to run:
  # 1. Generate PIT episodes
  python src/generate_pit_dataset.py dataset.n_episodes=5000
  # 2. Train
  python src/train.py training.dataset_dir=./data/pit_episodes
  # 3. Evaluate vs. classical baselines (synthetic GP episodes)
  python eval/runners/eval_checkpoint.py --ckpt ./checkpoints/copula_transformer/step_0029999_final.pt

  # 3b. Evaluate on real-world datasets (UCI Beijing PM2.5, California Housing)
  python eval/runners/run_benchmarks.py

python src/train_on_datasets.py --config conf/config.yaml --ckpt ./checkpoints/copula-tabicl/

Spatial-correlation diagnostics (real ERA5 + synthetic-kernel ground truth), one CLI:
  # One-shot: sweep every registered checkpoint (real + synthetic) -> baseline curve
  # fits -> report figures. Auto-fetches/caches ERA5, zero required flags.
  python eval/runners/spatial_correlation_eval.py all

  # Individual subcommands (see --help on each for full flag list):
  python eval/runners/spatial_correlation_eval.py diagnose --ckpt kernel-sweep-all-tabicl-retrain-60k --mode real --region western_europe --grid-size 24
  python eval/runners/spatial_correlation_eval.py sweep --mode synthetic --checkpoints all
  python eval/runners/spatial_correlation_eval.py baseline --mode real
  python eval/runners/spatial_correlation_eval.py report

  # Real ERA5 marginal-quantile calibration (independence-copula + per-quantile ECE
  # diagnostics), real TabICL, auto-fetches a small ERA5 sample if --nc-path omitted:
  python eval/runners/era5_calibration_eval.py

Checkpoint families, named regions, and shared constants for the above live in
eval/configs/ (checkpoints.py, regions.py, constants.py) — add a new checkpoint family
there rather than hardcoding a path in a script.
