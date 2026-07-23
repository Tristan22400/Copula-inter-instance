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
 