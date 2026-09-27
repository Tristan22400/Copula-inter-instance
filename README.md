# Copula Inter

Inter-instance Gaussian copula training and evaluation. Use Python 3.12. The
checked-in `tabicl_upstream` source is included in this project's install.

```bash
uv sync --locked --extra dev --extra cpu
uv run --no-sync pre-commit install      # ruff check + format on every commit
uv run --no-sync python -m pytest -q -n auto -m "not slow and not gpu and not pretrained and not external_data"
```

The checked-in `uv.lock` pins the tested environment. The `cpu` extra selects
the smaller PyTorch CPU wheel for local checks; use `--extra gpu` on a CUDA
machine.
A standard editable install is also supported with
`python -m pip install -e '.[dev]'`. With the shared conda env, run
`source scripts/_env.sh` instead so the current checkout is imported.

To generate a small on-disk dataset and train from it:

```bash
python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.dataset_dir=./data/pilot
python -m copula_inter.train training.live_generation=false training.dataset_dir=./data/pilot/pit
```

Generation with the default `data.z_train_source=tabicl` needs the configured
marginal weights. Add `data.z_train_source=analytic` to generate PIT values
without those weights; training still needs the selected copula backbone and
its configured validation resources. Data generation
creates a manifest and rejects resuming into a directory with different
generation settings or marginal checkpoint bytes. Use a fresh directory for
a different dataset. The training command is a full run; override
`training.steps` for a bounded experiment.

Evaluate a trained checkpoint with:

```bash
python -m eval.runners.eval_checkpoint ckpt=kernel-sweep-all-tabicl-retrain-15k
```

Eval runners take Hydra `key=value` overrides; add `--cfg job` to list every
key. Use `output.dump_episodes=scores.json` during evaluation to save per-episode
scores; `python -m eval.results scores.json` renders totals later without
loading either model.

The CPU pull-request gate is `.github/workflows/ci.yml`; it runs the test
command above. Tests that need a GPU, pretrained weights or external ERA5 data
carry the `gpu` / `pretrained` / `external_data` markers; run the full suite on
a Grid5000 GPU node with `oarsub -S ./scripts/test_full.sh` before merging.
Add `training.startup_probes=false` to a training command to skip the fixed
validation probes built at startup (kernel_fit/*, era5_fit/*, oracle_diag/*). Configuration lives in `conf/`; use
`python -m copula_inter.train --cfg job` to inspect the composed settings without
starting training. See [AGENTS.md](AGENTS.md) for the code map and
[CLAUDE.md](CLAUDE.md) for experiment-specific notes.
