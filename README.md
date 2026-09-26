# Copula Inter

Inter-instance Gaussian copula training and evaluation. Use Python 3.12. The
checked-in `tabicl_upstream` source is included in this project's install.

```bash
uv sync --locked --extra dev --extra cpu
uv run --no-sync python -m pytest -q tests/test_loss.py tests/test_data.py tests/test_eval_checkpoint.py
```

The checked-in `uv.lock` pins the tested environment. The `cpu` extra selects
the smaller PyTorch CPU wheel for local checks; use `--extra gpu` on a CUDA
machine.
A standard editable install is also supported with
`python -m pip install -e '.[dev]'`.

To generate a small on-disk dataset and train from it:

```bash
python src/generate_pit_dataset.py data.n_tasks=5000 data.dataset_dir=./data/pilot
python src/train.py training.live_generation=false training.dataset_dir=./data/pilot/pit
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
python eval/runners/eval_checkpoint.py --ckpt checkpoints/copula_transformer/step_0029999_final.pt
```

Use `--dump_episodes scores.json` during evaluation to save per-episode
scores; `python -m eval.results scores.json` renders totals later without
loading either model.

The CPU pull-request gate is `.github/workflows/ci.yml`. The full suite can
require pretrained models, a GPU, or external ERA5 data; run those tests in
the corresponding environment. Configuration lives in `conf/`; use
`python src/train.py --cfg job` to inspect the composed settings without
starting training. See [AGENTS.md](AGENTS.md) for the code map and
[CLAUDE.md](CLAUDE.md) for experiment-specific notes.
