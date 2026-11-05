# Workflows

How to run each stage of the project, from a fresh environment to evaluation
tables. Every Python entry point is a Hydra app: pass settings as `key=value`
overrides, and append `--cfg job` to print the composed configuration without
running anything. See the [glossary](glossary.md) for terms and
[architecture](architecture.md) for where the code lives.

## Environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `uv.lock`
pins the tested environment.

```bash
uv sync --locked --extra dev --extra cpu    # laptop / CI (CPU PyTorch wheel)
uv sync --locked --extra dev --extra gpu    # CUDA machine, e.g. a Grid5000 node
uv run --no-sync pre-commit install         # ruff check + format on every commit
```

Run commands through the venv with `uv run --no-sync python -m ...` (or
activate `.venv`). The job scripts under `scripts/` source `scripts/_env.sh`,
which:
- activates the checkout's `.venv`, or the main checkout's `.venv` when run from
  a git worktree, or `$COPULA_VENV` when set;
- puts this checkout's sources first on `PYTHONPATH`;
- excludes GPU models known not to work.

**One-time Grid5000 setup.** Run this on a node, from the checkout on your
storage:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs uv into ~/.local/bin
cd ~/thoth_storage/copula-inter
uv sync --locked --extra dev --extra gpu
```

The `.venv` lives in the checkout, so every OAR job on that storage reuses it.

**TabPFN-3.5 and Kumo (optional).** The two extra marginal backends are uv
extras: `--extra tabpfn` (`tabpfn35_fast`, PriorLabs TabPFN-3.5 Fast) and
`--extra kumo` (`kumo_small`, NVIDIA Kumo Tabular small). Add them to the
`uv sync` command and keep them on every later `uv sync`, which otherwise
uninstalls them. TabPFN also needs a PriorLabs API key in `TABPFN_TOKEN` whose
account accepted the TabPFN-3.5 licence (see [architecture](architecture.md)).
Without an extra, its tests are skipped and selecting it raises an ImportError.
If your laptop mounts the same storage (sshfs), don't run `uv sync` through the
mount. That would overwrite the cluster's GPU venv with a CPU one linked to the
laptop's Python. Keep a laptop venv elsewhere:

```bash
UV_PROJECT_ENVIRONMENT=~/.local/share/venvs/copula-inter-cpu uv sync --locked --extra dev --extra cpu
```

## 1. Generate an on-disk dataset

```bash
python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.dataset_dir=./data/pilot
oarsub -S "./scripts/generate_dataset.sh data.n_tasks=5000"        # on Grid5000
```

`data.z_train_source` selects the marginal that produces the PIT: `analytic`
(the exact-GP oracle, no model weights needed), `tabicl` (default),
`tabicl_split`, `tabpfn35_fast` or `kumo_small`. The last two read Phase-A
weights from `data.z_train_marginal_ckpt` (default: pretrained). They need no
GPU at generation time; they are just slower.

Generation writes a manifest. It refuses to resume into a directory created with
different settings or different marginal checkpoint bytes, so use a fresh
directory for a different dataset.

## 2. Train the copula model

```bash
python -m copula_inter.train training.live_generation=false training.dataset_dir=./data/pilot/pit
oarsub -S "./scripts/train.sh <overrides>"                         # on Grid5000
```

Without `training.live_generation=false`, episodes are generated live in
DataLoader workers. Useful overrides:
- `training.steps=500` for a bounded run;
- `training.startup_probes=false` to skip building the fixed validation probes
  (`kernel_fit/*`, `era5_fit/*`, `oracle_diag/*`);
- `WANDB_MODE=disabled` to run offline.

### Fine-tune on real ERA5

Instead of synthetic GP kernels, this samples a random region and grid
resolution for every episode. Fetch the corpus once, then resume from a
checkpoint:

```bash
python -m copula_inter.era5.fetch_global --start 2022-01 --n-months 24
python -m copula_inter.train experiment=finetune_era5 \
    training.resume_ckpt=kernel-sweep-all-tabicl-retrain-15k model.rank=32
oarsub -S "./scripts/finetune_era5.sh training.resume_ckpt=<ckpt> model.rank=32"
```

`model.*` must match the checkpoint. That family is rank 32, while the default
preset is rank 128. `conf/experiment/finetune_era5.yaml` lists the usual extras.

## 3. Evaluate against classical baselines

### Synthetic GP episodes

```bash
python -m eval.runners.eval_checkpoint ckpt=kernel-sweep-all-tabicl-retrain-15k
oarsub -S "./scripts/eval_checkpoint.sh ckpt=<ckpt>"
```

`ckpt=` takes a family name from `copula_inter/checkpoint_registry.py` or a path.
The run prints two tables. The **total Y-space NLL** table (marginal + copula) is
the comparison to read, because it is a proper scoring rule. The copula-only
table compares correlations on a shared `z_test`.

`marginal.z_train_source` decides where the context PIT comes from:
- `tabicl` (default): K-fold TabICL, as in deployment;
- `oracle`: the exact-GP LOO, an idealized upper bound;
- `tabpfn35_fast` / `kumo_small`: score against the marginal a run was trained
  with. Its weights default to the checkpoint's own `data.z_train_marginal_ckpt`
  (override with `marginal.backend_ckpt=<path>`). Also pass
  `autoregressive.enabled=false`.

`output.dump_episodes=scores.json` saves per-episode scores, and
`python -m eval.results scores.json` re-renders the totals later without loading
any model.

### Real ERA5 episodes

These are the same baselines, nested-CV best-of-baselines, tables and caches, on
real 2 m temperature (`eval/data/era5_episodes.py`):

```bash
python -m copula_inter.era5.fetch_global --start 2023-01 --n-months 12 \
    --cache-dir ./eval/data/cache/era5_global_val            # once: the held-out year
python -m eval.runners.eval_checkpoint era5.enabled=true n_episodes=400 \
    ckpt=kernel-sweep-all-tabicl-retrain-15k
oarsub -S "./scripts/eval_checkpoint_era5.sh ckpt=<ckpt>"
```

Differences from synthetic data:
- Real data has no generating kernel, so the "Oracle (prior)" row and the
  analytic GP rows are `nan`, and `marginal.z_train_source=oracle` is rejected.
- The shared `z_test` is the frozen-TabICL K-fold PIT, not a ground-truth
  marginal. Read the total table.
- Targets are z-scored per episode before the baselines are fitted
  (`era5.standardize_y`, on by default), and their NLLs are converted back to
  raw-Kelvin nats. The baselines' GP hyperpriors assume O(1) targets, and our
  model normalizes internally, so raw ~280 K targets would handicap the
  baselines on units alone.

Fitting the baselines takes about 98 % of the runtime and doesn't depend on the
checkpoint. To compare a second checkpoint on the same episodes, share
`baselines.cache` and give each checkpoint its own `output.results_cache`; only
the ICL forward pass is redone.

### The autoregressive row

This row is on by default for the TabICL marginal. It uses the same marginal
with no copula head, revealing the query points one at a time so that each
prediction conditions on the ones already revealed:

    log p(y_1..y_N) = sum_i log p(y_s(i) | context, y_s(<i))

That sum is an exact factorization of a joint density, so the row is comparable
to every other row of the total table. It is the copula-free reference the
copula head has to beat:
- its Marginal column is the one-shot (independence) marginal;
- its Copula column is therefore exactly what the sequencing bought;
- step 0 reproduces the one-shot PIT's `log_pdf_test` bit for bit
  (`tests/test_autoregressive.py`).

| Override | Effect |
|---|---|
| `autoregressive.enabled=false` | Turn the row off |
| `autoregressive.order=natural` | Reveal in grid order instead of a seeded per-episode permutation. An ICL model is not a coherent joint, so the total depends on the order; `natural` gives nearly every step a just-revealed neighbour and reads as a best case |
| `autoregressive.conditioning=sample` | Append a draw instead of the true y (ancestral sampling). The number is then **not** a density of `y_test` and not comparable to the other rows; the table prints a warning |
| `autoregressive.max_context=K` | Cap the chain's context. The episode's own P rows are always kept, and the oldest revealed rows are dropped first |
| `autoregressive.n_episodes=M` | Run the chain on the first M episodes only |

It runs on the GPU, batched over `era5.pit_batch` episodes; measured at about
1.5 s/episode on an RTX PRO 6000 Blackwell and 3.6 s/episode on an RTX A5000.
It is TabICL-only: the other backends have no one-query-at-a-time entry point
and raise an error rather than silently dropping the row.

### Real-world tabular benchmarks

```bash
python -m eval.runners.run_benchmarks        # UCI Beijing PM2.5, California Housing
```

## 4. Phase A: fine-tune the marginal

The training loss is copula + marginal, but the marginal comes from a frozen
model, so no copula run can improve its term. Phase A fine-tunes that
standalone marginal; the two phases meet only at a checkpoint path.

1. **Measure the defect** (no training). The headline number is the marginal-NLL
   gap to the analytic GP oracle:

   ```bash
   python -m eval.runners.marginal_calibration_eval ckpt=pretrained
   ```

2. **Fine-tune.** Phase A logs to its own wandb project, `copula-inter-marginal`:

   ```bash
   python -m copula_inter.finetune_marginal
   oarsub -S ./scripts/finetune_marginal.sh
   ```

   By default (`marginal.lora_all_layers=true`), LoRA at one shared rank goes on
   every 2-D weight matrix of the selected backbone. Measured at rank 8:

   | Backbone | Matrices | Trainable parameters |
   |---|---|---|
   | tabicl | 153 | 854K (2.91 %) |

   `marginal.backbone` picks the model: `tabicl` (default), `tabpfn35_fast` or
   `kumo_small`. The tier ladder is an ablation:
   `marginal.lora_all_layers=false marginal.tier=1`. Only tabicl can climb it
   (`backend_registry.Backend.max_tier`); the other two stop at tier 0.

   Each model is scored on a 999-level grid (`marginal.probs_n=null`): TabICL's
   and Kumo's native decoder levels, TabPFN's bar distribution read at the same
   levels. The objective is therefore comparable across backbones. Set
   `probs_n=<int>` only to resample onto a coarser grid. The generic backbones'
   Phase-A forward is the function their PIT reads
   (`finetune_adapters.backend_quantiles`), so a fine-tuned model is scored as
   it was trained.

3. **Re-measure and gate on real data.** The point of a TabICL marginal is
   non-Gaussian tabular transfer, which GP-only training can destroy, so real
   data must not regress. Then hand the checkpoint to a copula run:

   ```bash
   python -m eval.runners.marginal_calibration_eval ckpt=<the _final.pt>
   python -m eval.runners.run_benchmarks
   python -m copula_inter.train tabicl.pit_ckpt=<the _final.pt>
   ```

   For `tabpfn35_fast` / `kumo_small` the copula run selects the backend and its
   weights together:

   ```bash
   python -m copula_inter.train data.z_train_source=kumo_small \
       data.z_train_marginal_ckpt=<the _final.pt>
   ```

Non-TabICL Phase-A checkpoints load back through
`copula_inter.marginal.backends.make_regressor(..., ckpt=<path>)`, not
`marginal.pit.load_tabicl`; a checkpoint written for another backbone is
refused. Register finished runs in `checkpoint_registry.py::MARGINAL_FAMILIES`.

## Spatial-correlation diagnostics

One CLI covers real ERA5 and synthetic kernels with known ground truth. ERA5
data is fetched and cached automatically.

```bash
# Everything: sweep all registered checkpoints, fit baseline curves, render the report.
python -m eval.runners.spatial_correlation_eval

# One subcommand at a time (command=<name> --cfg job lists its keys):
python -m eval.runners.spatial_correlation_eval command=diagnose \
    command.ckpt=kernel-sweep-all-tabicl-retrain-15k command.mode=real \
    command.region=western_europe command.grid_size=24
python -m eval.runners.spatial_correlation_eval command=sweep command.mode=synthetic command.checkpoints=all
python -m eval.runners.spatial_correlation_eval command=baseline command.mode=real
python -m eval.runners.spatial_correlation_eval command=report

# Marginal-quantile calibration on real ERA5 (per-quantile ECE, independence-copula checks):
python -m eval.runners.era5_calibration_eval
```

A real-mode sweep also scores the total (marginal + copula) joint NLL on
held-out ERA5 points. The correlation-curve `model_r2` can't say how many nats
worse the predictive density is.

## Debug pipeline

`debug/` holds a staged investigation of why training with
`data.z_train_source=tabicl` plateaus. See [debug/README.md](../debug/README.md).

## Tests

```bash
uv run --no-sync python -m pytest -q -n auto -m "not slow and not gpu and not pretrained and not external_data"
oarsub -S ./scripts/test_full.sh                 # full suite on a GPU node, before merging
```

CI (`.github/workflows/ci.yml`) runs the first command, plus ruff and mypy.
