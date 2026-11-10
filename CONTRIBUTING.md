# Contributing

Read [docs/architecture.md](docs/architecture.md) first for the package map and
the layering rule, then [docs/glossary.md](docs/glossary.md) for the vocabulary.

## Setup

```bash
uv sync --locked --extra dev --extra cpu     # --extra gpu on a CUDA machine
uv run --no-sync pre-commit install
```

## Checks

CI (`.github/workflows/ci.yml`) runs exactly these, and all of them must pass:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync python -m pytest -q -n auto -m "not slow and not gpu and not pretrained and not external_data"
```

Before merging anything that touches a backend, a model or numerics, also run
the full suite on a GPU node: `oarsub -S ./scripts/test_full.sh`.

## Conventions

- **Types.** Every function, tests included, has parameter and return
  annotations. `mypy` runs with the strict flags in `pyproject.toml` and must
  report nothing.
- **Layering.** `copula_inter` imports only itself and `tabicl`; `inference`
  imports `copula_inter`; `eval` imports both. If library code needs something
  from `eval/`, move that code into the library.
  `test_package_layering` fails otherwise.
- **Configuration.** Settings are Hydra keys, never argparse flags.
  - Training and generation: add the key to `conf/config.yaml` or a group under `conf/`.
  - An eval runner: add a field to its dataclass config (for `eval_checkpoint`, `eval/runners/eval_args.py`) and its default in `conf/eval/<runner>.yaml`.
- **Imports and paths.** Nothing edits `sys.path`. Default locations come
  from `copula_inter/paths.py`, and checkpoint names from
  `copula_inter/checkpoint_registry.py`. Don't hardcode a path in a script.
- **Tests.** Mark a test that needs a GPU, pretrained weights or data outside
  the checkout `gpu`, `pretrained` or `external_data`. Mark it `slow` when it
  takes more than a few seconds on CPU.
- **Comments** explain why the code is the way it is. History (what changed,
  when, what was benchmarked) belongs in the commit message.

### Scientific contracts

These invariants are easy to break and hard to notice:

- Episode tensors align on P context and N query rows, and `test_mask` excludes
  padded query rows.
- Kernel reconstruction must use the saved raw or normalized coordinates
  consistently.
- Cache keys must change when dataset or model bytes change: dataset manifests,
  baseline fingerprints, results caches.
- The seed-to-episode mapping stays deterministic when generation batching
  changes.
- `tests/test_refactor_golden.py` pins the numerics of generation, losses, the
  model and the optimizer against a recorded golden. A change that moves those
  numbers on purpose needs a new golden and a sentence in the commit saying why.

## Recipes

### Add a GP kernel

1. Implement it in `src/copula_inter/data/gp_kernels.py`, add it to
   `KERNEL_REGISTRY`, and give it a hyperprior in `_kernel_prior_spec`. Add it
   to `_COMPOSABLE_KERNELS` if it should appear in `A+B` / `A*B` composites.
2. If the classical GP-MLE baseline should fit it, mirror the prior in
   `src/copula_inter/baselines/gp_mle.py` (`_kernel_priors`, `ARD_ELIGIBLE`).
3. Run `python -m copula_inter.diag_kernels` to check that its `R_star` is
   neither collapsed to independence nor saturated. Then run
   `tests/test_diag_kernels.py` and the dataset-correlation tests.

### Add a marginal backend

1. Register it in `src/copula_inter/backend_registry.py` (`Backend(...)`) with
   its capabilities: the highest Phase-A tier, whether it can be a copula
   backbone, and its batched PIT entry point.
2. Implement `make_regressor` and `quantiles` branches in
   `src/copula_inter/marginal/backends.py`. `quantiles` returns an
   `(n_query, Q)` grid in raw y units; the K-fold PIT is generic.
3. Optionally add a batched path, `src/copula_inter/marginal/<name>_batched.py`,
   on top of `batched_pit.py`.
4. For Phase A, add an adapter in `src/copula_inter/marginal/finetune_adapters.py`.
5. Follow `tests/test_marginal_backends.py` and `tests/test_marginal_backbones.py`:
   check that the batched path matches the per-episode path.

### Add a setting to an eval runner

Add a typed field (with a default) to the runner's dataclass config, mirror the
default in `conf/eval/<runner>.yaml`, and read it from the spec.
`--cfg job` should then list it. `tests/test_eval_runner_configs.py` checks that
the YAML and the schema agree.

### Register a checkpoint

Add an entry to `CHECKPOINT_FAMILIES` (copula runs) or `MARGINAL_FAMILIES`
(Phase-A runs) in `src/copula_inter/checkpoint_registry.py`. It is then usable
as `ckpt=<name>` or `ckpt=<name>:<step>` everywhere. Keep the two registries
separate: `spatial_correlation_eval command.checkpoints=all` iterates the copula
one.
