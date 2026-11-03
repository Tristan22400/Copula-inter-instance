# Architecture

This page maps the code: what each package owns, which way dependencies point,
and where data flows from an episode to a loss. Terms in *italics* are defined
in the [glossary](glossary.md).

## The model in one paragraph

For an *episode* of P context points and N query points, the joint predictive
density of the N query targets is split with Sklar's theorem into a
**marginal** part and a **copula** part:

    log p(y_1..y_N | context) = sum_i log f_i(y_i)  +  log c(z_1..z_N; Sigma)
                                 marginal NLL          copula NLL
    z_i = Phi^{-1}(F_i(y_i))

The marginals F_i come from a frozen tabular in-context-learning model (TabICL
by default) through a *PIT*. The copula is Gaussian with a low-rank-plus-diagonal
correlation Sigma. A second ICL trunk (the *copula backbone*) predicts that
correlation from the context features and the context's `z_train`.
`copula_inter.train` trains the copula backbone and head end to end
(`model.unfreeze_backbone`). The marginal stays frozen there and is improved
separately by *Phase A* (`copula_inter.finetune_marginal`).

## Data flow

```
 GP prior (data/)            or  ERA5 corpus (era5/)
   kernel + hyperparameters        random region, resolution, day
   x (P+N, d), y (P+N)             x = lon/lat (+ static fields), y = t2m
          |                                  |
          +-------------> episode <----------+
                             |
          marginal/  PIT: K-fold over the P context rows, full context for the N queries
                             |   z_train (P,), z_test (N,), log_pdf_test (N,)
                             v
          copula/    backbone(x, z_train) -> per-query (w_i in R^r, s_i)
                     Sigma = renormalize(W W^T + diag(softplus(s)))
                             |
          copula/loss.py     y_space_nll = copula NLL(z_test; Sigma) + marginal NLL(log_pdf_test)
                             |
          training/          optimizer step, validation probes, checkpoints
```

An episode is a dict of aligned tensors. P context rows and N query rows are
padded per batch, and `test_mask` marks the real query rows. The keys and shapes
are checked at the boundaries in `data/contracts.py`, not inside the training step.

## Packages and the layering rule

| Package | Owns | May import |
|---|---|---|
| `copula_inter` (`src/copula_inter/`) | The library and the training/generation CLIs | `tabicl` only |
| `inference/` | A small public API over a trained checkpoint (load, PIT, correlation, sampling) | `copula_inter` |
| `eval/` | Evaluation runners, baselines tables, metrics, eval-only datasets and plots | `copula_inter`, `inference` |
| `debug/`, `scripts/` | One-off diagnostics, job scripts | anything |

`tests/test_engineering_contracts.py::test_package_layering` enforces the first
three rows. When library code needs something that lives in `eval/`, move it
into `copula_inter` (and have `eval/` import it from there). Don't add the
import.

### `copula_inter`

Top-level modules are the command-line entry points plus a few shared utilities:

| Module | Role |
|---|---|
| `train.py` | `python -m copula_inter.train`: setup phases, then the training loop |
| `generate_pit_dataset.py` | `python -m copula_inter.generate_pit_dataset`: write episodes to disk as shards |
| `finetune_marginal.py` | `python -m copula_inter.finetune_marginal`: Phase-A marginal fine-tuning |
| `diag_kernels.py` | `python -m copula_inter.diag_kernels`: sanity-check every kernel's `R_star` |
| `backend_registry.py` | Which marginal/copula backends exist and what each supports |
| `checkpoint_registry.py` | Named checkpoint families (`ckpt=<name>`), copula and marginal kept separate |
| `paths.py` | Default locations: repo root, `checkpoints/`, ERA5 caches |
| `artifacts.py`, `config_path.py`, `rng.py`, `type_aliases.py` | Atomic file writes, Hydra config lookup, seeding/device, shared type aliases |

Subpackages:

- **`data/`** synthesizes and stores episodes.
  - `gp_episodes.py`: draw a GP episode end to end (`generate_gp_batch`).
  - `gp_kernels.py`: kernel classes, hyperpriors, composites.
  - `kernel_sampling.py`: per-call prior sampling (feature count, kernel chain, mean function).
  - `feature_transforms.py`, `structural_warps.py`: input warps applied before the kernel.
  - `dataset.py`, `manifest.py`: on-disk shards and their identity manifests.
  - `live_dataset.py`, `era5_live_dataset.py`: episodes generated on the fly in DataLoader workers.
  - `contracts.py`: episode schema checks.
- **`era5/`** accesses real ARCO-ERA5 2 m temperature.
  - `fetch.py`, `fetch_static.py`, `fetch_global.py`: download and cache regional grids, static fields, the global monthly corpus.
  - `corpus.py`: sample episodes from the global corpus.
  - `io.py`: NetCDF reading and geometry.
  - `regions.py`: named regions.
  - `spatial_stats.py`: empirical correlation, distance binning.
  - `probe.py`: the frozen per-region probe used by training validation and the spatial eval.
- **`marginal/`** turns targets into *z*.
  - `pit.py`: PIT through TabICL, batched and differentiable variants, and the exact-GP equivalent.
  - `kfold.py`: backend-agnostic PIT from any quantile grid.
  - `backends.py`: the K-fold PIT for every marginal backend (TabICL, TabPFN).
  - `batched_pit.py`, `tabpfn_batched.py`: fast batched path for TabPFN.
  - `tabicl_regressor.py`: helpers around the sklearn-style `TabICLRegressor`.
  - `calibration.py`: quantile ECE.
  - Phase A: `finetune_adapters.py` (per-architecture parameter selection and checkpoint format), `tiers.py`, `lora.py`, `objective.py`, `finetune_data.py`.
- **`copula/`** is the copula model.
  - `model.py`: `CopulaTabICL`, `build_sigma`.
  - `backbones.py`: build the trunk and strip its decoder.
  - `correlation.py`: low-rank correlation parametrizations.
  - `loss.py`: copula, marginal and Y-space NLLs and metrics.
- **`baselines/`** holds classical references that training validation also needs.
  - `gp_mle.py`: MAP-fitted gpytorch GP.
  - `classical_kernels.py`: fixed-kernel correlations.
- **`training/`** runs the copula training loop.
  - `setup.py`: build the data loaders, model, optimizer and probes.
  - `step.py`: schedule and one optimizer step.
  - `validation.py`: `validate()`.
  - `probe_batches.py`, `era5_probes.py`: fixed validation batches.
  - `adaptive_sampling.py`: kernel weights and the TabICL mix.
  - `checkpointing.py`, `muon.py`: checkpoint I/O and the Muon optimizer.
- **`viz/`**
  - `field_grids.py`: lon/lat field figures logged during training and by the spatial eval.

### `eval/`

| Path | Role |
|---|---|
| `runners/` | One Hydra entry point per runner (`python -m eval.runners.<name>`), each with a typed dataclass config composed from `conf/eval/<name>.yaml` |
| `runners/eval_checkpoint.py` | The main comparison: model vs. classical baselines, synthetic or real-ERA5 episodes |
| `baselines/` | Baseline tables (`classical.py`, with its per-episode cache), CV best-of-baselines (`prefit.py`), the autoregressive row |
| `data/era5_episodes.py` | Real-ERA5 evaluation episodes |
| `datasets/` | Real-world benchmark datasets (`run_benchmarks`) |
| `metrics/` | Joint NLL from a quantile grid, energy score |
| `spatial/` | Spatial-correlation diagnostics, sweeps, calibration plots |
| `configs/constants.py` | Eval knobs; values the library also needs are re-exported from their owner |
| `results.py`, `io.py`, `viz/` | Result caches and summaries, file I/O, plots |

## Configuration

Everything that runs is configured by Hydra. `conf/config.yaml` composes a
model preset (`conf/model/*.yaml`, which sets both `model.*` and `tabicl.*`),
a data preset (`conf/data/*.yaml`) and an optional `experiment=` preset over
both. `conf/finetune_marginal.yaml` is Phase A's own root, and
`conf/eval/<runner>.yaml` configures each eval runner. Append `--cfg job` to
any command to print the composed configuration without running it.

## Checkpoints

A copula checkpoint stores the composed config and the state dict. Phase-A
checkpoints use TabICL's own `{"config", "state_dict"}` format (plus a `"backbone"`
tag for non-TabICL backbones), so they load wherever a TabICL checkpoint does.
`checkpoint_registry.py` maps short names to paths under `checkpoints/`:
`CHECKPOINT_FAMILIES` holds copula runs (iterated by `spatial_correlation_eval command.checkpoints=all`) and
`MARGINAL_FAMILIES` holds Phase-A runs. Keep the two separate.

## Third-party code

- **`tabicl_upstream/`** is a vendored copy of TabICL v2.2.0 with one local
  patch: commit `dfca17b` chunks oversized flattened-batch SDPA calls so they
  stay under CUDA's grid limit (upstream lacks it). `pyproject.toml` installs it
  as the `tabicl` package, and `tests/test_tabicl_vendored.py` checks that
  `import tabicl` resolves to this copy. To upgrade, replace the tree with the
  new upstream tag, re-apply `dfca17b`, and rerun the full suite.
- TabPFN is licence-gated. It needs a PriorLabs API key in `TABPFN_TOKEN`
  (the job scripts read it from `~/.config/tabpfn_token`). Never pass the key
  on a command line or commit it.
