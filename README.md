# Copula Inter

Joint predictive distributions for tabular in-context learning, built from a
learned **marginal** and an inter-instance **Gaussian copula**.

Tabular foundation models such as TabICL predict each query point on its own,
so their predictions over a set of query points are independent. This project
adds the missing dependence. A frozen tabular ICL model gives each query
point's marginal distribution, the targets are mapped to Gaussian scores
through the probability integral transform (PIT), and a second ICL trunk with a
low-rank correlation head predicts how those scores co-vary:

    log p(y_1..y_N | context) = sum_i log f_i(y_i)  +  log c(z_1..z_N; Sigma)
                                 marginal NLL          copula NLL
    z_i = Phi^{-1}(F_i(y_i))

The copula model is trained on synthetic Gaussian-process episodes, where the
true correlation is known. It can be fine-tuned on real ERA5 temperature
fields, and is evaluated against classical GP baselines on synthetic and real
data. The marginal model can be improved separately ("Phase A").

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra dev --extra cpu     # --extra gpu on a CUDA machine
uv run --no-sync python -m pytest -q -n auto -m "not slow and not gpu and not pretrained and not external_data"
```

Generate a small dataset with the exact-GP marginal (no model weights needed),
then train on it:

```bash
uv run --no-sync python -m copula_inter.generate_pit_dataset \
    data.n_tasks=5000 data.dataset_dir=./data/pilot data.z_train_source=analytic
uv run --no-sync python -m copula_inter.train \
    training.live_generation=false training.dataset_dir=./data/pilot/pit training.steps=500
```

Training still loads the copula backbone's pretrained TabICL weights, and by
default builds validation probes that fetch ERA5 data. Add
`training.startup_probes=false` to skip the probes on a first run.

Evaluate a checkpoint against the classical baselines:

```bash
uv run --no-sync python -m eval.runners.eval_checkpoint ckpt=kernel-sweep-all-tabicl-retrain-15k
```

Every command is a Hydra app. Pass settings as `key=value`, and add `--cfg job`
to print the full configuration without running.

## Repository layout

| Path | What it is |
|---|---|
| `src/copula_inter/` | The library and the training CLIs: `data/` episodes, `era5/` real data, `marginal/` PIT and Phase A, `copula/` model and losses, `training/` the loop |
| `eval/` | Evaluation runners, baseline tables and metrics |
| `inference/` | A small API for using a trained checkpoint |
| `conf/` | Hydra configuration |
| `scripts/` | Grid5000 OAR job scripts (`oarsub -S ./scripts/<job>.sh`) |
| `debug/` | A staged diagnostic pipeline |
| `tests/` | CPU tests plus `gpu` / `pretrained` / `external_data` / `slow` tiers |
| `tabicl_upstream/` | A vendored, lightly patched TabICL v2.2.0 |

## Documentation

- [docs/architecture.md](docs/architecture.md): the model, data flow, package map and layering rule
- [docs/workflows.md](docs/workflows.md): every stage end to end (generation, training, ERA5 fine-tuning, evaluation, Phase A, diagnostics) and the Grid5000 setup
- [docs/glossary.md](docs/glossary.md): PIT, episode, `z_train_source`, oracle, tiers, …
- [CONTRIBUTING.md](CONTRIBUTING.md): checks, conventions and recipes for common extensions
