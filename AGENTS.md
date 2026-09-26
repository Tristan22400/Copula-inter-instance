# Repository map

Code lives in the `copula_inter` package (`src/copula_inter/`); run entry points
as `python -m copula_inter.<module>` with the checkout on `PYTHONPATH`
(`source scripts/_env.sh`) or from its own `uv` venv. Nothing edits `sys.path`.

- `data_gen.py`: GP kernels and episode sampling; `feature_transforms.py`: input
  and hidden feature warps; `pit.py`: PIT conversion; `episode_contracts.py`:
  episode shapes and boundary checks; `dataset.py`, `generate_pit_dataset.py`,
  `dataset_manifest.py`: on-disk episodes, shards and manifests;
  `live_dataset.py`, `era5_live_dataset.py`: live GP / ERA5 training data.
- `model.py`, `copula_backbones.py`, `correlation_factory.py`: the copula model;
  `loss.py`: NLLs and metrics.
- `train.py`: `main()` (setup phases, then the training loop);
  `training_core.py`: schedule, loss and optimizer step; `validation.py`:
  `validate()`; `probe_batches.py`, `era5_probes.py`: fixed validation probes;
  `adaptive_sampling.py`: kernel weights and TabICL mix; `checkpointing.py`.
- `finetune_marginal.py`, `marginal_backbones.py`, `lora.py`: Phase-A marginal
  fine-tuning.
- `backend_registry.py`: supported marginal and copula backbones and their
  capabilities. Add a backend here, then implement its adapter under
  `eval/spatial/` or `src/copula_inter/` and run the relevant backend tests.
- `eval/runners/eval_checkpoint.py`: evaluation CLI (`run_evaluation`);
  `eval/baselines/classical.py`: baseline fits and their cache;
  `eval/baselines/prefit.py`: parallel prefit and CV best-baseline;
  `eval/runners/eval_tables.py`: printed tables; `eval/results.py`: summaries.
- `conf/`: Hydra configuration; `scripts/`: OAR job scripts (`_env.sh` holds the
  shared setup); `tests/`: CPU and optional integration tests;
  `.github/workflows/ci.yml`: fast CPU gate.

Use Python 3.12 and `uv sync --locked --extra dev --extra cpu` for CPU work.
Run the focused tests
for changed code, then the CI command for cross-module changes. Training and
generation examples in `README.md` and `CLAUDE.md` use Hydra keys `data.n_tasks`
and `training.live_generation=false` for on-disk training.

Scientific contracts: episode tensors align on P context and N query rows;
`test_mask` excludes padded query rows; kernel reconstruction must use the
saved raw/normalized coordinates consistently; cache keys must change when
dataset or model bytes change. Preserve deterministic seed-to-episode mapping
when changing generation batches.
