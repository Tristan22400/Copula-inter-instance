# Repository map

- `src/copula_inter/data_gen.py`: GP kernels and raw episode sampling;
  `src/copula_inter/feature_transforms.py`: input and hidden feature warps; `src/copula_inter/pit.py`:
  PIT conversion; `src/copula_inter/episode_contracts.py`: raw/PIT/padded shapes and boundary
  checks; `src/copula_inter/dataset.py` and `src/copula_inter/generate_pit_dataset.py`: on-disk episodes,
  shard loading, and manifests.
- `src/copula_inter/training_core.py`: schedule, loss, and optimizer step shared by training
  entrypoints; `src/copula_inter/train.py`: training orchestration and checkpointing.
- `src/copula_inter/backend_registry.py`: supported marginal and copula backbones and
  their capabilities. Add a backend here, then implement its adapter under
  `eval/spatial/` or `src/` and run the relevant backend tests.
- `eval/runners/eval_checkpoint.py`: evaluation CLI and orchestration;
  `eval/baselines/classical.py`: fitted baselines and their cache;
  `eval/results.py`: result summaries and coverage rules.
- `conf/`: Hydra configuration; `tests/`: focused CPU and optional integration
  tests; `.github/workflows/ci.yml`: fast CPU gate.

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
