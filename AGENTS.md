# Notes for coding agents

Start with [docs/architecture.md](docs/architecture.md) (package map, data flow,
layering rule) and [CONTRIBUTING.md](CONTRIBUTING.md) (checks, conventions,
scientific contracts, recipes). Commands are in
[docs/workflows.md](docs/workflows.md).

- Use Python 3.12 and `uv sync --locked --extra dev --extra cpu` for CPU work.
  Run entry points as `python -m copula_inter.<module>` or
  `python -m eval.runners.<runner>`; nothing edits `sys.path`.
- `src/copula_inter` imports only itself and `tabicl`; `inference` imports
  `copula_inter`; `eval` imports both. Move shared code into the library rather
  than adding an upward import.
- Every function carries parameter and return annotations. `ruff check`,
  `ruff format --check` and `mypy` must be clean.
- Run the focused tests for changed code, then the CI command for cross-module
  changes:
  `pytest -n auto -m "not slow and not gpu and not pretrained and not external_data"`.
  Mark a new test `gpu`, `pretrained` or `external_data` when it needs one, and
  `slow` when it takes more than a few seconds on CPU.
- Every `python -m` command in the docs and `scripts/*.sh` is composed by
  `tests/test_eval_runner_configs.py`, so keep documented commands valid.
