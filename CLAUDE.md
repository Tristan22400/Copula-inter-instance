Environment: uv. `uv sync --locked --extra dev --extra gpu --extra tabpfn --extra kumo --extra era5`
(or `--extra cpu`; keep every extra on each sync, or uv uninstalls that backend), then
`.venv/bin/python` / `uv run --no-sync ...`. Job scripts source `scripts/_env.sh`
(activates the checkout's `.venv`, or the main checkout's from a git worktree,
and puts this checkout's src/, root and tabicl_upstream/src first on PYTHONPATH).
This checkout may be an sshfs mount of Grid5000 storage: never `uv sync` into its
`.venv` from a laptop (that is the cluster's GPU venv) -- set
`UV_PROJECT_ENVIRONMENT` to a local path instead.

Where things are:
  - docs/workflows.md     every command: generate, train, ERA5 finetune, eval
                          (synthetic, ERA5, autoregressive row), Phase A, spatial
                          diagnostics, tests, Grid5000 setup
  - docs/architecture.md  package map, data flow, layering rule, checkpoints,
                          vendored code
  - docs/glossary.md      vocabulary (PIT, z_train_source, oracle, tiers, ...)
  - CONTRIBUTING.md       checks, conventions, scientific contracts, recipes

Rules that are easy to break:
  - Layering: src/copula_inter imports only itself and tabicl; inference imports
    copula_inter; eval imports both (tests/test_engineering_contracts.py::
    test_package_layering). Move shared code into the library instead.
  - Settings are Hydra keys (eval runners: typed dataclass fields +
    conf/eval/<runner>.yaml), never argparse. Every `python -m` command in the
    docs and scripts is composed by tests/test_eval_runner_configs.py.
  - Paths come from copula_inter/paths.py; checkpoint names from
    copula_inter/checkpoint_registry.py (CHECKPOINT_FAMILIES = copula runs,
    MARGINAL_FAMILIES = Phase-A runs; do not mix the two).
  - Read the TOTAL (marginal + copula) Y-space NLL when comparing methods; the
    copula-only table needs a shared z_test, and real ERA5 has no oracle.
  - Every function is annotated; ruff check, ruff format --check and mypy must be
    clean (the .claude/hooks/check_python.sh hook runs them after each edit).
