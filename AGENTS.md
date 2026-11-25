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

## Working alongside other agents

Several agents (Claude, Codex, ...) work on this repository at the same time.

Shared layout:

| Path | Role |
|---|---|
| `.worktrees/<task>/` | the only worktree root, for every agent |
| `.claude/worktrees`, `.codex/worktrees`, `.codex/worktree`, `.agent/worktrees` | symlinks to `.worktrees`; never replace them with directories |
| `.agents/skills/<name>/SKILL.md` | the only skills folder; `.claude/skills` is a symlink to it |
| `checkpoints/`, `data/`, `logs/` (main checkout) | shared; worktrees link `checkpoints -> ../../checkpoints` |

Rules:

1. One task gets one worktree and one branch, `<agent>/<task>` (e.g. `codex/era5-eval`).
   Create it on the cluster and lock it:
   `ssh grenoble.g5k.remote 'cd ~/thoth_storage/copula-inter && git worktree add .worktrees/<task> -b <agent>/<task> && git worktree lock --reason "<agent>: <task>" .worktrees/<task> && ln -s ../../checkpoints .worktrees/<task>/checkpoints'`
2. Do not edit the main checkout (`main`). It holds other people's uncommitted work.
   Change it only when the user asks you to. Merging into `main` is up to the user.
3. Leave other agents' worktrees and branches alone: do not edit, check out, reset,
   rebase, stash, force-push or delete them. If you need their code, read it or
   merge their branch into yours.
4. Do not run commands that discard work: `git worktree prune`, `git clean`,
   `git reset --hard`, `git checkout -- .`, `git stash`, `git branch -D`,
   `git push --force` on a branch that is not yours. In particular, never prune
   from the laptop: over sshfs every cluster worktree looks missing, so prune
   deletes their metadata.
5. Run git commands that write (add, commit, worktree) on the cluster frontend or
   inside your own worktree. If the laptop sees "index corrompu"/corrupt index,
   it is a stale sshfs cache: retry, and do not try to repair the index.
6. Shared outputs: write only under a subfolder named after your task
   (`checkpoints/<task>/`, `logs/<task>/`, `data/<task>/`). Never delete or
   overwrite files that another task wrote.
7. Grid5000 jobs: start the OAR job name with your task (`oarsub -n <task>-...`)
   and `oardel` only your own jobs. Reuse a node only after checking that nothing
   else is running on it (see the `g5k-launch` skill).
8. Skills: add or edit them only in `.agents/skills/`. Hooks: `.claude/hooks/` and
   `.codex/hooks/` hold the same `check_python.sh`; keep them identical.
9. Do not `uv sync` into the shared `.venv` (it is the cluster GPU venv).
10. When you finish, commit on your branch and report the worktree path and
    branch. Unlock or remove a worktree only after the user agrees, and only
    from the cluster.
