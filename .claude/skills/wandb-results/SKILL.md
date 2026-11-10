---
name: wandb-results
description: Read and compare copula training runs from wandb (project iliass-lasri-team/copula-inter-instance) — full NLL split into marginal + copula against independence and the GP oracle, best steps, the size_fit (P, N) grid — and say which run is best. Use when asked for the results, status or comparison of training jobs, or which model/marginal/training recipe is best.
---

# Reading copula training results from wandb

`eval/runners/wandb_report.py` does the reading. This skill covers how to run it on Grid5000 and how to read its tables.

## 1. Run the report on a compute node

The Grenoble frontend can't import numpy, so run on a node. Reuse one of the user's running jobs; it only needs a little CPU. The wandb credentials are in `~/.netrc` on the cluster.

```bash
ssh grenoble.g5k.remote 'oarstat -u'                                  # pick a running job id
ssh grenoble.g5k.remote 'oarstat -j <id> -f | grep assigned_hostnames' # its node
ssh grenoble.g5k.remote 'OAR_JOB_ID=<id> oarsh <node> "cd ~/thoth_storage/copula-inter/<checkout> && source scripts/_env.sh >/dev/null 2>&1 && python -m eval.runners.wandb_report filter=<regex> since=<YYYY-MM-DD>" 2>&1 | grep -v ^wandb:'
```

- `<checkout>` is a checkout that contains this runner: `.claude/worktrees/size-grid-metrics` until that branch is merged.
- For worktree checkouts, prefix the command with `COPULA_VENV=$HOME/thoth_storage/copula-inter/.claude/worktrees/marginal-backend-choice/.venv`; the main checkout's venv lacks the kumo/tabpfn extras.

Keys (Hydra `key=value` overrides, never `--flags`):

| Key | Meaning |
|---|---|
| `filter=<regex>` | Searched in the run label, which is the basename of `training.ckpt_dir` (e.g. `FT_nano_PN_tabpfn`) |
| `ids=[a,b]` | Explicit wandb run ids; filter, since and states are then ignored |
| `since=YYYY-MM-DD` | Only runs created on or after this date |
| `states=[running,finished]` | Default; add `crashed` or `killed` to include those |
| `sort_by=full` | Or `copula_exact`, `excess`, `marginal`, `grid` |
| `out=<file>.json` | Also write the per-run records |

Hydra's override grammar rejects `(`, `|` and similar characters in unquoted values. Prefer plain substrings such as `filter=_PN_` or `filter=_nano_`. If you need a real regex, quote it inside the override: `'filter="^(FT|PT)_"'`.

## 2. What the numbers mean

All values are nats per test point, read at the last validation step. Lower is better.

$$\text{full} = \underbrace{-\tfrac1N\textstyle\sum_i\log f(y_i)}_{\text{marginal}} + \underbrace{c}_{\text{copula}},\qquad c = -\tfrac1N\log c_R(\mathbf u)$$

- **Independence is $c=0$:** the marginal column is the run's full NLL without a copula. A copula $c<0$ helps; $c>0$ is worse than independence.
- **The oracle is the GP posterior:** oracle full = oracle marginal + oracle copula, with oracle copula $= -$headroom.
- **Excess** over the oracle = marginal excess + copula excess. Use it to see which part to improve.
- **copula exact z** is the copula NLL on analytic $z$, where the marginal is exact. It measures how much GP correlation the model learned. The `copula` column is measured on the learned marginal's $z$, so it also rewards dependence left in the marginal's PIT residuals. A large live-$z$ copula gain with an exact-$z$ value near 0 means the copula is fitting marginal errors, not the GP.
- **Size grid** (runs with `baselines.size_grid_enabled=true`): the copula NLL on exact $z$ for each context size $P$, averaged over the test sizes $N$. It is computed as $c_{P,N}=\texttt{gap\_nll}-H_{P,N}$.
  - Independence is 0 and the oracle is $-H$.
  - `captured` = grid mean / oracle grid mean.
  - `first logged` is step 0 when `training.val_at_start=true`.
  - A value above 0 at some $P$ means the model does worse than independence at that size.

## 3. Comparing runs fairly

1. **Compare full NLLs only within one validation table.** The report starts a new table for each oracle full value. Fixed-size runs ($P=32$, $N=256$) and variable-size runs (`data.P_min`/`P_max`) use different validation sets.
2. **Check the marginal setup list before crediting a marginal.**
   - A full-NLL difference between two runs with different `marginal_ckpt` or `pit_ckpt` is mostly a marginal difference.
   - Runs resumed from `F_nano_*_A` use short Phase-A marginals (200–300 steps), not the ERA5-finetuned ones. Compare them to the FT runs on the copula columns and the grid only.
3. **Report the full NLL** (the project rule), then its split.
4. **Noise:** the grid uses 32 episodes per cell. Differences below about 0.005 nats in the grid mean, or about 0.002 in the exact-$z$ copula, are not reliable.
5. **Running runs** are read at their latest step. Say so, and give the step.

## 4. Answering "which is best"

- **Best predictive model:** the lowest full NLL within one validation set, among runs at comparable steps.
- **Best copula:** the lowest copula on exact $z$, plus the grid mean, with no cell above 0 at small $P$.

Give both when they disagree, and say whether the gap from the leader comes from the marginal or the copula. Keep the answer to tables plus a short verdict, following the user's style.
