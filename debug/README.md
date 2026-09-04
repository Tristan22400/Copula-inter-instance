# Debug pipeline: `data.z_train_source=tabicl` plateau

Investigates why training with `data.z_train_source=tabicl` plateaus far from
the oracle. Built as composable stages, each answering one question and
writing `results/<run_id>/<stage>.json`, so the same measurements re-run
after a change to the data prior (`conf/data/gp_tasks.yaml`) or the
architecture (`src/model.py`, rank, parametrization) — pass `--baseline
<old_run_id>` to `report.py` to see what moved.

## Quick start

```bash
# Diagnostics that need no trained checkpoint (S0-S3):
python debug/run_debug.py all --n-episodes 200

# Include the frozen-checkpoint stages (S5, S6) too:
python debug/run_debug.py all --n-episodes 200 --ckpt kernel-sweep-all-tabicl-retrain-15k

# One stage at a time, full control over its flags:
python debug/run_debug.py s1 --n-episodes 64 --ranks 8,16,32,64
python debug/stages/s1_rank_ceiling.py --help

# After changing conf/data/gp_tasks.yaml or the architecture, diff against
# the run before the change:
python debug/run_debug.py all --n-episodes 200 --run-id after_change
python debug/report.py --run-id after_change --baseline <before_change_run_id>
```

All Hydra-style overrides (`data.P_max=64`, `model.rank=64`, ...) work as
positional args on every stage, same syntax as `train.py`/`train_fast.py`.

## The headline metric, decomposed

`oracle_diag/gap_nll` (the number you watch in wandb) is
`total − oracle_posterior_total`, which bundles a copula term the model can
improve with a marginal term it can't (the marginal comes from a *separate
frozen* TabICL, `data_gen.py`'s `z_train_source=tabicl` override). `src/
train.py::validate()` now also logs `oracle_diag/copula_gap` and
`oracle_diag/marginal_gap` so the split is visible without re-deriving it
(plus `oracle_diag/copula_headroom`, the Bayes-optimal copula reward the gap
is measured against).

## What the evidence said before this pipeline existed

From `wandb/run-*/files/wandb-summary.json` (per-point nats, P=32 unless noted):

| z_src | rank | step | model copula | oracle post. copula | copula gap | `gap_nll` |
|---|---|---|---|---|---|---|
| tabicl | 32 | 18k | −0.164 | −0.321 | **0.158** | 0.815 |
| tabicl | 8 | 51k | −0.088 | −0.321 | **0.234** | 0.892 |

More steps at lower rank did *worse* — a capacity signature, not
undertraining. That's why S1 (rank ceiling) is the stage to run first.

## Stages

| stage | question | needs `--ckpt`? |
|---|---|---|
| **S0** `s0_signal.py` | How much copula signal does this prior even offer, as a function of context size P? | no |
| **S1** `s1_rank_ceiling.py` | **The key stage.** Exact rank-r ceiling: the best a `covnorm` factor model of rank r *could* do against `R_post`, no sampling noise. Sweeps r; model rank stays fixed at 32 by design (see "Rank is fixed" below). | no |
| **S1b** `s1b_rank_gap_decomp.py` | S1's ceiling re-expressed as a **gap to the exact GP posterior**, which is what says whether rank actually binds. Also fits an alternative basis — a sparse-GP Schur complement `K_theta - K_su(K_uu+D)^-1 K_us`, i.e. prior kernel minus rank-m correction, the shape `K_post` genuinely has — and reports `n_params` alongside, since a head must emit these per episode. | no |
| **S2** `s2_uspace.py` | u-space (PIT) audit: calibration (KS/ECE/reliability curve) + a **clamping census** — how much mass piles onto the hard `_probit` clamp (`u≤1e-6`) and TabICL's spline-knot boundary (`u≤1e-3`), pooled and per-episode. | no |
| **S3** `s3_pit_floor.py` | Attainable copula floor once TabICL's own PIT (not the oracle) is the marginal — decomposes the gap into rank-ceiling loss vs. PIT-distortion loss. | no |
| **S4** `s4_overfit.py` | Single-episode overfit sanity check. `--target prior\|posterior`, `--z-source oracle\|tabicl`. | no (trains from scratch) |
| **S5** `s5_kfold.py` | K-fold noise impact on `z_train`, frozen checkpoint (K-folding doesn't touch `z_test`, so this needs no retraining). | **yes** |
| **S6** `s6_guards.py` | Covnorm escape ratio (what actually sets reachable \|ρ\|) + Cholesky jitter-escalation / non-finite-input counts. | optional (fresh model if omitted) |
| **S7a** `s7_backbone.py` | z_train-gap diagnostic across marginal backends (tabicl/tabpfn/exaone/tabm) on a frozen copula head. Moved from `eval/runners/compare_marginal_backbones.py`. | **yes** |
| **S7b** `s7b_backend_train.py` | Actually **trains** fresh models under different marginal backends (`--backends tabicl,tabpfn`) and compares gap trajectories. Debug-scoped, not a production knob — see the module docstring for why. | no (trains from scratch) |
| **S8** `s8_single_kernel.py` | Forces a single kernel family via `train_fast.py`. Most informative once S1/S4 have ruled rank out — a negative control otherwise. | launches training |

## Design notes

- **Rank is fixed at 32.** S1's sweep measures how much correlation
  structure a low-rank approximation can carry — it is a diagnostic, not a
  proposal to raise the model's rank.
- **`R_post`, not `R_star`, is the training target.** Under
  `oracle_mode: prior` (the only supported mode), `R_star` is the
  *unconditioned* prior correlation. The conditioned object is
  `gp_analytical_posterior`'s `R_post` — every oracle comparison in this
  pipeline uses that, not `R_star`.
- **K-fold (S5) needs no retraining.** K-folding only affects `z_train`
  (the model's input); `z_test`/`log_pdf_test` always come from one
  non-folded TabICL forward regardless of K.
- **Saturation guards, narrowed.** The jitter ceiling is
  `1/(1+sigma_jitter) ≈ 0.9999` and rank imposes no *pairwise* correlation
  ceiling (only a joint-structure one) — neither is worth instrumenting.
  What S6 actually measures: the covnorm escape ratio
  `‖W‖²/softplus(s)`, and real Cholesky escalation/fallback counts via
  monkey-patching (no `src/` edits).
- **`overfit_single.py` and `compare_marginal_backbones.py` moved here**
  (`s4_overfit.py`, `s7_backbone.py`) — nothing outside this package
  imported either, so this is their one home now.

## Reused, not reimplemented

`src/pit.py::gp_analytical_posterior`, `src/loss.py::{y_space_nll,
oracle_copula_nll, _safe_cholesky}`, `src/model.py::{low_rank_correlation,
build_sigma}`, `src/data_gen.py::generate_gp_batch`,
`src/dataset.py::collate_fn`, `src/pit.py::{load_tabicl, resolve_pit_ckpt,
run_pit_batched, _probit}`, `eval/spatial/calibration.py::
compute_quantile_ece`, `eval/spatial/marginal_backends.py`,
`eval/metrics/joint_nll.py::{compute_pit, kfold_loo_pit}`,
`eval/configs/checkpoints.py::resolve_checkpoint`.

## Tests

`tests/test_debug_pipeline.py` — the rank-ceiling fitter recovers a known
low-rank `R` near-exactly; the clamping census counts a synthetic
all-saturated batch at 100%; `config.py`'s override parser round-trips
Hydra-style dotted overrides.
