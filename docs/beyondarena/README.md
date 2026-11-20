# BeyondArena study: CopulaTabICL on grouped and temporal regression

How to reproduce the study. The hypotheses and their decision rules are in
`preregistration.md`, and the settings were frozen in `frozen_settings.yaml`
before any confirmatory run. `deviations.md` lists every departure from the
official protocol. The commands below run on Grid5000 nodes, never on a laptop
or a frontend.

## 0. Separate BeyondArena venv (once)

TabArena pins its own torch, tabpfn and autogluon versions, so it gets its own
venv. The project's `.venv` and `uv.lock` are left untouched.

```bash
oarsub -l host=1/core=8,walltime=1:30:00 ./scripts/beyondarena/install_tabarena.sh   # tabarena cc507b84
```

## 1. Official track (H1)

Run TabArena's TA-TabICLv2 with the fine-tuned weights, on core2k regression
under the official protocol, sharded by dataset. A rerun resumes where it
stopped.

```bash
./scripts/beyondarena/official_track.sh method=ft "datasets=[<name>,...]"
```

Control: rerun TA-TabICLv2 on the lite split, to check that this environment
reproduces the cached baseline.

```bash
./scripts/beyondarena/official_track.sh method=repro splits=lite "datasets=[...]"
```

Elo and per-split errors against the cached baselines, using TabArena's own
`compare()`; this writes `results/beyondarena/official/compare/<scope>/`:

```bash
python scripts/beyondarena/official_compare.py   # in the tabarena venv
```

## 2. Joint track (H2, H3)

Export the official splits after TabArena's model-agnostic preprocessing (in
the tabarena venv). This writes `data/beyondarena/<dataset>/r<repeat>_f<fold>.npz`
and `meta.json`.

```bash
python scripts/beyondarena/export_splits.py
```

Fit the mixed-effects baseline on the grouped tasks (in the tabarena venv, for
statsmodels). This writes `results/beyondarena/lmm/`.

```bash
python scripts/beyondarena/fit_lmm.py
```

Run each model on a shard of datasets (`scripts/beyondarena/shards/`). Each run
also scores indep, exch and lmm on the same blocks.

```bash
./scripts/beyondarena/joint_batch.sh <model> <shard>
```

Models: `tabicl-ft`, `tabpfn-ft`, `kumo-ft`, `tabicl-pre`, `tabpfn-pre`, `kumo-pre`.
Shards: `grouped_small`, `temporal_small`, `iid_small`, `large_a`, `large_b`.

## 3. Ablations (H4 and the rest), primary model, grouped and temporal shards

```bash
TAG=xfit-random ./scripts/beyondarena/joint_batch.sh tabicl-ft grouped_small cross_fit=random
TAG=drop-group  ./scripts/beyondarena/joint_batch.sh tabicl-ft grouped_small drop_group=true
TAG=rank4       ./scripts/beyondarena/joint_batch.sh tabicl-ft grouped_small rank=4      # also rank 1, 16, 64
TAG=pcop256     ./scripts/beyondarena/joint_batch.sh tabicl-ft grouped_small p_cop=256   # also 4000, 50000
```

Run the same commands with the `temporal_small` shard as well.

Exploratory transfer test, not a main result: run a copula on a marginal it
was not trained with.

```bash
TAG=transfer ./scripts/beyondarena/joint.sh <dataset> tabicl-ft marginal=tabpfn35_fast tabicl_ckpt=null allow_unpaired=true
```

## 4. Report

Tables, hypothesis verdicts (`hypotheses.json`) and figures, in
`results/beyondarena/report/`:

```bash
python -m eval.runners.beyondarena_report
```

Code: `eval/runners/beyondarena_joint_eval.py` (joint track),
`eval/runners/beyondarena_report.py` (statistics, figures), `scripts/beyondarena/`
(tabarena-side scripts and job wrappers). The runner reuses the marginal,
copula and NLL helpers of `eval/runners/ctr23_full_eval.py`.
