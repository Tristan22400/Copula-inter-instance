# Workflows

How to run each stage of the project, from a fresh environment to evaluation
tables. Every Python entry point is a Hydra app: pass settings as `key=value`
overrides, and append `--cfg job` to print the composed configuration without
running anything. See the [glossary](glossary.md) for terms and
[architecture](architecture.md) for where the code lives.

## Environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `uv.lock`
pins the tested environment.

```bash
uv sync --locked --extra dev --extra cpu    # laptop / CI (CPU PyTorch wheel)
uv sync --locked --extra dev --extra gpu --extra tabpfn --extra kumo --extra era5    # CUDA machine, e.g. a Grid5000 node
uv run --no-sync pre-commit install         # ruff check + format on every commit
```

Run commands through the venv with `uv run --no-sync python -m ...` (or
activate `.venv`). The job scripts under `scripts/` source `scripts/_env.sh`,
which:
- activates the checkout's `.venv`, or the main checkout's `.venv` when run from
  a git worktree, or `$COPULA_VENV` when set;
- puts this checkout's sources first on `PYTHONPATH`;
- excludes GPU models known not to work.

**One-time Grid5000 setup.** Run this on a node, from the checkout on your
storage:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs uv into ~/.local/bin
cd ~/thoth_storage/copula-inter
uv sync --locked --extra dev --extra gpu --extra tabpfn --extra kumo --extra era5
```

The `.venv` lives in the checkout, so every OAR job on that storage reuses it.

**TabPFN-3.5 and Kumo (optional).** The two extra marginal backends are uv
extras: `--extra tabpfn` (`tabpfn35_fast`, PriorLabs TabPFN-3.5 Fast) and
`--extra kumo` (`kumo_small`, NVIDIA Kumo Tabular small); `--extra era5` is the
real-data loader. The Grid5000 command above includes all of them: keep every
extra on every later `uv sync` (including repairs with `--reinstall-package`),
which otherwise uninstalls the ones left out. TabPFN also needs a PriorLabs API key whose account accepted the TabPFN-3.5
licence (see **Credentials** below).
Without an extra, its tests are skipped and selecting it raises an ImportError.
If your laptop mounts the same storage (sshfs), don't run `uv sync` through the
mount. That would overwrite the cluster's GPU venv with a CPU one linked to the
laptop's Python. Keep a laptop venv elsewhere:

```bash
UV_PROJECT_ENVIRONMENT=~/.local/share/venvs/copula-inter-cpu uv sync --locked --extra dev --extra cpu
```

**Credentials.** `copula_inter.credentials` resolves the TabPFN API key
(`TABPFN_TOKEN`) and the path of your Grid5000 SSH key (`COPULA_SSH_KEY`);
nothing secret is committed. The first source that holds a value wins:

1. the environment variable;
2. `.env` at the checkout root (git-ignored; template `.env.example`);
3. a one-line file in `~/.config/copula-inter/` (`tabpfn_token`, `ssh_key_path`;
   `COPULA_CONFIG_DIR` moves the directory);
4. for TabPFN only, the legacy `~/.config/tabpfn_token` (or `$TABPFN_TOKEN_FILE`).

Every file must be mode 600 or the loader refuses it. Never pass a key on a
command line, because `ps`/`oarstat` show job command lines to every cluster user.

```bash
cp .env.example .env && chmod 600 .env       # then fill in TABPFN_TOKEN, COPULA_SSH_KEY
# or, outside the checkout (shared by every worktree):
install -d -m 700 ~/.config/copula-inter
install -m 600 /dev/null ~/.config/copula-inter/tabpfn_token && $EDITOR ~/.config/copula-inter/tabpfn_token
# Load the SSH key into the running ssh-agent (once per agent; asks the passphrase):
uv run --no-sync python -c "from copula_inter.credentials import ssh_add; ssh_add(lifetime_s=8*3600)"
```

### Other Grid5000 sites (when the thoth GPUs are full)

`scripts/g5k_gpus.py` finds GPUs you can use on every Abaca site (Grenoble,
Nancy, Rennes, Sophia) and on the SLICES-FR `default` queue (Lille, Lyon,
Nantes), and submits where the job starts soonest. It runs on any
frontend; from a laptop with the sshfs checkout it re-runs itself on the
frontend behind the mount.

```bash
python3 scripts/g5k_gpus.py status --walltime 24 [--gpus 2] [--besteffort] [--abaca-only] [--all]
python3 scripts/g5k_gpus.py submit --walltime 40 -- scripts/train.sh model=copula_nano
python3 scripts/g5k_gpus.py submit --dry-run --walltime 12 -- scripts/eval_checkpoint.sh ckpt=<ckpt>
python3 scripts/g5k_gpus.py jobs                          # your jobs on every site
python3 scripts/g5k_gpus.py setup-site nancy rennes sophia lille lyon nantes [--copy-secrets]   # once
python3 scripts/g5k_gpus.py submit --walltime 0:20 -- scripts/g5k_smoke.sh    # check a site
```

**Priority.** Each Abaca *nodeset* (e.g. `vercors9`, `gres`, `abacus25`) grants
each group a level: `p1` > `p2` > `p3` > `p4` > `besteffort`. The tool reads the
group's levels from `https://api.grid5000.fr/stable/accesses` and takes the best
over your unix groups. The queue sets the walltime cap: p1 168 h, p2 96 h, p3 48 h,
p4 24 h. A node's own `max_walltime` can be lower (Nancy graffiti, granite,
grele, grue and gruss: 24 h; Sophia `musa` H100: 6 h). Within a queue, OAR orders
jobs by Karma, a fair-share score that rises with your recent usage
(`oarstat -j <id> -f`, `message`). A
`besteffort` job runs only on idle GPUs and is killed as soon as a regular job
needs them. `submit` adds `-t idempotent` to besteffort jobs so OAR resubmits
them, but `train.sh` then restarts from scratch unless given `training.resume_ckpt`.
For thoth on 2026-11-13:

| level | GPU nodesets |
|---|---|
| p1 | Grenoble vercors9/10 (A5000), 11 (Quadro RTX 8000), 12 (TITAN RTX), 13 (A6000), 14 (RTX 6000 Ada), 18 (RTX PRO 6000) |
| p2 | Grenoble vercors16 (L4), vercors17 (L40S); Sophia musa (H100 NVL, 6 h nodes) |
| p3 | Nancy granite (RTX PRO 6000), gres (L40S), gruss (A40), grat (A100), graffiti, grele, grue, gratouille; Rennes abacus1/2/4/8/9/11/14/16/25/28/30-B (V100, A40, A5000, P100); Sophia esterel10/19/20/27/28/29 |
| besteffort | most other esterel (A100 80GB, H100, A40, L40S...), abacus (H100, A100, L40S...), vercors2-8, kinovis |

**SLICES-FR `default` queue.** Every account can use these clusters (checked
for thoth on 2026-11-13). Abaca users must name the queue (`-q default`), which
`submit` does. Exotic clusters also get `-t exotic`. The NVIDIA x86 clusters are:

| site | cluster | GPUs |
|---|---|---|
| Lille | chuc | 8 × 4 A100-SXM4 40GB |
| Lille | chifflot | V100 32GB / P100 |
| Lille | chicoree (exotic) | 4 H200 NVL 140GB |
| Lyon | sirius (exotic) | 8 A100 40GB |
| Lyon | gemini (exotic) | 2 × 8 V100 32GB |
| Nantes | ecotaxe (exotic) | 2 × 3 A100 80GB |
| Nancy | grouille (exotic) | 2 × 2 A100 40GB |

The usage policy limits them by the clock (Europe/Paris), and the tool caps the
walltime to what a job starting now may use:
- on a weekday, a job may not cross 09:00 or 19:00; one submitted from 17:00
  may cross 19:00;
- outside weekends a job lasts at most 14 h;
- a job starting Friday evening or at the weekend may run until Monday 09:00.

Use them for nights and weekends, or for short daytime jobs. `--abaca-only`
skips them.

Other GPUs need a different venv and are filtered out by `--arch x86_64` and
`--min-cc 7`:
- AMD GPUs (a ROCm torch build): Luxembourg larochette (MI210) and vianden
  (MI300X), Lyon neowise (MI50);
- ARM nodes: Lyon hydra (GH200), Toulouse estats;
- ppc64le: Grenoble drac.

The Lille frontend is a VM whose CPU lacks x86-64-v2. numpy can't load there,
but it works on the Lille nodes.

**How `submit` decides.** The tool queries the API resources, `oarstat` on each
site frontend, and the OAR Gantt (`/status?waiting=yes`). From them it ranks
targets by:

1. whether the walltime fits the cap;
2. whether some host has `--gpus` GPUs free for the whole walltime;
3. the estimated wait;
4. besteffort last;
5. compute capability, then VRAM;
6. Grenoble first on ties.

`submit` then submits to the first target (`-q <level> -p "nodeset='<x>'"`). It
reads OAR's `scheduled_start` and keeps the job if it starts within
`--max-wait` minutes. Otherwise it tries the next target (`--tries`), keeps the
job with the earliest start and cancels the others. It submits the script as a
command, not with `-S`, so the `#OAR` lines in `train.sh` (`-q p1`, 96 h) don't
apply; their `-n`, `gpu=` and `walltime=` serve as defaults. GPU filters default
to the project's exclusions (TITAN RTX, Quadro RTX 8000, L4), `--min-vram 20`
and `--min-cc 7`. Logs go to `logs/g5k/<name>.<site>.<jobid>.out`; job ids are
per site.

**What makes other sites work.** The thoth group storage
(`/srv/storage/thoth1@storage4.grenoble.grid5000.fr`) is NFS-mounted on every
site, so the checkout, venv, checkpoints and HF cache are shared. The home
directory is per site. `setup-site` recreates on the remote homes what jobs
need from the Grenoble home:

- the `~/thoth_storage` link;
- the uv-managed Python the venv links to: `.venv/bin/python` points to
  `~/.local/share/uv/python/...`. The tool mirrors it to
  `<storage>/.g5k-sites/uv-python` and links it.

With `--copy-secrets` it also copies `~/.netrc` (wandb) and the TabPFN key
files under `~/.config` at mode 600. A key kept in the checkout's `.env` is on
the shared storage, so every site sees it without a copy. `submit` forwards `HF_HOME` and
`UV_CACHE_DIR` as absolute storage paths, because remote `.bashrc` files don't
set them. Rerun `setup-site` after a `uv python` upgrade.

Measured from Nancy (`g5k_smoke.sh`, L40S, 2026-11-13):
- environment setup: 142 s, mostly the torch import over the cross-site NFS;
- reads from the Grenoble storage: 35 MB/s.

Live generation and checkpoint loads are fine at that rate. For an on-disk
`data.dataset_dir`, copy the dataset to the node's local disk first.

## 1. Generate an on-disk dataset

```bash
python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.dataset_dir=./data/pilot
oarsub -S "./scripts/generate_dataset.sh data.n_tasks=5000"        # on Grid5000
```

`data.z_train_source` selects the marginal that produces the PIT: `analytic`
(the exact-GP oracle, no model weights needed), `tabicl` (default),
`tabicl_split`, `tabpfn35_fast` or `kumo_small`. The last two read Phase-A
weights from `data.z_train_marginal_ckpt` (default: pretrained). They need no
GPU at generation time; they are just slower.

Generation writes a manifest. It refuses to resume into a directory created with
different settings or different marginal checkpoint bytes, so use a fresh
directory for a different dataset.

## 2. Train the copula model

```bash
python -m copula_inter.train training.live_generation=false training.dataset_dir=./data/pilot/pit
oarsub -S "./scripts/train.sh <overrides>"                         # on Grid5000
```

Without `training.live_generation=false`, episodes are generated live in
DataLoader workers. Useful overrides:
- `training.steps=500` for a bounded run;
- `training.startup_probes=false` to skip building the fixed validation probes
  (`kernel_fit/*`, `era5_fit/*`, `oracle_diag/*`);
- `baselines.size_grid_enabled=true` to add `val/size_fit/*`: gap to the GP
  posterior and copula capture on a fixed grid of context sizes P and test-set
  sizes N (`baselines.size_grid_P`, `baselines.size_grid_N`), for runs trained
  on varying `data.P_*`/`data.N_*`;
- `training.val_at_start=true` to validate once before the first update, e.g.
  to log the parent checkpoint's scores at step 0 of a post-training run;
- `WANDB_MODE=disabled` to run offline.

`scripts/train.sh` starts a private CUDA MPS daemon for the job (`start_cuda_mps`
in `scripts/_env.sh`) so the workers' kernels share the GPU concurrently instead
of time-slicing; `CUDA_MPS=0` turns it off.

### Fine-tune on real ERA5

Instead of synthetic GP kernels, this samples a random region and grid
resolution for every episode. Fetch the corpus once, then resume from a
checkpoint:

```bash
python -m copula_inter.era5.fetch_global --start 2022-01 --n-months 24
python -m copula_inter.train experiment=finetune_era5 \
    training.resume_ckpt=copula-nano-finetune-marginal-float32 model=copula_nano model.rank=512
oarsub -S "./scripts/finetune_era5.sh training.resume_ckpt=<ckpt> model=copula_nano model.rank=512"
```

`model.*` must match the checkpoint. That family is the nano backbone at rank 512,
while the nano preset is rank 128. `conf/experiment/finetune_era5.yaml` lists the usual extras.

## 3. Evaluate against classical baselines

### Synthetic GP episodes

```bash
python -m eval.runners.eval_checkpoint ckpt=copula-nano-finetune-marginal-float32
oarsub -S "./scripts/eval_checkpoint.sh ckpt=<ckpt>"
```

`ckpt=` takes a family name from `copula_inter/checkpoint_registry.py` or a path.
The run prints two tables. The **total Y-space NLL** table (marginal + copula) is
the comparison to read, because it is a proper scoring rule. The copula-only
table compares correlations on a shared `z_test`.

`marginal.z_train_source` decides where the context PIT comes from:
- `tabicl` (default): K-fold TabICL, as in deployment;
- `oracle`: the exact-GP LOO, an idealized upper bound;
- `tabpfn35_fast` / `kumo_small`: score against the marginal a run was trained
  with. Its weights default to the checkpoint's own `data.z_train_marginal_ckpt`
  (override with `marginal.backend_ckpt=<path>`). Also pass
  `autoregressive.enabled=false`.

`output.dump_episodes=scores.json` saves per-episode scores, and
`python -m eval.results scores.json` re-renders the totals later without loading
any model.

**GP baselines.** Every GP-MLE, DKL and zero-mean GP row predicts its
posterior given the context (`baselines.gp_predict=posterior`), with a learned
mean (`baselines.gp_mean=constant`; `zero` or `linear` also work). Tables made
before 2026-11-13 used `gp_predict=prior`: those rows scored the fitted kernel's
prior at the queries and ignored the context, so do not compare them with newer
tables. Rows that pick or mix GPs:

| Row | Selects with | Role |
|---|---|---|
| GP-MLE (BIC-selected) | context only: $\max_k \log p(y_C\mid\hat\theta_k) - \tfrac{k}{2}\log P$ | competitor |
| GP-MLE (BIC-weighted mixture) | context only, weights $\propto e^{\mathrm{BIC}_k}$ | competitor |
| Marginal + GP-MLE posterior corr. (BIC) | the same marginal as our model, $R_{po}(\hat\theta)$ as copula | matched competitor of the copula head |
| Marginal + oracle posterior corr. | the same marginal, $R_{po}(\theta)$ of the true GP | reference (synthetic only) |
| Best-of-Baselines, Best GP | query labels | reference, not a method |

The z-space "Oracle" row is the copula of the known-θ GP posterior, $R_{po}$:
under exact marginals $z_{\text{test}}\sim\mathcal N(0, R_{po})$.

### Real ERA5 episodes

These are the same baselines, nested-CV best-of-baselines, tables and caches, on
real 2 m temperature (`eval/data/era5_episodes.py`):

```bash
python -m copula_inter.era5.fetch_global --start 2023-01 --n-months 12 \
    --cache-dir ./eval/data/cache/era5_global_val            # once: the held-out year
python -m eval.runners.eval_checkpoint era5.enabled=true n_episodes=400 \
    ckpt=copula-nano-finetune-marginal-float32
oarsub -S "./scripts/eval_checkpoint_era5.sh ckpt=<ckpt>"
```

Differences from synthetic data:
- Real data has no generating kernel, so the "Oracle" rows and the
  analytic GP rows are `nan`, and `marginal.z_train_source=oracle` is rejected.
- The shared `z_test` is the frozen-TabICL K-fold PIT, not a ground-truth
  marginal. Read the total table.
- Targets are z-scored per episode before the baselines are fitted
  (`era5.standardize_y`, on by default), and their NLLs are converted back to
  raw-Kelvin nats. The baselines' GP hyperpriors assume O(1) targets, and our
  model normalizes internally, so raw ~280 K targets would handicap the
  baselines on units alone.

Fitting the baselines takes about 98 % of the runtime and doesn't depend on the
checkpoint. To compare a second checkpoint on the same episodes, share
`baselines.cache` and give each checkpoint its own `output.results_cache`; only
the ICL forward pass is redone.

### Real-data benchmarks

`real.dataset=<name>` swaps the episode source for one of `eval/data/real/`,
with the same baselines, tables and caches as ERA5 (no oracle rows, the
shared `z_test` is the marginal's PIT, read the total Y-space table). Raw
files live under `REAL_DATA_DIR/<name>/` (default `eval/data/cache/`; from a
git worktree, point `REAL_DATA_DIR` at the main checkout's cache). Episode
`ep_i` is drawn from `default_rng([seed, ep_i])`, so `episode_offset` shards
reproduce the unsharded stream; smoke runs use `seed=43` so none of their
episodes is a reported one. Pair every copula checkpoint with the marginal it
was trained with (`tabicl.pit_ckpt` in its config) via `marginal.tabicl_ckpt`.

#### SDWPF wind farm

One timestamp per episode, rows = turbines, target `Patv` (kW); five 49-day
folds, CI cluster = day (`eval/data/real/sdwpf.py` documents the episode
contract, the cleaning and the partial-load rule).

Feature set: **F0 = (x, y) only.** F1b (positions rotated by the farm wind
direction) is dropped: the audit (`sdwpf_audit`, 2026-11-04) found the
per-timestamp circular std of $\phi_i = (\mathrm{Ndir}_i + \mathrm{Wdir}_i) \bmod 360$
across turbines at about 73° in every fold (68 of 134 turbines biased more than
30° from the farm mean), far above the pre-declared 30° limit, so all five
cross-fitted decisions rejected it. The code path stays, for the audit and
its tests, but no reported run uses it.

```bash
python -m eval.runners.sdwpf_audit output_dir=results/sdwpf_audit    # model-free, per fold
python -m eval.runners.eval_checkpoint ckpt=<ckpt> marginal.tabicl_ckpt=<its marginal> \
    real.dataset=sdwpf real.sdwpf.features=f0 real.sdwpf.mask=random n_episodes=400 \
    output.dump_episodes=results/sdwpf/f0_random.json
python -m eval.runners.paired_compare a=results/sdwpf/f0_random.json \
    b=results/sdwpf/f1b_random.json
python -m eval.runners.paired_compare a=results/sdwpf/f0_random.json
```

The second command is F1b minus F0 on identical episodes (it fails if the
episode meta differs); the third gives the copula's within-run contrasts
against independence, the autoregressive row and best_baseline. Every delta
carries a 95 % cluster-bootstrap CI (2,000 replicates) and a win rate;
`folds=[...]` restricts both runs to the listed folds for cross-fitting.

Timestamp budget: episode j of a fold owns `timestamp_tries` hourly candidates,
so a fold supports at most ⌊candidates / timestamp_tries⌋ episodes (400 episodes
= 80 per fold). The P=64 cell runs with `real.sdwpf.timestamp_tries=7`: fold 0
has 580 candidates at that P + N, so 8 × 80 = 640 > 580, while 7 × 80 = 560 fits.
Changing `timestamp_tries` changes the slice boundaries, so the P=64 episodes
are a different stream from the other cells (they are not paired across P
anyway). Every other cell keeps the default of 8.

#### Real-data grid protocol (fixed 2026-11-05, before any grid score)

Written before the grid ran; nothing below changes after scores are seen.

- **Episodes.** Reported seed 42 (smoke runs used 43). Every drawn episode is
  reported; no held-out half. 400 episodes per cell, except Meuse (201 = 67 ×
  3 folds) and CTR23 (100 per dataset and mask: the unit there is the dataset,
  24 of them). Cells run as shards (`scripts/real_grid.sh <cell> <offset> <n>`);
  `paired_compare` and `real_calibration_gate` read a shard glob.
- **Checkpoints.** `copula-nano-finetune-marginal-float32` (630k) with its
  marginal `marginal_finetune_era5_run1`, zero-shot. A second checkpoint runs
  with its own marginal and its own baseline caches (`REAL_CKPT_TAG`).
- **Cells** (`./scripts/real_grid.sh list`), P/N as in each dataset section:
  SDWPF F0 random, block; KAUST {1a_2, 1b_2, 1a_1, 1b_1} × {random, block};
  EMNIST test split, P ∈ {50, 200} × {random, block}; California {f0, f1} ×
  {random, block}; Meuse {f0, f1}; CTR23 the 24 kept datasets × {dyadic_knn,
  random}.
- **P sweep** (½P, 2P) on the main mask: `random` for SDWPF (P = 16, 64), KAUST
  1a_2 (P = 64, 256) and California F0 (P = 64, 256); EMNIST's two P values are
  its sweep. The CTR23 sweep (dyadic_knn, P = 64, 256) is deferred until the
  main CTR23 cells are in.
- **Statistics per cell.** The total Y-space NLL table; paired Δ for copula −
  independence, copula − autoregressive, copula − best_baseline, each with a
  95 % cluster-bootstrap CI (2,000 replicates, the dataset's CI cluster) and a
  win rate; Meuse reports mean ± sd and no CI; CTR23 per-dataset Δ then a
  Wilcoxon signed-rank test across datasets. The calibration gate runs on
  every cell; a z_test std outside [0.8, 1.25] is flagged.
- **Cross-fitted choices.** SDWPF only: F1b was dropped by all five
  cross-fitted direction-audit decisions, so every SDWPF cell is F0. No other
  dataset makes a score-based choice.
- **Exclusions fixed in advance.** SDWPF cleaning and partial load; California
  top-code; CTR23 atom rule (11 of 35 datasets excluded).

#### Shared tools for every real dataset

```bash
python -m eval.runners.real_audit real.dataset=kaust real.kaust.variant=1a_2    # model-free audit, smoke seed 43
python -m eval.runners.real_calibration_gate dumps=[results/real/kaust/1a_2_random.json]
```

`real_audit` verifies and loads the data as `eval_checkpoint` does, prints the
module's dataset checks (cleaning counts, atoms and ties, layout), then draws
episodes with the smoke seed and reports skips, shapes, clusters and folds.
`real_calibration_gate` is the marginal gate: mean, std, |z| > 3 fraction, KS
statistic against N(0, 1) and quantile ECE of the stored `z_test`, flagging a
std outside [0.8, 1.25]. `scripts/eval_checkpoint_real.sh` runs one cell on a
whole GPU node; set `REAL_CELL` per cell so baseline caches never mix cells.

#### KAUST 2022 (sub-competitions 1a, 1b)

One square window of side `1 / real.kaust.window_cells` (default 1/15) per
episode, inside one of five vertical strips (fold = strip = `ep_i % 5`);
P = 128, N = 64, masks `random` and `block`, features (x, y). CI cluster: the
window-grid cell holding the window's centre. Variants `*_2` (zero-mean
nonstationary Matérn GP) are the main result; `*_1` (deterministic mean plus
iid noise) are the negative control where the copula must not hurt
(`eval/data/real/kaust.py`).

```bash
python -m eval.runners.real_audit real.dataset=kaust real.kaust.variant=1b_2
python -m eval.runners.eval_checkpoint ckpt=<ckpt> marginal.tabicl_ckpt=<its marginal> \
    real.dataset=kaust real.kaust.variant=1a_2 real.kaust.mask=block n_episodes=400 \
    output.dump_episodes=results/real/kaust/1a_2_block.json
```

#### EMNIST (balanced)

One image per episode (each episode a different image), rows = pixels,
features (row, col), target $y = (\text{pixel} + U)/256$ with $U \sim U(0,1)$
drawn per episode; P ∈ {50, 200}, N = 64, masks `random` and `block`. Report on
`real.emnist.split=test`, debug on `train`. CI cluster: the image. The gate
also prints bits/pixel, $(\text{NLL} + \ln 256)/\ln 2$.

```bash
python -m eval.runners.eval_checkpoint ckpt=<ckpt> marginal.tabicl_ckpt=<its marginal> \
    real.dataset=emnist real.emnist.n_context=200 real.emnist.mask=block n_episodes=400 \
    output.dump_episodes=results/real/emnist/p200_block.json
```

#### California Housing

`fetch_california_housing` into `REAL_DATA_DIR/california`. Rows at the
$500,001 top-code (the maximum, 5.00001, checked to be ≥
`real.california.censor_cap` = 5.0) are dropped once, at the dataset level;
rows at exactly 5.0 ($500,000) are kept. P = 128, N = 64, masks `random` and `block`
(kNN in latitude/longitude); `features=f0` (all 8) or `f1` (without
latitude/longitude, the ablation; same episodes). CI cluster: the block
anchor, or the episode for `random`.

```bash
python -m eval.runners.eval_checkpoint ckpt=<ckpt> marginal.tabicl_ckpt=<its marginal> \
    real.dataset=california real.california.mask=block real.california.features=f1 n_episodes=400 \
    output.dump_episodes=results/real/california/f1_block.json
```

#### Meuse

Repeated 3-fold over the 155 points: 67 repetitions × 3 folds = 201 episodes,
query = one fold (51–52 points), context = the rest; every point is queried 67
times. Target log(zinc); `features=f0` (x, y) or `f1` (x, y, dist, elev). The
folds share points, so there is **no CI**: `paired_compare` reports mean ± sd.

```bash
python -m eval.runners.eval_checkpoint ckpt=<ckpt> marginal.tabicl_ckpt=<its marginal> \
    real.dataset=meuse real.meuse.features=f1 n_episodes=201 \
    output.dump_episodes=results/real/meuse/f1.json
```

#### OpenML-CTR23 (suite 353)

One dataset per run (`real.ctr23.data_id`, one of the 35 pinned in
`eval/data/real/ctr23.py`), fetched by `fetch_openml` into
`REAL_DATA_DIR/ctr23`; categoricals ordinal-encoded and numerics
mean-imputed as TabICL does. A dataset whose most common target value holds
more than 5 % of its rows is excluded (`real.ctr23.max_atom_frac`; a real point
mass, not mere rounding; fixed before any CTR23 score was seen; the audit lists
them). P = 128, N = 64; `dyadic_knn` (the N/2 nearest neighbours of each of
two anchors in standardized feature space) is the main mask, `random` the
negative control. The unit is the dataset: a per-dataset mean Δ, then a
Wilcoxon signed-rank test across datasets.

```bash
python -m eval.runners.real_audit real.dataset=ctr23 ctr23_all=true
python -m eval.runners.eval_checkpoint ckpt=<ckpt> marginal.tabicl_ckpt=<its marginal> \
    real.dataset=ctr23 real.ctr23.data_id=44957 real.ctr23.mask=dyadic_knn n_episodes=100 \
    output.dump_episodes=results/real/ctr23/44957_dyadic_knn.json
python -m eval.runners.paired_compare across=[results/real/ctr23/44957_dyadic_knn.json]
```

#### OpenML-CTR23 on the full datasets (decided with the user on 2026-11-09)

A second protocol, next to the episodes: the task's **official 10-fold CV**
(OpenML split file, repeat 0, fetched once into `REAL_DATA_DIR/ctr23`), with
the **whole training fold as context** and the **whole test fold as queries**,
so the marginal is comparable with published CTR23 results and the copula is
tested far beyond its training P. Same 24 datasets (atom rule), same
preprocessing and checkpoint pair. Per fold:

- the marginal PITs the context with its 10-fold pass (context rows shuffled
  first) and predicts the test fold: marginal NLL per point, RMSE and R² of the
  quantile mean, MAE of the median, CRPS on the quantile grid, all in raw units;
- the copula's Σ over the test rows (a test row attends only to the context, so
  the forward runs in exact `query_batch` slices) scores the copula NLL over the
  whole fold and over two partitions into chunks of 64 (`random`, and
  `dyadic_knn`: two kNN clusters per chunk), each test row in exactly one chunk.
  A chunk's NLL uses Σ's sub-block, the copula's own joint over that chunk.
  Independence has copula NLL 0, so `copula_*` is the copula minus independence.

No GP baselines and no autoregressive row: neither scales to P ≈ 6·10⁴. The
JSON holds each fold and the mean over folds; a `.npz` beside it holds the
per-row predictions, log-densities, z_test and Σ's factor rows. A rerun skips
finished folds.

**Other foundation models (decided with the user on 2026-11-09).** With
`ckpt` unset the runner scores a marginal alone, i.e. the independence model
(copula NLL 0 over every query set), on the test fold only: `marginal=tabicl`
with `tabicl_ckpt=pretrained` (TabICL v2), or `marginal=tabpfn35_fast` /
`kumo_small` (pretrained; `marginal_ckpt` takes a Phase-A checkpoint). The
fine-tuned TabICL independence row is the marginal half of the copula run (the
same test forward). `compare_methods` ranks the methods per dataset on total NLL
(whole fold, or a chunk partition), CRPS or RMSE: mean and median rank, the
Friedman test and Holm-corrected pairwise Wilcoxon tests, over the datasets
every method completed; a run that failed for memory on a 48 GB GPU leaves its
dataset out of the ranking (`"failed"` in its JSON). TabPFN and Kumo need a venv
with the `tabpfn` / `kumo` extras (`COPULA_VENV`) and, for TabPFN, the API key
(see **Credentials** under Environment).

A copula on another backend takes `ckpt` with `marginal` and `marginal_ckpt`; the
runner checks against the copula's training config (`data.z_train_source`,
`data.z_train_marginal_ckpt` or `tabicl.pit_ckpt`) that it is scored with its own
marginal, where a null `z_train_marginal_ckpt` means the pretrained weights.
`scripts/ctr23_full.sh` names the pairs: `kumo_ft_nano` / `tabpfn_ft_nano` (copula
on the ERA5-fine-tuned marginal) and `pre_tabicl_nano` / `pre_tabpfn_nano` /
`pre_kumo_nano` (the PRE_nano_PN_* copulas, on the pretrained marginal).

```bash
python -m eval.runners.ctr23_full_eval ckpt=<ckpt> tabicl_ckpt=<its marginal> data_id=44957 \
    out_json=results/real/ctr23_full/44957_nano.json
python -m eval.runners.ctr23_full_eval marginal=tabpfn35_fast data_id=44957 \
    out_json=results/real/ctr23_full/44957_tabpfn35_fast.json
./scripts/ctr23_full.sh 44957               # the copula with the ERA5-report pair, on an OAR GPU node
./scripts/ctr23_full.sh 44957 tabicl_base   # or tabpfn35_fast, kumo_small: the marginal alone
```

### The autoregressive row

This row is on by default for the TabICL marginal. It uses the same marginal
with no copula head, revealing the query points one at a time so that each
prediction conditions on the ones already revealed:

    log p(y_1..y_N) = sum_i log p(y_s(i) | context, y_s(<i))

That sum is an exact factorization of a joint density, so the row is comparable
to every other row of the total table. It is the copula-free reference the
copula head has to beat:
- its Marginal column is the one-shot (independence) marginal;
- its Copula column is therefore exactly what the sequencing bought;
- step 0 reproduces the one-shot PIT's `log_pdf_test` bit for bit
  (`tests/test_autoregressive.py`).

| Override | Effect |
|---|---|
| `autoregressive.enabled=false` | Turn the row off |
| `autoregressive.order=natural` | Reveal in grid order instead of a seeded per-episode permutation. An ICL model is not a coherent joint, so the total depends on the order; `natural` gives nearly every step a just-revealed neighbour and reads as a best case |
| `autoregressive.conditioning=sample` | Append a draw instead of the true y (ancestral sampling). The number is then **not** a density of `y_test` and not comparable to the other rows; the table prints a warning |
| `autoregressive.max_context=K` | Cap the chain's context. The episode's own P rows are always kept, and the oldest revealed rows are dropped first |
| `autoregressive.n_episodes=M` | Run the chain on the first M episodes only |

It runs on the GPU, batched over `era5.pit_batch` episodes; measured at about
1.5 s/episode on an RTX PRO 6000 Blackwell and 3.6 s/episode on an RTX A5000.
It is TabICL-only: the other backends have no one-query-at-a-time entry point
and raise an error rather than silently dropping the row.

### Real-world tabular benchmarks

```bash
python -m eval.runners.run_benchmarks        # UCI Beijing PM2.5, California Housing
```

## 4. Phase A: fine-tune the marginal

The training loss is copula + marginal, but the marginal comes from a frozen
model, so no copula run can improve its term. Phase A fine-tunes that
standalone marginal; the two phases meet only at a checkpoint path.

1. **Measure the defect** (no training). The headline number is the marginal-NLL
   gap to the analytic GP oracle:

   ```bash
   python -m eval.runners.marginal_calibration_eval ckpt=pretrained
   ```

2. **Fine-tune.** Phase A logs to its own wandb project, `copula-inter-marginal`:

   ```bash
   python -m copula_inter.finetune_marginal
   oarsub -S ./scripts/finetune_marginal.sh
   ```

   By default (`marginal.lora_all_layers=true`), LoRA at one shared rank goes on
   every 2-D weight matrix of the selected backbone. Measured at rank 8:

   | Backbone | Matrices | Trainable parameters |
   |---|---|---|
   | tabicl | 153 | 854K (2.91 %) |

   `marginal.backbone` picks the model: `tabicl` (default), `tabpfn35_fast` or
   `kumo_small`. The tier ladder is an ablation:
   `marginal.lora_all_layers=false marginal.tier=1`. Only tabicl can climb it
   (`backend_registry.Backend.max_tier`); the other two stop at tier 0.

   Each model is scored on a 999-level grid (`marginal.probs_n=null`): TabICL's
   and Kumo's native decoder levels, TabPFN's bar distribution read at the same
   levels. The objective is therefore comparable across backbones. Set
   `probs_n=<int>` only to resample onto a coarser grid. The generic backbones'
   PIT (`finetune_adapters.backend_pit`) reads the same model forward as their
   Phase-A quantiles (`backend_quantiles`), as a full predictive distribution:
   TabPFN's bar distribution with its half-normal tails, Kumo's 999 quantiles
   through TabICL's `QuantileToDistribution`. So a fine-tuned model is scored
   as it was trained, with a proper density like TabICL's.

3. **Re-measure and gate on real data.** The point of a TabICL marginal is
   non-Gaussian tabular transfer, which GP-only training can destroy, so real
   data must not regress. Then hand the checkpoint to a copula run:

   ```bash
   python -m eval.runners.marginal_calibration_eval ckpt=<the _final.pt>
   python -m eval.runners.run_benchmarks
   python -m copula_inter.train tabicl.pit_ckpt=<the _final.pt>
   ```

   For `tabpfn35_fast` / `kumo_small` the copula run selects the backend and its
   weights together:

   ```bash
   python -m copula_inter.train data.z_train_source=kumo_small \
       data.z_train_marginal_ckpt=<the _final.pt>
   ```

Non-TabICL Phase-A checkpoints load back through
`copula_inter.marginal.backends.make_regressor(..., ckpt=<path>)`, not
`marginal.pit.load_tabicl`; a checkpoint written for another backbone is
refused. Register finished runs in `checkpoint_registry.py::MARGINAL_FAMILIES`.

## Spatial-correlation diagnostics

One CLI covers real ERA5 and synthetic kernels with known ground truth. ERA5
data is fetched and cached automatically.

```bash
# Everything: sweep all registered checkpoints, fit baseline curves, render the report.
python -m eval.runners.spatial_correlation_eval

# One subcommand at a time (command=<name> --cfg job lists its keys):
python -m eval.runners.spatial_correlation_eval command=diagnose \
    command.ckpt=copula-nano-finetune-marginal-float32 command.mode=real \
    command.region=western_europe command.grid_size=24
python -m eval.runners.spatial_correlation_eval command=sweep command.mode=synthetic command.checkpoints=all
python -m eval.runners.spatial_correlation_eval command=baseline command.mode=real
python -m eval.runners.spatial_correlation_eval command=report

# Marginal-quantile calibration on real ERA5 (per-quantile ECE, independence-copula checks):
python -m eval.runners.era5_calibration_eval
```

A real-mode sweep also scores the total (marginal + copula) joint NLL on
held-out ERA5 points. The correlation-curve `model_r2` can't say how many nats
worse the predictive density is.

## Compare training runs from wandb

`eval.runners.wandb_report` reads runs from wandb. It prints three things:

- **Full NLL split:** the full validation NLL as marginal + copula, for the run and for the oracle. The marginal alone is the run's full NLL under independence.
- **Best steps:** the step with the best full NLL, and the step with the best copula on analytic $z$.
- **Size grid:** the analytic-$z$ copula NLL for each context size $P$.

Run it on a compute node: the frontend's CPU can't import numpy. It reads credentials from `~/.netrc`.

```bash
python -m eval.runners.wandb_report filter=_PN_ since=2026-11-09
python -m eval.runners.wandb_report ids=[splmz6y3,0jkldstb] sort_by=grid out=report.json
```

The `.claude/skills/wandb-results` skill explains how to read the tables.

## Debug pipeline

`debug/` holds a staged investigation of why training with
`data.z_train_source=tabicl` plateaus. See [debug/README.md](../debug/README.md).

## Tests

```bash
uv run --no-sync python -m pytest -q -n auto -m "not slow and not gpu and not pretrained and not external_data"
oarsub -S ./scripts/test_full.sh                 # full suite on a GPU node, before merging
```

CI (`.github/workflows/ci.yml`) runs the first command, plus ruff and mypy.
