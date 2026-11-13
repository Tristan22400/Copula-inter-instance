---
name: g5k-launch
description: Decide where on Grid5000 to run a test, training or eval job, and launch it there — reuse a running node, else find free GPUs on Grenoble/Nancy/Rennes/Sophia with scripts/g5k_gpus.py and submit where the job starts soonest. Use whenever you need to run pytest or any GPU/heavy command, launch or relaunch a training/eval job, when the thoth GPUs (Grenoble vercors) are full or a job waits in the queue, or when asked where GPUs are free.
---

# Launching tests and jobs on the right Grid5000 site

Never run tests or heavy commands on the laptop or on a frontend. Frontends only
run light commands: `oarsub`, `oarstat`, git and the `g5k_gpus.py` queries.
`scripts/g5k_gpus.py` does the site choice. It runs from the laptop's sshfs
checkout and re-executes itself on the Grenoble frontend. Full reference:
docs/workflows.md, "Other Grid5000 sites".

## 1. Pick the route

| Need | Route |
|---|---|
| A few targeted tests (`pytest tests/test_x.py -k ...`), minutes, CPU or 1 GPU | **Reuse a running job** (step 2), else step 3 with `test_full.sh` |
| Full suite / gpu / slow / pretrained tiers | Step 3: `submit -- scripts/test_full.sh [pytest args]` |
| Training, fine-tune, eval job | Step 3 with an explicit `--walltime` |
| "Where are GPUs free?" | `python3 scripts/g5k_gpus.py status --walltime <h>` and report the table |

## 2. Reuse a running job (fastest, no queue)

```bash
ssh grenoble.g5k.remote 'oarstat -u'                                   # running jobs (Grenoble only)
ssh grenoble.g5k.remote 'oarstat -j <id> -f | grep assigned_hostnames'
ssh grenoble.g5k.remote 'OAR_JOB_ID=<id> oarsh <node> "cd ~/thoth_storage/copula-inter && source scripts/_env.sh && python -m pytest -q <tests>"'
```

`python3 scripts/g5k_gpus.py jobs` lists jobs on every site. A job on another
site is reached the same way through that site's frontend:
`ssh grenoble.g5k.remote 'ssh nancy "OAR_JOB_ID=<id> oarsh <node> ..."'`.
Don't load a node the user's training is saturating with a heavy test run.

## 3. Find GPUs and submit

```bash
python3 scripts/g5k_gpus.py status --walltime 2                       # look first (~15 s)
python3 scripts/g5k_gpus.py submit -- scripts/test_full.sh tests/test_x.py -k foo
python3 scripts/g5k_gpus.py submit -- scripts/test_full.sh -m "slow or gpu or pretrained"
python3 scripts/g5k_gpus.py submit --walltime 40 -- scripts/train.sh model=copula_nano <overrides>
python3 scripts/g5k_gpus.py submit --dry-run --walltime 12 -- scripts/eval_checkpoint.sh ckpt=<ckpt>
```

- `submit` takes the job's default `-n`, `gpu=` and `walltime=` from the script's
  `#OAR` lines and ignores the rest (`-q p1`, `-p`).
- It submits to the best target and reads OAR's predicted start. If the job
  starts later than `--max-wait` minutes (default 20), it tries the next target
  (`--tries`, default 4) and keeps the job that starts earliest. Never submit
  the same job by hand to several sites.
- Read the result from the `kept <site>:<jobid>` line. The log is at
  `logs/g5k/<name>.<site>.<jobid>.out` in the checkout, readable from the laptop.
- Watch: `ssh grenoble.g5k.remote 'ssh <site> oarstat -j <id> -f'`.
  Cancel: `... 'ssh <site> oardel <id>'`. Job ids are per site, so always keep
  the site with the id.

### Walltime caps (thoth's levels)

| Level | Where | Cap |
|---|---|---|
| p1 | Grenoble vercors9/10/13/14/18 | 168 h |
| p2 | Grenoble vercors17 (L40S); Sophia musa (H100) | 96 h; musa nodes 6 h |
| p3 | Nancy (granite, gres, gruss, grat...), Rennes abacus*, Sophia esterel10-29 | 48 h; Nancy granite, gruss, grele, grue and graffiti nodes 24 h |
| besteffort (`--besteffort`) | most other esterel/abacus (H100, A100 80GB...) | preempted at any time |

`train.sh` defaults to 96 h, which fits only Grenoble. For other sites pass
`--walltime 48` (or 24) and plan to resume with `training.resume_ckpt=<ckpt>`.
Don't use besteffort for training without a resume plan: OAR resubmits a killed
job (`-t idempotent`), but it restarts from scratch.

## 4. Remote-site checks (Nancy / Rennes / Sophia)

- `setup-site` already ran for nancy, rennes and sophia on 2026-11-13. It
  linked the uv Python and `~/thoth_storage` into those homes. For any other
  site, run `python3 scripts/g5k_gpus.py setup-site <site>` first, and rerun it
  after a `uv python` upgrade.
- Secrets live in each site's home. If `~/.netrc` (wandb) or
  `~/.config/tabpfn_token` is missing on that site, wandb and TabPFN fail. Ask
  the user to run `setup-site <site> --copy-secrets`; don't copy credentials
  yourself. Never fall back to `WANDB_MODE=offline` for training.
  `test_full.sh` disables wandb, so tests don't need `.netrc`.
- Check: `ssh grenoble.g5k.remote 'ssh <site> test -f ~/.netrc && echo ok'`,
  or run the smoke test: `submit --walltime 0:20 -- scripts/g5k_smoke.sh`.
- Expect about 2.5 min of startup (torch import over the cross-site NFS) and
  35 MB/s reads from the Grenoble storage. Live generation and checkpoints are
  fine. Copy an on-disk `data.dataset_dir` to the node's local disk before
  reading it in a loop.

## 5. Report

Tell the user the site, cluster, queue, GPU model, job id, expected start and
log path. If you sent a job off Grenoble or to besteffort, say so.
