# BeyondArena study: pre-registration

Written 2026-11-18, before any BeyondArena run. Changes after the first
confirmatory run are logged in `deviations.md` with a date and a reason.

## Models

| id | role | marginal | copula checkpoint |
|---|---|---|---|
| `tabicl-ft` | **primary** | TabICLv2 fine-tuned (`era5-run1`) | `copula_nano/size-robust-tabicl/step_0100000.pt` |
| `tabpfn-ft` | secondary | TabPFN-3.5 Fast fine-tuned | `copula_nano/size-robust-tabpfn35_fast/step_0100000.pt` |
| `kumo-ft` | secondary | Kumo small fine-tuned | `copula_nano/size-robust-kumo_small/step_0100000.pt` |
| `tabicl-pre` | secondary | TabICLv2 pretrained | `copula_nano/pretrained-marginal-tabicl/step_0100000.pt` |
| `tabpfn-pre` | secondary | TabPFN-3.5 Fast pretrained | `copula_nano/pretrained-marginal-tabpfn35_fast/step_0100000.pt` |
| `kumo-pre` | secondary | Kumo small pretrained | `copula_nano/pretrained-marginal-kumo_small/step_0100000.pt` |

Each copula runs only with the marginal it was trained on
(`ctr23_full_eval.check_marginal_pairing`). Crossing a copula onto another
marginal is the exploratory transfer test and is never a confirmatory result.

Joint-track comparators on the same marginal and the same blocks:
- `indep`: the marginal with independent sampling, i.e. $\Sigma = I$.
- `exch`: an exchangeable Gaussian copula, $\Sigma = (1-\rho)I + \rho\mathbf 1\mathbf 1^\top$.
  $\rho$ is the within-block intraclass correlation of the context's
  out-of-fold PITs $z_{\text{train}}$, formed into blocks by the same rule as the
  test blocks.
- `lmm` (grouped tasks only): a random-intercept linear mixed model,
  $y = X\beta + b_g + \varepsilon$, with $b_g \sim \mathcal N(0,\sigma_b^2)$ and
  $\varepsilon \sim \mathcal N(0,\sigma^2)$. For an unseen test group its joint
  predictive is $\mathcal N(X\hat\beta,\ \hat\sigma_b^2\mathbf 1\mathbf 1^\top + \hat\sigma^2 I)$.
  It has its own marginal, so it is compared on joint metrics only.

## Blocks

A block $S$ is a set of test rows scored jointly.
- **Grouped**: one block per test group, capped at $B_{\max}=128$ rows
  (a seeded subsample when the group is larger). Groups with fewer than 2 test
  rows are dropped.
- **Temporal**: test rows sorted by time, then cut into consecutive windows of
  $B$ rows.
- **IID**: random partitions into blocks of $B$ rows.
- Primary block size: $B=32$ for temporal and IID tasks; the natural group size
  for grouped tasks. The size sweep is $B \in \{2,4,8,16,32,64,128\}$.

## Metrics, per block, averaged within a dataset

Throughout, $S$ is a test block, $|S|$ its size, and $y_S$ the observed targets
in it. $F_i$ is the marginal CDF of row $i$, $z_i = \Phi^{-1}(F_i(y_i))$ its
normal score (PIT), and $\Sigma_{SS}$ the block's copula correlation. $M=1000$
joint samples are drawn as $y^{(m)}_i = F_i^{-1}(\Phi(\tilde z^{(m)}_i))$ with
$\tilde z^{(m)} \sim \mathcal N(0,\Sigma_{SS})$.

- **Marginal CRPS** comes from the quantile grid, as in
  `ctr23_full_eval.quantile_metrics`. It is therefore identical for every
  copula by construction. As an implementation check, the sample-based CRPS of
  `indep` and of each copula must agree within 3 Monte Carlo standard errors.
- **Energy score**: $\mathrm{ES}=\frac1M\sum_m\lVert y^{(m)}-y_S\rVert-\frac1{2M^2}\sum_{m,m'}\lVert y^{(m)}-y^{(m')}\rVert$,
  computed with `eval.metrics.energy_score`.
- **Variogram score** of order $p=0.5$:
  $\mathrm{VS}=\sum_{i<j}\big(|y_i-y_j|^{p}-\hat{\mathbb E}|Y_i-Y_j|^{p}\big)^2$.
- **Joint NLL**: total Y-space NLL per point, marginal plus copula, with
  copula part $\tfrac12(\log\det\Sigma_{SS}+z_S^\top\Sigma_{SS}^{-1}z_S-z_S^\top z_S)$.
- **Block mean**: $\bar y_S=\frac1{|S|}\sum_{i\in S}y_i$. Report the CRPS of
  $\bar y_S$ from the samples, and whether $\bar y_S$ falls inside the central
  90% sample interval (coverage).
- **Correlation calibration**: the mean predicted off-diagonal $\bar\rho_S$ of
  $\Sigma_{SS}$, against the empirical within-block correlation of the test
  $z$. The empirical value is the ICC of $z_{\text{test}}$ over the blocks of a
  dataset.
- **Runtime**: wall-clock seconds per block for the copula forward and for
  sampling, on the GPU type recorded in the output.

Scale: every $y$ metric is computed after dividing $y$ by the context
standard deviation, so that datasets can be averaged.

## Hypotheses and decision rules

Let $d$ index datasets. Each metric is first averaged over the splits and
blocks of a dataset. The confidence interval (CI) is a 95% percentile bootstrap
over datasets ($10^4$ resamples, seed 0). Paired tests are Wilcoxon signed-rank
tests over datasets. The relative ES gain is
$g_d = 1-\mathrm{ES}_d^{\text{cop}}/\mathrm{ES}_d^{\text{indep}}$.

**H1 (official track, non-inferiority).** `tabicl-ft` is run as TabArena's own
`TA-TabICLv2` default config, unchanged except for `model_path` pointing to
the fine-tuned weights. Baselines come from the cache: `TA-TabICLv2`,
`TA-TabPFN-3.5-Fast` and `TA-Kumo-Tabular-Small`. Datasets where a cached
baseline is imputed are flagged in the tables and dropped from the paired test
against that baseline. Let $r_d=\log(\mathrm{RMSE}^{\text{tabicl-ft}}_d/\mathrm{RMSE}^{\text{TabICLv2}}_d)$
over all BeyondArena regression datasets.
- *Supported* if the upper end of the CI on $\mathrm{median}_d\, r_d$ is below $\log 1.02$ (a 2% margin).
- *Refuted* if the lower end of the CI is above 0.
- *Inconclusive* otherwise.

The Elo difference from the TabICLv2 baseline is reported with its CI, as
secondary evidence. The same test is reported for TabPFN-ft and Kumo-ft against
their pretrained versions, also as secondary evidence.

**H2 (joint gain on grouped and temporal tasks).** For `tabicl-ft` on grouped
and on temporal datasets, each regime tested separately:
- *Supported* in a regime if $\mathrm{median}_d\,g_d>0$, its CI excludes 0, and
  the one-sided Wilcoxon test gives $p<0.05$.
- Secondary checks: the same test on joint NLL and on block-mean coverage
  error $|\text{cov}-0.9|$. For the five secondary copulas, Holm's correction
  is applied across them.

**H3 (near-independence on IID tasks).** On IID datasets, for `tabicl-ft`:
- *Supported* if the CI on $\mathrm{median}_d\,g_d$ lies inside $[-1\%,+1\%]$
  (an equivalence test) and $\mathrm{median}_d\,|\bar\rho|<0.05$.

**H4 (structure-aware cross-fitting).** The context PITs come from a $K$-fold
pass, with folds formed either at random or by group or time (`GroupKFold`, or
contiguous time blocks). The two are compared on grouped and temporal datasets.
- *Supported* if $g_d^{\text{struct}}-g_d^{\text{random}}$ has a median $>0$,
  a CI that excludes 0, and a one-sided Wilcoxon $p<0.05$.
- Mechanism check: under random folds, the context ICC of $z_{\text{train}}$
  should be lower than the test ICC of $z_{\text{test}}$.

## Ablations (secondary, `tabicl-ft`, grouped and temporal tasks)

1. Cross-fitting: random vs structure-aware (this is H4).
2. Group identifier kept as a feature vs dropped.
3. Copula rank: the factor $U$ truncated post hoc to its top-$r$ singular
   directions, $r\in\{1,4,16,64,\text{full}\}$. This is post hoc; no copula is
   retrained.
4. Context size $P\in\{256,1000,4000,\text{full}\}$, using a seeded subsample
   of the training rows. The copulas were trained with $P\le1000$, so $P>1000$
   is extrapolation.

## Benchmark inventory (tabarena commit `cc507b84`, `core` subset)

There are 44 regression datasets: 26 IID (`task_type=random`), 7 grouped (all
`group_labels=per_sample`) and 11 temporal. Blocks use the task's `group_on`
and `time_on` columns. Six datasets have at least 800k training rows:
electric_motor (grouped), rossmann and four `*_1m` tables (temporal), and
mercari (IID).

Power: with $n$ datasets, the smallest one-sided Wilcoxon $p$ is $2^{-n}$. That
is $2^{-7}=0.008$ for grouped and $2^{-11}=5\times10^{-4}$ for temporal. A single
grouped dataset going the wrong way already lifts the attainable $p$ to about
0.04, so H2-grouped is fragile by construction. The per-dataset results are
the main evidence there.

## Settings fixed a priori (not tuned)

| setting | value | reason |
|---|---|---|
| $K$ (context PIT folds) | 10 | the repository default (`ctr23_full_eval`) |
| marginal context $P_{\text{marg}}$ | $\min(P_{\text{train}}, 50{,}000)$, seeded subsample | GPU memory |
| copula context $P_{\text{cop}}$ | 1,000 rows of the marginal context | upper end of the copulas' training range $P\in[10,1000]$ |
| $M$ (joint samples) | 1,000 | |
| variogram order $p$ | 0.5 | Scheuerer & Hamill (2015) |
| $B$, $B_{\max}$ | 32, 128 | |
| blocks per split | at most 200, a seeded draw | bounds the runtime on 1M-row tables |
| `exch` $\rho$ | one-way ANOVA ICC of context $z_{\text{train}}$, clipped to $[0, 0.99]$ | |

## Development subset

The grouped and temporal datasets are too few to spend any on tuning. The
development set is therefore 3 IID datasets with 1k to 10k training rows,
drawn with seed 0. It is used only for engineering checks: memory, runtime,
the sample-vs-grid CRPS check, and the exchangeable estimator. No statistical
setting above is changed on its basis without a logged deviation. These 3
datasets are excluded from the H3 test and reported separately. No grouped or
temporal data is looked at before the confirmatory run.

The draw is `np.random.default_rng(0).choice(sorted(candidates), 3)`, where the
candidates are the 7 IID datasets with 1k to 10k training rows. It gave
**miami_housing, qsar_tid_11, wine_quality**.

## Reproducibility

- Every run is seeded with `default_rng([seed, dataset_id, split, block])`.
- Each split's results are written as soon as the split finishes, and a rerun
  skips completed splits (the `ctr23_full_eval` pattern).
- Results are saved as one parquet table per track.
- Negative results are reported with the same prominence as positive ones.

## Amendment 1 (2026-11-18): the joint track moves inside the BeyondArena framework

**Why.** At the user's request, the joint metrics become an extension of the BeyondArena
evaluation rather than a separate protocol. The block caps and the context subsampling of the
first design were my own choices; BeyondArena does not use them.

**What changes.**
- **Unit of evaluation.** Each model is fitted on the whole training fold and predicts the
  whole official test fold. The joint scores are computed on that whole fold, one value per
  split. TabArena's `compare()` then aggregates them like any metric: per-dataset averages,
  ranks and Elo.
- **Protocol.** The joint methods run through `BeyondArenaContext.build_and_run_jobs`, with:
  - core2k regression;
  - the enforced validation protocol;
  - TabArena preprocessing;
  - bagging with `refit_folds=True`, as TabArena's TabICLv2, TabPFN-3.5 and Kumo wrappers do:
    the fold models give the out-of-fold validation predictions, and one model refit on all
    the training rows predicts the test fold.
- **Context.** The copula context is the whole training fold, and so is the marginal context.
  No subsampling.
- **Methods.** One TabArena method per marginal (TabICLv2, TabPFN-3.5 Fast, Kumo small, each
  pretrained and ERA5-fine-tuned), each scored as:
  - `indep`: the marginal without a copula;
  - `exch`: block-exchangeable, $\rho$ from the context;
  - `copula`: its trained copula.

  The three variants share the fit and the normal draws. On grouped tasks, a random-intercept
  LMM is also run, as its own TabArena method.
- **Scores on the whole test fold.**
  - energy score: Monte-Carlo estimator over the $N$-vector, reported divided by $\sqrt N$;
  - variogram score: over uniform and structured (same-group / time-adjacent) pair samples;
  - exact joint NLL (Woodbury);
  - grid and sample CRPS;
  - means of whole test groups, of time windows of $B\in\{2,\dots,128\}$ rows, and of the
    whole fold: their CRPS and 90% coverage;
  - predicted vs empirical within-block correlation.
- **Decision rules.** H2 to H4 keep their rules, applied to the per-dataset scores above.
  The primary score is the energy score.

**Disclosure.** Before this amendment I had seen the stopped first design's interim results on
grouped and temporal datasets (reported to the user on 2026-11-18). The amendment was made for
the protocol reason above, not in response to them.
