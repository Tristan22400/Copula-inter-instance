# An ERA5-like prior for the copula head — design, indicators, and validation

Status: **proposal**. Phase 0 (the measurement harness and the baseline gap table)
is implemented and run; Phases 1–4 are specified but not implemented.

Code added by Phase 0:
- `eval/prior_similarity/indicators.py` — the indicator suite
- `eval/prior_similarity/bundles.py` — the three episode sources reduced to one representation
- `eval/runners/prior_similarity_eval.py` — CLI, writes `eval/reports/prior_similarity/`

```bash
python eval/runners/prior_similarity_eval.py \
    --sources era5,synthetic_current,lattice_matern_probe \
    --n-bundles 40 --n-realizations 3000 --era5-months 110 \
    --grid-min 24 --grid-max 24 --device cuda
```

---

## 0. What "looks like ERA5" has to mean here

The copula head maps `(x_train, z_train, x_test)` to a correlation matrix `R` over the
test points, and is scored by the Gaussian-copula log-density of `z_test` under `R`
(plus a marginal term with no trainable parameters). So a prior "looks like ERA5" iff
**the conditional correlation an episode induces — `R_post = Corr(y_test | context)` —
has ERA5's geometry, spectrum, anisotropy and non-stationarity**, and the marginal/tail
behaviour stresses the Gaussian-copula + TabICL-marginal factorization the same way.
It does *not* mean the sampled fields render like a temperature map. Judging by how
smooth a sample looks actively rewards overconfidence, which is how the current
checkpoints were misread once already.

Two consequences drive the whole design:

- **The copula is invariant under monotone marginal transforms.** We can therefore
  match ERA5's marginals as aggressively as we like — skew, heavy tails, bounded
  support — by pushing a Gaussian field through a monotone warp, and the analytic
  Gaussian-copula target (`R_star` / `R_prior`) stays exactly correct. This is free
  realism and it is currently unused.
- **The copula is emphatically *not* invariant to the design.** Where the sample points
  sit relative to the correlation length determines whether `R_post` is smooth and
  low-rank or rough and near-full-rank. This is what today's prior gets most wrong.

---

## 1. Diagnosis — where today's prior actually differs

All numbers from `eval/reports/prior_similarity/summary.md`, 40 bundles per source,
D = 576 (24×24) for every source so nothing is confounded by episode size, 3000
realizations each (ERA5 days / GP draws), identical indicator code on both sides.
Distances are always in units of the design's own median nearest-neighbour spacing —
the only unit in which a scattered 9-dimensional point cloud and a lat/lon lattice are
comparable at all.

Headline numbers, median [q25, q75], and the gap in reference standard deviations
(`(median_synth − median_era5) / (IQR_era5 / 1.349)`). The full 50-indicator table is
`eval/reports/prior_similarity/summary.md`.

| indicator | ERA5 | current prior | gap | lattice+Matérn probe | gap |
|---|---|---|---|---|---|
| `nn_cv` | 0.095 | 1.19 | **+11.4** | 0.00 | −1.0 |
| `spread_over_nn` | 16.6 | 8.4 | **−1.8** | 12.2 | −1.0 |
| `rho_at_1nn` | 0.983 | 0.847 | **−7.8** | 0.986 | +0.2 |
| `rho_far` | 0.697 | 0.033 | **−3.6** | 0.671 | −0.1 |
| `decorrelates_in_box` | 0.0 | 1.0 | **qualitative** | 0.0 | 0.0 |
| `od_mu` | 0.803 | 0.223 | **−4.5** | 0.770 | −0.3 |
| `matern_nu` | 0.47 | 1.02 | **+4.5** | 0.47 | −0.1 |
| `prior_eff_rank_frac` | 0.0047 | 0.0443 | **+15.1** | 0.0052 | +0.2 |
| `post05_var_ratio` | 0.041 | 0.267 | **+6.5** | 0.042 | +0.0 |
| `post05_rel_change` | 0.966 | 0.882 | **−6.5** | 0.964 | −0.2 |
| `post05_range_over_nn` | 2.18 | 0.774 | **−2.0** | 1.81 | −0.5 |
| `post05_evr32` | 0.619 | 0.493 | **−2.3** | 0.645 | +0.5 |
| `post20_rel_change` | 0.990 | 0.947 | **−16.2** | 0.988 | −0.8 |
| `increment_exkurt` | 4.62 | 2.72 | −0.4 | 3.42 | −0.2 |
| `chi_095_excess` | 0.039 | 0.006 | **−1.4** | 0.006 | −1.4 |
| `spatial_exkurt` | −0.43 | +0.45 | +1.3 | +1.78 | **+3.2** |

Two independent checks that the harness is measuring the right thing: ERA5's
`post05_evr32` comes out at 0.619, against 0.63 measured at D=600 by a completely
different route (per-day exact-GP-MLE fit) in the earlier rank-ceiling analysis; and the
ERA5 correlogram is fit by a Matérn with ν ≈ 0.47, i.e. near-exponential, which is the
shape the existing baseline curve-fitting already lands on.

### D1 — Design geometry (confirmed, largest single gap)

`data_gen._generate_gp_batch_raw:3345` draws `x_raw ~ N(0, I_d)` with
`d ~ LogNormal(log 10, 0.4)` and lets the kernel read `k` of those columns; ERA5 episodes
are a `grid_size × grid_size` lattice of (lon, lat) plus four static covariates. Measured
`nn_cv` 1.19 vs 0.095 — **+11.4σ, the largest gap in the suite**. (ERA5 is not exactly 0
because lat/lon spacing is anisotropic in km and varies with latitude.)

The downstream consequence is visible one row down: the synthetic prior *decorrelates
inside the episode* (`rho_far` 0.033, `decorrelates_in_box` = 1 for every bundle) while
ERA5 never does (`rho_far` 0.697, `decorrelates_in_box` = 0 for every bundle). These are
qualitatively different regimes, not two points on one continuum.

### D2 — Conditioning barely does anything, and leaves near-independence

The core finding, and it is sharper than expected.

- Context removes 96% of the variance in an ERA5 episode and 73% in a synthetic one
  (`post05_var_ratio` 0.041 vs 0.267, +6.5σ).
- Conditioning changes the correlation far less (`post20_rel_change` gap −16.2σ).
- **After conditioning, synthetic test points are essentially independent**:
  `post05_range_over_nn` is 0.774 — the residual correlation length is *shorter than one
  sample spacing* — versus 2.18 for ERA5. So the head spends most of training on episodes
  whose correct answer is close to the identity matrix, then is asked at deployment to
  emit substantial off-diagonal structure.

One precision worth stating because it is easy to get backwards: the *target definition*
is not the problem. `oracle_mode: prior` governs `R_star`/`Sigma_star`, which feed only the
`aux_mae` head (weight 0 by default); the copula loss is scored on `z_test`, standardized
by the posterior marginals, so the graded quantity is already the conditional correlation
(`data_gen.py:3502-3512`, `pit.py::gp_analytical_posterior`). The problem is the data, not
the label.

### D3 — Spectral concentration and single-scale-ness

`prior_eff_rank_frac` 0.044 vs 0.0047 — the synthetic prior's spectrum is **~10× flatter**
(+15.1σ). ERA5 boxes are dominated by a few large-scale modes with structure underneath;
the synthetic prior spreads its variance evenly across many modes. This is the same
one-scale-vs-two-scale story as D2 seen in the spectral domain, and it is what §2.2 targets.

Correlation *shape* differs too: ERA5 is near-exponential (`matern_nu` 0.47) while the
synthetic prior is smoother (1.02, +4.5σ).

### D4 — Non-Gaussianity: **hypothesis not supported, deprioritise**

I expected `increment_exkurt` ≈ 0 for the synthetic prior "by construction" — a GP draw has
Gaussian increments. That is wrong for *this* prior: measured 2.72 against ERA5's 4.62, a
−0.4σ gap that is not significant at 40 bundles. The existing feature machinery
(`tabiclv2_warp_features`, `apply_structural_feature_warp`, `apply_mlp_feature_mixing`,
`mean_fn`) already makes episodes substantially non-Gaussian in space. **The marginal warp
in §2.5 is therefore a low-priority nice-to-have, not a fix for a measured defect**, and I
would not build it before Phases 1–2.

What *is* different in Tier 3 is tail dependence: `chi_095_excess` 0.039 (ERA5) vs 0.006
(synthetic), i.e. ERA5 neighbours co-exceed high quantiles more often than a Gaussian
copula at the same correlation predicts. Neither synthetic source reproduces this, and a
Gaussian-copula model structurally cannot — see §7.

---

## 2. The proposed prior family

An `era5_like` geometry + kernel branch, six components, each attached to one measured
gap and each independently switchable so the ablation is clean.

### 2.1 Lattice design sampler — fixes D1

`data.geometry: {gaussian_cloud | lattice2d}`, chosen per `generate_gp_batch` call with
probability `data.lattice2d_frac`.

- `grid_size ~ U{8..28}`; jitter each point by `ε · spacing`, `ε ~ U[0, 0.15]`, so the
  model learns spatial structure rather than a memorised lattice.
- With some probability, keep an irregular random subset of the lattice instead of all of
  it — real deployment includes station-like designs, and it prevents overfitting to
  perfect regularity.
- Two columns carry the lattice; `q` further columns carry synthetic *static covariates*:
  smooth random fields with a heavy-tailed marginal (orography-like) and one thresholded
  binary field (land-sea-like). **These must be the same latent fields that modulate the
  kernel (§2.4)** — otherwise they are noise columns and the model correctly learns to
  ignore them, which is what happens with the real static columns today.
- Context fraction `P/D ~ U[0.05, 0.4]`, matching `era5_live`.

### 2.2 Two-scale correlation — fixes D2, and part of D3

`C = w · K_large(L_large) + (1 − w) · K_small(L_small)`, with `w ~ Beta`,
`L_small ∈ [1.5, 8]` sample spacings and `L_large / L_small ~ U[5, 40]`.

The large-scale component is what the context removes; the small-scale component is what
the model then has to predict. This is the single most important change for
post-conditioning rank, and it is what makes `post05_var_ratio` movable.

### 2.3 Anisotropy — and retiring the ripple artifact

A deformation `A = R(θ) diag(1, 1/κ) R(θ)ᵀ` with `κ ~ LogNormal` truncated to `[1, 3]` and
`θ` biased zonal (real mid-latitude fields have genuine E–W excess coherence).

This should *replace*, not stack on, the ARD/periodic anisotropy already in the bank. The
lattice-locked ripple in the model's predicted correlogram was proven to be
separable-kernel aliasing at exactly the pixel period, so for `lattice2d` episodes set
`ard: false` on the spatial columns and exclude `periodic`/`cosine`. Continuous rotation
of the deformation angle gives orientation structure without a lattice-locked period.

### 2.4 Non-stationarity — fixes D3

Two mechanisms, both standard geostatistics, both cheap:

1. **Covariate-modulated parameters.** `log σ(x)` and `log L(x)` affine in the synthetic
   static covariates plus a smooth random field. This is what makes the covariate columns
   informative.
2. **Interface.** With probability ~0.3, a level set of a smooth random field partitions
   the box into two regimes with different `(σ, L)` and partial decorrelation across the
   boundary — a synthetic coastline.

Build the non-stationary covariance as `diag(σ) · K_deformed · diag(σ)` (PSD by
construction), or by the Paciorek–Schervish closed form if range variation needs to be
stronger than a deformation can express.

### 2.5 Marginal warp — fixes D4, and is free

`y = g(f)` with `g` monotone (sinh–arcsinh with sampled skew/tail parameters, or a random
monotone spline). **The Gaussian copula of `y` is exactly that of `f`**, so `R_star` /
`R_prior` and the whole analytic oracle are untouched. It also gives the frozen-TabICL
marginal branch the non-Gaussian, non-standard marginals it will meet in deployment,
which the current Gaussian-plus-mean-function prior never does.

Honest caveat: this fixes *marginal* non-Gaussianity, not *joint*. Increment kurtosis
comes largely from fronts, a joint effect; a monotone warp raises it but not to the
observed level. Recommendation: do the warp, measure how much of the gap it closes, and
document the residual as a ceiling rather than chasing it with a non-Gaussian field that
would invalidate the analytic target.

### 2.6 Nugget / representativeness noise

ERA5 episodes *decimate* a 0.25° field, aliasing sub-grid variance into the sample —
visible as `spec_nyquist_excess`. Calibrate the nugget to that measured excess instead of
keeping `nugget_lognormal_loc: -4.63` for this family.

---

## 3. Integration into the existing prior

Deliberately additive: with the new flags off, every byte of the current path is unchanged.

1. **`conf/data/gp_tasks.yaml`** — a `geometry:` block (`lattice2d_frac`, grid/jitter/
   covariate knobs) and an `era5_like:` block (two-scale, anisotropy, non-stationarity,
   warp, nugget).
2. **`src/data_gen.py::_generate_gp_batch_raw`** — factor the single line
   `x_raw = torch.randn(B, T, d)` into `_sample_design(cfg, B, T, d)` returning coordinates
   plus optional grid metadata, and add the lattice branch. Everything downstream
   (normalization, PIT, oracle, packing) is untouched — **except** that
   `apply_structural_feature_warp` and `apply_mlp_feature_mixing` must be gated off (or
   restricted to the covariate columns) for lattice episodes, since they would destroy the
   geometry we just built. That gating is the one real code risk in the change. The same
   applies to `apply_kernel_hidden_warp` (`data_gen.py:3364`), which exists specifically to
   stop the model solving `R_star` from its own input — for this family, the geometry→
   correlation map *is* the thing we want learned, so it must stay off
   (`kernel_hidden_enabled: false`, which is already the default).
3. **Kernel side** — a `_build_era5_like_kernel` returning a dense `B×T×T` Gram; the
   `_DenseComposedKernel` / `_evaluate_kernel_dense` path already supports this. Register
   it as one more chain family so `systematic_composition` and `adaptive_kernel_sampling`
   can weight it like any other.
4. **`aux_mae` and `oracle_mode`.** No change is *required*: the copula loss already scores
   against the conditional correlation via `z_test`'s posterior PIT, and §2.2 fixes the
   real problem (prior ≈ posterior in these episodes) on the data side rather than the
   target side. But if the `aux_mae` head is ever turned on for this family it must not be
   trained against `R_star`, which under `oracle_mode: prior` is the *unconditional* `K_ss`.
   Note that `oracle_mode: posterior` is **not** a config flip — it was removed from
   `data_gen.py` because the float64 Schur complement cast back to float32 left `R_star`
   marginally non-PSD for composite kernels and the discard mask never checked for it. If
   we want it back, it needs reimplementing *with* an eigenvalue guard in the discard path.
5. **Mixing granularity** — per `generate_gp_batch` call, which already fixes
   kernel/P/N/d for its whole batch, so `collate_fn` needs no change. Lattice episodes have
   `d = 2 + q`, so a call is all-lattice or all-cloud; that is exactly how
   `_sample_d_features` already behaves.

---

## 4. The indicator suite

Every indicator is a function of a `FieldBundle` alone — `(coords, fields, grid_shape)` —
so real and synthetic go through identical code and no indicator can special-case a source.
A bundle is *R realizations of a field on a shared D-point design*: one realization is one
ERA5 day, or one GP draw. That correspondence is the right one, because the model's prior
is over episodes.

### Tier 0 — design geometry (before any field value is read)

| indicator | what it diagnoses |
|---|---|
| `nn_cv` | regularity of the design: 0 for a lattice, ~0.52 for Poisson, higher for a Gaussian cloud |
| `spread_over_nn` | median pair distance in sample spacings: `~√D` for a lattice, far smaller under concentration of measure |
| `maxdist_over_nn` | the same at the 99th percentile — the dynamic range of distances the kernel is asked to span |

### Tier 1 — prior second-order structure

| indicator | what it diagnoses |
|---|---|
| `rho_at_1nn` | correlation between adjacent samples — the "how oversampled is the field" number |
| `range_over_nn`, `range_e_over_nn` | correlation length in sample spacings (0.5 and 1/e crossings) |
| `decorrelates_in_box`, `rho_far` | whether the field decorrelates at all inside the window, and the far-field level if not |
| `matern_L_over_nn`, `matern_nu`, `matern_r2` | is the decay law even in the same family, and how well does a stationary Matérn describe it |
| `det_*` (same, plane-detrended) | the same after the dominant large-scale mode is removed — the part conditioning leaves behind |
| `od_mu`, `frac_pairs_gt_03` | overall dependence strength |
| `aniso_diff`, `orient_spread` | orientation dependence at matched distance: real zonal excess vs. lattice-locked ARD artifact |
| `nonstat_var_cv`, `nonstat_range_cv` | do different parts of *one* episode have different variance / range — stationarity, directly |

### Tier 2 — the posterior, i.e. what the head must emit

| indicator | what it diagnoses |
|---|---|
| `post05_evr32` / `evr64` / `evr128` | **the headline.** Share of the conditional correlation a rank-32/64/128 covnorm head can represent at best; `1 − evr32` is the white-noise floor it is forced to leave behind |
| `post05_eff_rank_frac` | entropy-based effective rank as a fraction of D — spectrum shape, not just one cut |
| `post05_var_ratio` | posterior/prior variance: how much the context actually removes |
| `post05_rel_change` | `‖R_post − R_prior‖_F / ‖R_prior‖_F` — is conditioning a non-trivial operation at all |
| `post05_range_over_nn` | conditional correlation length: screening / locality after conditioning |
| `post20_*` | the same at a denser context, to check the *trend* not one operating point |
| `prior_det_evr32`, `lw_shrinkage` | prior spectral concentration and how estimable it is |

### Tier 3 — marginal shape and Gaussian-copula adequacy

| indicator | what it diagnoses |
|---|---|
| `spatial_skew_abs`, `spatial_exkurt`, `spatial_gauss_ks` | per-episode spatial marginal — what TabICL is asked to fit |
| `increment_exkurt` | front/intermittency signature; identically 0 for any GP, so it measures how far the field is from *being* a GP |
| `chi_095`, `chi_095_excess` | upper-tail dependence versus the Gaussian-copula prediction at the same correlation — how misspecified the copula family itself is |
| `spec_slope` | large-to-small-scale variance balance; can be wrong while the correlogram looks right |
| `spec_nyquist_excess` | decimation aliasing (real) and lattice-locked ripple (artifact) |
| `morans_i` | single-snapshot smoothness, one number, cheap |

### Tier 4 — aggregate

- **C2ST.** A gradient-boosted classifier, 5-fold cross-validated, on the per-bundle
  indicator vector: can anything tell a real bundle from a synthetic one? AUC 0.5 means
  indistinguishable on everything measured. Its permutation importances are the ranked
  to-do list — this is the indicator that says *what to fix next*, and it is the one worth
  re-running after every phase.
- **Correlogram MMD / per-bin Wasserstein** between the binned `ρ(r)` curve vectors, for a
  distributional (not median-only) comparison of curve shape.

---

## 5. Validation protocol and gates

Indicator gates, cheap, run per phase:

- **Gate A (geometry).** `nn_cv`, `spread_over_nn` medians inside ERA5's IQR.
- **Gate B (rank).** `post05_evr32` median inside ERA5's IQR — this is the gate the whole
  exercise exists for.
- **Gate C (structure).** `det_range_over_nn`, `nonstat_var_cv`, `aniso_diff` medians inside
  ERA5's IQR; `det_matern_r2` comparable.
- **Gate D (indistinguishability).** C2ST AUC ≤ 0.65 on Tier 0–3.

Transfer gates, expensive, run at the end of a phase group:

- **Gate E — the only one that decides anything.** Held-out real-ERA5 *total*
  (marginal + copula) joint NLL from `eval/runners/spatial_correlation_eval.py sweep --mode real`,
  against independence (`Σ = I`) and against the exact-GP-MLE floor. **A prior change that
  improves every indicator but not this number is a failed prior change.**
- **Non-regression.** `eval/runners/eval_checkpoint.py` on synthetic GP episodes (don't lose
  the general GP prior) and `eval/runners/run_benchmarks.py` on UCI Beijing PM2.5 /
  California Housing (don't lose non-spatial tabular transfer — the concrete risk of
  over-fitting a prior to 2m temperature).

**Validating the indicators themselves.** Across the ≥5 prior variants the phases produce,
report Spearman ρ between each indicator's real-vs-synthetic gap and the resulting Gate-E
NLL gap. Indicators that do not predict transfer get demoted to diagnostics. Without this
step the suite is a similarity score nobody has shown matters; with it, it becomes a cheap
proxy that can be optimised between expensive training runs.

---

## 6. Phasing

| phase | content | expected to move |
|---|---|---|
| **0** ✅ | indicator harness + baseline gap table (this document) | — |
| **1** | lattice design + two-scale Matérn | `nn_cv`, `rho_far`, `od_mu`, `prior_eff_rank_frac`, `post05_var_ratio`, `post05_evr32` — **measured below to close nearly all of them** |
| **2** | anisotropy + non-stationarity + informative static covariates | `nonstat_var_cv`, `nonstat_range_cv`, `aniso_diff`, `orient_spread` |
| **3** | tail dependence + nugget calibration; marginal warp only if Phase 1–2 leaves a marginal gap | `chi_095_excess`, `spec_nyquist_excess` |
| **4** | mix into the production prior, one full training run, Gates A–E | Gate E |

### Phase 1 is already validated, before writing it

`lattice_matern_probe` in `bundles.py` is a standalone reference implementation of §2.1 +
§2.2 + a first cut of §2.3/§2.4, deliberately *not* wired into `src/data_gen.py`. Running
it through the same suite is the cheapest possible test of the plan's central claim, and
it passes: the probe's gap to ERA5 is **under 1σ on 13 of the 16 headline indicators**,
including every Tier-2 posterior indicator — `post05_var_ratio` +0.0σ (from +6.5σ),
`post05_rel_change` −0.2σ (from −6.5σ), `post05_evr32` +0.5σ (from −2.3σ), `rho_far` −0.1σ
(from −3.6σ), `od_mu` −0.3σ (from −4.5σ), `matern_nu` −0.1σ (from +4.5σ),
`prior_eff_rank_frac` +0.2σ (from +15.1σ).

That is strong evidence that a lattice design plus a two-scale Matérn is the right shape
for Phase 1, and that most of the measured distance to ERA5 is carried by those two
ingredients rather than by the more elaborate machinery in §2.3–§2.6.

Where the probe *fails*, and what Phases 2–3 therefore have to own:

- `spatial_exkurt` +3.2σ and `spatial_skew_abs` +2.5σ — the probe's log-variance
  modulation overshoots and makes the spatial marginal far too heavy-tailed. ERA5's
  within-episode spatial marginal is mildly *platykurtic* (−0.43), which is the opposite of
  what I assumed when writing §2.5. Non-stationarity strength needs a calibrated, not a
  generous, prior.
- `chi_095_excess` −1.4σ — the probe reproduces ERA5's tail dependence no better than the
  current prior does. Expected: it is a Gaussian field.
- `nn_cv` −1.0σ — a perfectly regular lattice is *too* regular; the jitter and irregular
  subsampling in §2.1 exist for this and are not in the probe.

---

## 7. What this plan cannot fix, and must be paired with

1. **Rank.** Even a perfect prior leaves a rank-32 covnorm head unable to represent an ERA5
   posterior: `post05_evr32` = 0.619 here, 0.63 at D=600 from the independent exact-GP
   analysis. ~38% of every point's variance is forced to independent noise, which *is* the
   visible speckle. Pair this work with a rank increase (64/128 — `post05_evr64` = 0.756,
   `evr128` = 0.869) or a structured local-plus-low-rank parameterization.

   **Correction to a plausible-sounding argument I had to drop.** I expected the synthetic
   prior to be *more* representable at rank 32 than ERA5 — "the prior never produces a
   target that needs the extra rank", which would have made rank work pointless until the
   prior was fixed. The measurement says the opposite: `post05_evr32` is **0.493 for the
   current prior versus 0.619 for ERA5**. Synthetic conditional correlation is *less*
   concentrated, because it is closer to the identity (D2) and near-identity spectra are
   flat. So the two workstreams are independent: rank is a real ERA5-side ceiling worth
   lifting on its own schedule, and it is not gated on the prior work.
2. **Gaussian-copula misspecification.** `chi_095_excess` bounds how much of the real-data
   NLL is unreachable for *any* Gaussian copula, however good the prior.
3. **The `val/y_nll_copula` discrepancy.** Independent scoring of held-out ERA5 and the
   in-training validation metric disagree in sign; that must be resolved before Gate E can
   be trusted as a steering signal.
4. **Scope of "ERA5".** This suite is calibrated on 2m temperature. A prior tuned to it will
   not automatically suit precipitation (zero-inflated, far heavier tailed) or wind
   (vector-valued). If multivariate deployment targets those, the indicator profile should be
   re-measured per variable before committing the prior's hyperparameters.
