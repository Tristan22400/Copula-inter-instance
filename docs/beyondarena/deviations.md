# Deviations from the official BeyondArena protocol

Each entry gives the date, what differs from the official protocol, and why.

## Official track (H1)

**None for the model run itself.**
- `TA-TabICLv2-ERA5FT` is TabArena's `TabICLv2Model`. Only the `ag_key` and the
  regression checkpoint (`model_path`) are changed.
- It runs through `BeyondArenaContext.build_and_run_jobs`, on `core2k`, under the
  enforced validation protocol (`8x1+tiny5x5<=500+task-specific`) with
  TabArena's preprocessing and bagging.

**Environment.**
- The run uses a separate venv (tabarena `cc507b84`, tabicl 2.2.0, torch 2.15),
  on Grid5000 GPUs (A5000 / L40S / A100 ...). The cached baselines ran on other
  hardware.
- A control reruns `TA-TabICLv2` (default config) on the lite split of every
  regression dataset, to check that this environment reproduces the cached
  numbers.

## Joint track (H2 to H4): outside the official protocol by design

1. **No bagging and no inner validation.** One marginal is fitted per split, by
   in-context prediction on the training rows.
2. **Preprocessing.** We fit `TabArenaModelAgnosticPreprocessing` on the training
   rows, with semantic text features disabled because no precomputed embedding
   cache is used. Then:
   - categorical columns become ordinal codes (unseen categories are NaN);
   - NaNs are mean-imputed with training means;
   - features are z-scored, as in `normalize_features`.

   This affects the datasets with text columns.
3. **Context subsampling.**
   - Training sets above 100k rows are stored as a seeded 100k-row subsample. The
     preprocessing is fitted on the full set.
   - The marginal context is at most 50k rows.
   - The copula context is 1,000 rows (the copulas were trained with $P \le 1000$).
4. **Test subsampling.** At most 200 blocks per split are scored. Grouped blocks
   are capped at 128 rows per group.
5. **LMM baseline.** It is fitted on at most 50k training rows, on principal axes
   of the standardized features, without the group column.
6. **Ablations.** They run only on the grouped and temporal datasets below 100k
   training rows (compute).
7. **TabPFN-3.5 feature cap (2026-11-18, found at run time).** TabPFN-3.5's input schema accepts at
   most 500 columns, and four datasets have more after preprocessing:
   - early_learning_predictors (741 columns);
   - santander_transaction_value (540);
   - qsar_tid_11 (1024, a development dataset);
   - maps_router_eta_1m (995).

   For `tabpfn-ft` and `tabpfn-pre` only, these datasets use a seeded random subset of 500 columns
   that always includes the group column (`max_features=500`). TabArena's own TabPFN wrapper
   subsamples features in a similar way. The subset feeds the marginal, the copula and every
   comparator, so the copula-vs-indep comparison stays paired.
8. **Group column in the stopped first design (found 2026-11-18).** TabArena's official
   preprocessing drops the group column (`GroupAggregationFeatureGenerator`), so no official
   model sees a group ID. The first design's exporter did not pass the group columns to the
   preprocessor, so its models did see it as a feature. The framework-native joint track
   (amendment 1) uses the official pipeline, so the group column is dropped. The group labels
   reach the runner only, which takes them from the task. It uses them to score the predictions,
   to form the exchangeable baseline's blocks and to fit the mixed model. The copula model never
   receives them: its cross-fitting folds are random.
9. **Mixed-model baseline fix (2026-11-19, found in the first full run).**
   - **Symptom:** on 12 of the 13 splits of early_learning_predictors and
     video_transcoding_time_prediction, the joint NLL was about $10^{12}$.
   - **Cause:** the predictive mean used REML's GLS fixed effects. Fitted on many near-collinear
     principal axes (721 on early_learning), they diverged on the test rows: median shift from the
     OLS mean of 8.6e3 and 1.0e6 predictive sd, recorded as `reml_mean_shift_sd`.
   - **Secondary problem:** the OLS point predictions also extrapolated, with an early_learning
     RMSE of 150–870 against about 13 for the foundation models.
   - **Fixes, applied before any rerun was seen:**
     - the predictive mean is the OLS one, i.e. the point prediction TabArena scores; the test
       groups are new, so the random intercept adds variance, not a shift. REML supplies only the
       variance components $s_u^2, s_e^2$;
     - a REML fit is rejected, and the ANOVA moment estimate used instead, when
       $s_u^2+s_e^2 \notin [0.1, 10]\,\hat v_{\text{OLS}}$ or $s_e^2 < 10^{-3}\,\hat v_{\text{OLS}}$,
       where $\hat v_{\text{OLS}}$ is the variance of the OLS residuals;
     - only principal axes with singular value $\ge 10^{-3}$ times the largest are kept (previously
       $10^{-6}$);
     - test rows are clipped to the training range on each axis.
   - **Rerun:** run `lmm_v2` on all 7 grouped datasets (93 splits) replaces the first run in
     `main`. The old results are kept in `main_lmm_v1`.
10. **Failure handling and resources for the large tables (2026-11-19).**
    - The first large-table jobs ran with TabArena's `raise_on_failure=True`, so a single failure
      ended the job:
      - TabPFN and Kumo on mercari hit `ag.max_time_limit = 14400 s`, the official per-fit limit;
      - Kumo-pre ran out of GPU memory on a 48 GB GPU;
      - TabICL was killed at the 252 GiB memory limit of a 1-GPU job on maps_router (1M rows,
        988 columns).
    - The relaunch uses `raise_on_failure=False`, TabArena's benchmarking behaviour: a failed split
      is not cached, and the compare step imputes it as it does for any method.
    - The time limit is unchanged.
    - Kumo runs on 96 GB GPUs, and TabICL's maps_router on a whole node (503 GB).
