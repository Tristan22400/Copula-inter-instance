# Glossary

Terms used across the code, configs and logs, in alphabetical order.

**Analytic / oracle.** When an episode is a pure GP draw, the correct marginal
posterior and the correct correlation are known in closed form from the saved
kernel. `data.z_train_source=analytic` (generation) and
`marginal.z_train_source=oracle` (eval) use this exact-GP PIT instead of a
learned marginal. It is an idealized upper bound and doesn't exist for real data.

**Autoregressive row.** An eval baseline that uses the marginal model with no
copula: it reveals the query points one at a time, each conditioned on the
already revealed ones. Its total NLL is an exact factorization of a joint
density, so it is the copula-free reference the copula head has to beat.

**Backend / backbone.** A *marginal backend* is a tabular foundation model used
for the PIT: `tabicl` or `tabpfn`, listed in
`backend_registry.py`. A *copula backbone* is the ICL trunk the copula head sits
on (`model.backbone`: only `tabicl`). The two choices are independent.

**Copula NLL / marginal NLL / total (Y-space) NLL.** The two terms of Sklar's
decomposition and their sum, in nats per query point. The total is a proper
scoring rule of the joint predictive density of `y_test`, so it is the number to
compare across methods. The copula NLL alone is only comparable between methods
that share the same `z_test`.

**Episode.** One task: P context (train) rows and N query (test) rows with
features `x` and target `y`. Batches pad P and N to `P_max`/`N_max`; `test_mask`
marks real query rows.

**ERA5 / ARCO-ERA5.** ECMWF's reanalysis, read from Google's public
Analysis-Ready Cloud-Optimized Zarr archive (no account needed). This project
uses 2 m temperature (`t2m`) on lon/lat grids as real spatial data.

**K-fold PIT.** How context rows get their `z_train`: they are split into K
folds, and each fold is scored with the other K−1 folds as context. This mimics
deployment, where query points are never in their own context. `DEFAULT_K_FOLDS`
is 10.

**`log_pdf_test`.** The marginal log-density of each query target, the
marginal NLL's input.

**Phase A.** Fine-tuning the standalone marginal model (`copula_inter.finetune_marginal`)
so its marginal NLL approaches the analytic GP oracle. It is separate from copula
training; the two meet only at a checkpoint path (`tabicl.pit_ckpt`).

**PIT (probability integral transform).** `z = Phi^{-1}(F(y | x, context))`.
Under a correct marginal, `z` is standard normal, and the copula models the
dependence between the `z` of different query points.

**`R_star`, `mu_star`, `sigma_star`.** The ground-truth correlation, mean and
standard deviation of an episode's query points under its generating GP
(synthetic data only). `R_prior` and `Sigma_star` are rebuilt from the saved
kernel rather than stored.

**Rank (`model.rank`).** The rank r of the low-rank factor W in
`Sigma ∝ W W^T + diag(D)`. It must match the checkpoint when resuming.

**Tier.** How much of the marginal model Phase A trains. Tier 0 covers the label
path, ICL norms and decoder; higher tiers add LoRA on more attention stages. The
default (`marginal.lora_all_layers=true`) instead puts LoRA on every 2-D weight,
which makes tiers an ablation.

**`z_train`, `z_test`.** PIT values of the context and query targets.
`z_train` is an input to the copula backbone, and `z_test` is what the copula
NLL scores.

**`z_train_source`.** Which marginal produces the PIT for generated episodes:
`analytic`, `tabicl`, `tabicl_split` (query scored against a disjoint
calibration split) or `tabpfn`.
