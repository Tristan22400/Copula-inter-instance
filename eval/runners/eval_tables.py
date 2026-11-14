"""Printed result tables for eval_checkpoint: per-method NLL, oracle Y-space rows, total NLL, and ranks."""

from __future__ import annotations

import numpy as np
import torch

from copula_inter.data.gp_kernels import _parse_composite
from eval.results import (
    NAN_PARTS as _NAN_PARTS,
)
from eval.results import (
    competition_ranks,
    numeric_summary,
    score_summary,
)

_METHOD_ORDER = [
    ("independence", "Independence"),
    ("gp_prior_rbf", "GP-Prior-RBF"),
    ("gp_mle_rbf", "GP-MLE-RBF"),
    ("gp_mle_ard_rbf", "GP-MLE-ARD-RBF"),
    ("gp_mle_matern32", "GP-MLE-Matern32"),
    ("gp_mle_ard_matern32", "GP-MLE-ARD-Matern32"),
    ("gp_mle_periodic", "GP-MLE-Periodic"),
    ("gp_mle_ard_periodic", "GP-MLE-ARD-Periodic"),
    ("gp_mle_rq", "GP-MLE-RQ"),
    ("gp_mle_ard_rq", "GP-MLE-ARD-RQ"),
    ("gp_mle_dot_product", "GP-MLE-DotProduct"),
    ("gp_mle_polynomial", "GP-MLE-Polynomial"),
    ("gp_evidence", "GP-MLE (BIC-selected, context only)"),
    ("gp_laplace", "GP-MLE (Laplace-evidence-selected, context only)"),
    ("gp_zeromean_rbf", "Marginal + Zero Mean GP (RBF)"),
    ("gp_zeromean_matern32", "Marginal + Zero Mean GP (Matern32)"),
    ("dkl_rbf", "Deep Kernel Learning (RBF)"),
    ("dkl_matern32", "Deep Kernel Learning (Matern32)"),
    ("dkl_rq", "Deep Kernel Learning (RQ)"),
    ("dkl_dot_product", "Deep Kernel Learning (DotProduct)"),
    ("per_ep_transformer", "PerEp-Transformer"),
    ("best_baseline", "Best-of-Baselines (nested CV on query labels)"),
    ("icl", "ICL (pretrained)"),
    ("oracle", "Oracle (posterior corr., known theta)"),
]


# oracle_prior (the generator's unconditioned R_star) appears only in the correlation-grid plot.
_METHOD_LABELS = dict(_METHOD_ORDER) | {"oracle_prior": "Oracle prior corr. (unconditioned)"}


def _kernel_composition_label(ep: dict) -> str:
    """Kernel-composition string for one episode (e.g. "rbf(ARD)+periodic, mlp-mixing") from its kernel metadata."""
    real_meta = ep.get("real_meta")
    if real_meta is not None:
        from eval.data.real import dataset_module

        return str(dataset_module(real_meta["dataset"]).label(real_meta))
    meta = ep.get("era5_meta")
    if meta is not None:
        # Real ERA5: report the episode's geography instead of a kernel.
        lat = meta.get("lat_bounds")
        lon = meta.get("lon_bounds")
        where = (
            f"lat[{lat[0]:.1f},{lat[1]:.1f}] lon[{lon[0]:.1f},{lon[1]:.1f}]"
            if lat is not None and lon is not None
            else "region n/a"
        )
        return f"ERA5 {where} grid={meta['grid_size']} P={meta['P']} N={meta['N']}"
    if "kernel" not in ep:
        return "unavailable (set dataset_dir to pre-generated metadata, or use live_generate=true)"

    if "kernel_components" in ep:
        parts = [ep["kernel_components"][0]]
        for op, comp_name in zip(ep["kernel_ops"], ep["kernel_components"][1:]):
            parts.append(op)
            parts.append(comp_name)
        label = " ".join(parts)
        ard_tags = [
            comp_name + "(ARD)"
            for comp_name, comp_params in zip(ep["kernel_components"], ep["kernel_component_params"])
            if torch.is_tensor(comp_params.get("l")) and comp_params["l"].numel() > 1
        ]
    else:
        label = ep["kernel"]
        composite = _parse_composite(ep["kernel"])
        ard_tags = []
        if composite is None:
            if torch.is_tensor(ep.get("l")) and ep["l"].numel() > 1:
                ard_tags.append(f"{ep['kernel']}(ARD)")
        else:
            name_a, _op, name_b = composite
            if torch.is_tensor(ep.get("l")) and ep["l"].numel() > 1:
                ard_tags.append(f"{name_a}(ARD)")
            if torch.is_tensor(ep.get("l_b")) and ep["l_b"].numel() > 1:
                ard_tags.append(f"{name_b}(ARD)")

    if ard_tags:
        label = f"{label}  [{', '.join(ard_tags)}]"
    if bool(ep.get("mlp_mixed", False)):
        label = f"{label}, mlp-mixing"
    return label


def _print_table(
    all_nlls: list[dict[str, float]],
    z_train_source: str = "tabicl",
    era5: bool = False,
    attempted: int | None = None,
    real_label: str | None = None,
) -> None:
    summaries = {k: score_summary(all_nlls, k) for k, _ in _METHOD_ORDER}
    attempted = len(all_nlls) if attempted is None else attempted

    col = max(22, max(len(label) for _, label in _METHOD_ORDER) + 2)
    total = col + 3 * 12
    print(f"\n{'─' * total}")
    print(f"Inter-instance copula NLL (z-space) — lower is better  [N={len(all_nlls)} episodes]")
    if era5:
        print(
            "Episodes: REAL ARCO-ERA5 2m-temperature. The shared z_test every row "
            "is scored against is the frozen-{s} K-fold PIT, NOT a ground-truth "
            "marginal (none exists on real data) — rows still differ only in their "
            "correlation matrix R, so the ranking is valid, but the Oracle row is "
            "structurally unavailable and prints nan.".format(s=z_train_source)
        )
    if real_label is not None:
        print(
            f"Episodes: REAL data, {real_label}. The shared z_test is the frozen-{z_train_source} "
            "K-fold PIT, not a ground-truth marginal, so this table ranks correlation matrices "
            "against an estimated marginal and must NOT be read across methods as the result; "
            "read the total Y-space NLL table. The Oracle row prints nan."
        )
    print(
        f"ICL z_train source: {z_train_source}"
        + ("  (exact GP-LOO PIT)" if z_train_source == "oracle" else f"  ({z_train_source} K-fold PIT estimate)")
    )
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean NLL':>12}{'Std NLL':>12}{'Valid/All':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}{'─' * 12}")
    for key, label in _METHOD_ORDER:
        m, s, n_valid = summaries[key]
        marker = ""
        if key == "best_baseline":
            marker = "  ← selected with the query labels (nested CV): a reference, not a method"
        elif key == "icl":
            marker = "  ← our model"
        elif key == "oracle":
            marker = "  ← copula of the known-theta GP posterior: the target under exact marginals"
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{f'{n_valid}/{attempted}':>12}{marker}")
    print(f"{'─' * total}\n")


def _print_y_space_oracle(y_space_nlls: list[dict[str, dict[str, float]]]) -> None:
    """Print the analytic GP prior vs posterior Y-space NLL table (NaN episodes excluded)."""
    prior_vals = [d["prior"]["total"] for d in y_space_nlls]
    post_vals = [d["posterior"]["total"] for d in y_space_nlls]
    prior_mean, prior_std, _ = numeric_summary(prior_vals)
    post_mean, post_std, n_valid = numeric_summary(post_vals)
    print(
        f"GP oracle total NLL (Y-space, marginal+copula) — lower is better, "
        f"posterior <= prior is a Bayes-optimality guarantee here "
        f"[valid for {n_valid}/{len(y_space_nlls)} episodes]"
    )
    print(f"  prior (unconditioned):      mean={prior_mean:.4f}  std={prior_std:.4f}")
    print(f"  posterior (Schur-conditioned): mean={post_mean:.4f}  std={post_std:.4f}\n")


# Fitted competitors (icl included) for the total table and both rank tables. gp_evidence and
# gp_laplace are left out: each copies one gp_mle_* row per episode, so ranking it would tie.
_SELECTION_COPIES = ("gp_evidence", "gp_laplace")
_RANK_KEYS = [
    (k, label)
    for k, label in _METHOD_ORDER
    if k not in _SELECTION_COPIES and k not in ("independence", "gp_prior_rbf", "best_baseline", "oracle")
]

# Context-only GP selections and the plug-in competitor of the copula head (total table only).
_GP_SELECTION_ROWS = [
    ("gp_evidence", "GP-MLE (BIC-selected, context only)"),
    ("gp_evidence_mix", "GP-MLE (BIC-weighted mixture)"),
    ("marg_gp_evidence", "Marginal + GP-MLE posterior corr. (BIC)"),
    ("gp_laplace", "GP-MLE (Laplace-evidence-selected, context only)"),
    ("gp_laplace_mix", "GP-MLE (Laplace BMA over kernels)"),
    ("marg_gp_laplace", "Marginal + GP-MLE posterior corr. (Laplace)"),
]


# Ordinary GP baselines, for the post-hoc best-GP competitor.
_GP_TOTAL_KEYS = tuple(k for k, _ in _RANK_KEYS if k.startswith("gp_mle_") or k.startswith("gp_zeromean_"))


# Total-NLL rank: adds the context-only GP selections (not the copies of a ranked row),
# independence, the autoregressive row and best_gp_total; no oracles.
_TOTAL_RANK_ORDER = (
    _RANK_KEYS
    + [row for row in _GP_SELECTION_ROWS if row[0] not in _SELECTION_COPIES]
    + [
        ("independence_marginal", "Independence (marginal only)"),
        ("autoregressive", "Autoregressive marginal (chain rule)"),
        ("best_gp_total", "Best GP (oracle pick on y_test)"),
    ]
)


# Total-table row order: competitors, independence, autoregressive, then the
# oracle references. autoregressive has no R, so it is only in Y-space tables.
_TOTAL_NLL_ORDER = (
    _RANK_KEYS
    + _GP_SELECTION_ROWS
    + [
        ("independence_marginal", "Independence (marginal only)"),
        ("autoregressive", "Autoregressive marginal (chain rule)"),
        ("marg_oracle_corr", "Marginal + oracle posterior corr. (known theta)"),
        ("marg_attainable_lowrank", "Marginal + attainable copula R-dagger (head rank, known theta)"),
        ("marg_attainable", "Marginal + attainable copula R-dagger (full rank, known theta)"),
        ("marg_bayes_attainable", "Marginal + Bayes attainable copula (theta marginalised)"),
        ("oracle_bayes", "Bayes oracle (theta marginalised, family known)"),
        ("oracle_prior", "Oracle (prior, unconditioned)"),
        ("oracle_posterior", "Oracle (posterior, Schur-conditioned)"),
    ]
)


# Copula-efficiency rows after the attainable ones: (key in meta["copula_efficiency"], label).
_EFFICIENCY_ROWS = (
    ("icl", "ICL copula head"),
    ("oracle_posterior", "R_po (known-theta posterior corr.)"),
    ("gp_laplace", "GP-MLE posterior corr. (Laplace pick)"),
    ("gp_evidence", "GP-MLE posterior corr. (BIC pick)"),
)


def _efficiency_rows(effs: list[dict]) -> list[tuple[str, str]]:
    """Attainable rows (any-copula bound, full rank, each tracked rank r), then _EFFICIENCY_ROWS."""
    head = {int(e["head_rank"]) for e in effs if "head_rank" in e}
    ranks = sorted({int(k.removeprefix("attainable_r")) for e in effs for k in e if k.startswith("attainable_r")})
    rows = [
        ("any_copula_bound", "Any copula: lower bound (= if calibrated)"),
        ("attainable", "Best Gaussian copula R-dagger (full rank)"),
    ]
    rows += [(f"attainable_r{r}", f"Best Gaussian copula, rank {r}" + (" (head)" if r in head else "")) for r in ranks]
    rows.append(("bayes_attainable", "Bayes attainable (theta marginalised)"))
    return rows + list(_EFFICIENCY_ROWS)


def _print_copula_efficiency(all_episode_meta: list[dict]) -> None:
    """Expected copula NLL l(R) under the frozen marginals, and the share of the attainable gain each R captures.

    l(R) = E[-log c_R(z)] / N with E[z z'] = M from the known-theta posterior (0 = independence).
    efficiency = mean l(R) / mean l(R_dagger) over the episodes where both exist (1 = the best
    Gaussian copula, 0 = no better than independence, < 0 = worse). The best copula of any family
    lies between the bound and R_dagger. Synthetic episodes with a TabICL marginal only.
    """
    effs = [m["copula_efficiency"] for m in all_episode_meta if m.get("copula_efficiency")]
    if not effs:
        return
    print(
        f"\nCopula efficiency under the frozen marginals ({len(effs)} episodes; l in nats per point, 0 = independence)"
    )
    print(f"  mean diag(M) = {np.mean([e['m_diag'] for e in effs]):.4f}  (1 for calibrated marginals)")
    print(f"  {'copula':<46}{'n':>5}{'mean l':>10}{'l - l(R-dagger)':>17}{'efficiency':>12}")
    for key, label in _efficiency_rows(effs):
        pairs = [(e[key], e["attainable"]) for e in effs if key in e and np.isfinite(e[key])]
        if not pairs:
            continue
        l_r, l_att = np.array(pairs).T
        eff = l_r.mean() / l_att.mean() if l_att.mean() != 0 else float("nan")
        print(f"  {label:<46}{len(pairs):>5}{l_r.mean():>+10.4f}{(l_r - l_att).mean():>+17.4f}{eff:>12.3f}")
    bayes = [e for e in effs if "bayes_attainable" in e and "icl" in e and np.isfinite(e["icl"])]
    if bayes:
        l_icl = np.mean([e["icl"] for e in bayes])
        l_bayes = np.mean([e["bayes_attainable"] for e in bayes])
        ess = np.array([e["bayes_ess"] for e in bayes])
        print(
            f"  ICL / Bayes attainable = {l_icl / l_bayes:.3f} over {len(bayes)} episodes"
            f"  (Bayes on its own M: {np.mean([e['bayes_attainable_self'] for e in bayes]):+.4f};"
            f" importance ESS median {np.median(ess):.0f}, < 10 in {np.mean(ess < 10):.0%})"
        )
    print()


def _ar_note(
    all_total_nlls: list[dict[str, dict[str, float]]], order: str, conditioning: str, max_context: int | None
) -> str | None:
    """Footnote for the autoregressive row (warns when conditioning="sample"), or None when no episode has one."""
    n_valid = sum(
        1 for m in all_total_nlls if not np.isnan(m.get("autoregressive", _NAN_PARTS).get("total", float("nan")))
    )
    if n_valid == 0:
        return None
    cap = "uncapped" if max_context is None else f"max_context={max_context}"
    note = (
        f"Autoregressive row: chain-rule joint density from the SAME marginal, "
        f"revealed in {order} order, {cap}, over {n_valid}/{len(all_total_nlls)} "
        f"episodes. Its Marginal column is the one-shot (independence) marginal, "
        f"so its Copula column is exactly what the sequencing bought."
    )
    if conditioning != "teacher_forcing":
        note += (
            "\n  *** WARNING: autoregressive.conditioning=sample. Each step appended a DRAW, "
            "not the true y, so this row is NOT a joint density of y_test and is "
            "NOT comparable to the other rows. Ancestral-sampling diagnostic only. ***"
        )
    return note


def _print_total_nll_table(
    all_total_nlls: list[dict[str, dict[str, float]]],
    z_train_source: str,
    era5: bool = False,
    ar_note: str | None = None,
    attempted: int | None = None,
    real_label: str | None = None,
) -> None:
    """Print the per-point total Y-space NLL table: each method's own marginal plus copula.

    Each entry is {total, marginal, copula}. The icl row is NaN with
    z_train_source="oracle". Oracle rows are divided by N here only.
    """
    attempted = len(all_total_nlls) if attempted is None else attempted

    def _col(part: str) -> dict[str, float]:
        return {
            k: numeric_summary([m.get(k, _NAN_PARTS).get(part, float("nan")) for m in all_total_nlls])[0]
            for k, _ in _TOTAL_NLL_ORDER
        }

    total_rows = [
        {k: m.get(k, _NAN_PARTS).get("total", float("nan")) for k, _ in _TOTAL_NLL_ORDER} for m in all_total_nlls
    ]
    total_summaries = {k: score_summary(total_rows, k) for k, _ in _TOTAL_NLL_ORDER}
    means_marginal = _col("marginal")
    means_copula = _col("copula")

    col = max(22, max(len(label) for _, label in _TOTAL_NLL_ORDER) + 2)
    total = col + 5 * 12
    print(f"\n{'─' * total}")
    print(
        f"Total NLL (Y-space, marginal+copula, own marginal per method) — "
        f"lower is better  [N={len(all_total_nlls)} episodes]"
    )
    if era5:
        print(
            "Episodes: REAL ARCO-ERA5 2m-temperature (Kelvin). THIS is the table to "
            "read on real data: every method supplies its own full predictive "
            "density and is scored at the same real y_test, which is a proper "
            "scoring rule regardless of whose marginal is whose. The two Oracle "
            "rows are nan by construction (no generating kernel behind ERA5)."
        )
    if real_label is not None:
        print(
            f"Episodes: REAL data, {real_label}, in raw target units. THIS is the table to read: "
            "every method supplies its own full predictive density and is scored at the same "
            "real y_test (a proper scoring rule). The two Oracle rows are nan by construction."
        )
    print(
        f"ICL z_train source: {z_train_source}"
        + (
            "  (icl row n/a — oracle mode has no learned ICL marginal to score)"
            if z_train_source == "oracle"
            else f"  ({z_train_source} K-fold PIT estimate)"
        )
    )
    if ar_note:
        print(f"  {ar_note}")
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean Total':>12}{'Std Total':>12}{'Mean Marg.':>12}{'Mean Cop.':>12}{'Valid/All':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}{'─' * 12}{'─' * 12}{'─' * 12}")
    for key, label in _TOTAL_NLL_ORDER:
        m, s, n_valid = total_summaries[key]
        mm, mc = means_marginal.get(key, float("nan")), means_copula.get(key, float("nan"))
        marker = "  ← our model" if key == "icl" else ""
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{mm:>12.4f}{mc:>12.4f}{f'{n_valid}/{attempted}':>12}{marker}")
    paired = [
        row
        for row in total_rows
        if np.isfinite(row.get("icl", float("nan"))) and np.isfinite(row.get("independence_marginal", float("nan")))
    ]
    if paired:
        icl_mean, _, _ = score_summary(paired, "icl")
        independent_mean, _, _ = score_summary(paired, "independence_marginal")
        print(
            f"Paired ICL/independence on {len(paired)}/{len(total_rows)} episodes: "
            f"ICL={icl_mean:.4f}, independence={independent_mean:.4f}"
        )
    print(f"{'─' * total}\n")


def _print_rank_table(
    values: list[dict[str, float]],
    order: list[tuple[str, str]],
    title: str,
    z_train_source: str,
) -> None:
    """Print mean and median per-episode rank of each method in order, sorted by mean rank."""
    keys = [k for k, _ in order]
    labels = dict(order)
    ranks = competition_ranks(values, keys)
    rows = [(k, float(np.mean(rs)), float(np.median(rs)), len(rs)) for k, rs in ranks.items() if rs]
    rows.sort(key=lambda r: r[1])
    if not rows:
        return

    col = max(22, max(len(labels[k]) for k, *_ in rows) + 2)
    total = col + 3 * 12
    print(f"\n{'─' * total}")
    print(f"{title}  [1 = best of {len(keys)} candidates that episode; lower is better]")
    print(f"ICL z_train source: {z_train_source}")
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean Rank':>12}{'Median Rank':>12}{'N':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}{'─' * 12}")
    for k, mean_r, med_r, n in rows:
        marker = "  ← our model" if k == "icl" else ""
        print(f"{labels[k]:<{col}}{mean_r:>12.2f}{med_r:>12.1f}{n:>12d}{marker}")
    print(f"{'─' * total}\n")
