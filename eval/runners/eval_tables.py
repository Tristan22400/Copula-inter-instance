"""Printed result tables for eval_checkpoint: per-method NLL, oracle Y-space rows, total NLL, and ranks."""

from __future__ import annotations


import numpy as np
import torch


from copula_inter.data_gen import _parse_composite  # noqa: E402
from eval.results import (
    NAN_PARTS as _NAN_PARTS,
)
from eval.results import (  # noqa: E402
    competition_ranks,
    numeric_summary,
    score_summary,
)

_METHOD_ORDER = [
    ("independence",        "Independence"),
    ("gp_prior_rbf",        "GP-Prior-RBF"),
    ("gp_mle_rbf",          "GP-MLE-RBF"),
    ("gp_mle_ard_rbf",      "GP-MLE-ARD-RBF"),
    ("gp_mle_matern32",     "GP-MLE-Matern32"),
    ("gp_mle_ard_matern32", "GP-MLE-ARD-Matern32"),
    ("gp_mle_periodic",     "GP-MLE-Periodic"),
    ("gp_mle_ard_periodic", "GP-MLE-ARD-Periodic"),
    ("gp_mle_rq",           "GP-MLE-RQ"),
    ("gp_mle_ard_rq",       "GP-MLE-ARD-RQ"),
    ("gp_mle_dot_product",  "GP-MLE-DotProduct"),
    ("gp_mle_polynomial",   "GP-MLE-Polynomial"),
    ("gp_zeromean_rbf",     "Marginal + Zero Mean GP (RBF)"),
    ("gp_zeromean_matern32", "Marginal + Zero Mean GP (Matern32)"),
    ("dkl_rbf",             "Deep Kernel Learning (RBF)"),
    ("dkl_matern32",        "Deep Kernel Learning (Matern32)"),
    ("dkl_rq",              "Deep Kernel Learning (RQ)"),
    ("dkl_dot_product",     "Deep Kernel Learning (DotProduct)"),
    ("per_ep_transformer",  "PerEp-Transformer"),
    ("best_baseline",       "Best-of-Baselines (per-episode)"),
    ("icl",                 "ICL (pretrained)"),
    ("oracle",              "Oracle (prior)"),
]


# oracle_posterior only ever appears in R_dict (the correlation-grid plot),
# never in `nlls`/_METHOD_ORDER's z-space table — see
# _eval_icl_episode's docstring for why its NLL isn't comparable in that
# table's units. Kept here only so the plot panel gets a readable title.
_METHOD_LABELS = dict(_METHOD_ORDER) | {"oracle_posterior": "Oracle (posterior)"}


def _kernel_composition_label(ep: dict) -> str:
    """Human-readable kernel-composition string for one episode (e.g.
    "rbf(ARD)+periodic, mlp-mixing"), built from the return_kernel_metadata
    fields generate_gp_batch attaches — absent entirely for episodes loaded
    from a pre-built dataset that didn't request that metadata (the common
    case for existing PIT datasets on disk today)."""
    meta = ep.get("era5_meta")
    if meta is not None:
        # Real ERA5: there is no kernel, so this column reports the episode's
        # geography instead (the closest analogue of "what generated it").
        lat = meta.get("lat_bounds")
        lon = meta.get("lon_bounds")
        where = (
            f"lat[{lat[0]:.1f},{lat[1]:.1f}] lon[{lon[0]:.1f},{lon[1]:.1f}]"
            if lat is not None and lon is not None else "region n/a"
        )
        return f"ERA5 {where} grid={meta['grid_size']} P={meta['P']} N={meta['N']}"
    if "kernel" not in ep:
        return "unavailable (pass --dataset_dir with pre-generated metadata, or use --live_generate)"

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


def _print_table(all_nlls: list[dict[str, float]], z_train_source: str = "tabicl",
                 era5: bool = False, attempted: int | None = None) -> None:
    summaries = {k: score_summary(all_nlls, k) for k, _ in _METHOD_ORDER}
    attempted = len(all_nlls) if attempted is None else attempted

    col = max(22, max(len(label) for _, label in _METHOD_ORDER) + 2)
    total = col + 3 * 12
    print(f"\n{'─' * total}")
    print(f"Inter-instance copula NLL (z-space) — lower is better  [N={len(all_nlls)} episodes]")
    if era5:
        print("Episodes: REAL ARCO-ERA5 2m-temperature. The shared z_test every row "
              "is scored against is the frozen-{s} K-fold PIT, NOT a ground-truth "
              "marginal (none exists on real data) — rows still differ only in their "
              "correlation matrix R, so the ranking is valid, but 'Oracle (prior)' is "
              "structurally unavailable and prints nan.".format(s=z_train_source))
    print(f"ICL z_train source: {z_train_source}"
          + ("  (exact GP-LOO PIT)" if z_train_source == "oracle"
             else f"  ({z_train_source} K-fold PIT estimate)"))
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean NLL':>12}{'Std NLL':>12}{'Valid/All':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}{'─' * 12}")
    for key, label in _METHOD_ORDER:
        m, s, n_valid = summaries[key]
        marker = ""
        if key == "best_baseline":
            marker = "  ← per-episode best baseline (nested CV)"
        elif key == "icl":
            marker = "  ← our model"
        elif key == "oracle":
            marker = "  ← unconditional kernel corr. among test pts (NOT Bayes-optimal; see GP oracle Y-space NLL below)"
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{f'{n_valid}/{attempted}':>12}{marker}")
    print(f"{'─' * total}\n")


def _print_y_space_oracle(y_space_nlls: list[dict[str, dict[str, float]]]) -> None:
    """Total (marginal + copula) Y-space multivariate-normal GP oracle NLL,
    prior vs. posterior — see gp_analytical_posterior's docstring for why
    this is a SEPARATE table from _print_table's z-space copula-only NLL
    (different units, not just a missing row there): this one directly
    answers "how much does conditioning on context help, in the units of a
    real predictive log-likelihood" and posterior <= prior is a real,
    provable guarantee here (Bayes-optimality of the posterior predictive
    under log-loss), unlike a same-z_test z-space copula comparison.

    Episodes where gp_analytical_posterior was unavailable (systematic-chain
    kernel with whole-chain outer sign modulation, or a --dataset_dir
    episode missing kernel metadata) contribute NaN to both columns and are
    excluded via nanmean/nanstd, same convention as _print_table.
    """
    prior_vals = [d["prior"]["total"] for d in y_space_nlls]
    post_vals  = [d["posterior"]["total"] for d in y_space_nlls]
    prior_mean, prior_std, _ = numeric_summary(prior_vals)
    post_mean, post_std, n_valid = numeric_summary(post_vals)
    print(f"GP oracle total NLL (Y-space, marginal+copula) — lower is better, "
          f"posterior <= prior is a Bayes-optimality guarantee here "
          f"[valid for {n_valid}/{len(y_space_nlls)} episodes]")
    print(f"  prior (unconditioned):      mean={prior_mean:.4f}  std={prior_std:.4f}")
    print(f"  posterior (Schur-conditioned): mean={post_mean:.4f}  std={post_std:.4f}\n")


# Methods with a genuine (own) fit/marginal, i.e. real competitors in a
# ranking sense — independence/gp_prior_rbf are non-fit floor references and
# best_baseline/oracle are derived after the fact, so all four are excluded
# here (same reasons as _NON_FITTED_EXCLUDED, but icl stays IN since it's the
# model under evaluation). Shared by _print_total_nll_table's row order and
# both rank tables (_print_rank_table calls in main()).
_RANK_KEYS = [
    (k, label) for k, label in _METHOD_ORDER
    if k not in ("independence", "gp_prior_rbf", "best_baseline", "oracle")
]


# These are the ordinary GP baselines, rather than DKL (which has a learned
# neural feature map).  The per-episode minimum is intentionally a post-hoc
# diagnostic competitor in the Y-space rank table, not a model-selection
# estimate: it sees that episode's test NLL when choosing its kernel.
_GP_TOTAL_KEYS = tuple(
    k for k, _ in _RANK_KEYS
    if k.startswith("gp_mle_") or k.startswith("gp_zeromean_")
)


# The total-NLL rank is a distinct competition from the displayed total table:
# it includes the marginal-only independence baseline and the chain-rule
# marginal, but excludes the two GP-oracle references.  Neither has a fair
# counterpart in the z-space correlation ranking.  ``best_gp_total`` is the
# requested per-episode, post-hoc best-GP competitor.
_TOTAL_RANK_ORDER = _RANK_KEYS + [
    ("independence_marginal", "Independence (marginal only)"),
    ("autoregressive", "Autoregressive marginal (chain rule)"),
    ("best_gp_total", "Best GP (per-episode, post hoc)"),
]


# Row order for _print_total_nll_table: fitted competitors, the marginal-only
# independence and autoregressive rows, then the two oracle references (which
# have no z-space-only counterpart in _RANK_KEYS since they're Y-space-only).
#
# "autoregressive" is appended HERE and is deliberately in neither
# _METHOD_ORDER nor _RANK_KEYS: it has no correlation matrix R, so it cannot
# appear in the z-space copula table (whose every row IS an R scored against a
# shared z_test), it is not a candidate for the best-of-baselines ranking, and
# it has nothing to contribute to the z-space rank table. Being in
# _TOTAL_NLL_ORDER does put it in the Y-space rank table, which is right: it
# supplies a full predictive density scored at the same y_test as every other
# row there. Its marginal column is the one-shot marginal every other row's
# ICL branch also uses, so its copula column reads directly as "what
# sequencing bought over independence" — see eval/baselines/autoregressive.py.
_TOTAL_NLL_ORDER = _RANK_KEYS + [
    ("independence_marginal", "Independence (marginal only)"),
    ("autoregressive", "Autoregressive marginal (chain rule)"),
    ("oracle_prior", "Oracle (prior, unconditioned)"),
    ("oracle_posterior", "Oracle (posterior, Schur-conditioned)"),
]


def _ar_note(all_total_nlls: list[dict[str, dict[str, float]]],
             order: str, conditioning: str, max_context: int | None) -> str | None:
    """The footnote _print_total_nll_table prints for the autoregressive row,
    or None when no episode carries one.

    It exists mainly for the ``sample`` case: that number is a log-density sum
    taken along a SAMPLED conditioning path, not a density of y_test (see
    eval/baselines/autoregressive.py), and a row sitting in a table of proper
    scoring rules with no warning attached is exactly how it would get
    compared to its neighbours by mistake.
    """
    n_valid = sum(
        1 for m in all_total_nlls
        if not np.isnan(m.get("autoregressive", _NAN_PARTS).get("total", float("nan")))
    )
    if n_valid == 0:
        return None
    cap = "uncapped" if max_context is None else f"max_context={max_context}"
    note = (f"Autoregressive row: chain-rule joint density from the SAME marginal, "
            f"revealed in {order} order, {cap}, over {n_valid}/{len(all_total_nlls)} "
            f"episodes. Its Marginal column is the one-shot (independence) marginal, "
            f"so its Copula column is exactly what the sequencing bought.")
    if conditioning != "teacher_forcing":
        note += ("\n  *** WARNING: --ar_conditioning=sample. Each step appended a DRAW, "
                 "not the true y, so this row is NOT a joint density of y_test and is "
                 "NOT comparable to the other rows. Ancestral-sampling diagnostic only. ***")
    return note


def _print_total_nll_table(
    all_total_nlls: list[dict[str, dict[str, float]]], z_train_source: str,
    era5: bool = False, ar_note: str | None = None, attempted: int | None = None,
) -> None:
    """Total (marginal + copula) Y-space NLL, EVERY method's own fitted/
    estimated marginal, all divided by that episode's own N (per-point,
    nats/point) — the genuinely cross-method-comparable counterpart to
    _print_table's shared-ground-truth-marginal copula-only table (see
    eval_baselines_episode's and _eval_icl_episode's docstrings for why
    each method supplying its own predictive density, scored at the same
    real y_test, is always a valid proper-scoring-rule comparison,
    regardless of how different the marginals are).

    Each `all_total_nlls[i][method]` is a {"total", "marginal", "copula"}
    dict (see eval_baselines_episode/_eval_icl_episode) — the Marginal/
    Copula columns below are each method's OWN split, not comparable to
    _print_table's shared-ground-truth-marginal copula NLL (see
    eval_baselines_episode's docstring, or the "NAMING TRAP" note in
    eval/metrics/joint_nll.py's module docstring, for why those are
    different quantities that happen to share a name).

    icl's row is nan whenever z_train_source == "oracle" (--z_train_source
    default): the oracle z_test the ICL model would otherwise be scored
    against IS the ground truth, so there is no learned marginal to score a
    total NLL against — --z_train_source=tabicl is required to populate it.

    oracle_prior/oracle_posterior here are gp_analytical_posterior's own
    nll_prior/nll_post (and their marginal/copula split), divided by N for
    this table only — the existing _print_y_space_oracle table's own
    numbers are NOT changed by this (kept unnormalized there for backward
    compatibility with any previously tracked output).
    """
    attempted = len(all_total_nlls) if attempted is None else attempted

    def _col(part: str) -> dict[str, float]:
        return {
            k: numeric_summary([
                m.get(k, _NAN_PARTS).get(part, float("nan")) for m in all_total_nlls
            ])[0]
            for k, _ in _TOTAL_NLL_ORDER
        }

    total_rows = [
        {k: m.get(k, _NAN_PARTS).get("total", float("nan")) for k, _ in _TOTAL_NLL_ORDER}
        for m in all_total_nlls
    ]
    total_summaries = {k: score_summary(total_rows, k) for k, _ in _TOTAL_NLL_ORDER}
    means_marginal = _col("marginal")
    means_copula = _col("copula")

    col = max(22, max(len(label) for _, label in _TOTAL_NLL_ORDER) + 2)
    total = col + 5 * 12
    print(f"\n{'─' * total}")
    print(f"Total NLL (Y-space, marginal+copula, own marginal per method) — "
          f"lower is better  [N={len(all_total_nlls)} episodes]")
    if era5:
        print("Episodes: REAL ARCO-ERA5 2m-temperature (Kelvin). THIS is the table to "
              "read on real data: every method supplies its own full predictive "
              "density and is scored at the same real y_test, which is a proper "
              "scoring rule regardless of whose marginal is whose. The two Oracle "
              "rows are nan by construction (no generating kernel behind ERA5).")
    print(f"ICL z_train source: {z_train_source}"
          + ("  (icl row n/a — oracle mode has no learned ICL marginal to score)"
             if z_train_source == "oracle" else f"  ({z_train_source} K-fold PIT estimate)"))
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
        row for row in total_rows
        if np.isfinite(row.get("icl", float("nan")))
        and np.isfinite(row.get("independence_marginal", float("nan")))
    ]
    if paired:
        icl_mean, _, _ = score_summary(paired, "icl")
        independent_mean, _, _ = score_summary(paired, "independence_marginal")
        print(
            f"Paired ICL/independence on {len(paired)}/{len(total_rows)} episodes: "
            f"ICL={icl_mean:.4f}, independence={independent_mean:.4f}"
        )
    print(f"{'─' * total}\n")


def _compute_ranks(values: list[dict[str, float]], keys: list[str]) -> dict[str, list[int]]:
    """Compatibility wrapper for the shared, tie-aware summary."""
    return competition_ranks(values, keys)


def _print_rank_table(
    values: list[dict[str, float]], order: list[tuple[str, str]], title: str,
    z_train_source: str,
) -> None:
    """Average and median per-episode rank for each method in `order`,
    computed by _compute_ranks over `values` (one {key: nll} dict per
    episode — either all_nlls' z-space NLLs or the "total" slice of
    all_total_nlls' Y-space NLLs, see the two call sites in main()). Mean
    rank rewards consistent placement, median is robust to the rare episode
    a normally-strong method fails or gets a pathological fit; reading both
    together separates "usually great, occasionally terrible" from
    "reliably mediocre". Sorted by mean rank ascending (best first).
    """
    keys = [k for k, _ in order]
    labels = dict(order)
    ranks = _compute_ranks(values, keys)
    rows = [
        (k, float(np.mean(rs)), float(np.median(rs)), len(rs))
        for k, rs in ranks.items() if rs
    ]
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
