"""report.py — aggregate debug/results/<run_id>/*.json into one markdown
report, optionally diffed against a --baseline run_id.

Each stage script already writes its own <stage>.json (via
common.py::save_stage_result); this module doesn't recompute anything, it
only extracts a short human-readable highlight per stage and renders a
markdown table. Re-run after a conf/data/gp_tasks.yaml or architecture
change with --baseline pointing at the run before the change to see what
moved.

Usage:
    python debug/report.py --run-id 20260826_120000_abcd123_all
    python debug/report.py --run-id <new_id> --baseline <old_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, "results")

STAGE_ORDER = ["s0_signal", "s1_rank_ceiling", "s2_uspace", "s3_pit_floor", "s5_kfold", "s6_guards", "s7b_backend_train"]


def _load(run_id: str, stage: str):
    path = os.path.join(RESULTS_ROOT, run_id, f"{stage}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _highlight_s0(r: dict) -> "list[str]":
    lines = ["| P | copula NLL/pt | marginal NLL/pt | |R| abs_mean | eff_rank@90% |", "|---|---|---|---|---|"]
    for row in r.get("per_P", []):
        if row["n_episodes_scored"] == 0:
            continue
        lines.append(
            f"| {row['P']} | {row['copula_nll_per_point']['mean']:.4f} | "
            f"{row['marginal_nll_per_point']['mean']:.4f} | {row['offdiag_R_post']['abs_mean']:.4f} | "
            f"{row['effective_rank']['at_90pct_variance_mean']:.1f} |"
        )
    return lines


def _highlight_s1(r: dict) -> "list[str]":
    if "error" in r:
        return [r["error"]]
    lines = ["| rank | ceiling copula NLL/pt |", "|---|---|"]
    for row in r.get("per_rank", []):
        lines.append(f"| {row['rank']} | {row['ceiling_copula_nll_per_point']['mean']:.4f} |")
    return lines


def _highlight_s2(r: dict) -> "list[str]":
    lines = []
    for label in ("analytic", "tabicl"):
        d = r.get(label, {})
        lines.append(f"**{label}**")
        for name in ("z_train", "z_test"):
            v = d.get(name, {})
            if "error" in v:
                continue
            cc = v.get("clamping_census", {})
            lines.append(
                f"- {name}: KS={v.get('ks_statistic_pooled', float('nan')):.4f} "
                f"ECE={v.get('pit_ece', float('nan')):.4f} "
                f"spline_sat={cc.get('pooled_frac_spline_saturated', float('nan')):.5f} "
                f"episodes>1%sat={cc.get('n_episodes_gt_1pct_saturated', '?')}/{cc.get('n_episodes_total', '?')}"
            )
    return lines


def _highlight_s3(r: dict) -> "list[str]":
    if "error" in r:
        return [r["error"]]
    s = r.get("summary", {})
    return [
        f"- floor copula NLL/pt on R_z (held-out): {s.get('floor_copula_nll_on_Rz_mean', float('nan')):.4f}",
        f"- copula NLL/pt of R_post on PIT z: {s.get('copula_nll_of_Rpost_on_pit_z_mean', float('nan')):.4f}",
        f"- rank-{r.get('rank')} ceiling on R_z: {s.get('rank_ceiling_on_Rz_mean', float('nan')):.4f}",
        f"- Ledoit-Wolf shrinkage (mean): {s.get('Rz_shrinkage_coefficient_mean', float('nan')):.4f}",
    ]


def _highlight_s5(r: dict) -> "list[str]":
    if "error" in r:
        return [r["error"]]
    lines = ["| K | copula NLL/pt | corr Pearson |", "|---|---|---|"]
    for label, row in r.get("results", {}).items():
        pear = row.get("corr_pearson_vs_Rpost")
        pear_str = f"{pear:.4f}" if pear is not None else "n/a"
        lines.append(f"| {label} | {row['copula_nll']:.4f} | {pear_str} |")
    return lines


def _highlight_s6(r: dict) -> "list[str]":
    e = r.get("escape_ratio") or {}
    return [
        f"- ||W_i||: mean={r['W_norm']['mean']:.4f} max={r['W_norm']['max']:.4f}",
        f"- escape ratio ||W||^2/softplus(s): mean={e.get('mean', float('nan')):.4f} frac>1={e.get('frac_gt_1', float('nan')):.4f}",
        f"- Sigma offdiag abs_mean: {r['sigma_offdiag']['abs_mean']:.4f}",
        f"- Cholesky escalations: {r['safe_cholesky_calls_escalated']}/{r['safe_cholesky_calls_total']}, "
        f"non-finite input slices: {r['nonfinite_input_slices']}",
    ]


def _highlight_s7b(r: dict) -> "list[str]":
    lines = ["| backend | eval_total | eval_copula | eval_marginal |", "|---|---|---|---|"]
    for backend, hist in r.get("history", {}).items():
        if not hist:
            continue
        last = hist[-1]
        lines.append(f"| {backend} | {last['eval_total']:.4f} | {last['eval_copula']:.4f} | {last['eval_marginal']:.4f} |")
    return lines


_HIGHLIGHTERS = {
    "s0_signal": _highlight_s0, "s1_rank_ceiling": _highlight_s1, "s2_uspace": _highlight_s2,
    "s3_pit_floor": _highlight_s3, "s5_kfold": _highlight_s5, "s6_guards": _highlight_s6,
    "s7b_backend_train": _highlight_s7b,
}


def build_report(run_id: str, baseline_id: "str | None" = None) -> str:
    run_dir = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    lines = [f"# Debug pipeline report — {run_id}", ""]
    if baseline_id:
        lines.append(f"Diffed against baseline: `{baseline_id}`")
        lines.append("")

    found_any = False
    for stage in STAGE_ORDER:
        payload = _load(run_id, stage)
        if payload is None:
            continue
        found_any = True
        lines.append(f"## {stage}")
        lines.append(f"*git_sha={payload.get('git_sha')}  n_episodes={payload.get('n_episodes')}  "
                      f"ckpt={payload.get('ckpt')}  overrides={payload.get('overrides')}*")
        lines.append("")
        highlighter = _HIGHLIGHTERS.get(stage)
        result = payload.get("result", {})
        if highlighter:
            lines.extend(highlighter(result))
        else:
            lines.append(f"```json\n{json.dumps(result, indent=2)[:2000]}\n```")
        lines.append("")

        if baseline_id:
            base_payload = _load(baseline_id, stage)
            if base_payload is not None:
                lines.append(f"**vs. baseline {baseline_id}:**")
                if highlighter:
                    lines.extend(highlighter(base_payload.get("result", {})))
                lines.append("")

    if not found_any:
        lines.append(f"*No stage JSON found under debug/results/{run_id}/*")

    out_path = os.path.join(run_dir, "report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--baseline", default=None)
    args = p.parse_args()
    out_path = build_report(args.run_id, baseline_id=args.baseline)
    print(f"Report written -> {out_path}")


if __name__ == "__main__":
    main()
