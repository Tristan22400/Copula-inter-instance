"""Tables for eval/runners/eeg_joint_eval.py, laid out like the two papers' EEG tables.

Reads out_dir/<unit>/<marginal>.{json,npz} and out_dir/timing/<marginal>.json; writes
out_dir/report.md. Published numbers are copied from the papers' LaTeX sources:
- Bruinsma et al. (ICLR 2023), Table "Normalised log-likelihoods on the EEG experiments": mean
  +- 1.96 SE over 4096 tasks, nats per target scalar, raw microvolts.
- Hassan et al. (ICLR 2026), Table 1 (M=16) and the appendix table (M=64): mean (SEM) over the
  evaluation trials x N in {8, 16, 32, 64, 128, 192}, standardised channels; AR rows log-mean-exp
  over 128 target orders.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

__all__ = ["ARCNP_PUBLISHED", "HASSAN_PUBLISHED", "load_results", "write_report"]

MARGINAL_LABELS = {"tabicl": "TabICLv2", "tabpfn35_fast": "TabPFN-3.5", "kumo_small": "Kumo"}

# (model, Int, Rec), each (mean, 95% half-width).
ARCNP_PUBLISHED: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
    ("ConvCNP", (-1.02, 0.01), (-2.07, 0.03)),
    ("ConvCNP (AR)", (-0.34, 0.01), (-0.63, 0.01)),
    ("ConvGNP", (-0.93, 0.01), (-1.45, 0.03)),
    ("FullConvGNP", (-0.35, 0.01), (-0.57, 0.01)),
    ("ConvLNP (ML)", (-1.04, 0.01), (-1.53, 0.02)),
    ("ConvLNP (ELBO)", (-1.20, 0.01), (-2.00, 0.06)),
]

# model -> {(mode, M): (mean, SEM)}; mode int / for.
HASSAN_PUBLISHED: dict[str, dict[tuple[str, int], tuple[float, float]]] = {
    "TNP-D (AR)": {
        ("int", 16): (0.51, 0.013),
        ("for", 16): (1.07, 0.004),
        ("int", 64): (0.88, 0.011),
        ("for", 64): (1.12, 0.003),
    },
    "TNP-D (Ind)": {
        ("int", 16): (0.36, 0.014),
        ("for", 16): (-0.74, 0.008),
        ("int", 64): (0.35, 0.014),
        ("for", 64): (-1.08, 0.007),
    },
    "TNP-ND": {
        ("int", 16): (0.46, 0.011),
        ("for", 16): (-0.04, 0.005),
        ("int", 64): (0.50, 0.010),
        ("for", 64): (-0.23, 0.004),
    },
    "TNP-A": {
        ("int", 16): (0.58, 0.014),
        ("for", 16): (1.23, 0.003),
        ("int", 64): (0.95, 0.012),
        ("for", 64): (1.20, 0.003),
    },
    "TNP buffer K=16": {
        ("int", 16): (0.52, 0.013),
        ("for", 16): (0.85, 0.004),
        ("int", 64): (0.90, 0.011),
        ("for", 64): (1.12, 0.003),
    },
    "TNP buffer K=4": {
        ("int", 16): (0.54, 0.014),
        ("for", 16): (1.17, 0.003),
        ("int", 64): (0.91, 0.011),
        ("for", 64): (1.18, 0.003),
    },
    "TNP buffer K=1": {
        ("int", 16): (0.54, 0.014),
        ("for", 16): (1.21, 0.003),
        ("int", 64): (0.91, 0.011),
        ("for", 64): (1.19, 0.003),
    },
}

_HASSAN_COLS = [("int", 16), ("for", 16), ("int", 64), ("for", 64)]
_METHODS = (("ll_indep", "indep"), ("ll_copula", "copula"), ("ll_ar", "AR"))


def load_results(out_dir: str) -> list[dict[str, Any]]:
    """Every scored (unit, marginal): its JSON blob plus the per-episode arrays under "arrays"."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "*.json"))):
        if os.path.basename(os.path.dirname(path)) == "timing":
            continue
        with open(path) as fh:
            blob = json.load(fh)
        with np.load(os.path.splitext(path)[0] + ".npz") as f:
            blob["arrays"] = {k: f[k] for k in f.files}
        rows.append(blob)
    return rows


def _mean_err(v: np.ndarray, z: float) -> tuple[float, float, int]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(v.mean()), z * se, int(v.size)


def _fmt_ci(m: float, e: float) -> str:
    return "—" if not np.isfinite(m) else f"{m:.2f} ± {e:.2f}"


def _fmt_sem(m: float, e: float) -> str:
    return "—" if not np.isfinite(m) else f"{m:.2f} ({e:.3f})"


def _arcnp_table(res: list[dict[str, Any]], marginals: list[str]) -> list[str]:
    by = {(r["unit"], r["marginal"]): r["arrays"] for r in res if r["family"] == "arcnp"}
    lines = [
        "| Model | Int. | Rec. |",
        "|---|---|---|",
        *[f"| {name} (published) | {_fmt_ci(*i)} | {_fmt_ci(*r)} |" for name, i, r in ARCNP_PUBLISHED],
    ]
    for m in marginals:
        for key, label in _METHODS:
            cells, ns = [], []
            for unit in ("arcnp_interpolation", "arcnp_reconstruction"):
                a = by.get((unit, m))
                if a is None:
                    cells.append("—")
                    continue
                mean, err, n = _mean_err(a[key], 1.96)
                cells.append(_fmt_ci(mean, err) + (_subset_note(a) if key == "ll_ar" else ""))
                ns.append(n)
            note = f" (n={'/'.join(map(str, ns))})" if key == "ll_ar" and ns else ""
            lines.append(f"| {MARGINAL_LABELS[m]} + {label}{note} | {cells[0]} | {cells[1]} |")
    return lines


def _subset_note(a: dict[str, np.ndarray]) -> str:
    """indep / copula means on the AR row's own episodes (the AR subset is not the whole task)."""
    sel = np.isfinite(a["ll_ar"])
    if not sel.any():
        return ""
    return f" [same eps: indep {a['ll_indep'][sel].mean():.2f}, copula {a['ll_copula'][sel].mean():.2f}]"


def _hassan_pooled(res: list[dict[str, Any]], marginal: str, mode: str, M: int, key: str) -> np.ndarray:
    parts = [
        r["arrays"][key]
        for r in res
        if r["family"] == "hassan"
        and r["marginal"] == marginal
        and r["unit"].startswith(f"hassan_{mode}_")
        and r["n_target"] == M
    ]
    return np.concatenate(parts) if parts else np.array([])


def _hassan_table(res: list[dict[str, Any]], marginals: list[str]) -> list[str]:
    head = "| Model | " + " | ".join(f"EEG-{mo.capitalize()} (M={M})" for mo, M in _HASSAN_COLS) + " |"
    lines = [head, "|---" * (len(_HASSAN_COLS) + 1) + "|"]
    for name, cols in HASSAN_PUBLISHED.items():
        lines.append(f"| {name} (published) | " + " | ".join(_fmt_sem(*cols[c]) for c in _HASSAN_COLS) + " |")
    for m in marginals:
        for key, label in _METHODS:
            cells, ns = [], []
            for mo, M in _HASSAN_COLS:
                mean, err, n = _mean_err(_hassan_pooled(res, m, mo, M, key), 1.0)
                sub = {k: _hassan_pooled(res, m, mo, M, k) for k in ("ll_ar", "ll_indep", "ll_copula")}
                cells.append(_fmt_sem(mean, err) + (_subset_note(sub) if key == "ll_ar" and n else ""))
                ns.append(n)
            note = f" (n={'/'.join(map(str, ns))})" if key == "ll_ar" else ""
            lines.append(f"| {MARGINAL_LABELS[m]} + {label}{note} | " + " | ".join(cells) + " |")
    return lines


def _gain_table(res: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Unit | Marginal | n | indep | copula | copula − indep (± 1.96 SE) | win rate | z_test std |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        res, key=lambda r: (r["unit"].split("_N")[0], r["n_target"] or 0, r["n_context"] or 0, r["marginal"])
    ):
        a = r["arrays"]
        gain = a["ll_copula"] - a["ll_indep"]
        g, e, n = _mean_err(gain, 1.96)
        lines.append(
            f"| {r['unit']} | {MARGINAL_LABELS[r['marginal']]} | {n} | {np.mean(a['ll_indep']):.3f} | "
            f"{np.mean(a['ll_copula']):.3f} | {g:+.3f} ± {e:.3f} | {np.mean(gain > 0):.2f} | {np.mean(a['z_std']):.2f} |"
        )
    return lines


def _ar_paired_table(res: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Unit | Marginal | n (AR episodes) | indep | copula | AR | copula − AR (± 1.96 SE) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        res, key=lambda r: (r["unit"].split("_N")[0], r["n_target"] or 0, r["n_context"] or 0, r["marginal"])
    ):
        a = r["arrays"]
        sel = np.isfinite(a["ll_ar"])
        if not sel.any():
            continue
        d, e, n = _mean_err(a["ll_copula"][sel] - a["ll_ar"][sel], 1.96)
        lines.append(
            f"| {r['unit']} | {MARGINAL_LABELS[r['marginal']]} | {n} | {a['ll_indep'][sel].mean():.3f} | "
            f"{a['ll_copula'][sel].mean():.3f} | {a['ll_ar'][sel].mean():.3f} | {d:+.3f} ± {e:.3f} |"
        )
    return lines


def _check_table(res: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Marginal | episodes | mean KS, copula draws | mean KS, independent draws | max abs(diag Σ − 1) |",
        "|---|---|---|---|---|",
    ]
    for m in MARGINAL_LABELS:
        rs = [r["arrays"] for r in res if r["marginal"] == m]
        if not rs:
            continue
        kc = np.concatenate([a["ks_copula"] for a in rs])
        ki = np.concatenate([a["ks_indep"] for a in rs])
        md = max(float(np.max(a["max_diag_err"])) for a in rs)
        ok = np.isfinite(kc)
        lines.append(
            f"| {MARGINAL_LABELS[m]} | {int(ok.sum())} | {kc[ok].mean():.4f} | {ki[ok].mean():.4f} | {md:.1e} |"
        )
    return lines


def _timing_table(out_dir: str) -> list[str]:
    lines = [
        "| Marginal | GPU | M (time pts) | target rows | B × S | indep (s) | copula (s) | AR (s) | AR / copula |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for path in sorted(glob.glob(os.path.join(out_dir, "timing", "*.json"))):
        with open(path) as fh:
            t = json.load(fh)
        for r in t["rows"]:
            lines.append(
                f"| {MARGINAL_LABELS[t['marginal']]} | {t['device']} | {r['n_target_time']} | {r['target_rows']} | "
                f"{r['episodes']} × {r['samples']} | {r['seconds_indep']:.3f} | {r['seconds_copula']:.3f} | "
                f"{r['seconds_ar']:.2f} | {r['seconds_ar'] / r['seconds_copula']:.0f}× |"
            )
    return lines


def write_report(out_dir: str) -> str:
    """Render out_dir/report.md from the scored units; returns its path."""
    res = load_results(out_dir)
    marginals = [m for m in MARGINAL_LABELS if any(r["marginal"] == m for r in res)]
    parts = [
        "# EEG joint log-likelihood: TFM marginals, independent vs copula vs autoregressive",
        "",
        "Nats per target scalar, higher is better. Our rows are zero-shot (pretrained marginals, copulas trained on",
        "synthetic GP data); the published models were trained on the EEG training subjects. indep and copula rows",
        "cover every task; AR rows cover an evenly spread subset (n), and [same eps: ...] gives indep and copula",
        "on exactly those episodes.",
        "",
        "## Bruinsma et al. (2023) EEG, exact protocol (mean ± 1.96 SE, raw µV)",
        "",
        *_arcnp_table(res, marginals),
        "",
        "## Hassan et al. (2026) EEG, protocol reconstructed (mean (SEM) over trials × N, standardised channels)",
        "",
        *_hassan_table(res, marginals),
        "",
        "## Copula vs independence, paired per episode",
        "",
        *_gain_table(res),
        "",
        "## Copula vs autoregressive, same marginal, on the AR episodes",
        "",
        *_ar_paired_table(res),
        "",
        "## Marginal preservation (copula draws vs independent draws, KS of Φ(z_i) to U(0,1))",
        "",
        *_check_table(res),
        "",
        "## Joint sampling cost (Hassan EEG-Int, N = 128 context time points)",
        "",
        *_timing_table(out_dir),
        "",
    ]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {path}")
    return path
