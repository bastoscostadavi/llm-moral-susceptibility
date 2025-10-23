#!/usr/bin/env python3
"""Generate LaTeX tables (by foundation) from per-foundation metrics CSV.

Reads results/moral_metrics_by_foundation.csv and writes two tables to
articles/table_susceptibility_by_foundation.tex and
articles/table_robustness_by_foundation.tex by default.

Numbers use 1 significant digit for uncertainties, and values are rounded to
the same decimal place.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DEFAULT_FOUNDATIONS: List[str] = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="by_foundation_csv",
        type=Path,
        default=Path("results") / "moral_metrics_by_foundation.csv",
        help="Path to per-foundation metrics CSV.",
    )
    parser.add_argument(
        "--articles-dir",
        dest="articles_dir",
        type=Path,
        default=Path("articles"),
        help="Directory to write LaTeX tables (default: articles/).",
    )
    parser.add_argument(
        "--foundations",
        nargs="*",
        default=DEFAULT_FOUNDATIONS,
        help="Override foundation column order.",
    )
    return parser.parse_args()


def round_sig(x: float, sig: int = 1) -> float:
    if not np.isfinite(x) or x == 0:
        return float(0)
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def format_value_err(value: float, err: float) -> str:
    """Format as value±err with 1-sig-fig err, value to same decimals."""
    if not np.isfinite(value):
        return "--"
    if (not np.isfinite(err)) or err <= 0:
        return f"{value:.3f}"
    e = abs(float(err))
    e1 = round_sig(e, 1)
    if e1 == 0:
        return f"{value:.3f}"
    exp = int(math.floor(math.log10(e1)))
    decimals = max(0, -exp)
    v = round(float(value), decimals)
    fmt = f"{{:.{decimals}f}}$\pm${{:.{decimals}f}}"
    return fmt.format(v, e1)


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def build_table(
    df: pd.DataFrame,
    foundations: List[str],
    value_col: str,
    err_col: str,
    caption: str,
    label: str,
    wide: bool = False,
) -> str:
    models = sorted(df["model"].unique().tolist())

    header_cols = " & ".join(["Model"] + foundations)
    begin_env = "table*" if wide else "table"
    lines = [
        f"\\begin{{{begin_env}}}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{l{'c'*len(foundations)}}}",
        "    \\toprule",
        f"    {header_cols} \\",
        "    \\midrule",
    ]

    for m in models:
        row_vals = [latex_escape(m)]
        sub = df[df["model"] == m]
        for f in foundations:
            cell = "--"
            subf = sub[sub["foundation"] == f]
            if not subf.empty:
                v = float(subf.iloc[0][value_col])
                e = float(subf.iloc[0][err_col])
                cell = format_value_err(v, e)
            row_vals.append(cell)
        lines.append("    " + " & ".join(row_vals) + " \\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"\\end{{{begin_env}}}",
        "",
    ]
    return "\n".join(lines)


def build_summary_table(df: pd.DataFrame) -> str:
    models = sorted(df["model"].unique().tolist())
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Overall susceptibility and robustness by model (mean $\pm$ SE).}",
        "  \\label{tab:summary_by_model}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Model & Susceptibility ($\pm$) & Robustness ($\pm$) \\",
        "    \\midrule",
    ]
    for m in models:
        sub = df[df["model"] == m].iloc[0]
        sus = format_value_err(float(sub["susceptibility"]), float(sub["s_uncertainty"]))
        rob = format_value_err(float(sub["robustness"]), float(sub["r_uncertainty"]))
        lines.append(f"    {latex_escape(m)} & {sus} & {rob} \\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not args.by_foundation_csv.exists():
        raise SystemExit(f"Per-foundation metrics CSV not found: {args.by_foundation_csv}")

    df = pd.read_csv(args.by_foundation_csv)
    required = {
        "model",
        "foundation",
        "susceptibility",
        "s_uncertainty",
        "robustness",
        "r_uncertainty",
    }
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required.difference(df.columns)))
        raise SystemExit(f"Per-foundation CSV missing columns: {missing}")

    articles_dir = args.articles_dir
    articles_dir.mkdir(parents=True, exist_ok=True)

    sus_tex = build_table(
        df,
        args.foundations,
        value_col="susceptibility",
        err_col="s_uncertainty",
        caption=(
            "Per-foundation moral susceptibility by model (mean $\pm$ SE across persona groups)."
        ),
        label="tab:susceptibility_by_foundation",
        wide=True,
    )
    (articles_dir / "table_susceptibility_by_foundation.tex").write_text(sus_tex)

    rob_tex = build_table(
        df,
        args.foundations,
        value_col="robustness",
        err_col="r_uncertainty",
        caption=(
            "Per-foundation moral robustness by model (inverse of average per-item uncertainty; error bars show propagated SE)."
        ),
        label="tab:robustness_by_foundation",
        wide=True,
    )
    (articles_dir / "table_robustness_by_foundation.tex").write_text(rob_tex)

    # Overall summary by model (from moral_metrics.csv)
    overall_csv = args.by_foundation_csv.parent / "moral_metrics.csv"
    if overall_csv.exists():
        overall = pd.read_csv(overall_csv)
        need = {"model", "susceptibility", "s_uncertainty", "robustness", "r_uncertainty"}
        if need.issubset(overall.columns):
            summary_tex = build_summary_table(overall)
            (articles_dir / "table_summary_by_model.tex").write_text(summary_tex)


if __name__ == "__main__":
    main()
