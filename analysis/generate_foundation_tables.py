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
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

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
    if not math.isfinite(x) or x == 0:
        return float(0)
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def format_value_err(value: float, err: float) -> str:
    """Format as $value \pm err$ with 1-sig-fig err.

    - Keeps both value and error inside a single math block to avoid
      alignment issues in tabular environments.
    - Rounds the error to 1 significant figure and value to the same
      number of decimal places.
    """
    if not math.isfinite(value):
        return "--"
    if (not math.isfinite(err)) or err <= 0:
        return f"{value:.3f}"
    e = abs(float(err))
    e1 = round_sig(e, 1)
    if e1 == 0:
        return f"{value:.3f}"
    exp = int(math.floor(math.log10(e1)))
    decimals = max(0, -exp)
    v = round(float(value), decimals)
    # Wrap both the value and the uncertainty in the same math block.
    return f"${v:.{decimals}f}\\pm {e1:.{decimals}f}$"


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
    records: Sequence[Dict[str, str]],
    foundations: Sequence[str],
    value_col: str,
    err_col: str,
    caption: str,
    label: str,
    wide: bool = False,
) -> str:
    LINEBREAK = r"\\"  # two backslashes in output
    models = sorted({m for m in (row.get("model") for row in records) if m})
    value_map = {
        (r.get("model"), r.get("foundation")): r
        for r in records
        if r.get("model") and r.get("foundation")
    }

    header_cols = " & ".join(["Model"] + foundations)
    begin_env = "table*" if wide else "table"
    LINEBREAK = r"\\"  # two backslashes in output
    lines = [
        f"\\begin{{{begin_env}}}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{l{'c'*len(foundations)}}}",
        "    \\toprule",
        f"    {header_cols} {LINEBREAK}",
        "    \\midrule",
    ]

    for m in models:
        row_vals = [latex_escape(m)]
        for f in foundations:
            cell = "--"
            entry = value_map.get((m, f))
            if entry is not None:
                v = parse_float(entry.get(value_col))
                e = parse_float(entry.get(err_col))
                cell = format_value_err(v, e)
            row_vals.append(cell)
        # Ensure each data row ends with a proper LaTeX line break.
        lines.append("    " + " & ".join(row_vals) + f" {LINEBREAK}")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"\\end{{{begin_env}}}",
        "",
    ]
    return "\n".join(lines)


def build_summary_table(records: Sequence[Dict[str, str]]) -> str:
    models = sorted({m for m in (row.get("model") for row in records) if m})
    model_map = {r.get("model"): r for r in records if r.get("model")}
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        r"  \caption{Overall susceptibility and robustness by model (mean $\pm$ SE).}",
        "  \\label{tab:summary_by_model}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        r"    Model & Susceptibility ($\pm$) & Robustness ($\pm$) \\",
        "    \\midrule",
    ]
    LINEBREAK = r"\\"  # two backslashes in output
    for m in models:
        data = model_map.get(m, {})
        sus = format_value_err(
            parse_float(data.get("susceptibility")),
            parse_float(data.get("s_uncertainty")),
        )
        rob = format_value_err(
            parse_float(data.get("robustness")),
            parse_float(data.get("r_uncertainty")),
        )
        lines.append(f"    {latex_escape(m)} & {sus} & {rob} {LINEBREAK}")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def parse_float(value: object | None) -> float:
    if value in {None, ""}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_csv_records(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV missing header: {path}")
        rows = [row for row in reader if any((cell or "").strip() for cell in row.values())]
    return rows


def main() -> None:
    args = parse_args()

    if not args.by_foundation_csv.exists():
        raise SystemExit(f"Per-foundation metrics CSV not found: {args.by_foundation_csv}")

    records = list(read_csv_records(args.by_foundation_csv))
    if not records:
        raise SystemExit(f"Per-foundation metrics CSV has no rows: {args.by_foundation_csv}")

    fieldnames: set[str] = set()
    for row in records:
        fieldnames.update(row.keys())
    required = {
        "model",
        "foundation",
        "susceptibility",
        "s_uncertainty",
        "robustness",
        "r_uncertainty",
    }
    if not required.issubset(fieldnames):
        missing = ", ".join(sorted(required.difference(fieldnames)))
        raise SystemExit(f"Per-foundation CSV missing columns: {missing}")

    articles_dir = args.articles_dir
    articles_dir.mkdir(parents=True, exist_ok=True)

    sus_tex = build_table(
        records,
        args.foundations,
        value_col="susceptibility",
        err_col="s_uncertainty",
        caption=(
            r"Per-foundation moral susceptibility by model (mean $\pm$ SE across persona groups)."
        ),
        label="tab:susceptibility_by_foundation",
        wide=True,
    )
    (articles_dir / "table_susceptibility_by_foundation.tex").write_text(sus_tex)

    rob_tex = build_table(
        records,
        args.foundations,
        value_col="robustness",
        err_col="r_uncertainty",
        caption=(
            r"Per-foundation moral robustness by model (inverse of average per-item standard deviation; error bars show propagated SE via delta method)."
        ),
        label="tab:robustness_by_foundation",
        wide=True,
    )
    (articles_dir / "table_robustness_by_foundation.tex").write_text(rob_tex)

    # Overall summary by model (from moral_metrics.csv)
    overall_csv = args.by_foundation_csv.parent / "moral_metrics.csv"
    if overall_csv.exists():
        overall_records = list(read_csv_records(overall_csv))
        if overall_records:
            columns: set[str] = set()
            for row in overall_records:
                columns.update(row.keys())
            need = {"model", "susceptibility", "s_uncertainty", "robustness", "r_uncertainty"}
            if need.issubset(columns):
                summary_tex = build_summary_table(overall_records)
                (articles_dir / "table_summary_by_model.tex").write_text(summary_tex)


if __name__ == "__main__":
    main()
