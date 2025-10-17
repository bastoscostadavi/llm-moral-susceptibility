#!/usr/bin/env python3
"""Collapse rerun MFQ results into persona-question means and uncertainties."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_csv",
        type=Path,
        required=True,
        help="Path to the raw MFQ CSV with persona/question reruns.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("results"),
        help="Directory where the summary CSV will be written (default: results).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = pd.read_csv(args.input_csv)
    expected_cols = {"persona_id", "question_id", "run_index", "rating"}
    missing = expected_cols.difference(raw.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV missing columns: {missing_str}")

    grouped = raw.groupby(["persona_id", "question_id"])["rating"]

    def _std(series: pd.Series) -> float:
        if len(series) <= 1:
            return 0.0
        return float(series.std(ddof=1))

    summary = (
        grouped.agg(average_score="mean", uncertainty=_std)
        .reset_index()
        .rename(columns={"persona_id": "persona", "question_id": "question"})
        .sort_values(["persona", "question"], ignore_index=True)
    )

    summary["uncertainty"] = summary["uncertainty"].fillna(0.0)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.input_csv.name

    summary.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

