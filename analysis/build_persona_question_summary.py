#!/usr/bin/env python3
"""Collapse rerun MFQ results into persona-question means, uncertainties, and relative scores.

This summary expects as input a persona-conditioned MFQ CSV (e.g.,
`data/gpt-4o-mini.csv`) and will automatically locate the matching
no-persona self-run file (e.g., `data/gpt-4o-mini_self.csv`). The output
contains one row per (persona, question) with:

- persona
- question
- average_score
- uncertainty
- relative_score = sqrt((average_score - R0)^2) where R0 is the mean
  self-run rating for the same question.

Notes:
Notes:
- All ratings, including -1, are aggregated; ensure you re-run missing
  slots to remove invalid entries before summarizing.
"""

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

    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")

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

    # Locate and load the corresponding self-run CSV to compute R0 per question
    input_name = args.input_csv.name  # e.g., gpt-4o-mini.csv
    stem = args.input_csv.stem        # e.g., gpt-4o-mini
    self_csv = args.input_csv.with_name(f"{stem}_self.csv")

    if not self_csv.exists():
        raise FileNotFoundError(
            f"Could not find matching self-run CSV: {self_csv}. Run run_mfq_self.py for this model first."
        )

    self_raw = pd.read_csv(self_csv)
    self_expected = {"question_id", "run_index", "rating"}
    self_missing = self_expected.difference(self_raw.columns)
    if self_missing:
        missing_str = ", ".join(sorted(self_missing))
        raise ValueError(f"Self-run CSV missing columns: {missing_str}")

    self_raw["rating"] = pd.to_numeric(self_raw["rating"], errors="coerce")

    r0_per_question = (
        self_raw.groupby("question_id")["rating"].mean().reset_index().rename(
            columns={"question_id": "question", "rating": "R0"}
        )
    )

    # Merge and compute relative_score = |average_score - R0|
    summary = summary.merge(r0_per_question, on="question", how="left")
    if summary["R0"].isna().any():
        # Questions missing in self-run file would prevent relative score computation
        missing_q = sorted(summary.loc[summary["R0"].isna(), "question"].unique().tolist())
        raise ValueError(
            f"Self-run file lacks ratings for questions: {missing_q}. Rerun self-run to cover all questions."
        )

    summary["relative_score"] = (summary["average_score"] - summary["R0"]).abs()
    summary = summary.drop(columns=["R0"])  # keep only requested columns

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.input_csv.name

    # Reorder columns as requested
    summary = summary[["persona", "question", "average_score", "uncertainty", "relative_score"]]
    summary.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
