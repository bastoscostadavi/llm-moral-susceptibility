#!/usr/bin/env python3
"""Summarize persona/question MFQ reruns while ignoring invalid ratings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw rerun CSV files (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("results"),
        help="Directory where summary CSVs will be written (default: results).",
    )
    return parser.parse_args()


def iter_input_files(data_dir: Path) -> Iterable[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for path in sorted(data_dir.glob("*.csv")):
        if path.name.endswith("_self.csv"):
            continue
        if path.is_file():
            yield path


def summarize_file(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)

    expected_cols = {"persona_id", "question_id", "run_index", "rating"}
    missing_cols = expected_cols.difference(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(f"{input_csv} missing required columns: {missing_str}")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    base_pairs = (
        df[["persona_id", "question_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(["persona_id", "question_id"], ignore_index=True)
    )

    valid = df[df["rating"].notna() & (df["rating"] != -1)]

    def _std(series: pd.Series) -> float:
        if len(series) <= 1:
            return 0.0
        return float(series.std(ddof=1))

    stats = (
        valid.groupby(["persona_id", "question_id"], as_index=False)["rating"]
        .agg(average_score="mean", standard_deviation=_std)
    )

    summary = base_pairs.merge(stats, on=["persona_id", "question_id"], how="left")
    summary["average_score"] = summary["average_score"].fillna(-1)
    summary["standard_deviation"] = summary["standard_deviation"].fillna(-1)

    summary = summary[["persona_id", "question_id", "average_score", "standard_deviation"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_csv.name
    summary.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    for input_csv in iter_input_files(args.data_dir):
        summarize_file(input_csv, args.output_dir)


if __name__ == "__main__":
    main()
