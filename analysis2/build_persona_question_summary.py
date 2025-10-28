#!/usr/bin/env python3
"""Summarise persona-question scores while discarding failed runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


EXPECTED_COLUMNS = {"persona_id", "question_id", "run_index", "rating"}
FAILURE_COLUMNS = ("failures", "failure", "failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_csvs",
        type=Path,
        nargs="*",
        help=(
            "Optional raw MFQ CSV paths. If omitted, all files matching --pattern "
            "inside --input-dir are processed."
        ),
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        type=Path,
        default=Path("data"),
        help="Directory to scan when --input is not supplied (default: data).",
    )
    parser.add_argument(
        "--pattern",
        dest="pattern",
        type=str,
        default="*.csv",
        help="Glob used with --input-dir to locate raw CSV files (default: *.csv).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("results2"),
        help="Directory where summary CSVs will be written (default: results2).",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip generating summaries that already exist in the output directory.",
    )
    parser.add_argument(
        "--exclude-persona",
        dest="exclude_personas",
        type=int,
        action="append",
        default=[],
        help="Persona IDs to omit entirely from the summaries (repeatable).",
    )
    return parser.parse_args()


def locate_inputs(inputs: Iterable[Path] | None, directory: Path, pattern: str) -> list[Path]:
    if inputs:
        resolved = [p for p in (Path(path) for path in inputs) if p.exists()]
        missing = [str(p) for p in inputs if not Path(p).exists()]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(f"Input file(s) not found: {missing_str}")
        return resolved

    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def discard_failures(frame: pd.DataFrame) -> pd.DataFrame:
    for col in FAILURE_COLUMNS:
        if col in frame.columns:
            failures = pd.to_numeric(frame[col], errors="coerce")
            mask = failures.isna() | (failures <= 0)
            return frame.loc[mask].copy()
    return frame.copy()


def summarise(input_path: Path, exclude_personas: Sequence[int]) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    missing = EXPECTED_COLUMNS.difference(raw.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV missing columns: {missing_str}")

    filtered = discard_failures(raw)
    if exclude_personas:
        exclude_set = set(int(pid) for pid in exclude_personas)
        filtered = filtered.loc[~filtered["persona_id"].isin(exclude_set)].copy()
    if filtered.empty:
        raise ValueError("No rows remain after filtering failed entries and exclusions.")

    filtered = filtered.copy()
    filtered["rating"] = pd.to_numeric(filtered["rating"], errors="coerce")

    grouped = filtered.groupby(["persona_id", "question_id"], dropna=False)["rating"]

    def _std(series: pd.Series) -> float:
        if len(series) <= 1:
            return 0.0
        return float(series.std(ddof=1))

    summary = (
        grouped.agg(average_score="mean", standard_deviation=_std)
        .reset_index()
        .rename(columns={"persona_id": "persona", "question_id": "question"})
        .sort_values(["persona", "question"], ignore_index=True)
    )

    summary["standard_deviation"] = summary["standard_deviation"].fillna(0.0)
    summary = summary[["persona", "question", "average_score", "standard_deviation"]]
    return summary


def main() -> None:
    args = parse_args()
    inputs = locate_inputs(args.input_csvs, args.input_dir, args.pattern)

    if not inputs:
        raise SystemExit("No input CSVs found to summarise.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_personas = args.exclude_personas or []

    for path in inputs:
        output_path = output_dir / path.name
        if args.skip_existing and output_path.exists():
            print(f"Skipping {path.name}: output already exists.")
            continue

        try:
            summary = summarise(path, exclude_personas)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        summary.to_csv(output_path, index=False)
        dropped_note = (
            f" (dropped personas: {sorted(set(exclude_personas))})"
            if exclude_personas
            else ""
        )
        print(f"Wrote {output_path}{dropped_note}")


if __name__ == "__main__":
    main()
