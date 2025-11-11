#!/usr/bin/env python3
"""Convert unbounded MFQ robustness/susceptibility metrics into bounded scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results") / "moral_metrics.csv",
        help="Input CSV with model-level metrics (default: results/moral_metrics.csv)",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results") / "moral_metrics_bounded.csv",
        help="Destination CSV for bounded model-level metrics.",
    )
    parser.add_argument(
        "--foundation-metrics",
        type=Path,
        default=Path("results") / "moral_metrics_per_foundation.csv",
        help="Input CSV with foundation-level metrics (default: results/moral_metrics_per_foundation.csv)",
    )
    parser.add_argument(
        "--foundation-output",
        type=Path,
        default=Path("results") / "moral_metrics_per_foundation_bounded.csv",
        help="Destination CSV for bounded foundation-level metrics.",
    )
    return parser.parse_args()


def _bound_series(series: pd.Series, eps: float = 1e-12) -> tuple[pd.Series, pd.Series]:
    mean_val = series.mean()
    denom = series + mean_val
    denom = denom.mask(denom.abs() < eps, eps)
    bounded = series / denom
    derivative = mean_val / (denom ** 2)
    return bounded, derivative


def _apply_bounds(
    df: pd.DataFrame,
    value_col: str,
    se_col: Optional[str],
    group_col: Optional[str] = None,
) -> None:
    if value_col not in df:
        return
    if group_col and group_col in df:
        group_iter = df.groupby(group_col, group_keys=False)
    else:
        group_iter = [(None, df)]

    for _, group in group_iter:
        series = group[value_col].astype(float)
        bounded, derivative = _bound_series(series)
        df.loc[group.index, f"bounded_{value_col}"] = bounded
        if se_col and se_col in df:
            se_series = group[se_col].astype(float)
            df.loc[group.index, f"bounded_{value_col}_uncertainty"] = se_series * derivative


def _process_file(
    input_path: Path,
    output_path: Path,
    group_col: Optional[str] = None,
) -> None:
    if not input_path.exists():
        return
    df = pd.read_csv(input_path)
    if df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return

    _apply_bounds(df, "robustness", "robustness_uncertainty", group_col)
    _apply_bounds(df, "susceptibility", "susceptibility_uncertainty", group_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    _process_file(args.metrics, args.metrics_output, group_col=None)
    _process_file(args.foundation_metrics, args.foundation_output, group_col="foundation")


if __name__ == "__main__":
    main()
