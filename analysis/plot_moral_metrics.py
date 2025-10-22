#!/usr/bin/env python3
"""Generate bar plots for susceptibility, relative susceptibility, and robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        dest="metrics_csv",
        type=Path,
        default=Path("results") / "moral_metrics.csv",
        help="Path to the moral metrics CSV produced by compute_moral_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("results"),
        help="Directory where plot images will be written (default: results).",
    )
    return parser.parse_args()


def ensure_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Metrics CSV missing columns: {missing_str}")


def plot_metric(
    frame: pd.DataFrame,
    value_col: str,
    error_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    models = frame["model"]
    values = frame[value_col]
    errors = frame[error_col]

    bars = ax.bar(models, values, yerr=errors, capsize=6, color="#4B8BBE")

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.metrics_csv.exists():
        raise SystemExit(f"Metrics CSV not found: {args.metrics_csv}")

    metrics = pd.read_csv(args.metrics_csv)
    ensure_columns(
        metrics,
        {
            "model",
            "susceptibility",
            "s_uncertainty",
            "relative_susceptibility",
            "rs_uncertainty",
            "robustness",
            "r_uncertainty",
        },
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        metrics,
        "susceptibility",
        "s_uncertainty",
        "Susceptibility",
        "Moral Susceptibility by Model",
        output_dir / "susceptibility.png",
    )

    plot_metric(
        metrics,
        "relative_susceptibility",
        "rs_uncertainty",
        "Relative Susceptibility",
        "Relative Susceptibility by Model",
        output_dir / "relative_susceptibility.png",
    )

    plot_metric(
        metrics,
        "robustness",
        "r_uncertainty",
        "Robustness",
        "Moral Robustness by Model",
        output_dir / "robustness.png",
    )


if __name__ == "__main__":
    main()
