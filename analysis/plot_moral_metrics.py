#!/usr/bin/env python3
"""Create bar plots for moral robustness and susceptibility (overall + foundations).

Requires the CSV outputs from compute_moral_metrics.py and writes twelve bar charts:
- overall robustness and susceptibility
- five foundation-specific robustness plots (one per foundation)
- five foundation-specific susceptibility plots (one per foundation)

All plots order models alphabetically on the x-axis and include uncertainty bars."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results") / "moral_metrics.csv",
        help="Path to overall moral metrics CSV (default: results/moral_metrics.csv).",
    )
    parser.add_argument(
        "--foundation-metrics",
        type=Path,
        default=Path("results") / "moral_metrics_per_foundation.csv",
        help=(
            "Path to foundation-level metrics CSV (default: results/moral_metrics_per_foundation.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where plot PNGs will be written (default: results).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots per inch for saved figures (default: 200).",
    )
    return parser.parse_args()


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_and_sort(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    ensure_columns(
        frame,
        required=[
            "model",
            "robustness",
            "robustness_uncertainty",
            "susceptibility",
            "susceptibility_uncertainty",
        ],
        label=str(path),
    )
    return frame.sort_values("model").reset_index(drop=True)


def slugify(value: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in value]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "foundation"


def format_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", color="#dddddd", linestyle="--", linewidth=0.7, alpha=0.7)


def format_value_with_uncertainty(value: float, err: float) -> str:
    if math.isnan(err) or err <= 0:
        return f"{value:.3f}"

    def round_sig(x: float, sig: int = 1) -> float:
        if x == 0:
            return 0.0
        return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))

    err_rounded = round_sig(abs(err), 1)
    if err_rounded == 0:
        return f"{value:.3f}"

    exp = int(math.floor(math.log10(err_rounded))) if err_rounded > 0 else 0
    decimals = max(0, -exp)
    value_r = round(value, decimals)
    return f"{value_r:.{decimals}f}±{err_rounded:.{decimals}f}"


def plot_metric(
    frame: pd.DataFrame,
    value_col: str,
    err_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    models = frame["model"].tolist()
    values = frame[value_col].astype(float).to_numpy()
    errors = frame[err_col].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, values, yerr=errors, capsize=5, color="#4B8BBE", edgecolor="#1F3056")
    format_axes(ax, title=title, ylabel=ylabel)

    # Expand y-limits to leave room for annotations
    current_ylim = ax.get_ylim()
    ymin, ymax = current_ylim
    top_padding = 0.1 * max(1.0, abs(ymax))
    bottom_padding = 0.1 * max(1.0, abs(ymin))
    ax.set_ylim(ymin - bottom_padding, ymax + top_padding)

    for bar, value, err in zip(bars, values, errors):
        label = format_value_with_uncertainty(value, err)
        err = abs(err)
        sign = 1 if value >= 0 else -1
        offset = 6 * sign
        y = value + sign * err
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def foundation_names(frame: pd.DataFrame) -> list[str]:
    if "foundation" not in frame.columns:
        raise ValueError("Foundation metrics CSV missing 'foundation' column.")
    return sorted(frame["foundation"].unique())


def main() -> None:
    args = parse_args()

    overall = load_and_sort(args.metrics)
    foundation_frame = pd.read_csv(args.foundation_metrics)
    ensure_columns(
        foundation_frame,
        required=[
            "model",
            "foundation",
            "robustness",
            "robustness_uncertainty",
            "susceptibility",
            "susceptibility_uncertainty",
        ],
        label=str(args.foundation_metrics),
    )

    overall_title_suffix = "(Overall)"
    plot_metric(
        overall,
        value_col="robustness",
        err_col="robustness_uncertainty",
        title=f"Moral Robustness {overall_title_suffix}",
        ylabel="Robustness",
        output_path=args.output_dir / "robustness_overall.png",
        dpi=args.dpi,
    )
    plot_metric(
        overall,
        value_col="susceptibility",
        err_col="susceptibility_uncertainty",
        title=f"Moral Susceptibility {overall_title_suffix}",
        ylabel="Susceptibility",
        output_path=args.output_dir / "susceptibility_overall.png",
        dpi=args.dpi,
    )

    foundations = foundation_names(foundation_frame)
    for foundation in foundations:
        subset = (
            foundation_frame[foundation_frame["foundation"] == foundation]
            .sort_values("model")
            .reset_index(drop=True)
        )
        slug = slugify(foundation)
        plot_metric(
            subset,
            value_col="robustness",
            err_col="robustness_uncertainty",
            title=f"Moral Robustness ({foundation})",
            ylabel="Robustness",
            output_path=args.output_dir / f"robustness_{slug}.png",
            dpi=args.dpi,
        )
        plot_metric(
            subset,
            value_col="susceptibility",
            err_col="susceptibility_uncertainty",
            title=f"Moral Susceptibility ({foundation})",
            ylabel="Susceptibility",
            output_path=args.output_dir / f"susceptibility_{slug}.png",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
