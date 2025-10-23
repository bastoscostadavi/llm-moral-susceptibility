#!/usr/bin/env python3
"""Generate bar plots for susceptibility, relative susceptibility, and robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

import math
import numpy as np
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
    parser.add_argument(
        "--metrics-by-foundation",
        dest="metrics_by_foundation_csv",
        type=Path,
        default=Path("results") / "moral_metrics_by_foundation.csv",
        help="Path to the per-foundation metrics CSV (default: results/moral_metrics_by_foundation.csv).",
    )
    return parser.parse_args()


def ensure_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Metrics CSV missing columns: {missing_str}")


def _format_value_with_uncertainty(value: float, err: float) -> tuple[str, int, float, float]:
    """Return a label string with 1-sig-fig uncertainty and matching value rounding.

    Returns (label, decimals, value_rounded, err_rounded).
    """
    if err is None or (isinstance(err, float) and (math.isnan(err) or err <= 0)):
        # Fallback: show value with 3 decimals, no uncertainty
        return f"{value:.3f}", 3, round(value, 3), float('nan')

    # Round uncertainty to 1 significant digit
    def round_sig(x: float, sig: int = 1) -> float:
        if x == 0:
            return 0.0
        return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))

    err_rounded = round_sig(abs(err), 1)
    if err_rounded == 0:
        return f"{value:.3f}", 3, round(value, 3), 0.0

    # Determine decimal places implied by err_rounded
    exp = int(math.floor(math.log10(err_rounded))) if err_rounded > 0 else 0
    decimals = max(0, -exp)
    value_rounded = round(value, decimals)

    fmt = f"{{:.{decimals}f}}±{{:.{decimals}f}}"
    label = fmt.format(value_rounded, err_rounded)
    return label, decimals, value_rounded, err_rounded


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
    values = frame[value_col].astype(float).to_numpy()
    errors = frame[error_col].astype(float).to_numpy() if error_col in frame.columns else np.zeros_like(values)

    bars = ax.bar(models, values, yerr=errors, capsize=6, color="#4B8BBE")

    # Set ylim to accommodate error bars and labels
    top = 0.0
    for v, e in zip(values, errors):
        e = 0.0 if (isinstance(e, float) and math.isnan(e)) else e
        top = max(top, v + max(e, 0.0))
    ax.set_ylim(0, top * 1.15 + 1e-9)

    # Annotate above error bars with formatted value±uncertainty
    for bar, value, err in zip(bars, values, errors):
        err = 0.0 if (isinstance(err, float) and math.isnan(err)) else err
        label, decimals, v_round, e_round = _format_value_with_uncertainty(value, err)
        y = value + max(err, 0.0)
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
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

    # Per-foundation plots (if available)
    if args.metrics_by_foundation_csv.exists():
        by_fnd = pd.read_csv(args.metrics_by_foundation_csv)
        required = {
            "model",
            "foundation",
            "susceptibility",
            "s_uncertainty",
            "relative_susceptibility",
            "rs_uncertainty",
            "robustness",
            "r_uncertainty",
        }
        missing = required.difference(by_fnd.columns)
        if missing:
            raise ValueError(
                f"Per-foundation metrics CSV missing columns: {', '.join(sorted(missing))}"
            )

        # Plot per foundation
        import re as _re

        def _slug(s: str) -> str:
            return _re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()

        for fnd, frame in by_fnd.groupby("foundation"):
            slug = _slug(fnd)

            plot_metric(
                frame,
                "susceptibility",
                "s_uncertainty",
                f"Susceptibility — {fnd}",
                f"Moral Susceptibility — {fnd}",
                output_dir / f"susceptibility_{slug}.png",
            )

            plot_metric(
                frame,
                "relative_susceptibility",
                "rs_uncertainty",
                f"Relative Susceptibility — {fnd}",
                f"Relative Susceptibility — {fnd}",
                output_dir / f"relative_susceptibility_{slug}.png",
            )

            plot_metric(
                frame,
                "robustness",
                "r_uncertainty",
                f"Robustness — {fnd}",
                f"Robustness — {fnd}",
                output_dir / f"robustness_{slug}.png",
            )


if __name__ == "__main__":
    main()
