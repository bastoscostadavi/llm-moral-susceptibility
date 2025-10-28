#!/usr/bin/env python3
"""Plot susceptibility and robustness metrics produced by the filtered pipeline."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("results2")

SUFFIX_PRIORITY = {
    "": 0,
    "mini": 1,
    "nano": 2,
    "fast": 3,
    "flash": 4,
    "lite": 5,
    "self": 6,
}


def model_sort_key(name: str) -> tuple[str, int, str]:
    match = re.match(r"^(.*?)(?:[-_](mini|nano|fast|flash|lite|self))?$", name)
    if match:
        base, suffix = match.groups()
        suffix = suffix or ""
        return base, SUFFIX_PRIORITY.get(suffix, 100), name
    return name, 0, name


def sort_by_model_order(frame: pd.DataFrame) -> pd.DataFrame:
    models = frame["model"].astype(str)
    order = {
        model: idx for idx, model in enumerate(sorted(models.unique(), key=model_sort_key))
    }
    ordered = frame.copy()
    ordered["__order"] = models.map(order)
    ordered = ordered.sort_values("__order").drop(columns=["__order"])
    return ordered.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        dest="metrics_csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "moral_metrics.csv",
        help="Path to the moral metrics CSV produced by compute_moral_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where plot images will be written (default: results2).",
    )
    parser.add_argument(
        "--metrics-by-foundation",
        dest="metrics_by_foundation_csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "moral_metrics_by_foundation.csv",
        help=(
            "Path to the per-foundation metrics CSV (default: results2/moral_metrics_by_foundation.csv)."
        ),
    )
    return parser.parse_args()


def ensure_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Metrics CSV missing columns: {missing_str}")


def _add_zscore_columns(
    frame: pd.DataFrame,
    value_col: str,
    error_col: str,
    z_value_col: str,
    z_error_col: str,
) -> None:
    values = frame[value_col].astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))

    if math.isclose(std, 0.0):
        frame[z_value_col] = 0.0
        if error_col in frame.columns:
            errs = frame[error_col].astype(float)
            frame[z_error_col] = 0.0 * errs
        else:
            frame[z_error_col] = np.nan
        return

    frame[z_value_col] = (values - mean) / std

    if error_col in frame.columns:
        errs = frame[error_col].astype(float)
        frame[z_error_col] = np.abs(errs / std)
    else:
        frame[z_error_col] = np.nan


def _format_value_with_uncertainty(value: float, err: float) -> tuple[str, int, float, float]:
    if err is None or (isinstance(err, float) and (math.isnan(err) or err <= 0)):
        return f"{value:.3f}", 3, round(value, 3), float("nan")

    def round_sig(x: float, sig: int = 1) -> float:
        if x == 0:
            return 0.0
        return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))

    err_rounded = round_sig(abs(err), 1)
    if err_rounded == 0:
        return f"{value:.3f}", 3, round(value, 3), 0.0

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
    zero_line: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    models = frame["model"]
    values = frame[value_col].astype(float).to_numpy()
    errors = frame[error_col].astype(float).to_numpy() if error_col in frame.columns else np.zeros_like(values)

    bars = ax.bar(models, values, yerr=errors, capsize=6, color="#4B8BBE")

    top = 0.0
    bottom = 0.0
    for v, e in zip(values, errors):
        e = 0.0 if (isinstance(e, float) and math.isnan(e)) else e
        top = max(top, v + max(e, 0.0))
        bottom = min(bottom, v - max(e, 0.0))

    if bottom >= 0.0:
        ymin = 0.0
        ymax = top * 1.15 + 1e-9
    elif top <= 0.0:
        ymax = 0.0
        ymin = bottom * 1.15 - 1e-9
    else:
        span = max(top - bottom, 1e-9)
        pad = span * 0.1
        ymin = bottom - pad
        ymax = top + pad

    if math.isclose(ymax, ymin):
        ymax = ymin + 1.0

    ax.set_ylim(ymin, ymax)

    if zero_line:
        ax.axhline(0.0, color="#333333", linewidth=1, linestyle="--", alpha=0.6)

    for bar, value, err in zip(bars, values, errors):
        err = 0.0 if (isinstance(err, float) and math.isnan(err)) else err
        label, _, _, _ = _format_value_with_uncertainty(value, err)
        if value >= 0:
            y = value + max(err, 0.0)
            offset = 6
            va = "bottom"
        else:
            y = value - max(err, 0.0)
            offset = -6
            va = "top"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
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
            "robustness",
            "r_uncertainty",
        },
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = sort_by_model_order(metrics.copy())
    _add_zscore_columns(
        metrics,
        "susceptibility",
        "s_uncertainty",
        "susceptibility_zscore",
        "s_z_uncertainty",
    )
    _add_zscore_columns(
        metrics,
        "robustness",
        "r_uncertainty",
        "robustness_zscore",
        "r_z_uncertainty",
    )

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
        "robustness",
        "r_uncertainty",
        "Robustness",
        "Moral Robustness by Model",
        output_dir / "robustness.png",
    )

    plot_metric(
        metrics,
        "susceptibility_zscore",
        "s_z_uncertainty",
        "Susceptibility z-score",
        "Moral Susceptibility z-score by Model",
        output_dir / "susceptibility_zscore.png",
        zero_line=True,
    )

    plot_metric(
        metrics,
        "robustness_zscore",
        "r_z_uncertainty",
        "Robustness z-score",
        "Moral Robustness z-score by Model",
        output_dir / "robustness_zscore.png",
        zero_line=True,
    )

    if args.metrics_by_foundation_csv.exists():
        by_fnd = pd.read_csv(args.metrics_by_foundation_csv)
        required = {
            "model",
            "foundation",
            "susceptibility",
            "s_uncertainty",
            "robustness",
            "r_uncertainty",
        }
        missing = required.difference(by_fnd.columns)
        if missing:
            raise ValueError(
                f"Per-foundation metrics CSV missing columns: {', '.join(sorted(missing))}"
            )

        import re as _re

        def _slug(s: str) -> str:
            return _re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()

        for fnd, frame in by_fnd.groupby("foundation"):
            slug = _slug(fnd)
            frame = sort_by_model_order(frame.copy())
            _add_zscore_columns(
                frame,
                "susceptibility",
                "s_uncertainty",
                "susceptibility_zscore",
                "s_z_uncertainty",
            )
            _add_zscore_columns(
                frame,
                "robustness",
                "r_uncertainty",
                "robustness_zscore",
                "r_z_uncertainty",
            )

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
                "robustness",
                "r_uncertainty",
                f"Robustness — {fnd}",
                f"Robustness — {fnd}",
                output_dir / f"robustness_{slug}.png",
            )

            plot_metric(
                frame,
                "susceptibility_zscore",
                "s_z_uncertainty",
                f"Susceptibility z-score — {fnd}",
                f"Moral Susceptibility z-score — {fnd}",
                output_dir / f"susceptibility_{slug}_zscore.png",
                zero_line=True,
            )

            plot_metric(
                frame,
                "robustness_zscore",
                "r_z_uncertainty",
                f"Robustness z-score — {fnd}",
                f"Moral Robustness z-score — {fnd}",
                output_dir / f"robustness_{slug}_zscore.png",
                zero_line=True,
            )


if __name__ == "__main__":
    main()
