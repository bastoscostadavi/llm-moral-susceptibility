#!/usr/bin/env python3
"""Plot MFQ moral foundation profiles for no-persona self-assessments using Matplotlib."""

from __future__ import annotations

import csv
import os
import sys
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure Matplotlib to use a repository-local config directory and Agg backend
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfq_questions import iter_questions

FOUNDATION_ORDER: List[str] = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "moral_foundations_relevance_profiles.png"

# Restrict plot to a fixed subset of models. Keep both possible Gemini slugs
# to accommodate dataset naming; only existing ones will be included.
ALLOWED_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "grok-4",
    "grok-4-fast",
    "gemini-2.5-flash-lite",
}

MODEL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MODEL_LINESTYLES = ["-", "--", "-.", ":"]


def _foundation_questions() -> Dict[int, str]:
    """Map MFQ item ids (relevance and agreement) to their foundations."""

    mapping: Dict[int, str] = {}
    for question in iter_questions():
        if question.question_type in {"relevance", "agreement"} and question.foundation:
            mapping[question.id] = question.foundation
    return mapping


def load_relevance_scores() -> Dict[str, Dict[str, List[float]]]:
    """Collect raw MFQ scores (relevance and agreement) from each *_self.csv file."""

    question_to_foundation = _foundation_questions()
    scores_by_model: Dict[str, Dict[str, List[float]]] = {}

    for csv_path in sorted(DATA_DIR.glob("*_self.csv")):
        model_slug = csv_path.stem.replace("_self", "")
        if ALLOWED_MODELS and model_slug not in ALLOWED_MODELS:
            continue
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "question_id" not in reader.fieldnames or "rating" not in reader.fieldnames:
                continue

            foundation_scores = {foundation: [] for foundation in FOUNDATION_ORDER}
            for row in reader:
                try:
                    question_id = int(row["question_id"])
                except (TypeError, ValueError):
                    continue

                foundation = question_to_foundation.get(question_id)
                if foundation is None:
                    continue

                failures_raw = (row.get("failures") or "").strip()
                if failures_raw:
                    try:
                        if int(float(failures_raw)) > 0:
                            continue
                    except ValueError:
                        continue

                try:
                    score = float(row["rating"])
                except ValueError:
                    continue
                foundation_scores[foundation].append(score)

        if any(len(values) == 0 for values in foundation_scores.values()):
            continue

        scores_by_model[model_slug] = foundation_scores

    return scores_by_model


def summarise_scores(
    scores_by_model: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute mean scores and standard errors per foundation for each model."""

    summaries: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for model, foundation_scores in scores_by_model.items():
        model_summary: Dict[str, Tuple[float, float]] = {}
        for foundation in FOUNDATION_ORDER:
            values = foundation_scores[foundation]
            n = len(values)
            mean_score = mean(values)
            se_score = stdev(values) / sqrt(n) if n > 1 else 0.0
            model_summary[foundation] = (mean_score, se_score)
        summaries[model] = model_summary
    return summaries


def _pairwise(points: Iterable[Tuple[float, float]]) -> Iterable[Tuple[Tuple[float, float], Tuple[float, float]]]:
    point_list = list(points)
    for idx in range(len(point_list) - 1):
        yield point_list[idx], point_list[idx + 1]


def plot_profiles(model_summaries: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    """Generate a Haidt-style line plot with error bars using Matplotlib."""

    if not model_summaries:
        raise ValueError("No complete self-assessment datasets were found.")

    sorted_models = sorted(model_summaries)
    style_map = {}
    for idx, model in enumerate(sorted_models):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        marker = MODEL_MARKERS[idx % len(MODEL_MARKERS)]
        linestyle = MODEL_LINESTYLES[idx % len(MODEL_LINESTYLES)]
        style_map[model] = (color, marker, linestyle)

    x_positions = list(range(len(FOUNDATION_ORDER)))

    legend_cols = min(5, max(1, len(sorted_models)))
    legend_rows = ceil(len(sorted_models) / legend_cols)

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[14, max(legend_rows, 1)], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')

    for model in sorted_models:
        color, marker, linestyle = style_map[model]
        foundation_summary = model_summaries[model]
        means = [foundation_summary[foundation][0] for foundation in FOUNDATION_ORDER]
        errors = [foundation_summary[foundation][1] for foundation in FOUNDATION_ORDER]

        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            label=model,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker=marker,
            markersize=7,
            capsize=5,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(FOUNDATION_ORDER, rotation=15, ha="right")
    ax.tick_params(axis="both", labelsize=14) 
    ax.set_ylim(0, 5.5)
    ax.set_xlim(-0.2, len(FOUNDATION_ORDER) - 0.8)
    ax.set_ylabel("Relevance to Moral Decisions", fontsize=16)
    ax.set_title("Moral Foundation Profile", fontsize=20)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    legend = legend_ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center",
        ncol=legend_cols,
        fontsize=14,
        columnspacing=1.2,
        handlelength=2.0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.0,
    )
    legend._legend_box.align = "left"

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    scores = load_relevance_scores()
    summaries = summarise_scores(scores)
    plot_profiles(summaries)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
