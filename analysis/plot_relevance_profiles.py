#!/usr/bin/env python3
"""Plot MFQ relevance profiles for no-persona model self-assessments using Matplotlib."""

from __future__ import annotations

import csv
import os
import sys
from math import sqrt
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

from mfq_questions import get_question

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


def _relevance_foundations() -> Dict[int, str]:
    """Map relevance question ids to the associated moral foundation."""

    mapping: Dict[int, str] = {}
    for question_id in range(1, 16):
        question = get_question(question_id)
        if question.question_type == "relevance":
            mapping[question_id] = question.foundation
    return mapping


def load_relevance_scores() -> Dict[str, Dict[str, List[float]]]:
    """Collect raw relevance scores from each *_self.csv file."""

    question_to_foundation = _relevance_foundations()
    scores_by_model: Dict[str, Dict[str, List[float]]] = {}

    for csv_path in sorted(DATA_DIR.glob("*_self.csv")):
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "question_id" not in reader.fieldnames or "rating" not in reader.fieldnames:
                continue

            foundation_scores = {foundation: [] for foundation in FOUNDATION_ORDER}
            for row in reader:
                foundation = question_to_foundation.get(int(row["question_id"]))
                if foundation is None:
                    continue
                try:
                    score = float(row["rating"])
                except ValueError:
                    continue
                foundation_scores[foundation].append(score)

        if any(len(values) == 0 for values in foundation_scores.values()):
            continue

        model_name = csv_path.stem.replace("_self", "")
        scores_by_model[model_name] = foundation_scores

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
    color_map = {model: MODEL_COLORS[idx % len(MODEL_COLORS)] for idx, model in enumerate(sorted_models)}

    x_positions = list(range(len(FOUNDATION_ORDER)))

    fig, ax = plt.subplots(figsize=(11, 6))
    

    for model in sorted_models:
        color = color_map[model]
        foundation_summary = model_summaries[model]
        means = [foundation_summary[foundation][0] for foundation in FOUNDATION_ORDER]
        errors = [foundation_summary[foundation][1] for foundation in FOUNDATION_ORDER]

        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            label=model,
            color=color,
            linewidth=2.5,
            marker="o",
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

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.5), fontsize=14)

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
