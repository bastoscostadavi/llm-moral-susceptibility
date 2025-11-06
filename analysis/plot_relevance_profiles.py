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
OUTPUT_PATH = RESULTS_DIR / "moral_foundations_relevance_profiles.pdf"

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
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "deepseek-chat-v3.1",
    "llama-4-maverick",
    "llama-4-scout",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "openrouter-gpt-5",
    "grok-4",
    "grok-4-fast",
}

MODEL_COLORS = {
    "claude-haiku-4-5": "#F9C784",
    "claude-sonnet-4-5": "#E67E22",
    "gpt-4.1-nano": "#D9F0D3",
    "gpt-4o-mini": "#A6DBA0",
    "gpt-4.1": "#52B788",
    "grok-4": "#BDA0E3",
    "gpt-4o": "#2F855A",
    "gpt-4.1-mini": "#74C69D",
    "gemini-2.5-flash": "#F2D16B",
    "gemini-2.5-flash-lite": "#F9E69F",
    "grok-4-fast": "#7E57C2",
    "gpt-5": "#E36B6B",
    "gpt-5-mini": "#F29B9B",
    "gpt-5-nano": "#F8C8C8",
    "openrouter-gpt-5": "#E36B6B",
    "deepseek-chat-v3.1": "#5B4B8A",
    "llama-4-maverick": "#4A90E2",
    "llama-4-scout": "#4A90E2",
}

MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MODEL_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_COLOR_CYCLE = plt.rcParams.get('axes.prop_cycle', None)
if DEFAULT_COLOR_CYCLE is not None:
    DEFAULT_COLOR_CYCLE = DEFAULT_COLOR_CYCLE.by_key().get('color', ["#1f77b4"])
else:
    DEFAULT_COLOR_CYCLE = ["#1f77b4"]


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
        question_scores: Dict[int, List[float]] = {}

        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required_headers = {"question_id", "rating", "run_index"}
            if not reader.fieldnames or required_headers.difference(reader.fieldnames):
                continue

            for row in reader:
                try:
                    question_id = int(row["question_id"])
                except (TypeError, ValueError):
                    continue

                foundation = question_to_foundation.get(question_id)
                if foundation is None:
                    continue

                run_index_raw = row.get("run_index")
                if run_index_raw is None:
                    continue

                run_index_raw = run_index_raw.strip()
                if not run_index_raw:
                    continue

                try:
                    int(run_index_raw)
                except (TypeError, ValueError):
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

                question_scores.setdefault(question_id, []).append(score)

        if not question_scores:
            continue

        foundation_scores = {foundation: [] for foundation in FOUNDATION_ORDER}
        for question_id, responses in question_scores.items():
            if not responses:
                continue

            foundation = question_to_foundation.get(question_id)
            if foundation is None:
                continue

            foundation_scores[foundation].append(mean(responses))

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


def plot_profiles(model_summaries: Dict[str, Dict[str, Tuple[float, float]]]) -> Path:
    """Generate a Haidt-style line plot with error bars using Matplotlib."""

    if not model_summaries:
        raise ValueError("No complete self-assessment datasets were found.")

    sorted_models = sorted(model_summaries)
    style_map = {}
    for idx, model in enumerate(sorted_models):
        display_name = model.replace("openrouter-", "")
        color = MODEL_COLORS.get(model)
        if color is None:
            fallback_palette = DEFAULT_COLOR_CYCLE or ["#1f77b4"]
            color = fallback_palette[idx % len(fallback_palette)]
        marker = MODEL_MARKERS[idx % len(MODEL_MARKERS)]
        linestyle = MODEL_LINESTYLES[idx % len(MODEL_LINESTYLES)]
        style_map[model] = (color, marker, linestyle, display_name)

    x_positions = list(range(len(FOUNDATION_ORDER)))

    legend_cols = min(5, max(1, len(sorted_models)))
    legend_rows = ceil(len(sorted_models) / legend_cols)

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[14, max(legend_rows, 1)], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')

    for model in sorted_models:
        color, marker, linestyle, display_name = style_map[model]
        foundation_summary = model_summaries[model]
        means = [foundation_summary[foundation][0] for foundation in FOUNDATION_ORDER]
        errors = [foundation_summary[foundation][1] for foundation in FOUNDATION_ORDER]

        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            label=display_name,
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
    pdf_path = OUTPUT_PATH if OUTPUT_PATH.suffix.lower() == ".pdf" else OUTPUT_PATH.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    return pdf_path


def main() -> None:
    scores = load_relevance_scores()
    summaries = summarise_scores(scores)
    pdf_path = plot_profiles(summaries)
    print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    main()
