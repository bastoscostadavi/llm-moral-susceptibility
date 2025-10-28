#!/usr/bin/env python3
"""Plot averaged MFQ moral foundation profiles for a random persona sample across models."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from math import ceil, sqrt
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "persona_moral_foundations_relevance_profiles.png"
DEFAULT_SAMPLE_SIZE = 5

SERIES_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SERIES_LINESTYLES = ["-", "--", "-.", ":"]


def _foundation_questions() -> Dict[int, str]:
    """Map MFQ item ids (relevance and agreement) to their foundations."""

    mapping: Dict[int, str] = {}
    for question in iter_questions():
        if question.question_type in {"relevance", "agreement"} and question.foundation:
            mapping[question.id] = question.foundation
    return mapping


def available_personas() -> List[int]:
    """Discover persona ids present in the results CSV files."""

    persona_ids: Set[int] = set()
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        if csv_path.name.startswith("moral_"):
            continue

        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "persona" not in reader.fieldnames:
                continue
            for row in reader:
                try:
                    persona_ids.add(int(row["persona"]))
                except (TypeError, ValueError):
                    continue

    return sorted(persona_ids)


def load_persona_scores(persona_filter: Set[int] | None = None) -> Dict[int, Dict[str, List[float]]]:
    """Aggregate per-foundation mean scores across models for selected personas."""

    question_to_foundation = _foundation_questions()
    persona_scores: Dict[int, Dict[str, List[float]]] = {}

    def empty_foundation_dict() -> Dict[str, List[float]]:
        return {foundation: [] for foundation in FOUNDATION_ORDER}

    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        if csv_path.name.startswith("moral_"):
            continue

        per_persona: Dict[int, Dict[str, List[float]]] = {}
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            required_fields = {"persona", "question", "average_score"}
            if not required_fields.issubset(reader.fieldnames):
                continue

            for row in reader:
                try:
                    persona_id = int(row["persona"])
                    question_id = int(row["question"])
                    score = float(row["average_score"])
                except (TypeError, ValueError):
                    continue

                if persona_filter is not None and persona_id not in persona_filter:
                    continue

                foundation = question_to_foundation.get(question_id)
                if foundation is None:
                    continue

                foundation_values = per_persona.setdefault(
                    persona_id,
                    {name: [] for name in FOUNDATION_ORDER},
                )
                foundation_values[foundation].append(score)

        for persona_id, foundation_values in per_persona.items():
            if persona_filter is not None and persona_id not in persona_filter:
                continue
            if any(len(foundation_values[foundation]) == 0 for foundation in FOUNDATION_ORDER):
                continue

            persona_summary = persona_scores.setdefault(persona_id, empty_foundation_dict())
            for foundation in FOUNDATION_ORDER:
                persona_summary[foundation].append(mean(foundation_values[foundation]))

    return {
        persona: foundation_scores
        for persona, foundation_scores in persona_scores.items()
        if all(len(values) > 0 for values in foundation_scores.values())
    }


def summarise_persona_scores(
    scores_by_persona: Dict[int, Dict[str, List[float]]]
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Compute mean scores and standard errors per foundation for each persona."""

    summaries: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for persona, foundation_scores in sorted(scores_by_persona.items()):
        foundation_summary: Dict[str, Tuple[float, float]] = {}
        for foundation in FOUNDATION_ORDER:
            values = foundation_scores[foundation]
            mean_score = mean(values)
            se_score = stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0
            foundation_summary[foundation] = (mean_score, se_score)

        summaries[persona] = foundation_summary
    return summaries


def plot_persona_profiles(
    persona_summaries: Dict[int, Dict[str, Tuple[float, float]]],
    output_path: Path,
) -> None:
    """Generate a line plot for averaged persona foundation profiles."""

    if not persona_summaries:
        raise ValueError("No persona datasets were found for plotting.")

    sorted_personas = sorted(persona_summaries)
    x_positions = list(range(len(FOUNDATION_ORDER)))

    legend_cols = min(5, max(1, len(sorted_personas)))
    legend_rows = ceil(len(sorted_personas) / legend_cols)

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[14, max(legend_rows, 1)], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')

    for idx, persona in enumerate(sorted_personas):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
        linestyle = SERIES_LINESTYLES[idx % len(SERIES_LINESTYLES)]
        foundation_summary = persona_summaries[persona]
        means = [foundation_summary[foundation][0] for foundation in FOUNDATION_ORDER]
        errors = [foundation_summary[foundation][1] for foundation in FOUNDATION_ORDER]

        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            label=f"Persona {persona}",
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
    ax.set_title(
        "Average Moral Foundation Profile Across Models (Random Personas)",
        fontsize=20,
    )
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
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged MFQ moral foundation profiles for a random sample of personas.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of personas to include in the plot (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed controlling which personas are sampled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    persona_ids = available_personas()
    if not persona_ids:
        raise ValueError("No persona data found in results directory.")

    if args.sample_size < 1:
        raise ValueError("Sample size must be at least 1.")
    if args.sample_size > len(persona_ids):
        raise ValueError(
            f"Sample size {args.sample_size} exceeds available personas ({len(persona_ids)})."
        )

    rng = Random(args.seed)
    sampled_personas = sorted(rng.sample(persona_ids, args.sample_size))

    scores = load_persona_scores(set(sampled_personas))
    summaries = summarise_persona_scores(scores)
    if not summaries:
        raise ValueError("No personas with complete data were selected for plotting.")

    missing_personas = [persona for persona in sampled_personas if persona not in summaries]
    if missing_personas:
        print(
            f"Excluded personas without complete data: {missing_personas}",
            file=sys.stderr,
        )

    plot_persona_profiles(summaries, args.output)
    print(f"Sampled personas: {sampled_personas}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
