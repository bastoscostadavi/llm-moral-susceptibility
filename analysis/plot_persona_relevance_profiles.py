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
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfq_questions import iter_questions
from plot_relevance_profiles import load_relevance_scores, summarise_scores

FOUNDATION_ORDER: List[str] = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "persona_moral_foundations_relevance_profiles.pdf"
DEFAULT_SAMPLE_SIZE = 14

# Reuse the same model allowlist as the self-assessment plots to keep figures aligned.
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
    "grok-4",
    "grok-4-fast",
}

SERIES_COLORS = [
    "#5E81B5",  # blue
    "#E19C24",  # orange
    "#8FB031",  # green
    "#EB6235",  # red
    "#8678B3",  # purple
    "#C46E1A",  # brown
    "#5D9DC7",  # light blue
    "#FFBF00",  # gold
    "#AD5F90",  # magenta
    "#6C7D47",  # olive
    "#E36B6B",  # GPT-5
    "#F29B9B",  # GPT-5-mini
    "#F8C8C8",  # GPT-5-nano
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
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.name.endswith("_self.csv"):
            continue

        model_slug = csv_path.stem
        if ALLOWED_MODELS and model_slug not in ALLOWED_MODELS:
            continue

        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "persona_id" not in reader.fieldnames:
                continue
            for row in reader:
                persona_raw = row.get("persona_id")
                if persona_raw is None:
                    continue
                try:
                    persona_ids.add(int(persona_raw))
                except (TypeError, ValueError):
                    continue

    return sorted(persona_ids)


def load_persona_scores(persona_filter: Set[int]) -> Dict[int, Dict[str, List[float]]]:
    """Collect per-foundation means for each persona by averaging across models and runs."""

    question_to_foundation = _foundation_questions()
    per_model_data: Dict[str, Dict[int, Dict[int, List[float]]]] = {}

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.name.endswith("_self.csv"):
            continue

        model_slug = csv_path.stem
        if ALLOWED_MODELS and model_slug not in ALLOWED_MODELS:
            continue

        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue

            required_fields = {"persona_id", "question_id", "rating", "run_index"}
            if required_fields.difference(reader.fieldnames):
                continue

            model_personas = per_model_data.setdefault(model_slug, {})

            for row in reader:
                persona_raw = row.get("persona_id")
                question_raw = row.get("question_id")
                rating_raw = row.get("rating")

                if persona_raw is None or question_raw is None or rating_raw is None:
                    continue

                try:
                    persona_id = int(persona_raw)
                except (TypeError, ValueError):
                    continue

                if persona_id not in persona_filter:
                    continue

                try:
                    question_id = int(question_raw)
                except (TypeError, ValueError):
                    continue

                if question_id not in question_to_foundation:
                    continue

                failures_raw = (row.get("failures") or "").strip()
                if failures_raw:
                    try:
                        if int(float(failures_raw)) > 0:
                            continue
                    except ValueError:
                        continue

                run_index_raw = row.get("run_index")
                if run_index_raw is None or not run_index_raw.strip():
                    continue

                try:
                    int(run_index_raw)
                except (TypeError, ValueError):
                    continue

                try:
                    rating = float(rating_raw)
                except (TypeError, ValueError):
                    continue

                persona_entries = model_personas.setdefault(persona_id, {})
                persona_entries.setdefault(question_id, []).append(rating)

    persona_scores: Dict[int, Dict[str, List[float]]] = {}

    for model_personas in per_model_data.values():
        for persona_id, question_scores in model_personas.items():
            foundation_question_means = {foundation: [] for foundation in FOUNDATION_ORDER}

            for question_id, scores in question_scores.items():
                if not scores:
                    continue

                foundation = question_to_foundation.get(question_id)
                if foundation is None:
                    continue

                foundation_question_means[foundation].append(mean(scores))

            if any(len(foundation_question_means[foundation]) == 0 for foundation in FOUNDATION_ORDER):
                continue

            persona_foundations = persona_scores.setdefault(
                persona_id,
                {foundation: [] for foundation in FOUNDATION_ORDER},
            )

            for foundation in FOUNDATION_ORDER:
                persona_foundations[foundation].append(
                    mean(foundation_question_means[foundation])
                )

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
    *,
    self_summary: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Path:
    """Generate a line plot for averaged persona foundation profiles."""

    if not persona_summaries:
        raise ValueError("No persona datasets were found for plotting.")

    sorted_personas = sorted(persona_summaries)
    x_positions = list(range(len(FOUNDATION_ORDER)))

    total_series = len(sorted_personas) + (1 if self_summary else 0)
    legend_cols = min(5, max(1, total_series))
    legend_rows = ceil(total_series / legend_cols)

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[14, max(legend_rows, 1)], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')

    if self_summary:
        means = [self_summary[foundation][0] for foundation in FOUNDATION_ORDER]
        errors = [self_summary[foundation][1] for foundation in FOUNDATION_ORDER]

        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            label="Self",
            color="#000000",
            linestyle="-",
            linewidth=3.0,
            marker="o",
            markersize=6,
            capsize=5,
        )

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
        "Average Moral Foundation Profile Across Models",
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
    pdf_path = output_path if output_path.suffix.lower() == ".pdf" else output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    return pdf_path


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

    self_scores = load_relevance_scores()
    self_model_summaries = summarise_scores(self_scores)
    self_summary: Optional[Dict[str, Tuple[float, float]]]
    if not self_model_summaries:
        self_summary = None
    else:
        aggregated: Dict[str, Tuple[float, float]] = {}
        for foundation in FOUNDATION_ORDER:
            means: List[float] = []
            ses: List[float] = []
            for model_summary in self_model_summaries.values():
                if foundation not in model_summary:
                    continue
                mean_val, se_val = model_summary[foundation]
                means.append(mean_val)
                ses.append(se_val)
            if not means:
                continue
            mean_avg = sum(means) / len(means)
            se_combined = sqrt(sum(se ** 2 for se in ses)) / len(ses)
            aggregated[foundation] = (mean_avg, se_combined)
        self_summary = aggregated if aggregated else None

    pdf_path = plot_persona_profiles(summaries, args.output, self_summary=self_summary)
    print(f"Sampled personas: {sampled_personas}")
    print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    main()
