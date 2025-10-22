#!/usr/bin/env python3
"""Compute robustness and moral susceptibility from persona-question summaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mfq_questions import MFQ_QUESTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        dest="summary_csv",
        type=Path,
        required=True,
        help="Persona-question summary CSV produced by build_persona_question_summary.py.",
    )
    parser.add_argument(
        "--metrics",
        dest="metrics_csv",
        type=Path,
        default=Path("results") / "moral_metrics.csv",
        help="Path to the aggregated metrics CSV (default: results/moral_metrics.csv).",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        default=None,
        help="Optional explicit model name overriding the inferred one.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=10,
        help="Number of personas per susceptibility bootstrap group.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="Number of persona groups to evaluate for susceptibility.",
    )
    return parser.parse_args()


def chunk(items: Iterable[int], size: int) -> List[List[int]]:
    items_list = list(items)
    return [items_list[i : i + size] for i in range(0, len(items_list), size)]


def valid_question_ids() -> List[int]:
    return [q.id for q in MFQ_QUESTIONS if q.foundation and q.foundation.lower() != "useless"]


def infer_model_name(summary_path: Path) -> str:
    stem = summary_path.stem
    stem = re.sub(r"\s+", "_", stem)
    return stem


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    summary = pd.read_csv(args.summary_csv)
    expected_cols = {"persona", "question", "average_score", "uncertainty", "relative_score"}
    missing = expected_cols.difference(summary.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Summary CSV missing columns: {missing_str}")

    summary = summary.copy()
    summary["persona"] = summary["persona"].astype(int)
    summary["question"] = summary["question"].astype(int)

    uncertainties = summary["uncertainty"].to_numpy()
    robustness = float(np.mean(uncertainties))
    robustness_uncertainty = (
        float(np.std(uncertainties, ddof=1))/len(uncertainties) if len(uncertainties) > 1 else 0.0
    )

    valid_questions = set(valid_question_ids())
    filtered = summary[summary["question"].isin(valid_questions)]
    if filtered.empty:
        raise ValueError("No valid questions remain after filtering foundations.")

    pivot = filtered.pivot(index="persona", columns="question", values="average_score").sort_index()
    if pivot.isnull().any().any():
        missing = pivot.isnull().sum().sum()
        raise ValueError(f"Summary contains {int(missing)} missing average_score values.")

    relative_pivot = filtered.pivot(index="persona", columns="question", values="relative_score").sort_index()
    if relative_pivot.isnull().any().any():
        missing = relative_pivot.isnull().sum().sum()
        raise ValueError(f"Summary contains {int(missing)} missing relative_score values.")

    persona_ids = list(pivot.index)
    if not persona_ids:
        raise ValueError("No personas available for susceptibility computation.")

    expected_personas = args.group_size * args.num_groups
    if len(persona_ids) != expected_personas:
        raise ValueError(
            "Persona count does not match group configuration: "
            f"{len(persona_ids)} personas vs {expected_personas} expected"
        )

    groups = chunk(persona_ids, args.group_size)
    if len(groups) != args.num_groups:
        raise ValueError(
            f"Expected {args.num_groups} groups but formed {len(groups)} from persona ids"
        )

    susceptibility_samples: List[float] = []
    relative_samples: List[float] = []
    for group in groups:
        block = pivot.loc[group]
        per_question_std = block.std(axis=0, ddof=1)
        susceptibility_samples.append(float(per_question_std.mean()))

        rel_block = relative_pivot.loc[group]
        rel_per_question_std = rel_block.std(axis=0, ddof=1)
        relative_samples.append(float(rel_per_question_std.mean()))

    susceptibility = float(np.mean(susceptibility_samples))
    susceptibility_uncertainty = (
        float(np.std(susceptibility_samples, ddof=1)) if len(susceptibility_samples) > 1 else 0.0
    )

    relative_susceptibility = float(np.mean(relative_samples))
    relative_uncertainty = (
        float(np.std(relative_samples, ddof=1)) if len(relative_samples) > 1 else 0.0
    )

    metrics_columns = [
        "model",
        "susceptibility",
        "s_uncertainty",
        "relative_susceptibility",
        "rs_uncertainty",
        "robustness",
        "r_uncertainty",
    ]

    model_name = args.model_name or infer_model_name(args.summary_csv)
    new_row = pd.DataFrame(
        {
            "model": [model_name],
            "susceptibility": [susceptibility],
            "s_uncertainty": [susceptibility_uncertainty],
            "relative_susceptibility": [relative_susceptibility],
            "rs_uncertainty": [relative_uncertainty],
            "robustness": [robustness],
            "r_uncertainty": [robustness_uncertainty],
        }
    )

    metrics_path = args.metrics_csv
    ensure_parent(metrics_path)

    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        for col in metrics_columns:
            if col not in existing.columns:
                existing[col] = np.nan
        existing = existing[metrics_columns]
        if "model" in existing.columns:
            existing = existing[existing["model"] != model_name]
        metrics = pd.concat([existing, new_row], ignore_index=True)
    else:
        metrics = new_row

    metrics = metrics[metrics_columns]
    metrics.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    main()
