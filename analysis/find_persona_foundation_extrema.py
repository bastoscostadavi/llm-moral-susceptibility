#!/usr/bin/env python3
"""Identify personas that maximise each MFQ foundation when averaging across models."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import DefaultDict, Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mfq_questions import iter_questions

FOUNDATIONS = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]


def build_foundation_lookup() -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for question in iter_questions():
        if question.foundation:
            lookup[question.id] = question.foundation
    return lookup


def list_model_csvs(data_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in data_dir.glob("*.csv")
        if path.is_file() and not path.name.endswith("_self.csv")
    )


def load_model_persona_means(csv_path: Path, foundation_lookup: Dict[int, str]) -> Dict[int, Dict[str, float]]:
    per_persona: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"persona_id", "question_id", "rating"}
        if not reader.fieldnames or required.difference(reader.fieldnames):
            return {}

        for row in reader:
            try:
                persona_id = int(row["persona_id"])
                question_id = int(row["question_id"])
            except (TypeError, ValueError):
                continue

            foundation = foundation_lookup.get(question_id)
            if foundation is None:
                continue

            rating_raw = row.get("rating")
            if rating_raw is None:
                continue
            try:
                rating = float(rating_raw)
            except (TypeError, ValueError):
                continue
            if rating < 0:
                continue

            per_persona[persona_id][foundation].append(rating)

    means: Dict[int, Dict[str, float]] = {}
    for persona_id, foundation_scores in per_persona.items():
        persona_means: Dict[str, float] = {}
        for foundation, scores in foundation_scores.items():
            if scores:
                persona_means[foundation] = mean(scores)
        if persona_means:
            means[persona_id] = persona_means
    return means


def compute_persona_foundation_scores(data_dir: Path) -> Dict[int, Dict[str, float]]:
    foundation_lookup = build_foundation_lookup()
    model_files = list_model_csvs(data_dir)
    if not model_files:
        raise FileNotFoundError(f"No persona CSV files found in {data_dir}")

    aggregated: DefaultDict[int, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for csv_path in model_files:
        model_means = load_model_persona_means(csv_path, foundation_lookup)
        for persona_id, foundation_means in model_means.items():
            for foundation, value in foundation_means.items():
                aggregated[persona_id][foundation].append(value)

    persona_scores: Dict[int, Dict[str, float]] = {}
    for persona_id, foundation_lists in aggregated.items():
        persona_scores[persona_id] = {
            foundation: mean(values)
            for foundation, values in foundation_lists.items()
            if values
        }
    return persona_scores


def find_max_per_foundation(persona_scores: Dict[int, Dict[str, float]]):
    results = {}
    for foundation in FOUNDATIONS:
        best_persona = None
        best_score = float("-inf")
        for persona_id, foundation_scores in persona_scores.items():
            score = foundation_scores.get(foundation)
            if score is None:
                continue
            if score > best_score or (score == best_score and (best_persona is None or persona_id < best_persona)):
                best_persona = persona_id
                best_score = score
        if best_persona is not None:
            results[foundation] = (best_persona, best_score)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing persona CSV files (default: data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    persona_scores = compute_persona_foundation_scores(args.data_dir)
    if not persona_scores:
        print("No persona scores available.")
        return

    results = find_max_per_foundation(persona_scores)
    if not results:
        print("No maxima found.")
        return

    print("Persona maxima per foundation (averaged across models):")
    for foundation in FOUNDATIONS:
        entry = results.get(foundation)
        if entry is None:
            print(f"- {foundation}: no data")
            continue
        persona_id, score = entry
        print(f"- {foundation}: persona {persona_id} with mean score {score:.3f}")


if __name__ == "__main__":
    main()
