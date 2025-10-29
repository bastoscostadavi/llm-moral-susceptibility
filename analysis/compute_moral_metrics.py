#!/usr/bin/env python3
"""Compute moral robustness and susceptibility using shared valid personas.

The script scans summary CSVs (default: results/*.csv) produced by
build_persona_question_summary, determines the personas that have valid
(`average_score`, `standard_deviation`) entries for every model, optionally
removing one persona to avoid a prime count, and then computes the moral
robustness and susceptibility metrics described in articles/mlsys2025.tex.

Outputs a CSV with four metric columns (value + uncertainty for each
capability) and one row per model ordered alphabetically."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mfq_questions import MFQ_QUESTIONS


@dataclass
class SummaryData:
    model: str
    path: Path
    frame: pd.DataFrame
    questions: set[int]


TARGET_GROUP_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing persona-question summary CSVs (default: results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "moral_metrics.csv",
        help="Destination CSV for model-level metrics (default: results/moral_metrics.csv).",
    )
    parser.add_argument(
        "--foundation-output",
        type=Path,
        default=Path("results") / "moral_metrics_per_foundation.csv",
        help=(
            "Destination CSV for foundation-level metrics (default: results/moral_metrics_per_foundation.csv)."
        ),
    )
    parser.add_argument(
        "--groups-output",
        type=Path,
        default=Path("results") / "persona_groups.csv",
        help=(
            "Destination CSV describing persona group assignments (default: results/persona_groups.csv)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit informational messages about persona selection and grouping.",
    )
    return parser.parse_args()


def iter_summary_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Summaries directory not found: {directory}")
    for path in sorted(directory.glob("*.csv")):
        if path.name.startswith("moral_metrics"):
            continue
        if path.name.endswith("_self.csv"):
            continue
        if path.is_file():
            yield path


def load_summaries(directory: Path) -> List[SummaryData]:
    summaries: List[SummaryData] = []
    for csv_path in iter_summary_files(directory):
        df = pd.read_csv(csv_path)
        required = {"persona_id", "question_id", "average_score", "standard_deviation"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {', '.join(sorted(missing))}")
        df = df.copy()
        df["persona_id"] = df["persona_id"].astype(int)
        df["question_id"] = df["question_id"].astype(int)
        questions = set(df["question_id"].unique())
        summaries.append(
            SummaryData(
                model=csv_path.stem,
                path=csv_path,
                frame=df,
                questions=questions,
            )
        )
    if not summaries:
        raise RuntimeError("No summary CSVs found in the specified directory.")
    return summaries


def intersect_questions(summaries: Sequence[SummaryData]) -> List[int]:
    question_sets = [s.questions for s in summaries]
    common = set.intersection(*question_sets)
    if not common:
        raise RuntimeError("No common question IDs across summaries.")
    return sorted(common)


def personas_with_valid_stats(summary: SummaryData, questions: Sequence[int]) -> tuple[set[int], set[int]]:
    q_set = set(questions)
    subset = summary.frame[summary.frame["question_id"].isin(q_set)].copy()
    counts = subset.groupby("persona_id")["question_id"].nunique()
    complete_personas = {pid for pid, cnt in counts.items() if cnt == len(q_set)}
    if not complete_personas:
        return set(), set()
    valid_mask = (
        subset["average_score"].notna()
        & subset["standard_deviation"].notna()
        & (subset["average_score"] != -1)
        & (subset["standard_deviation"] != -1)
    )
    invalid_personas = set(subset.loc[~valid_mask, "persona_id"].unique())
    valid_personas = complete_personas.difference(invalid_personas)
    return complete_personas, valid_personas


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n)) + 1
    for candidate in range(3, limit, 2):
        if n % candidate == 0:
            return False
    return True


def choose_grouping(num_personas: int, target_size: int = TARGET_GROUP_SIZE) -> tuple[int, int] | None:
    candidates: List[tuple[float, int, int]] = []
    for groups in range(2, num_personas // 2 + 1):
        if num_personas % groups != 0:
            continue
        group_size = num_personas // groups
        if group_size < 2:
            continue
        score = abs(group_size - target_size)
        candidates.append((score, groups, group_size))
    if not candidates:
        return None
    candidates.sort()
    _, best_groups, best_size = candidates[0]
    return best_groups, best_size


def build_groups(personas: Sequence[int], verbose: bool = False) -> tuple[List[List[int]], List[int]]:
    ordered = sorted(personas)
    removed: List[int] = []
    attempt = 0
    while True:
        count = len(ordered)
        if count < 4:
            raise RuntimeError(
                f"Need at least 4 personas to form equal-size groups with size >=2; have {count}."
            )
        grouping = choose_grouping(count)
        if grouping is not None:
            groups, group_size = grouping
            break
        if not is_prime(count):
            raise RuntimeError(
                "Unable to partition personas into equal-size groups (non-prime size but no valid grouping)."
            )
        if attempt >= 1:
            raise RuntimeError("Dropping one persona did not resolve prime-size grouping.")
        # Remove the highest-index persona to break primality
        removed_persona = ordered.pop()
        removed.append(removed_persona)
        attempt += 1
    if verbose and removed:
        print(
            f"Removed personas to enable grouping: {', '.join(map(str, removed))}",
        )
    groups_list = [ordered[i : i + group_size] for i in range(0, len(ordered), group_size)]
    if verbose:
        print(
            f"Formed {len(groups_list)} groups of size {group_size} from {len(ordered)} personas.",
        )
    return groups_list, removed


def compute_robustness(std_values: np.ndarray) -> tuple[float, float]:
    if std_values.size == 0:
        raise RuntimeError("No standard deviation values available for robustness computation.")
    mean_unc = float(np.mean(std_values))
    if std_values.size > 1:
        sd_unc = float(np.std(std_values, ddof=1))
        se_unc = sd_unc / math.sqrt(std_values.size)
    else:
        se_unc = 0.0
    eps = 1e-9
    if mean_unc <= 0:
        robustness = float("inf")
        robustness_se = 0.0
    else:
        robustness = 1.0 / mean_unc
        robustness_se = se_unc / (mean_unc ** 2) if se_unc > 0 else 0.0
    return robustness, robustness_se


def compute_susceptibility(pivot: pd.DataFrame, groups: Sequence[Sequence[int]]) -> tuple[float, float]:
    samples: List[float] = []
    for group in groups:
        block = pivot.loc[list(group)]
        per_question_std = block.std(axis=0, ddof=1)
        if per_question_std.isna().any():
            raise RuntimeError("Encountered NaN susceptibility standard deviation for a group.")
        samples.append(float(per_question_std.mean()))
    susceptibility = float(np.mean(samples))
    if len(samples) > 1:
        susceptibility_se = float(np.std(samples, ddof=1) / math.sqrt(len(samples)))
    else:
        susceptibility_se = 0.0
    return susceptibility, susceptibility_se


def main() -> None:
    args = parse_args()

    summaries = load_summaries(args.summaries_dir)
    common_questions = intersect_questions(summaries)
    foundation_map = {
        q.id: q.foundation
        for q in MFQ_QUESTIONS
        if q.id in common_questions and q.foundation is not None
    }
    if not foundation_map:
        raise RuntimeError("No foundation metadata available for the shared questions.")
    missing_foundations = sorted(set(common_questions).difference(set(foundation_map)))
    if missing_foundations:
        raise RuntimeError(
            "Missing foundation labels for questions: " + 
            ", ".join(map(str, missing_foundations))
        )
    foundations = sorted(set(foundation_map.values()))

    stats_sets = [personas_with_valid_stats(summary, common_questions) for summary in summaries]
    complete_sets = [complete for complete, _ in stats_sets]
    valid_sets = [valid for _, valid in stats_sets]
    for summary, (complete, valid) in zip(summaries, stats_sets, strict=True):
        if args.verbose:
            print(f"{summary.model}: {len(valid)} valid personas (of {len(complete)} complete).")
    if not valid_sets or any(len(v) == 0 for v in valid_sets):
        raise RuntimeError("A model has no personas with valid stats across all questions.")
    valid_personas = set.intersection(*valid_sets)
    if not valid_personas:
        raise RuntimeError("No personas remain after intersecting valid sets across models.")

    shared_complete = set.intersection(*complete_sets) if complete_sets else set()
    dropped_for_invalid = sorted(shared_complete.difference(valid_personas))
    if dropped_for_invalid:
        print("Personas dropped due to invalid stats across models:", ", ".join(map(str, dropped_for_invalid)))

    groups, removed_personas = build_groups(sorted(valid_personas), verbose=args.verbose)
    retained_personas = [pid for pid in sorted(valid_personas) if pid not in removed_personas]
    if removed_personas:
        print("Personas removed to balance groups:", ", ".join(map(str, removed_personas)))

    # Persist persona grouping assignments for downstream use
    group_rows: list[dict[str, int]] = []
    for group_id, group in enumerate(groups):
        for persona_id in group:
            group_rows.append({"group_id": group_id, "persona_id": persona_id})
    group_df = pd.DataFrame(group_rows).sort_values(["group_id", "persona_id"])
    args.groups_output.parent.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(args.groups_output, index=False)
    if args.verbose:
        print(f"Wrote persona grouping to {args.groups_output}")

    metrics_rows = []
    foundation_metrics_rows = []
    for summary in summaries:
        df = summary.frame
        filtered = df[
            df["question_id"].isin(common_questions)
            & df["persona_id"].isin(retained_personas)
        ].copy()
        if filtered.empty:
            raise RuntimeError(f"No data remaining for model {summary.model} after filtering personas/questions.")
        std_values = filtered["standard_deviation"].to_numpy(dtype=float)
        robustness, robustness_se = compute_robustness(std_values)

        pivot = (
            filtered.pivot(index="persona_id", columns="question_id", values="average_score")
            .loc[retained_personas]
        )
        if pivot.isnull().any().any():
            missing = int(pivot.isnull().sum().sum())
            raise RuntimeError(
                f"Model {summary.model} has {missing} missing average scores after filtering."
            )
        susceptibility, susceptibility_se = compute_susceptibility(pivot, groups)

        metrics_rows.append(
            {
                "model": summary.model,
                "robustness": robustness,
                "robustness_uncertainty": robustness_se,
                "susceptibility": susceptibility,
                "susceptibility_uncertainty": susceptibility_se,
            }
        )
        for foundation in foundations:
            foundation_questions = [
                qid for qid, name in foundation_map.items() if name == foundation
            ]
            subset_f = filtered[filtered["question_id"].isin(foundation_questions)].copy()
            if subset_f.empty:
                continue
            std_values_f = subset_f["standard_deviation"].to_numpy(dtype=float)
            robustness_f, robustness_se_f = compute_robustness(std_values_f)
            pivot_f = (
                subset_f.pivot(index="persona_id", columns="question_id", values="average_score")
                .loc[retained_personas]
            )
            if pivot_f.isnull().any().any():
                missing = int(pivot_f.isnull().sum().sum())
                raise RuntimeError(
                    f"Model {summary.model} has {missing} missing averages for foundation {foundation}."
                )
            susceptibility_f, susceptibility_se_f = compute_susceptibility(pivot_f, groups)
            foundation_metrics_rows.append(
                {
                    "model": summary.model,
                    "foundation": foundation,
                    "robustness": robustness_f,
                    "robustness_uncertainty": robustness_se_f,
                    "susceptibility": susceptibility_f,
                    "susceptibility_uncertainty": susceptibility_se_f,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("model").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output, index=False)

    if foundation_metrics_rows:
        foundation_df = (
            pd.DataFrame(foundation_metrics_rows)
            .sort_values(["model", "foundation"])
            .reset_index(drop=True)
        )
        args.foundation_output.parent.mkdir(parents=True, exist_ok=True)
        foundation_df.to_csv(args.foundation_output, index=False)
        if args.verbose:
            model_count = foundation_df["model"].nunique()
            print(f"Wrote foundation metrics for {model_count} models to {args.foundation_output}")

    if args.verbose:
        print(f"Wrote metrics for {len(metrics_df)} models to {args.output}")


if __name__ == "__main__":
    main()
