#!/usr/bin/env python3
"""Bootstrap moral robustness and susceptibility metrics via resampling.

This script mirrors the filtering/grouping logic in compute_moral_metrics but
estimates the uncertainty of the bounded robustness/susceptibility indices with
non-parametric bootstraps over the underlying persona/question statistics. The
procedure is:

1. Load per-model summary CSVs and determine the shared persona/question set.
2. For each model, collect the set of within-persona standard deviations
   (``u_{pq}``) and group-level susceptibility samples (``S_g``).
3. Draw ``B`` bootstrap replicates of the mean ``u_{pq}`` and ``S_g`` by
   resampling those base quantities with replacement.
4. For every replicate, recompute the shared baselines ``c`` and ``c_S`` and
   apply the bounded transform to obtain replicated robustness/susceptibility
   values.
5. Report the Monte Carlo mean and sample standard deviation across replicates
   as the point estimate and uncertainty for each model (overall and per
   foundation).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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


@dataclass
class FoundationBootstrapData:
    u_values: np.ndarray
    s_samples: np.ndarray


@dataclass
class ModelBootstrapData:
    model: str
    u_values: np.ndarray
    s_samples: np.ndarray
    foundation_stats: Dict[str, FoundationBootstrapData]


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
        default=Path("results") / "moral_metrics_bootstrap.csv",
        help="Destination CSV for model-level metrics (default: results/moral_metrics_bootstrap.csv).",
    )
    parser.add_argument(
        "--foundation-output",
        type=Path,
        default=Path("results") / "moral_metrics_per_foundation_bootstrap.csv",
        help=(
            "Destination CSV for foundation-level metrics (default: results/moral_metrics_per_foundation_bootstrap.csv)."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap replicates for each model (default: 5000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Bootstrap batch size to control memory usage (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for bootstrap resampling (default: 1337).",
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


def load_summaries(directory: Path, verbose: bool = False) -> List[SummaryData]:
    summaries: List[SummaryData] = []
    skipped: List[tuple[Path, set[str]]] = []
    for csv_path in iter_summary_files(directory):
        df = pd.read_csv(csv_path)
        required = {"persona_id", "question_id", "average_score", "standard_deviation"}
        missing = required.difference(df.columns)
        if missing:
            base_cols = {"persona_id", "question_id"}
            if base_cols.issubset(df.columns):
                raise ValueError(
                    f"{csv_path} missing required columns: {', '.join(sorted(missing))}"
                )
            skipped.append((csv_path, missing))
            if verbose:
                print(
                    f"Skipping {csv_path} missing required columns: {', '.join(sorted(missing))}",
                    file=sys.stderr,
                )
            continue
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
    if skipped and not verbose:
        skipped_str = ", ".join(
            f"{path.name} ({', '.join(sorted(missing))})" for path, missing in skipped
        )
        print(
            f"Skipping non-summary CSVs missing required columns: {skipped_str}",
            file=sys.stderr,
        )
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


def compute_group_s_samples(pivot: pd.DataFrame, groups: Sequence[Sequence[int]]) -> np.ndarray:
    samples: List[float] = []
    for group in groups:
        block = pivot.loc[list(group)]
        per_question_std = block.std(axis=0, ddof=1)
        if per_question_std.isna().any():
            raise RuntimeError("Encountered NaN susceptibility standard deviation for a group.")
        samples.append(float(per_question_std.mean()))
    return np.asarray(samples, dtype=float)


def bootstrap_means(
    values: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    *,
    chunk_size: int,
) -> np.ndarray:
    if draws <= 0:
        raise ValueError("Number of bootstrap samples must be positive.")
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise RuntimeError("Cannot bootstrap an empty set of values.")
    result = np.empty(draws, dtype=float)
    remaining = draws
    start = 0
    while remaining > 0:
        batch = min(remaining, chunk_size)
        idx = rng.integers(0, values.size, size=(batch, values.size))
        sampled = values[idx]
        result[start : start + batch] = sampled.mean(axis=1)
        start += batch
        remaining -= batch
    return result


def main() -> None:
    args = parse_args()

    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    summaries = load_summaries(args.summaries_dir, verbose=args.verbose)
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
            "Missing foundation labels for questions: "
            + ", ".join(map(str, missing_foundations))
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

    model_data: List[ModelBootstrapData] = []
    for summary in summaries:
        df = summary.frame
        filtered = df[
            df["question_id"].isin(common_questions)
            & df["persona_id"].isin(retained_personas)
        ].copy()
        if filtered.empty:
            raise RuntimeError(
                f"No data remaining for model {summary.model} after filtering personas/questions."
            )
        std_values = filtered["standard_deviation"].to_numpy(dtype=float)
        if std_values.size == 0:
            raise RuntimeError(f"Model {summary.model} has no standard deviation values.")

        pivot = (
            filtered.pivot(index="persona_id", columns="question_id", values="average_score")
            .loc[retained_personas]
        )
        if pivot.isnull().any().any():
            missing = int(pivot.isnull().sum().sum())
            raise RuntimeError(
                f"Model {summary.model} has {missing} missing average scores after filtering."
            )
        s_samples = compute_group_s_samples(pivot, groups)

        foundation_stats: Dict[str, FoundationBootstrapData] = {}
        for foundation in foundations:
            foundation_questions = [
                qid for qid, name in foundation_map.items() if name == foundation
            ]
            subset_f = filtered[filtered["question_id"].isin(foundation_questions)].copy()
            if subset_f.empty:
                continue
            std_values_f = subset_f["standard_deviation"].to_numpy(dtype=float)
            pivot_f = (
                subset_f.pivot(index="persona_id", columns="question_id", values="average_score")
                .loc[retained_personas]
            )
            if pivot_f.isnull().any().any():
                missing = int(pivot_f.isnull().sum().sum())
                raise RuntimeError(
                    f"Model {summary.model} has {missing} missing averages for foundation {foundation}."
                )
            s_samples_f = compute_group_s_samples(pivot_f, groups)
            foundation_stats[foundation] = FoundationBootstrapData(
                u_values=std_values_f,
                s_samples=s_samples_f,
            )

        model_data.append(
            ModelBootstrapData(
                model=summary.model,
                u_values=std_values,
                s_samples=s_samples,
                foundation_stats=foundation_stats,
            )
        )

    rng = np.random.default_rng(args.seed)
    draws = args.bootstrap_samples

    overall_rob_boot = []
    overall_susc_boot = []
    for data in model_data:
        overall_rob_boot.append(
            bootstrap_means(data.u_values, draws, rng, chunk_size=args.chunk_size)
        )
        overall_susc_boot.append(
            bootstrap_means(data.s_samples, draws, rng, chunk_size=args.chunk_size)
        )

    overall_rob_boot = np.asarray(overall_rob_boot, dtype=float)
    overall_susc_boot = np.asarray(overall_susc_boot, dtype=float)

    rob_baseline = overall_rob_boot.mean(axis=0)
    susc_baseline = overall_susc_boot.mean(axis=0)

    bounded_robust = rob_baseline[None, :] / (overall_rob_boot + rob_baseline[None, :])
    bounded_susc = overall_susc_boot / (overall_susc_boot + susc_baseline[None, :])

    metrics_rows = []
    for data, rob_vals, susc_vals in zip(
        model_data,
        bounded_robust,
        bounded_susc,
        strict=True,
    ):
        metrics_rows.append(
            {
                "model": data.model,
                "robustness": float(rob_vals.mean()),
                "robustness_uncertainty": float(rob_vals.std(ddof=1)),
                "susceptibility": float(susc_vals.mean()),
                "susceptibility_uncertainty": float(susc_vals.std(ddof=1)),
            }
        )

    foundation_metrics_rows = []
    for foundation in foundations:
        foundation_models: List[str] = []
        rob_matrix: List[np.ndarray] = []
        susc_matrix: List[np.ndarray] = []
        for data in model_data:
            stats = data.foundation_stats.get(foundation)
            if stats is None:
                continue
            foundation_models.append(data.model)
            rob_matrix.append(
                bootstrap_means(stats.u_values, draws, rng, chunk_size=args.chunk_size)
            )
            susc_matrix.append(
                bootstrap_means(stats.s_samples, draws, rng, chunk_size=args.chunk_size)
            )
        if not foundation_models:
            continue
        rob_matrix = np.asarray(rob_matrix, dtype=float)
        susc_matrix = np.asarray(susc_matrix, dtype=float)
        rob_baseline_f = rob_matrix.mean(axis=0)
        susc_baseline_f = susc_matrix.mean(axis=0)
        bounded_rob_f = rob_baseline_f[None, :] / (rob_matrix + rob_baseline_f[None, :])
        bounded_susc_f = susc_matrix / (susc_matrix + susc_baseline_f[None, :])
        for model_name, rob_vals, susc_vals in zip(
            foundation_models,
            bounded_rob_f,
            bounded_susc_f,
            strict=True,
        ):
            foundation_metrics_rows.append(
                {
                    "model": model_name,
                    "foundation": foundation,
                    "robustness": float(rob_vals.mean()),
                    "robustness_uncertainty": float(rob_vals.std(ddof=1)),
                    "susceptibility": float(susc_vals.mean()),
                    "susceptibility_uncertainty": float(susc_vals.std(ddof=1)),
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
        print(
            f"Wrote bootstrap metrics for {len(metrics_df)} models to {args.output}"
        )
        if foundation_metrics_rows:
            print(
                f"Wrote bootstrap foundation metrics to {args.foundation_output}"
            )


if __name__ == "__main__":
    main()
