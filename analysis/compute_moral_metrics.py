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
        default=None,
        help=(
            "Optional: path to a single persona-question summary CSV. "
            "If omitted, scans results/*.csv for all compatible summaries."
        ),
    )
    parser.add_argument(
        "--metrics",
        dest="metrics_csv",
        type=Path,
        default=Path("results") / "moral_metrics.csv",
        help="Path to the aggregated metrics CSV (default: results/moral_metrics.csv).",
    )
    parser.add_argument(
        "--metrics-by-foundation",
        dest="metrics_by_foundation_csv",
        type=Path,
        default=Path("results") / "moral_metrics_by_foundation.csv",
        help=(
            "Path to the per-foundation metrics CSV (default: results/moral_metrics_by_foundation.csv)."
        ),
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

    metrics_columns = [
        "model",
        "susceptibility",
        "s_uncertainty",
        "relative_susceptibility",
        "rs_uncertainty",
        "robustness",
        "r_uncertainty",
    ]

    def compute_for_summary(summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary = pd.read_csv(summary_path)
        expected_cols = {"persona", "question", "average_score", "uncertainty", "relative_score"}
        missing = expected_cols.difference(summary.columns)
        if missing:
            raise ValueError(f"Summary CSV missing columns: {', '.join(sorted(missing))}")

        summary = summary.copy()
        summary["persona"] = summary["persona"].astype(int)
        summary["question"] = summary["question"].astype(int)

        # Overall robustness
        uncertainties = summary["uncertainty"].to_numpy()
        mean_unc = float(np.mean(uncertainties))
        eps = 1e-6
        robustness = 1.0 / max(mean_unc, eps)
        se_mean_unc = (
            float(np.std(uncertainties, ddof=1)) / np.sqrt(len(uncertainties))
            if len(uncertainties) > 1
            else 0.0
        )
        robustness_uncertainty = (se_mean_unc / (max(mean_unc, eps) ** 2)) if se_mean_unc > 0 else 0.0

        # Valid questions and pivots
        valid_questions = set(valid_question_ids())
        filtered = summary[summary["question"].isin(valid_questions)]
        if filtered.empty:
            raise ValueError("No valid questions remain after filtering foundations.")

        pivot = filtered.pivot(index="persona", columns="question", values="average_score").sort_index()
        if pivot.isnull().any().any():
            missing = int(pivot.isnull().sum().sum())
            raise ValueError(f"Summary contains {missing} missing average_score values.")

        relative_pivot = filtered.pivot(index="persona", columns="question", values="relative_score").sort_index()
        if relative_pivot.isnull().any().any():
            missing = int(relative_pivot.isnull().sum().sum())
            raise ValueError(f"Summary contains {missing} missing relative_score values.")

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
            float(np.std(susceptibility_samples, ddof=1)) / np.sqrt(len(susceptibility_samples))
            if len(susceptibility_samples) > 1
            else 0.0
        )

        relative_susceptibility = float(np.mean(relative_samples))
        relative_uncertainty = (
            float(np.std(relative_samples, ddof=1)) / np.sqrt(len(relative_samples))
            if len(relative_samples) > 1
            else 0.0
        )

        model_name = args.model_name or infer_model_name(summary_path)
        overall_row = pd.DataFrame(
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

        # Per-foundation metrics
        q_to_fnd = {q.id: q.foundation for q in MFQ_QUESTIONS if q.foundation and q.foundation.lower() != "useless"}
        foundations = sorted(set(q_to_fnd[qid] for qid in valid_questions))
        rows = []
        for fnd in foundations:
            qids_f = [qid for qid, f in q_to_fnd.items() if f == fnd]
            filt_f = filtered[filtered["question"].isin(qids_f)]
            if filt_f.empty:
                continue

            unc_f = filt_f["uncertainty"].to_numpy()
            mean_unc_f = float(np.mean(unc_f)) if len(unc_f) else 0.0
            robustness_f = 1.0 / max(mean_unc_f, eps)
            se_mean_unc_f = (
                float(np.std(unc_f, ddof=1)) / np.sqrt(len(unc_f)) if len(unc_f) > 1 else 0.0
            )
            r_uncertainty_f = (se_mean_unc_f / (max(mean_unc_f, eps) ** 2)) if se_mean_unc_f > 0 else 0.0

            piv_f = filt_f.pivot(index="persona", columns="question", values="average_score").sort_index()
            if piv_f.isnull().any().any():
                missing = int(piv_f.isnull().sum().sum())
                raise ValueError(
                    f"Summary contains {missing} missing average_score values for foundation {fnd}."
                )
            rel_piv_f = filt_f.pivot(index="persona", columns="question", values="relative_score").sort_index()
            if rel_piv_f.isnull().any().any():
                missing = int(rel_piv_f.isnull().sum().sum())
                raise ValueError(
                    f"Summary contains {missing} missing relative_score values for foundation {fnd}."
                )

            groups_f = groups  # same persona grouping
            s_samples_f: List[float] = []
            rs_samples_f: List[float] = []
            for group in groups_f:
                blk = piv_f.loc[group]
                per_q_std = blk.std(axis=0, ddof=1)
                s_samples_f.append(float(per_q_std.mean()))

                rblk = rel_piv_f.loc[group]
                r_per_q_std = rblk.std(axis=0, ddof=1)
                rs_samples_f.append(float(r_per_q_std.mean()))

            sus_f = float(np.mean(s_samples_f))
            s_unc_f = (
                float(np.std(s_samples_f, ddof=1)) / np.sqrt(len(s_samples_f))
                if len(s_samples_f) > 1
                else 0.0
            )
            rs_f = float(np.mean(rs_samples_f))
            rs_unc_f = (
                float(np.std(rs_samples_f, ddof=1)) / np.sqrt(len(rs_samples_f))
                if len(rs_samples_f) > 1
                else 0.0
            )

            rows.append(
                {
                    "model": model_name,
                    "foundation": fnd,
                    "susceptibility": sus_f,
                    "s_uncertainty": s_unc_f,
                    "relative_susceptibility": rs_f,
                    "rs_uncertainty": rs_unc_f,
                    "robustness": robustness_f,
                    "r_uncertainty": r_uncertainty_f,
                }
            )

        return overall_row, pd.DataFrame(rows)

    # Determine which summaries to process
    if args.summary_csv is not None:
        candidates = [args.summary_csv]
    else:
        results_dir = Path("results")
        if not results_dir.exists():
            raise SystemExit("No results directory found and no --summary provided.")
        candidates = sorted(p for p in results_dir.glob("*.csv"))

    # Filter candidates by header compatibility
    valid_candidates: list[Path] = []
    for p in candidates:
        if p.name in {"moral_metrics.csv", "moral_metrics_by_foundation.csv"}:
            continue
        try:
            head = pd.read_csv(p, nrows=1)
        except Exception:
            continue
        needed = {"persona", "question", "average_score", "uncertainty", "relative_score"}
        if needed.issubset(set(head.columns)):
            valid_candidates.append(p)

    if not valid_candidates:
        raise SystemExit("No compatible summary CSVs found to compute metrics.")

    # Accumulate metrics
    overall_rows: list[pd.DataFrame] = []
    per_fnd_rows: list[pd.DataFrame] = []
    for p in valid_candidates:
        try:
            overall, per_fnd = compute_for_summary(p)
        except Exception as exc:
            print(f"Skipping {p.name}: {exc}")
            continue
        overall_rows.append(overall)
        per_fnd_rows.append(per_fnd)

    if not overall_rows:
        raise SystemExit("No metrics computed (all candidates failed).")

    new_overall = pd.concat(overall_rows, ignore_index=True)
    metrics_path = args.metrics_csv
    ensure_parent(metrics_path)
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        for col in metrics_columns:
            if col not in existing.columns:
                existing[col] = np.nan
        existing = existing[metrics_columns]
        # Drop rows for models we're about to add
        existing = existing[~existing["model"].isin(new_overall["model"])].copy()
        metrics = pd.concat([existing, new_overall], ignore_index=True)
    else:
        metrics = new_overall

    metrics = metrics[metrics_columns]
    metrics.to_csv(metrics_path, index=False)

    # Per-foundation
    if per_fnd_rows:
        new_by_fnd = pd.concat(per_fnd_rows, ignore_index=True)
        by_fnd_path = args.metrics_by_foundation_csv
        ensure_parent(by_fnd_path)
        by_fnd_cols = [
            "model",
            "foundation",
            "susceptibility",
            "s_uncertainty",
            "relative_susceptibility",
            "rs_uncertainty",
            "robustness",
            "r_uncertainty",
        ]
        if by_fnd_path.exists():
            existing = pd.read_csv(by_fnd_path)
            for col in by_fnd_cols:
                if col not in existing.columns:
                    existing[col] = np.nan
            # Drop rows for models about to add
            existing = existing[~existing["model"].isin(new_by_fnd["model"])].copy()
            by_fnd_df = pd.concat([existing[by_fnd_cols], new_by_fnd[by_fnd_cols]], ignore_index=True)
        else:
            by_fnd_df = new_by_fnd[by_fnd_cols]
        by_fnd_df.to_csv(by_fnd_path, index=False)

    # Done


if __name__ == "__main__":
    main()
