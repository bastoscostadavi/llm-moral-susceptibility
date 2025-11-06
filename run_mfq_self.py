#!/usr/bin/env python3
"""
Run the MFQ-30 without any persona conditioning.

This script mirrors the flow of `run_mfq_experiment.py` but asks each
MFQ item directly (no persona wrapper) and writes a per-model CSV that
omits the persona column. The CSV filename appends `_self` to the
sanitized model name (or the custom name defined in
`run_mfq_experiment.CUSTOM_MODEL_FILENAMES`) to distinguish from
persona-conditioned runs. Existing files are incrementally updated: only
missing or invalid slots are rerun, and each row tracks how many failed
attempts preceded the current rating via a `failures` column.
"""

import csv
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Set, List, Any, Callable

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from mfq_questions import iter_questions
from llm_interface import get_llm_response

# Reuse helpers from the persona runner for consistency
from run_mfq_experiment import (
    extract_rating,
    prompt_for_model_selection,
    resolve_model_filename,
)


def run_mfq_self(
    model_type: str,
    model_name: str,
    n: int = 10,
    csv_writer: Optional[csv.DictWriter] = None,
    csv_file=None,
    existing_valid_slots: Optional[Set[Tuple[int, int]]] = None,
    slot_failures: Optional[Dict[Tuple[int, int], int]] = None,
    collect_new_rows: bool = False,
    row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **model_kwargs,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Run MFQ without personas.

    - When ``csv_writer`` is provided, rows are streamed directly to disk.
    - ``existing_valid_slots`` allows skipping slots that already have valid ratings.
    - ``slot_failures`` records how many invalid attempts (rating -1) were seen per slot.
    - ``collect_new_rows`` stores new rows for callers that prefer manual persistence.
    - ``row_callback`` is called with each produced row (e.g., for incremental writes).
    """

    if csv_writer is None and not collect_new_rows and row_callback is None:
        raise ValueError(
            "run_mfq_self requires a csv_writer unless collect_new_rows or row_callback is provided"
        )

    questions = list(iter_questions())
    questions_processed = 0
    responses_written = 0
    existing_valid_slots = existing_valid_slots or set()
    slot_failures = slot_failures or {}
    new_rows: List[Dict[str, Any]] = []

    print(f"Running MFQ (self) using {model_type}:{model_name}")

    for q_idx, question in enumerate(questions, start=1):
        print(f"\nProgress: question {q_idx}/{len(questions)} (id={question.id})")
        questions_processed += 1

        # Prompt is just the MFQ question text with rating instructions
        prompt = question.prompt

        for run_index in range(1, n + 1):
            slot_key = (question.id, run_index)
            if slot_key in existing_valid_slots:
                continue

            response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
            rating = extract_rating(response)
            response_text = response.strip() if isinstance(response, str) else str(response)

            prior_failures = slot_failures.get(slot_key, 0)
            failures = prior_failures + (1 if rating < 0 else 0)

            row = {
                "question_id": question.id,
                "run_index": run_index,
                "rating": rating,
                "failures": failures,
                "response": response_text,
                "collected_at": datetime.now().isoformat(),
            }

            if csv_writer is not None:
                csv_writer.writerow(row)
                responses_written += 1
                if csv_file is not None:
                    csv_file.flush()
            else:
                responses_written += 1

            slot_failures[slot_key] = failures

            if row_callback is not None:
                row_callback(dict(row))

            if collect_new_rows:
                new_rows.append(dict(row))

            if rating >= 0:
                existing_valid_slots.add(slot_key)

    return questions_processed, responses_written, new_rows


def main():
    parser = argparse.ArgumentParser(description="Run MFQ (self) with no persona conditioning")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of times to answer each question",
    )

    args = parser.parse_args()

    model_type, model_name, selection_kwargs = prompt_for_model_selection()

    model_kwargs = {
        "temperature": 0.1,
        "max_tokens": 150,
    }
    model_kwargs.update(selection_kwargs)

    print(f"Selected model: {model_type}:{model_name}")

    output_path = resolve_model_filename(model_name, suffix="_self")
    file_exists = output_path.exists()

    fieldnames = [
        "question_id",
        "run_index",
        "rating",
        "failures",
        "response",
        "collected_at",
    ]

    existing_valid_slots: Set[Tuple[int, int]] = set()
    slot_failures: Dict[Tuple[int, int], int] = {}
    rows_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    had_missing_failures = False

    if file_exists:
        try:
            with open(output_path, "r", newline="", encoding="utf-8") as existing_file:
                reader = csv.DictReader(existing_file)
                for row in reader:
                    try:
                        question_id = int(row["question_id"])
                        run_index = int(row["run_index"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    rating_value = row.get("rating", -1)
                    try:
                        rating = int(rating_value)
                    except (TypeError, ValueError):
                        rating = -1

                    raw_failures = row.get("failures")
                    if raw_failures in (None, ""):
                        failures = 0
                        had_missing_failures = True
                    else:
                        try:
                            failures = int(raw_failures)
                        except (TypeError, ValueError):
                            failures = 0
                            had_missing_failures = True

                    if rating < 0 and failures <= 0:
                        failures = 1

                    row_dict = {
                        "question_id": question_id,
                        "run_index": run_index,
                        "rating": rating,
                        "failures": failures,
                        "response": row.get("response", ""),
                        "collected_at": row.get("collected_at", ""),
                    }

                    rows_by_key[(question_id, run_index)] = row_dict

                    if rating >= 0:
                        existing_valid_slots.add((question_id, run_index))

                    slot_failures[(question_id, run_index)] = failures

            if existing_valid_slots:
                print(
                    f"Found {len(existing_valid_slots)} self slots with valid ratings. Only missing or invalid entries will be re-run."
                )

        except FileNotFoundError:
            file_exists = False

    if file_exists:
        def write_rows_to_disk() -> None:
            if not rows_by_key:
                return
            tmp_path = output_path.parent / f"{output_path.name}.tmp"
            with open(tmp_path, "w", newline="", encoding="utf-8") as tmp_file:
                writer = csv.DictWriter(tmp_file, fieldnames=fieldnames)
                writer.writeheader()
                for key in sorted(rows_by_key.keys()):
                    writer.writerow(rows_by_key[key])
            os.replace(tmp_path, output_path)

        def handle_new_row(row: Dict[str, Any]) -> None:
            key = (row["question_id"], row["run_index"])
            rows_by_key[key] = row
            slot_failures[key] = row.get("failures", 0)
            write_rows_to_disk()

        questions_processed, responses_written, _ = run_mfq_self(
            model_type,
            model_name,
            n=args.n,
            csv_writer=None,
            csv_file=None,
            existing_valid_slots=set(existing_valid_slots),
            slot_failures=slot_failures,
            collect_new_rows=False,
            row_callback=handle_new_row,
            **model_kwargs,
        )

        if responses_written == 0 and had_missing_failures and rows_by_key:
            write_rows_to_disk()

    else:
        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            questions_processed, responses_written, _ = run_mfq_self(
                model_type,
                model_name,
                n=args.n,
                csv_writer=writer,
                csv_file=csv_file,
                existing_valid_slots=None,
                slot_failures=slot_failures,
                collect_new_rows=False,
                row_callback=None,
                **model_kwargs,
            )

    if file_exists and responses_written == 0:
        print("\nNo new self-run slots were required; all questions already had valid ratings.")

    print(
        f"\nSelf run completed! Processed {questions_processed} questions and logged {responses_written} responses to {output_path}."
    )


if __name__ == "__main__":
    main()
