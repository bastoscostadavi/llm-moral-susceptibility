#!/usr/bin/env python3
"""
Run the MFQ-30 without any persona conditioning.

This script mirrors the flow of `run_mfq_experiment.py` but asks each
MFQ item directly (no persona wrapper) and writes a per-model CSV that
omits the persona column. The CSV filename appends `_self` to the
sanitized model name to distinguish from persona-conditioned runs.
"""

import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from mfq_questions import iter_questions
from llm_interface import get_llm_response

# Reuse helpers from the persona runner for consistency
from run_mfq_experiment import extract_rating, prompt_for_model_selection


def run_mfq_self(
    model_type: str,
    model_name: str,
    n: int = 10,
    csv_writer: Optional[csv.DictWriter] = None,
    csv_file=None,
    **model_kwargs,
) -> Tuple[int, int]:
    """Run MFQ without personas, streaming rows to the provided CSV writer.

    Returns a tuple of (questions_processed, responses_written).
    """

    if csv_writer is None:
        raise ValueError("run_mfq_self requires a csv_writer")

    questions = list(iter_questions())
    questions_processed = 0
    responses_written = 0

    print(f"Running MFQ (self) using {model_type}:{model_name}")

    for q_idx, question in enumerate(questions, start=1):
        print(f"\nProgress: question {q_idx}/{len(questions)} (id={question.id})")
        questions_processed += 1

        # Prompt is just the MFQ question text with rating instructions
        prompt = question.prompt

        for run_index in range(1, n + 1):
            response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
            rating = extract_rating(response)
            response_text = response.strip() if isinstance(response, str) else str(response)

            row = {
                "question_id": question.id,
                "run_index": run_index,
                "rating": rating,
                "response": response_text,
                "collected_at": datetime.now().isoformat(),
            }

            csv_writer.writerow(row)
            responses_written += 1
            if csv_file is not None:
                csv_file.flush()

    return questions_processed, responses_written


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
        "max_tokens": 2,
    }
    model_kwargs.update(selection_kwargs)

    print(f"Selected model: {model_type}:{model_name}")

    model_suffix = model_name.replace(":", "_").replace("/", "_")
    output_path = Path("data") / f"{model_suffix}_self.csv"
    file_exists = output_path.exists()

    fieldnames = [
        "question_id",
        "run_index",
        "rating",
        "response",
        "collected_at",
    ]

    with open(output_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        questions_processed, responses_written = run_mfq_self(
            model_type,
            model_name,
            n=args.n,
            csv_writer=writer,
            csv_file=csv_file,
            **model_kwargs,
        )

    print(
        f"\nSelf run completed! Processed {questions_processed} questions and logged {responses_written} responses to {output_path}."
    )


if __name__ == "__main__":
    main()

