#!/usr/bin/env python3
"""
Simple MFQ experiment runner with different personas
"""

import json
import csv
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
import re

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

AVAILABLE_MODELS = [
    {
        "key": "1",
        "label": "Claude Haiku 4.5 (Anthropic API)",
        "model_type": "anthropic",
        "model_name": "claude-haiku-4-5-20251001",
        "model_kwargs": {
             "max_tokens": 2,
        },
    },
    {
        "key": "2",
        "label": "Claude Sonnet 4.5 (Anthropic API)",
        "model_type": "anthropic",
        "model_name": "claude-sonnet-4-5-20250929",
        "model_kwargs": {
             "max_tokens": 2,
        },
    },
    {
        "key": "3",
        "label": "Gemini 2.5 Flash Lite",
        "model_type": "google",
        "model_name": "gemini-2.5-flash-lite",
        "model_kwargs": {},
    },
    {
        "key": "4",
        "label": "Gemini 2.5 Flash (Google)",
        "model_type": "google",
        "model_name": "gemini-2.5-flash",
        "model_kwargs": {},
    },
    {
        "key": "5",
        "label": "OpenAI GPT-4.1",
        "model_type": "openai",
        "model_name": "gpt-4.1",
        "model_kwargs": {},
    },
    {
        "key": "6",
        "label": "OpenAI GPT-4.1 Mini",
        "model_type": "openai",
        "model_name": "gpt-4.1-mini",
        "model_kwargs": {},
    },
    {
        "key": "7",
        "label": "OpenAI GPT-4.1 Nano",
        "model_type": "openai",
        "model_name": "gpt-4.1-nano",
        "model_kwargs": {},
    },
    {
        "key": "8",
        "label": "OpenAI GPT-4o Mini",
        "model_type": "openai",
        "model_name": "gpt-4o-mini",
        "model_kwargs": {},
    },
    {
        "key": "9",
        "label": "OpenAI GPT-5 (minimal reasoning)",
        "model_type": "openai",
        "model_name": "gpt-5",
        "model_kwargs": {
            "reasoning_effort": "minimal",
            "use_responses_api": True,
            "max_tokens": 16,
        },
    },
    {
        "key": "10",
        "label": "OpenAI GPT-5 Mini (minimal reasoning)",
        "model_type": "openai",
        "model_name": "gpt-5-mini",
        "model_kwargs": {
            "reasoning_effort": "minimal",
            "use_responses_api": True,
            "max_tokens": 16,
        },
    },
    {
        "key": "11",
        "label": "xAI Grok-3 Mini",
        "model_type": "xai",
        "model_name": "grok-3-mini",
        "model_kwargs": {},
    },
    {
        "key": "12",
        "label": "xAI Grok-4",
        "model_type": "xai",
        "model_name": "grok-4",
        "model_kwargs": {},
    },
    {
        "key": "13",
        "label": "xAI Grok-4-Fast",
        "model_type": "xai",
        "model_name": "grok-4-fast",
        "model_kwargs": {},
    },
]

# Map model names to custom CSV filenames (without path). Update this dictionary
# whenever you rename or relocate a model's results file. Filenames are assumed
# to reside under the ``data/`` directory; the standard ``_self`` suffix is
# appended automatically for no-persona runs.
CUSTOM_MODEL_FILENAMES: Dict[str, str] = {
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5.csv",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5.csv",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite.csv",
    "gemini-2.5-flash": "gemini-2.5-flash.csv",
    "gpt-4.1": "gpt-4.1.csv",
    "gpt-4o-mini": "gpt-4o-mini.csv",
    "gpt-4.1-mini": "gpt-4.1-mini.csv",
    "gpt-5-mini": "gpt-5-mini.csv",
    "gpt-5": "gpt-5.csv",
    "grok-3-mini": "grok-3-mini.csv",
    "grok-4": "grok-4.csv",
    "grok-4-fast": "grok-4-fast.csv",
}

from mfq_questions import iter_questions
from llm_interface import get_llm_response

def create_persona_prompt(persona: str, question: str) -> str:
    """Create a prompt that includes the persona and the MFQ question"""
    return f"""You are roleplaying as the following persona: {persona}

Please answer the following question from the perspective of this persona.

{question}"""

def extract_rating(response: str) -> int:
    """Extract numeric rating from LLM response"""
    try:
        number = re.search(r'\b[0-5]\b', response.strip())
        if number:
            return int(number.group())
        print(f"No valid rating found in response: {response}")
        return -1
    except Exception:
        print(f"Could not parse response: {response}")
        return -1


def resolve_model_filename(model_name: str, suffix: str = "", directory: Optional[Path] = None) -> Path:
    """Return the CSV path for the given model, applying any custom mapping."""

    base_dir = directory or Path("data")
    custom = CUSTOM_MODEL_FILENAMES.get(model_name)
    if custom:
        stem = custom[:-4] if custom.lower().endswith(".csv") else custom
    else:
        stem = model_name.replace(":", "_").replace("/", "_")

    filename = f"{stem}{suffix}.csv"
    return base_dir / filename


def run_mfq_experiment(
    personas: List[str],
    model_type: str,
    model_name: str,
    n: int = 10,
    csv_writer: Optional[csv.DictWriter] = None,
    csv_file=None,
    existing_valid_slots: Optional[Set[Tuple[int, int, int]]] = None,
    collect_new_rows: bool = False,
    slot_failures: Optional[Dict[Tuple[int, int, int], int]] = None,
    row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **model_kwargs,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Run the MFQ experiment.

    - If ``csv_writer`` is supplied, rows are streamed directly to the CSV.
    - ``existing_valid_slots`` marks (persona_id, question_id, run_index) entries
      that already have valid (>=0) ratings and should be skipped.
    - When ``collect_new_rows`` is True, the function returns any newly
      generated rows so the caller can handle persistence (e.g., rewrite files).
    - ``slot_failures`` tracks how many invalid attempts (rating -1) have been
      recorded for each (persona, question, run_index) slot.
    """

    if csv_writer is None and not collect_new_rows and row_callback is None:
        raise ValueError(
            "run_mfq_experiment requires a csv_writer unless collect_new_rows or row_callback is provided"
        )

    questions = list(iter_questions())

    personas_processed = 0
    responses_written = 0
    existing_valid_slots = existing_valid_slots or set()
    slot_failures = slot_failures or {}
    new_rows: List[Dict[str, Any]] = []

    print(f"Running MFQ experiment with {len(personas)} personas using {model_type}:{model_name}")

    for persona_id, persona in enumerate(personas):
        persona_text = str(persona)
        print(f"\nProgress: {persona_id + 1}/{len(personas)} - {persona_text[:50]}...")
        personas_processed += 1

        for question in questions:
            prompt = create_persona_prompt(persona_text, question.prompt)

            for run_index in range(1, n + 1):
                slot_key = (persona_id, question.id, run_index)
                if slot_key in existing_valid_slots:
                    continue

                response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
                rating = extract_rating(response)
                response_text = response.strip() if isinstance(response, str) else str(response)

                prior_failures = slot_failures.get(slot_key, 0)
                failures = prior_failures + (1 if rating < 0 else 0)

                row = {
                    "persona_id": persona_id,
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
                    # Store a copy to avoid accidental mutation
                    new_rows.append(dict(row))

                if rating >= 0:
                    existing_valid_slots.add(slot_key)

    return personas_processed, responses_written, new_rows


def prompt_for_model_selection() -> Tuple[str, str, Dict[str, Any]]:
    """Interactively prompt the user to select a model."""

    print("Select the model to run:")
    for option in AVAILABLE_MODELS:
        print(f"  {option['key']}. {option['label']}")

    while True:
        choice = input("Enter the number of the model to use: ").strip()
        for option in AVAILABLE_MODELS:
            if choice == option["key"]:
                return (
                    option["model_type"],
                    option["model_name"],
                    option.get("model_kwargs", {}),
                )
        print("Invalid selection. Please try again.")

def main():
    
    parser = argparse.ArgumentParser(description="Run MFQ experiments with different personas")
    
    parser.add_argument("--limit", type=int, default=100,
                        help="Limit number of personas to test")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of times each persona answers each question")


    args = parser.parse_args()

    # Load personas
    try:
        with open("personas.json", 'r', encoding='utf-8') as f:
            personas = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find personas file")
        return

    if args.limit:
        personas = personas[:args.limit]

    print(f"Loaded {len(personas)} personas")

    model_type, model_name, selection_kwargs = prompt_for_model_selection()

    model_kwargs = {
        "temperature": 0.1,
        "max_tokens": 100,
    }
    model_kwargs.update(selection_kwargs)

    print(f"Selected model: {model_type}:{model_name}")

    output_path = resolve_model_filename(model_name)
    file_exists = output_path.exists()

    fieldnames = [
        "persona_id",
        "question_id",
        "run_index",
        "rating",
        "failures",
        "response",
        "collected_at",
    ]

    existing_rows: List[Dict[str, Any]] = []
    existing_valid_slots: Set[Tuple[int, int, int]] = set()
    slot_failures: Dict[Tuple[int, int, int], int] = {}
    rows_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    had_missing_failures = False

    if file_exists:
        try:
            with open(output_path, "r", newline="", encoding="utf-8") as existing_file:
                reader = csv.DictReader(existing_file)
                for row in reader:
                    try:
                        persona_id = int(row["persona_id"])
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
                        "persona_id": persona_id,
                        "question_id": question_id,
                        "run_index": run_index,
                        "rating": rating,
                        "failures": failures,
                        "response": row.get("response", ""),
                        "collected_at": row.get("collected_at", ""),
                    }

                    existing_rows.append(row_dict)
                    rows_by_key[(persona_id, question_id, run_index)] = row_dict

                    if rating >= 0:
                        existing_valid_slots.add((persona_id, question_id, run_index))

                    slot_failures[(persona_id, question_id, run_index)] = failures

            if existing_valid_slots:
                print(
                    f"Found {len(existing_valid_slots)} previously completed slots. Only missing or invalid entries will be re-run."
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
            key = (row["persona_id"], row["question_id"], row["run_index"])
            rows_by_key[key] = row
            slot_failures[key] = row.get("failures", 0)
            write_rows_to_disk()

        personas_processed, responses_written, _ = run_mfq_experiment(
            personas,
            model_type,
            model_name,
            n=args.n,
            csv_writer=None,
            csv_file=None,
            existing_valid_slots=set(existing_valid_slots),
            collect_new_rows=False,
            slot_failures=slot_failures,
            row_callback=handle_new_row,
            **model_kwargs,
        )

        if responses_written == 0 and had_missing_failures and rows_by_key:
            write_rows_to_disk()

    else:
        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            personas_processed, responses_written, _ = run_mfq_experiment(
                personas,
                model_type,
                model_name,
                n=args.n,
                csv_writer=writer,
                csv_file=csv_file,
                existing_valid_slots=None,
                collect_new_rows=False,
                slot_failures=slot_failures,
                **model_kwargs,
            )

    if file_exists and responses_written == 0:
        print("\nNo new runs were required; all slots were already filled with valid ratings.")

    print(
        f"\nExperiment completed! Processed {personas_processed} personas and logged {responses_written} responses to {output_path}."
    )

if __name__ == "__main__":
    main()
