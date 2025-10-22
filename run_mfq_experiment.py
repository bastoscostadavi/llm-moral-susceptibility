#!/usr/bin/env python3
"""
Simple MFQ experiment runner with different personas
"""

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import re

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

AVAILABLE_MODELS = [
    {
        "key": "1",
        "label": "Mistral-7B Instruct v0.3 Q8_0 (local gguf)",
        "model_type": "local",
        "model_name": "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
        "model_kwargs": {
            "model_dir": "../models",
        },
    },
    {
        "key": "2",
        "label": "Meta Llama 3.1 8B Instruct (local gguf)",
        "model_type": "local",
        "model_name": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "model_kwargs": {
            "model_dir": "../models",
        },
    },
    {
        "key": "3",
        "label": "Qwen2.5 7B Instruct (local gguf)",
        "model_type": "local",
        "model_name": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "model_kwargs": {
            "model_dir": "../models",
        },
    },
    {
        "key": "4",
        "label": "Claude Sonnet 4.5 (Anthropic API)",
        "model_type": "anthropic",
        "model_name": "claude-sonnet-4-5-20250929",
        "model_kwargs": {},
    },
    {
        "key": "5",
        "label": "OpenAI GPT-5 Mini",
        "model_type": "openai",
        "model_name": "gpt-5-mini",
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
        "label": "OpenAI GPT-4o Mini",
        "model_type": "openai",
        "model_name": "gpt-4o-mini",
        "model_kwargs": {},
    },
    {
        "key": "8",
        "label": "Claude Haiku 4.5 (Anthropic API)",
        "model_type": "anthropic",
        "model_name": "claude-haiku-4-5-20251001",
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
        "label": "xAI Grok-4-Fast",
        "model_type": "xai",
        "model_name": "grok-4-fast",
        "model_kwargs": {},
    },
    {
        "key": "11",
        "label": "OpenAI GPT-5 Mini (minimal reasoning)",
        "model_type": "openai",
        "model_name": "gpt-5-mini",
        "model_kwargs": {
            "reasoning_effort": "minimal",
            "use_responses_api": True,
            "max_tokens": 16,
        },
    },
]

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


def run_mfq_experiment(
    personas: List[str],
    model_type: str,
    model_name: str,
    n: int = 10,
    csv_writer: Optional[csv.DictWriter] = None,
    csv_file=None,
    **model_kwargs,
) -> Tuple[int, int]:
    """Run the MFQ experiment, streaming rows to the provided CSV writer."""

    if csv_writer is None:
        raise ValueError("run_mfq_experiment requires a csv_writer")

    questions = list(iter_questions())

    personas_processed = 0
    responses_written = 0

    print(f"Running MFQ experiment with {len(personas)} personas using {model_type}:{model_name}")

    for persona_id, persona in enumerate(personas):
        persona_text = str(persona)
        print(f"\nProgress: {persona_id + 1}/{len(personas)} - {persona_text[:50]}...")
        personas_processed += 1

        for question in questions:
            prompt = create_persona_prompt(persona_text, question.prompt)

            for run_index in range(1, n + 1):
                response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
                rating = extract_rating(response)
                response_text = response.strip() if isinstance(response, str) else str(response)

                row = {
                    "persona_id": persona_id,
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

    return personas_processed, responses_written


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
        "max_tokens": 2,
    }
    model_kwargs.update(selection_kwargs)

    print(f"Selected model: {model_type}:{model_name}")

    model_suffix = model_name.replace(":", "_").replace("/", "_")
    output_path = Path("data") / f"{model_suffix}.csv"
    file_exists = output_path.exists()

    fieldnames = [
        "persona_id",
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

        personas_processed, responses_written = run_mfq_experiment(
            personas,
            model_type,
            model_name,
            n=args.n,
            csv_writer=writer,
            csv_file=csv_file,
            **model_kwargs,
        )

    print(
        f"\nExperiment completed! Processed {personas_processed} personas and logged {responses_written} responses to {output_path}."
    )

if __name__ == "__main__":
    main()
