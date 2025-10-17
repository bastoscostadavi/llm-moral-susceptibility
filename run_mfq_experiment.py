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
        "model_name": "claude-4.5-sonnet",
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
        "label": "Claude Haiku 3.5 (Anthropic API)",
        "model_type": "anthropic",
        "model_name": "claude-3-5-haiku-latest",
        "model_kwargs": {},
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
    parser.add_argument("--model-type", choices=["openai", "anthropic", "ollama", "local"], default=None,
                        help="Type of model to use (skip to choose interactively)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Specific model name (skip to choose interactively)")
    parser.add_argument("--output", type=str, default="mfq_results",
                        help="Output file prefix")
    parser.add_argument("--limit", type=int, default=100,
                        help="Limit number of personas to test")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--personas-file", type=str, default="personas.json",
                        help="Path to personas JSON file")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of times each persona answers each question")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory containing local GGUF models")

    # Model-specific parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=5,
                        help="Maximum tokens in response")

    args = parser.parse_args()

    model_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.list_models:
        print("Available models:")
        for option in AVAILABLE_MODELS:
            print(f"  {option['label']} -> type: {option['model_type']}, name: {option['model_name']}")
        return

    # Load personas
    try:
        with open(args.personas_file, 'r', encoding='utf-8') as f:
            personas = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find personas file {args.personas_file}")
        print("Please run download_personas.py first")
        return

    if args.limit:
        personas = personas[:args.limit]

    print(f"Loaded {len(personas)} personas")

    model_type = args.model_type
    model_name = args.model_name
    selection_kwargs: Dict[str, Any] = {}

    if not model_type or not model_name:
        model_type, model_name, selection_kwargs = prompt_for_model_selection()
    else:
        selection_kwargs = {}

    if model_type == "local":
        selection_kwargs.setdefault("model_dir", args.models_dir)

    model_kwargs.update(selection_kwargs)

    print(f"Selected model: {model_type}:{model_name}")

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_suffix = model_name.replace(":", "_").replace("/", "_")
    csv_path = output_dir / f"{args.output}_{model_suffix}.csv"
    file_exists = csv_path.exists()

    fieldnames = [
        "persona_id",
        "question_id",
        "run_index",
        "rating",
        "response",
        "collected_at",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
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
        f"\nExperiment completed! Processed {personas_processed} personas and logged {responses_written} responses to {csv_path}."
    )

if __name__ == "__main__":
    main()
