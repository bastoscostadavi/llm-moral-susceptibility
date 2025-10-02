#!/usr/bin/env python3
"""
Simple MFQ experiment runner with different personas
"""

import json
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

from mfq_questions import get_all_mfq_questions, format_relevance_question, format_agreement_question
from llm_interface import get_llm_response, check_model_availability, list_available_models

def create_persona_prompt(persona: str, question: str) -> str:
    """Create a prompt that includes the persona and the MFQ question"""
    return f"""You are roleplaying as the following persona: {persona}

Please answer the following question from the perspective of this persona. Give only a numeric rating from 0 to 5.

{question}

Your rating (0-5):"""

def extract_rating(response: str) -> int:
    """Extract numeric rating from LLM response"""
    try:
        # Try to find a number in the response
        import re
        numbers = re.findall(r'\b[0-5]\b', response.strip())
        if numbers:
            return int(numbers[0])
        else:
            print(f"No valid rating found in response: {response}")
            return -1
    except Exception:
        print(f"Could not parse response: {response}")
        return -1

def run_mfq_experiment(personas: List[Dict], model_type: str, model_name: str, **model_kwargs) -> List[Dict[str, Any]]:
    """
    Main experiment loop: for each persona, run all MFQ questions

    Args:
        personas: List of persona dictionaries
        model_type: Type of model ('openai', 'anthropic', 'ollama')
        model_name: Specific model name
        **model_kwargs: Additional model parameters

    Returns:
        List of results for each persona
    """

    questions = get_all_mfq_questions()
    all_results = []

    print(f"Running MFQ experiment with {len(personas)} personas using {model_type}:{model_name}")

    for i, persona in enumerate(personas):
        persona_text = persona.get("persona", "")
        print(f"\nProgress: {i+1}/{len(personas)} - {persona_text[:50]}...")

        result = {
            "persona_id": i,
            "persona": persona_text,
            "model_type": model_type,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "relevance_scores": {},
            "agreement_scores": []
        }

        # Process relevance questions for each moral foundation
        for foundation, foundation_questions in questions["relevance_questions"].items():
            result["relevance_scores"][foundation] = []

            for question in foundation_questions:
                formatted_question = format_relevance_question(question)
                prompt = create_persona_prompt(persona_text, formatted_question)

                # Get LLM response
                response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
                rating = extract_rating(response)
                result["relevance_scores"][foundation].append(rating)

                # Small delay to avoid overwhelming the model
                time.sleep(0.1)

        # Process agreement questions
        for statement in questions["agreement_questions"]:
            formatted_question = format_agreement_question(statement)
            prompt = create_persona_prompt(persona_text, formatted_question)

            # Get LLM response
            response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
            rating = extract_rating(response)
            result["agreement_scores"].append(rating)

            # Small delay to avoid overwhelming the model
            time.sleep(0.1)

        all_results.append(result)

    return all_results

def save_results(results: List[Dict[str, Any]], output_prefix: str):
    """Save results to JSON and CSV files"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"{output_prefix}_{timestamp}.json"
    csv_file = f"{output_prefix}_{timestamp}.csv"

    # Save full results as JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary as CSV
    if results:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            header = ['persona_id', 'persona', 'model_type', 'model_name', 'timestamp']

            # Add relevance score columns
            for foundation in results[0]['relevance_scores'].keys():
                for i in range(len(results[0]['relevance_scores'][foundation])):
                    header.append(f"relevance_{foundation}_{i+1}")

            # Add agreement score columns
            for i in range(len(results[0]['agreement_scores'])):
                header.append(f"agreement_{i+1}")

            writer.writerow(header)

            # Write data
            for result in results:
                row = [
                    result['persona_id'],
                    result['persona'][:100],  # Truncate for CSV
                    result['model_type'],
                    result['model_name'],
                    result['timestamp']
                ]

                # Add relevance scores
                for foundation in result['relevance_scores'].keys():
                    row.extend(result['relevance_scores'][foundation])

                # Add agreement scores
                row.extend(result['agreement_scores'])

                writer.writerow(row)

    print(f"Results saved to {json_file} and {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Run MFQ experiments with different personas")
    parser.add_argument("--model-type", choices=["openai", "anthropic", "ollama"], required=True,
                        help="Type of model to use")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Specific model name")
    parser.add_argument("--personas-file", type=str, default="personas/personas_sample_01.json",
                        help="JSON file containing personas")
    parser.add_argument("--output", type=str, default="mfq_results",
                        help="Output file prefix")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of personas to test")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")

    # Model-specific parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Maximum tokens in response")

    args = parser.parse_args()

    # List available models if requested
    if args.list_models:
        models = list_available_models(args.model_type)
        if models:
            print(f"Available {args.model_type} models:")
            for model in models:
                print(f"  - {model}")
        else:
            print(f"No {args.model_type} models available or service not running")
        return

    # Check model availability
    model_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }

    if not check_model_availability(args.model_type, args.model_name, **model_kwargs):
        print(f"Model {args.model_name} not available for {args.model_type}")
        if args.model_type == "ollama":
            print("Make sure Ollama is running and the model is downloaded:")
            print(f"  ollama pull {args.model_name}")
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

    # Run experiment
    results = run_mfq_experiment(personas, args.model_type, args.model_name, **model_kwargs)

    # Save results
    output_prefix = f"data/{args.output}_{args.model_type}_{args.model_name.replace(':', '_')}"
    save_results(results, output_prefix)

    print(f"\nExperiment completed! Processed {len(results)} personas.")

if __name__ == "__main__":
    main()