#!/usr/bin/env python3
"""Export top persona failure counts by model to a CSV report."""

from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "results" / "top_persona_failures.csv"
PERSONAS_JSON = PROJECT_ROOT / "personas.json"


def load_personas() -> List[str]:
    if PERSONAS_JSON.exists():
        try:
            return json.loads(PERSONAS_JSON.read_text())
        except json.JSONDecodeError:
            pass
    return []


def aggregate_failures() -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    total_by_persona: Dict[int, float] = {}
    breakdown: Dict[int, Dict[str, float]] = {}

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.name.endswith("_self.csv"):
            continue

        model = csv_path.stem
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "persona_id" not in reader.fieldnames or "failures" not in reader.fieldnames:
                continue

            for row in reader:
                persona_raw = row.get("persona_id", "")
                failure_raw = row.get("failures", "")

                try:
                    persona_id = int(float(persona_raw))
                except (TypeError, ValueError):
                    continue

                try:
                    failures = float(failure_raw) if failure_raw else 0.0
                except (TypeError, ValueError):
                    continue

                if failures <= 0:
                    continue

                total_by_persona[persona_id] = total_by_persona.get(persona_id, 0.0) + failures
                model_breakdown = breakdown.setdefault(persona_id, {})
                model_breakdown[model] = model_breakdown.get(model, 0.0) + failures

    return total_by_persona, breakdown


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=4, help="Number of personas to include (default: 4)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    total_by_persona, breakdown = aggregate_failures()
    if not total_by_persona:
        raise SystemExit("No persona failures found in dataset")

    personas = load_personas()

    sorted_personas = sorted(total_by_persona.items(), key=lambda kv: kv[1], reverse=True)
    top_personas = sorted_personas[: max(args.top, 0)]

    if not top_personas:
        raise SystemExit("No personas selected")

    models = sorted({model for pid, _ in top_personas for model in breakdown.get(pid, {})})

    header = ["persona_id", "persona_name", *models, "total_failures"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for persona_id, total in top_personas:
            name = personas[persona_id] if persona_id < len(personas) else f"persona_{persona_id}"
            row = [persona_id, name]
            model_values = breakdown.get(persona_id, {})
            row.extend(model_values.get(model, 0.0) for model in models)
            row.append(total)
            writer.writerow(row)

    print(f"Wrote {len(top_personas)} personas to {args.output}")


if __name__ == "__main__":
    main()
