#!/usr/bin/env python3
"""
Generate 10 different persona samples of 100 personas each for testing stability
"""

import pandas as pd
import json
from datasets import load_dataset

def generate_multiple_persona_samples(num_samples=10, personas_per_sample=100):
    """
    Generate multiple random persona samples for stability testing

    Args:
        num_samples: Number of different samples to generate
        personas_per_sample: Number of personas in each sample
    """

    print(f"Loading Persona Hub dataset...")

    try:
        # Load the persona subset from Persona Hub
        dataset = load_dataset("proj-persona/PersonaHub", "persona")
        personas_data = dataset['train']

        print(f"Total personas available: {len(personas_data)}")

        # Convert to pandas DataFrame
        df = personas_data.to_pandas()

        if len(df) < personas_per_sample:
            print(f"Warning: Only {len(df)} personas available")
            personas_per_sample = len(df)

        # Generate multiple samples with different random seeds
        for i in range(1, num_samples + 1):
            print(f"Generating personas_sample_{i:02d}.json...")

            # Sample with different random seed for each sample
            sampled_df = df.sample(n=personas_per_sample, random_state=i*42)

            # Convert to list of dictionaries
            personas_list = sampled_df.to_dict('records')

            # Save to JSON file
            output_file = f"personas_sample_{i:02d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(personas_list, f, indent=2, ensure_ascii=False)

            print(f"  ✅ Saved {len(personas_list)} personas to {output_file}")

        print(f"\n🎉 Successfully generated {num_samples} persona samples!")
        print(f"Each sample contains {personas_per_sample} personas")
        print(f"Use these for testing moral susceptibility stability across different persona sets")

        return True

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    generate_multiple_persona_samples(10, 100)