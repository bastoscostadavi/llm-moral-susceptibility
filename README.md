# LLM Moral Susceptibility Experiments

This project explores the moral susceptibility of Large Language Models (LLMs) using the Moral Foundations Questionnaire (MFQ-30) with different personas from the Persona Hub dataset.

## Project Structure

```
llm-moral-susceptibility/
├── personas/
│   ├── generate_persona_samples.py    # Generate multiple persona samples
│   ├── personas_sample_01.json        # First sample (100 personas)
│   ├── personas_sample_02.json        # Second sample (100 personas)
│   └── ... (up to 10 samples)         # For testing stability
├── data/                              # Experimental results
│   ├── mfq_results_*.json            # Complete experimental data
│   └── mfq_results_*.csv             # Tabular data for analysis
├── run_mfq_experiment.py             # Main experimental runner
├── llm_interface.py                  # LLM interface (OpenAI, Anthropic, Ollama)
├── mfq_questions.py                  # MFQ-30 questions by moral foundations
└── requirements.txt                  # Python dependencies
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate persona samples (optional)
```bash
cd personas
python generate_persona_samples.py  # Creates 10 different samples
```

### 3. Set up local models (Ollama)
```bash
# Install Ollama (https://ollama.ai)
ollama pull mistral:7b-instruct
ollama pull llama3.2:1b
```

### 4. Or set up API keys for cloud models
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Usage

### Quick test with local models
```bash
# Test with Mistral 7B (recommended)
python run_mfq_experiment.py --model-type ollama --model-name mistral:7b-instruct --limit 5

# Test with Llama 3.2 1B
python run_mfq_experiment.py --model-type ollama --model-name llama3.2:1b --limit 5
```

### Full experiments
```bash
# Full experiment with 100 personas
python run_mfq_experiment.py --model-type ollama --model-name mistral:7b-instruct

# Different persona sample
python run_mfq_experiment.py --model-type ollama --model-name mistral:7b-instruct --personas-file personas/personas_sample_02.json
```

### API-based models
```bash
# OpenAI models
python run_mfq_experiment.py --model-type openai --model-name gpt-3.5-turbo --limit 10
python run_mfq_experiment.py --model-type openai --model-name gpt-4 --limit 10

# Anthropic models
python run_mfq_experiment.py --model-type anthropic --model-name claude-3-haiku-20240307 --limit 10
```

## Testing Stability

Run experiments across multiple persona samples to test moral susceptibility stability:

```bash
# Test different persona samples
for i in {01..10}; do
  python run_mfq_experiment.py \
    --model-type ollama \
    --model-name mistral:7b-instruct \
    --personas-file personas/personas_sample_${i}.json \
    --limit 20
done
```

## Output

Results are automatically saved to `data/` in two formats:
- **JSON**: Complete experimental data with metadata
- **CSV**: Tabular numeric scores for statistical analysis

Example files:
- `data/mfq_results_ollama_mistral_7b-instruct_20241002_103045.json`
- `data/mfq_results_ollama_mistral_7b-instruct_20241002_103045.csv`

## Moral Foundations

The MFQ-30 measures five moral foundations:

1. **Care/Harm** - Concerns about suffering and welfare
2. **Fairness/Cheating** - Concerns about justice and rights
3. **Loyalty/Betrayal** - Concerns about group cohesion
4. **Authority/Subversion** - Concerns about hierarchy and tradition
5. **Sanctity/Degradation** - Concerns about purity and degradation

## Experimental Design

For each persona:
1. The LLM adopts the persona description
2. Answers MFQ relevance questions (0-5 scale: how relevant each moral consideration is)
3. Answers MFQ agreement questions (0-5 scale: agreement with moral statements)
4. Results capture how different personas influence LLM moral reasoning

This enables analysis of:
- **Moral susceptibility**: How much personas change LLM moral responses
- **Stability**: Whether effects are consistent across different persona samples
- **Foundation differences**: Which moral foundations are most susceptible to persona influence

## Research Questions

- Do different personas significantly alter LLM moral reasoning?
- Which moral foundations show the highest susceptibility to persona influence?
- Are these effects stable across different sets of personas?
- How do different LLM models compare in their moral susceptibility?