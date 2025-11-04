# LLM Moral Susceptibility Experiments

This repository explores how persona conditioning alters large language model responses on the 30-item Moral Foundations Questionnaire (MFQ-30). The runner will impersonate each persona and ask every MFQ question multiple times, logging the numeric rating returned by the model.

## Repository Layout

```
.
├── data/                         # Streaming CSV outputs (one per model)
├── results/                      # Optional sandbox for downstream analyses
├── generate_persona_samples.py   # Utility to slice persona datasets
├── llm_interface.py              # Shared LLM access layer (local + API providers)
├── mfq_questions.py              # Canonical MFQ-30 catalog and prompt helpers
├── personas.json                 # Default persona list (array of descriptions)
└── run_mfq_experiment.py         # Experiment entry point / CLI
```

## Prerequisites

1. **Python packages**

   ```bash
   pip install -r requirements.txt
   pip install llama-cpp-python  # required for the local GGUF workflow
   ```

2. **Configure environment variables**

   Create or edit the provided `.env` file and add any API keys you plan to use. For Claude Sonnet 4.5 you must supply `ANTHROPIC_API_KEY`:

   ```bash
   echo "ANTHROPIC_API_KEY=your-key-here" >> .env
   ```

   You can also set keys via your shell environment; `python-dotenv` loads `.env` automatically when present.

   For xAI Grok models, set `XAI_API_KEY` (the client uses OpenAI-compatible endpoints at `https://api.x.ai/v1`):

   ```bash
   echo "XAI_API_KEY=your-xai-key" >> .env
   ```

   To reach OpenRouter-hosted models (e.g., GPT-5), provide your key via

   ```bash
   echo "OPENROUTER_API_KEY=your-openrouter-key" >> .env
   echo "OPENROUTER_APP_NAME=Your App Name" >> .env        # optional but recommended
   echo "OPENROUTER_APP_URL=https://your-app.example" >> .env  # optional but recommended
   ```

   OpenRouter honours the `HTTP-Referer` and `X-Title` headers for rate
   bumping; populate the optional variables above (or set
   `OPENROUTER_HTTP_REFERER`) so the client can forward them automatically.

3. **MFQ personas** (already provided as `personas.json`). Use `generate_persona_samples.py` if you need alternative subsets.

3. **Local GGUF model**

   Download `Mistral-7B-Instruct-v0.3-Q8_0.gguf` (or another chat-tuned model) and place it in `../models/` relative to this project, or choose any directory and pass it via `--models-dir`.

   ```bash
   mkdir -p ../models
   mv /path/to/Mistral-7B-Instruct-v0.3-Q8_0.gguf ../models/
   ```

   Other providers (OpenAI, Anthropic, etc.) remain supported through `llm_interface.py`, provided their Python SDKs and API keys are configured.

## Running Experiments

The CLI now prompts for a model when no `--model-type/--model-name` are supplied. With the GGUF above in `../models`, you can simply run:

```bash
python run_mfq_experiment.py --limit 5
```

or provide everything explicitly:

```bash
python run_mfq_experiment.py \
  --model-type local \
  --model-name Mistral-7B-Instruct-v0.3-Q8_0.gguf \
  --models-dir ../models \
  --personas-file personas.json \
  --n 10 \
  --limit 100
```

To query Claude Sonnet 4.5 through the Anthropic API, ensure `ANTHROPIC_API_KEY` is set in your environment (or `.env`) and run:

```bash
python run_mfq_experiment.py \
  --model-type anthropic \
  --model-name claude-4.5-sonnet \
  --n 5 \
  --limit 20
```

To use xAI Grok-4-Fast, set `XAI_API_KEY` and select “xAI Grok-4-Fast” from the interactive menu (works in both persona and self runners). The preset uses `max_tokens=2` to keep outputs short.

To run GPT-5 through OpenRouter, configure the API variables above and select
“OpenRouter GPT-5” when prompted (or pass
`--model-type openrouter --model-name openai/gpt-5` on the CLI). The preset uses
`max_tokens=8` with a conservative temperature; adjust `model_kwargs` in
`run_mfq_experiment.py` if you need different decoding parameters.

For GPT‑5 family with minimal reasoning, select either “OpenAI GPT‑5 (minimal reasoning)” or “OpenAI GPT‑5 Mini (minimal reasoning)”. These use the Responses API and `reasoning: {effort: "minimal"}`.

Key flags:

- `--n`: number of times each persona answers a given MFQ item (default 10).
- `--limit`: cap the number of personas processed (useful for smoke tests).
- `--models-dir`: directory that contains the target GGUF model when using `--model-type local`.
- `--output`: filename prefix for the CSV stored in `data/` (defaults to `mfq_results`).

### Custom CSV filenames

If you rename the per-model CSVs under `data/`, update the
`CUSTOM_MODEL_FILENAMES` dictionary at the top of `run_mfq_experiment.py`.
Entries map the model name (as passed to the runner) to the desired CSV
filename. The self-run script reuses the same mapping, automatically
appending `_self` when writing the no-persona results.

## Result Format

Every response is appended to a per-model CSV in `data/`. The filename is derived from the output prefix and sanitized model name, e.g. `data/mfq_results_Mistral-7B-Instruct-v0.3-Q8_0.gguf.csv`.

Each row captures a single rating:

| Column        | Description                                                         |
|---------------|---------------------------------------------------------------------|
| `persona_id`  | Zero-based index of the persona in the loaded list (self runs omit) |
| `question_id` | Canonical MFQ item id (1-32; 6 and 22 are official MFQ filler items) |
| `run_index`   | 1-based counter of repeated queries for that persona/question pair  |
| `rating`      | Parsed integer 0–5 returned by the model (−1 if no rating detected) |
| `failures`    | Count of failed attempts (rating −1) recorded before this row       |
| `response`    | First few tokens of the model’s raw reply after whitespace trimming |
| `collected_at`| ISO8601 timestamp written immediately after receiving the response  |

Rows are flushed to disk as they are gathered, so partial runs still yield usable data.

Self-run CSVs (`*_self.csv`) use the same schema without the `persona_id`
column.

## MFQ Catalog Helpers

`mfq_questions.py` now exposes a single ordered list of `MFQQuestion` dataclasses. Use these helpers when analysing results:

```python
from mfq_questions import iter_questions, get_question

# Iterate in MFQ canonical order
for question in iter_questions():
    print(question.id, question.question_type, question.foundation)

# Lookup metadata for a specific question id
q5 = get_question(5)
print(q5.text)
# Filler items (6 and 22) report foundation='useless'
```

This structure eliminates the need to reconstruct foundation-specific groupings when aligning CSV question ids with the original prompts.

## Personas and Samples

- `personas.json` contains the default persona descriptions (plain strings). The runner implicitly uses each list position as the persona identifier.
- Generate alternative slices with `python generate_persona_samples.py` if you wish to test stability across different persona cohorts.

## Troubleshooting

- If you see `Local model error: ...`, double-check the GGUF path or install status of `llama-cpp-python`.
- Ratings of `-1` indicate the model produced no parsable score; inspect those responses manually to adjust prompts or retry with different parameters.
- Existing CSVs are opened in append mode. Remove or rename old files if you want a clean slate for a new run.

## License

Released under the MIT License. See `LICENSE` for details.
