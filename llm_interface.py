#!/usr/bin/env python3
"""
LLM utilities for different model interfaces
"""

import os
import time
from typing import Optional, Dict, Any

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_CACHE: Dict[str, Any] = {}

def get_llm_response(model_type: str, model_name: str, prompt: str, **kwargs) -> str:
    """
    Get response from different LLM models

    Args:
        model_type: Type of model ('openai', 'anthropic', 'ollama')
        model_name: Specific model name
        prompt: Input prompt
        **kwargs: Additional parameters

    Returns:
        Model response as string
    """

    if model_type == "openai":
        return _openai_response(model_name, prompt, **kwargs)
    elif model_type == "anthropic":
        return _anthropic_response(model_name, prompt, **kwargs)
    elif model_type == "ollama":
        return _ollama_response(model_name, prompt, **kwargs)
    elif model_type == "local":
        return _local_response(model_name, prompt, **kwargs)
    elif model_type == "xai":
        return _xai_response(model_name, prompt, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def _openai_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from OpenAI API.

    Supports both Chat Completions and the Responses API. If
    `reasoning_effort` is provided (or `use_responses_api=True`), the
    Responses API is used and the effort level is passed through.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"))

        reasoning_effort = kwargs.get("reasoning_effort")
        # Accept both cookbook-style "minimal" and older "low" synonyms
        if reasoning_effort in {"minimal", "low", "medium", "high"}:
            pass  # keep as provided
        elif reasoning_effort is not None:
            # Unknown effort hint; coerce to minimal to be safe
            reasoning_effort = "minimal"

        is_gpt5 = model_name.startswith("gpt-5") or "gpt-5" in model_name
        use_responses_api = bool(kwargs.get("use_responses_api") or reasoning_effort or is_gpt5)

        if use_responses_api:
            # Use the newer Responses API to support reasoning parameters
            req: dict = {
                "model": model_name,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
                "max_output_tokens": kwargs.get("max_tokens", 5),
            }
            # GPT-5 does not use temperature; omit it in that case
            if not is_gpt5 and "temperature" in kwargs:
                req["temperature"] = kwargs.get("temperature", 0.1)
            if reasoning_effort:
                req["reasoning"] = {"effort": reasoning_effort}

            try:
                response = client.responses.create(**req)
            except Exception as exc:
                # For GPT-5, do not fall back to Chat Completions
                if is_gpt5:
                    print(f"OpenAI Responses API error for GPT-5: {exc}")
                    return "ERROR"
                # Otherwise, try Chat Completions as a fallback
                response = None
                print(f"OpenAI Responses API error: {exc}. Falling back to chat.completions.")

            if response is not None:
                # Prefer unified accessor when available
                text = getattr(response, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()

                # Fallback: concatenate any text items in the output
                try:
                    parts = []
                    for item in getattr(response, "output", []) or []:
                        content = item.get("content") if isinstance(item, dict) else None
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    parts.append(str(c.get("text", "")))
                    joined = "\n".join(p for p in parts if p)
                    if joined.strip():
                        return joined.strip()
                except Exception:
                    pass
                # As a last resort, try .model_dump_json() if available
                try:
                    return response.model_dump_json()
                except Exception:
                    return ""

        # Default: Chat Completions API
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 2),
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "ERROR"

def _anthropic_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from Anthropic API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model_name,
            max_tokens=kwargs.get("max_tokens", 5),
            temperature=kwargs.get("temperature", 0.1),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    except ImportError:
        raise ImportError("Please install anthropic: pip install anthropic")
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return "ERROR"

def _ollama_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from Ollama local API"""
    try:
        import requests

        base_url = kwargs.get("base_url", "http://localhost:11434")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 5)
            }
        }

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=kwargs.get("timeout", 30)
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "ERROR").strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return "ERROR"

    except ImportError:
        raise ImportError("Please install requests: pip install requests")
    except Exception as e:
        print(f"Ollama API error: {e}")
        return "ERROR"


def _resolve_model_path(model_name: str, model_dir: Optional[str] = None) -> str:
    """Resolve the absolute path to a local GGUF model."""

    if os.path.isabs(model_name):
        return model_name

    search_dir = model_dir or DEFAULT_MODELS_DIR
    return os.path.join(search_dir, model_name)


def _load_local_model(model_path: str, **kwargs):
    """Load and cache a local llama.cpp model."""

    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python") from exc

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Local model file not found: {model_path}")

    model = Llama(
        model_path=model_path,
        n_ctx=kwargs.get("n_ctx", 2048),
        n_gpu_layers=kwargs.get("n_gpu_layers", -1),
        verbose=kwargs.get("verbose", False),
    )

    _MODEL_CACHE[model_path] = model
    return model


def _local_response(model_name: str, prompt: str, **kwargs) -> str:
    """Generate a response from a local GGUF model via llama.cpp."""

    model_dir = kwargs.get("model_dir")
    model_path = _resolve_model_path(model_name, model_dir)

    try:
        model = _load_local_model(model_path, **kwargs)
    except Exception as exc:
        print(f"Local model error: {exc}")
        return "ERROR"

    try:
        result = model.create_completion(
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 5),
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 0.95),
            repeat_penalty=kwargs.get("repeat_penalty", 1.1),
            stream=False,
        )
    except Exception as exc:
        print(f"Local model generation error: {exc}")
        return "ERROR"

    choices = result.get("choices", []) if isinstance(result, dict) else []
    if choices:
        return choices[0].get("text", "").strip()

    return ""

def _xai_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from xAI (Grok) via OpenAI-compatible API.

    Uses the OpenAI SDK with base_url 'https://api.x.ai/v1'.
    Expects XAI_API_KEY in the environment or passed as api_key.
    """
    try:
        import openai
        api_key = kwargs.get("api_key") or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("Missing XAI_API_KEY for xAI Grok API")

        base_url = kwargs.get("base_url") or os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1"
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 5),
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.choices[0].message.content.strip()

    except ImportError as exc:
        raise ImportError("Please install openai: pip install openai") from exc
    except Exception as e:
        print(f"xAI API error: {e}")
        return "ERROR"

def check_model_availability(model_type: str, model_name: str, **kwargs) -> bool:
    """Check if a model is available and accessible"""

    if model_type == "ollama":
        try:
            import requests
            base_url = kwargs.get("base_url", "http://localhost:11434")

            # Check if Ollama is running
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if specific model is available
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            return model_name in available_models

        except Exception:
            return False

    elif model_type == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        return api_key is not None

    elif model_type == "anthropic":
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        return api_key is not None

    elif model_type == "local":
        model_dir = kwargs.get("model_dir")
        model_path = _resolve_model_path(model_name, model_dir)
        return os.path.isfile(model_path)

    elif model_type == "xai":
        api_key = kwargs.get("api_key") or os.getenv("XAI_API_KEY")
        return api_key is not None

    return False

def list_available_models(model_type: str, **kwargs) -> list:
    """List available models for a given model type"""

    if model_type == "ollama":
        try:
            import requests
            base_url = kwargs.get("base_url", "http://localhost:11434")

            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]

        except Exception:
            pass

    elif model_type == "local":
        model_dir = kwargs.get("model_dir") or DEFAULT_MODELS_DIR
        try:
            return [
                filename
                for filename in os.listdir(model_dir)
                if filename.lower().endswith(".gguf")
            ]
        except OSError:
            return []

    return []
