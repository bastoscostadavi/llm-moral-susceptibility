#!/usr/bin/env python3
"""
LLM utilities for different model interfaces
"""

import os
import random
import time
import uuid
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
    elif model_type == "google":
        return _google_response(model_name, prompt, **kwargs)
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

        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)

        # Normalise advanced knobs so we only send recognised fields.
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort in {"minimal", "low", "medium", "high"}:
            pass
        elif reasoning_effort is not None:
            reasoning_effort = "minimal"

        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p")
        presence_penalty = kwargs.get("presence_penalty")
        frequency_penalty = kwargs.get("frequency_penalty")
        max_tokens = kwargs.get("max_tokens", 8)
        system_prompt = kwargs.get("system_prompt") or kwargs.get("system")
        instructions = kwargs.get("instructions")

        is_gpt5 = model_name.startswith("gpt-5") or "gpt-5" in model_name
        use_responses_api = bool(kwargs.get("use_responses_api") or reasoning_effort or is_gpt5)

        def _build_responses_input(user_prompt: str) -> list:
            messages = []
            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": system_prompt},
                        ],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                    ],
                }
            )
            return messages

        def _extract_response_text(response: Any) -> str:
            text = getattr(response, "output_text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

            try:
                parts = []
                for item in getattr(response, "output", []) or []:
                    content = item.get("content") if isinstance(item, dict) else None
                    if isinstance(content, list):
                        for piece in content:
                            if isinstance(piece, dict) and piece.get("type") in {"output_text", "message"}:
                                candidate = piece.get("text") or piece.get("content")
                                if isinstance(candidate, str):
                                    parts.append(candidate)
            except Exception:
                parts = []

            if parts:
                combined = "\n".join(part for part in parts if part)
                if combined.strip():
                    return combined.strip()

            try:
                return response.model_dump_json()
            except Exception:
                return ""

        if use_responses_api:
            if is_gpt5:
                # GPT-5 prefers the simplified Responses payload without extra tuning knobs.
                messages: list = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                req: Dict[str, Any] = {
                    "model": model_name,
                    "input": messages,
                }

                if max_tokens is not None:
                    req["max_output_tokens"] = max_tokens
                if instructions:
                    req["instructions"] = instructions
                if reasoning_effort:
                    req["reasoning"] = {"effort": reasoning_effort}

            else:
                req = {
                    "model": model_name,
                    "input": _build_responses_input(prompt),
                }
                if max_tokens is not None:
                    req["max_output_tokens"] = max_tokens
                if temperature is not None:
                    req["temperature"] = temperature
                if top_p is not None:
                    req["top_p"] = top_p
                if presence_penalty is not None:
                    req["presence_penalty"] = presence_penalty
                if frequency_penalty is not None:
                    req["frequency_penalty"] = frequency_penalty
                if instructions:
                    req["instructions"] = instructions
                if reasoning_effort:
                    req["reasoning"] = {"effort": reasoning_effort}

            response = None
            last_exc: Optional[Exception] = None

            max_attempts = kwargs.get("max_retries") or (5 if is_gpt5 else 1)
            backoff = kwargs.get("initial_backoff") or 1.0
            idem_key = kwargs.get("idempotency_key") or str(uuid.uuid4())

            extra_headers = dict(kwargs.get("extra_headers", {}) or {})
            if is_gpt5 and "Idempotency-Key" not in extra_headers:
                extra_headers["Idempotency-Key"] = idem_key

            for attempt in range(max_attempts):
                create_kwargs = dict(req)
                if extra_headers:
                    create_kwargs["extra_headers"] = extra_headers

                try:
                    response = client.responses.create(**create_kwargs)
                    break
                except Exception as exc:
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    should_retry = is_gpt5 and status in {429, 500, 502, 503, 504}
                    if not should_retry or attempt + 1 >= max_attempts:
                        break
                    print(
                        "OpenAI Responses API error for GPT-5 "
                        f"(attempt {attempt + 1}/{max_attempts}): {exc}"
                    )
                    sleep_time = backoff + random.uniform(0, 0.25)
                    time.sleep(sleep_time)
                    backoff *= 2

            if response is not None:
                return _extract_response_text(response)

            if last_exc is not None:
                print(f"OpenAI Responses API error: {last_exc}")

            if is_gpt5:
                return "ERROR"

        # Default: Chat Completions API fallback (older models and general safety net)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_base_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        chat_max_tokens = kwargs.get("chat_max_tokens")
        if chat_max_tokens is None:
            chat_max_tokens = max_tokens

        last_chat_error: Optional[Exception] = None
        for token_param in ("max_completion_tokens", "max_tokens"):
            chat_kwargs = dict(chat_base_kwargs)
            if chat_max_tokens is not None:
                chat_kwargs[token_param] = chat_max_tokens

            try:
                response = client.chat.completions.create(**chat_kwargs)
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_chat_error = exc
                details = str(exc)
                if f"Unsupported parameter: '{token_param}'" not in details:
                    raise

        if last_chat_error is not None:
            raise last_chat_error

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

def _google_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from Google Gemini via google-generativeai SDK.

    Expects GOOGLE_API_KEY in environment or passed as api_key.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError("Please install google-generativeai: pip install google-generativeai") from exc

    api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY for Google Gemini API")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        gen_cfg = {}
        if "temperature" in kwargs:
            gen_cfg["temperature"] = kwargs.get("temperature")
        if "max_tokens" in kwargs:
            gen_cfg["max_output_tokens"] = kwargs.get("max_tokens")

        thinking_cfg = kwargs.get("thinking_config") or None
        if "thinkingBudget" in kwargs:
            thinking_cfg = dict(thinking_cfg or {})
            thinking_cfg["max_thinking_tokens"] = kwargs.get("thinkingBudget")

        call_kwargs = {"generation_config": gen_cfg or None}
        # For thinking-enabled models, allow passing explicit thinking_config.
        if thinking_cfg is not None:
            call_kwargs["thinking_config"] = thinking_cfg

        response = model.generate_content(prompt, **call_kwargs)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Fallback: try candidates
        try:
            for cand in getattr(response, "candidates", []) or []:
                parts = []
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", []) or []:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(str(t))
                if parts:
                    return "\n".join(parts).strip()
        except Exception:
            pass
        return ""
    except Exception as e:
        print(f"Google Gemini API error: {e}")
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

    elif model_type == "google":
        api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
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
