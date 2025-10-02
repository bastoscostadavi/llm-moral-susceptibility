#!/usr/bin/env python3
"""
LLM utilities for different model interfaces
"""

import os
import time
from typing import Optional

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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def _openai_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from OpenAI API"""
    try:
        import openai
        client = openai.OpenAI(api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 50),
            temperature=kwargs.get("temperature", 0.1)
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
            max_tokens=kwargs.get("max_tokens", 50),
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
                "num_predict": kwargs.get("max_tokens", 10)
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

    return []