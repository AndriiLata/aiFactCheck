from __future__ import annotations
from typing import List, Dict, Optional

from ...config import Settings

settings = Settings()

if settings.PROVIDER_IN_USE == "azure":
    # Azure OpenAI
    from openai import AzureOpenAI as _OpenAIClient

    _CLIENT = _OpenAIClient(
        api_version="2025-01-01-preview",
        azure_endpoint=settings.AZURE_ENDPOINT,
        api_key=settings.AZURE_API_KEY,
    )
    _MODEL = "gpt-4o"
else:
    # vanilla OpenAI
    from openai import OpenAI as _OpenAIClient

    _CLIENT = _OpenAIClient(api_key=settings.OPENAI_API_KEY)
    _MODEL = "gpt-4o-mini"  # cheap chat model

_MAX_TOKENS = 4096


def chat(messages: List[Dict], functions: Optional[List[Dict]] = None):
    """
    Unified chat wrapper – same call-signature regardless of provider.
    """
    kwargs = {
        "model": _MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": _MAX_TOKENS,
    }

    if functions:
        kwargs["tools"] = [{"type": "function", "function": f} for f in functions]
        kwargs["tool_choice"] = "auto"

    resp = _CLIENT.chat.completions.create(**kwargs)
    return resp.choices[0].message
