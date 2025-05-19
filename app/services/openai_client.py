from __future__ import annotations

from openai import OpenAI
from ..config import Settings

settings = Settings()

# instantiate once (thread-safe)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

MODEL_NAME = "gpt-4.1-nano"          # cheapest text-completion model
MAX_TOKENS = 256                     # cost guardrail


def chat(messages: list[dict], functions: list[dict] | None = None):
    """
    Wrapper that talks to the new chat API.
    If *functions* is supplied, they’re converted to the new 'tools' format.
    Returns the *first* message object (assistant or tool call).
    """
    kwargs: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }

    if functions:
        kwargs["tools"] = [{"type": "function", "function": f} for f in functions]
        kwargs["tool_choice"] = "auto"

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message
