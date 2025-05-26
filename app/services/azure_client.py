import os
from openai import AzureOpenAI
from ..config import Settings

settings = Settings()

model_name = "gpt-4o"
deployment = "gpt-4o"

endpoint = settings.AZURE_ENDPOINT
subscription_key=settings.AZURE_API_KEY
api_version = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


MODEL_NAME = model_name          # cheapest text-completion model
MAX_TOKENS = 4096                   # cost guardrail

def chat_a(messages: list[dict], functions: list[dict] | None = None):
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