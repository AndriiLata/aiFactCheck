"""
Hybrid triple extraction:
1. Try Stanford-OpenIE locally (fast, free).
2. Fallback to GPT only if OpenIE produced < 2 triples.
"""
from __future__ import annotations

import json
from typing import Optional, List

from .openai_client import chat
from .azure_client import chat_a
from ..config      import Settings

from .models import Triple
              # starts an internal CoreNLP JVM

settings = Settings()

def get_chat_func():
    return chat_a if settings.PROVIDER_IN_USE.lower() == "azure" else chat

def _openie_extract(claim: str) -> List[Triple]:
    # ToDO!!!

    return []


# --------------------------------------------------------------------------- #
# 1) GPT fallback
# --------------------------------------------------------------------------- #
_FUNC_SCHEMA = [{
    "name": "extract_triples",
    "description": (
        "Parse the input claim into ONE OR MORE subject-predicate-object triples "
        "that fully capture the factual assertions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject":   {"type": "string"},
                        "predicate": {"type": "string"},
                        "object":    {"type": "string"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            }
        },
        "required": ["triples"],
    },
}]


def _llm_extract(claim: str) -> List[Triple]:
    sys = {
        "role": "system",
        "content": (
            "You are a precise OpenIE system. Return ONLY a tool call that extracts "
            "every factual triple in the claim. Use concise noun phrases."
        ),
    }
    usr = {"role": "user", "content": claim}
    chat_func = get_chat_func()
    msg = chat_func([sys, usr], functions=_FUNC_SCHEMA)

    if not getattr(msg, "tool_calls", None):
        return []
    args = json.loads(msg.tool_calls[0].function.arguments)
    return [Triple(**t) for t in args.get("triples", [])]


# --------------------------------------------------------------------------- #
# 2) Public helper
# --------------------------------------------------------------------------- #
def parse_claim_to_triple(claim: str) -> Optional[Triple]:
    """
    • Run OpenIE first.
    • If OpenIE produced < 2 triples, call GPT.
    • Return the FIRST triple, else None.
    """
    triples = _openie_extract(claim)
    if len(triples) < 2:                         # 0 or 1 → use GPT fallback
        triples = _llm_extract(claim)
    return triples[0] if triples else None
