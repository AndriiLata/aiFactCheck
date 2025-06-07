from __future__ import annotations
import json
from typing import List, Optional

from ...infrastructure.llm.llm_client import chat
from ...models import Triple



_FUNC_SCHEMA = [
    {
        "name": "extract_triples",
        "description": "Extract ONE OR MORE factual triples from the claim.",
        "parameters": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                }
            },
            "required": ["triples"],
        },
    }
]


def _llm_extract(claim: str) -> List[Triple]:
    sys = {
        "role": "system",
        "content": (
            "You are a precise OpenIE system. Return ONLY a tool call that "
            "extracts every factual triple in the claim."
        ),
    }
    usr = {"role": "user", "content": claim}
    msg = chat([sys, usr], functions=_FUNC_SCHEMA)

    if not getattr(msg, "tool_calls", None):
        return []

    args = json.loads(msg.tool_calls[0].function.arguments)
    return [Triple(**t) for t in args.get("triples", [])]


# --------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------- #
def parse_claim_to_triple(claim: str) -> Optional[Triple]:

    triples = _llm_extract(claim)

    return triples[0] if triples else None
