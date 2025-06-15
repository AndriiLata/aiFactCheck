from __future__ import annotations
import json
import re
from typing import List, Optional

from ...infrastructure.llm.llm_client import chat
from ...models import Triple

from hashlib import sha256

cache = {}

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
    claim = re.sub(r"\b(a|an|the)\b\s+", "", claim, flags=re.IGNORECASE)
    key = sha256(claim.encode()).hexdigest()
    if key in cache:
        return cache[key]
    sys = {
        "role": "system",
        "content": (
            "You are a precise OpenIE system. Extract canonical factual triples from the claim. "
            "Avoid including articles like 'a', 'the', etc. in subject or object."
        )
    }
    usr = {"role": "user", "content": claim}
    msg = chat([sys, usr], functions=_FUNC_SCHEMA)

    if not getattr(msg, "tool_calls", None):
        return []

    args = json.loads(msg.tool_calls[0].function.arguments)
    
    cache[key] = [Triple(**t) for t in args.get("triples", [])]
    return [Triple(**t) for t in args.get("triples", [])]


# --------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------- #
def parse_claim_to_triple(claim: str) -> Optional[Triple]:

    triples = _llm_extract(claim)

    return triples[0] if triples else None
