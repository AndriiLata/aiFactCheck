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
            "You are a precise OpenIE system. Extract every factual triple from the claim. "
            "Return ONLY a tool call that extracts all factual triples. "
            "If there are no factual triples, return an empty list. "
            "A factual triple is a (subject, predicate, object) statement that can be verified."
        ),
    }
    usr = {
        "role": "user",
        "content": (
            "Claim: The Eiffel Tower is located in Paris.\n"
            "Expected triples: [{'subject': 'The Eiffel Tower', 'predicate': 'is located in', 'object': 'Paris'}]\n\n"
            "Claim: Albert Einstein was born in Ulm and won the Nobel Prize in Physics.\n"
            "Expected triples: ["
            "{'subject': 'Albert Einstein', 'predicate': 'was born in', 'object': 'Ulm'}, "
            "{'subject': 'Albert Einstein', 'predicate': 'won', 'object': 'the Nobel Prize in Physics'}]\n\n"
            f"Claim: {claim}"
        )
    }
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
    # Post-process: filter out incomplete triples
    triples = [t for t in triples if t.subject and t.predicate and t.object]

    return triples if triples else None
