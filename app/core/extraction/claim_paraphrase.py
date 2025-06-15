"""
LLM-powered claim → search-query paraphraser.

The main entry-point is `paraphrase_claim(claim: str) -> str`, which returns
ONE high-quality Google/Bing query tailored for maximising recall of
relevant evidence.  Internally we ask the model for a *set* of queries
(3-5) and pick the first; feel free to experiment with ranking/selection.
"""
from __future__ import annotations

import json
from typing import List

from ...infrastructure.llm.llm_client import chat

_FUNC_SCHEMA = [
    {
        "name": "paraphrase_for_search",
        "description": (
            "Reformulate the given *claim* into 3-5 concise, high-recall "
            "web-search queries.  Each query should:\n"
            "  • be ≤ ~12 words\n"
            "  • keep critical named entities / dates / numbers\n"
            "  • add quotation marks for exact phrases when it helps\n"
            "  • avoid hashtags or advanced operators other than quotes\n"
            "Return a JSON object: {\"queries\": [ ... ]}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["queries"],
        },
    }
]


def _llm_paraphrase(claim: str) -> List[str]:
    """Internal helper – call the LLM functionally."""
    sys = {
        "role": "system",
        "content": (
            "You are an expert fact-checking assistant who writes superb "
            "Google queries."
        ),
    }
    usr = {"role": "user", "content": claim}

    msg = chat([sys, usr], functions=_FUNC_SCHEMA)

    # Fallback – the LLM might answer normally if tool call fails
    if not getattr(msg, "tool_calls", None):
        return [msg.content.strip() or claim]

    args = json.loads(msg.tool_calls[0].function.arguments)
    queries = [q.strip() for q in args.get("queries", []) if q.strip()]
    return queries or [claim]


# --------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------- #
def paraphrase_claim(claim: str) -> str:
    """
    Return the *single* best search query for the claim.
    Currently we simply take the first suggestion.
    """
    return _llm_paraphrase(claim)[0]
