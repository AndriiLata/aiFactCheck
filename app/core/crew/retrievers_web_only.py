"""
Asynchronous evidence retriever that **only** leverages web search.

It drops all KG logic and relies on `claim_paraphrase.paraphrase_claim`
to build the query string.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List

from ..extraction.claim_paraphrase import paraphrase_claim
from ..verification.web_verifier import WebVerifier
from .trust import score_for_url


async def _collect_web(claim: str, top_k: int = 100) -> List[Dict]:
    """Search the web using a paraphrased query and normalise snippets."""
    query = paraphrase_claim(claim)
    wv = WebVerifier(num_results=top_k)
    results = wv._search(query)

    evidence = []
    for r in results:
        snip = r.get("snippet") or ""
        if not snip:
            continue
        link = r.get("link") or ""
        evidence.append(
            {
                "snippet": snip,
                "source": link,
                "trust": score_for_url(link),
            }
        )
    return evidence


# ------------------------------------------------------------------ #
# Public helper
# ------------------------------------------------------------------ #
async def gather_evidence_web_only(claim: str) -> Dict[str, List[Dict]]:
    """
    Thin wrapper that mirrors the structure returned by the old
    `gather_evidence`, but with ONLY the 'web' key populated.
    """
    web = await _collect_web(claim)
    return {"web": web}
