"""
Light-weight evidence collectors for the *new* multi-agent pipeline.

Agents
------
KGEvidenceRetriever  – entity-link → KG path retrieval  (DBpedia)
WebEvidenceRetriever – claim-paraphrase → web snippets  (SerpAPI)

Both classes are synchronous; the surrounding pipeline decides
whether to run them sequentially or concurrently.
"""
from __future__ import annotations

from typing import List, Tuple, Dict

from ..linking.entity_linker2 import EntityLinker2
from ...infrastructure.kg.kg_client2 import KGClient2
from ..verification.web_verifier import WebVerifier
from ..extraction.claim_paraphrase import paraphrase_claim
from .trust import score_for_url
from ...models import Edge


# --------------------------------------------------------------------- #
# KG agent
# --------------------------------------------------------------------- #
class KGEvidenceRetriever:
    """Link entities in the *claim* and pull 1-hop DBpedia edges."""

    def __init__(self, *, max_hops: int = 1) -> None:
        self._linker = EntityLinker2()
        self._kg     = KGClient2()
        self._max_hops = max_hops

    # public ----------------------------------------------------------- #
    def retrieve(self, claim: str) -> Tuple[List[str], List[List[Edge]]]:
        """
        Returns
        -------
        dbpedia_uris : List[str]
        paths        : List[List[Edge]]
        """
        dbpedia_uris = self._linker.link(claim)
        if not dbpedia_uris:
            return [], []

        paths = self._kg.fetch_paths(dbpedia_uris, max_hops=self._max_hops)
        return dbpedia_uris, paths


# --------------------------------------------------------------------- #
# Web / RAG agent
# --------------------------------------------------------------------- #
class WebEvidenceRetriever:
    """
    Claim-paraphrase → Google (via SerpAPI) → normalised evidence dicts.
    Intended **only** as a *fallback* when the KG agent cannot decide.
    """

    def __init__(self, *, top_k: int = 100) -> None:
        self._top_k = top_k
        self._wv    = WebVerifier(num_results=top_k)

    # public ----------------------------------------------------------- #
    def retrieve(self, claim: str) -> List[Dict]:
        query = paraphrase_claim(claim)
        search_results = self._wv._search(query)

        evidence = []
        for r in search_results:
            snip = r.get("snippet") or ""
            if not snip:
                continue
            link = r.get("link") or ""
            evidence.append(
                {
                    "snippet": snip,
                    "source":  link,
                    "trust":   score_for_url(link),
                }
            )
        return evidence
