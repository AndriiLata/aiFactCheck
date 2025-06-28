"""
Unified **multi-agent** fact-checking pipeline used by `/verify_crewAI`.

Strategy
--------
1. KG-agent
   • Entity linking  (EntityLinker2)
   • 1-hop KG paths  (KGClient2)
   • Evidence rank   (EvidenceRanker2)
   • Verdict         (Verifier2)

2. If the KG verdict is *Not Enough Info*  →  Web / RAG-agent
   • Paraphrase + Google (SerpAPI or Brave Search)
   • MiniLM similarity filtering
   • Batch NLI           (DeBERTa-v3 MNLI)
   • Weighted vote aggregation

Public helper
-------------
    verify_claim_crew(claim: str) -> dict
        Synchronous – safe to call from Flask.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .retrievers import KGEvidenceRetriever, WebEvidenceRetriever
from ..ranking.evidence_ranker2 import EvidenceRanker2
from ..verification.verifier2 import Verifier2
from .synthesiser import synthesise
from .nli import batch_nli
from .verdict import _aggregate as aggregate
from ...models import Edge


# ------------------------------------------------------------------ #
# Helper utilities
# ------------------------------------------------------------------ #
def _flatten_edges(ranked_paths: List[Tuple[List[Edge], float]], k: int) -> List[Edge]:
    """Take the first *k* individual edges from the ranked paths list."""
    edges: List[Edge] = []
    for path, _ in ranked_paths:
        edges.extend(path)
        if len(edges) >= k:
            break
    return edges[:k]


# ------------------------------------------------------------------ #
# Main orchestration
# ------------------------------------------------------------------ #
def verify_claim_crew(claim: str) -> Dict:
    """
    Multi-agent reasoning wrapper – **no async/await needed**.
    Returns a JSON-serialisable dict ready for `Flask.jsonify`.
    """
    # ---------- 1.  KG AGENT ---------------------------------------- #
    kg_ret   = KGEvidenceRetriever()
    uris, paths = kg_ret.retrieve(claim)

    if paths:
        ranker  = EvidenceRanker2(claim_text=claim)
        ranked  = ranker.top_k(paths, k=3, use_bi_encoder=False)
        edges   = _flatten_edges(ranked, k=3)

        verifier = Verifier2()
        label, reason = verifier.classify(claim, edges)
    else:
        ranked, label, reason = [], "Not Enough Info", "No KG evidence retrieved."

    # ---------- 2.  SUCCESS on KG branch --------------------------- #
    if label in ("Supported", "Refuted"):
        return {
            "claim": claim,
            "label": label,
            "reason": reason,
            "all_top_evidence_paths": [
                [e.__dict__ for e in p] for p, _ in ranked
            ],
            "entity_linking": {
                "candidates": uris,
            },
        }

    # ---------- 3.  FALLBACK → WEB / RAG agent -------------------- #
    web_ret  = WebEvidenceRetriever()
    web_ev   = web_ret.retrieve(claim)

    if not web_ev:
        return {
            "claim": claim,
            "label": "Not Enough Info",
            "reason": "Neither KG nor web search produced usable evidence.",
            "evidence": [],
        }

    syn_ev   = synthesise(claim, web_ev, top_k=100)
    nli_out  = batch_nli(claim, [e["snippet"] for e in syn_ev])
    lbl, conf, annotated_ev = aggregate(syn_ev, nli_out)

    return {
        "claim": claim,
        "label": lbl,
        "confidence": conf,
        "evidence": annotated_ev,
        "fallback_used": True,
    }
