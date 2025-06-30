"""
Unified **multi-agent** fact-checking pipeline used by `/verify_crewAI`.

Strategy
--------
1. KG-agent (only in hybrid mode)
   • Entity linking  (EntityLinker2)
   • 1-hop KG paths  (KGClient2)
   • Evidence rank   (EvidenceRanker2)
   • Verdict         (Verifier2)

2. Web / RAG-agent (always runs, or only in web_only mode)
   • Paraphrase + Google (SerpAPI/Serper/Brave)
   • MiniLM similarity filtering
   • Batch NLI           (DeBERTa-v3 MNLI)
   • Weighted vote aggregation

Public helper
-------------
    verify_claim_crew(claim: str, mode: str = "hybrid") -> dict
        Mode options: "hybrid" (KG + Web fallback) or "web_only" (Web only)
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
def verify_claim_crew(claim: str, mode: str = "web_only", use_cross_encoder: bool = False) -> Dict:
    """
    Multi-agent reasoning wrapper with ranking method support.
    
    Parameters:
    - claim: The claim to verify
    - mode: "hybrid" (KG first, Web fallback) or "web_only" (Web only)
    
    Returns a JSON-serialisable dict ready for `Flask.jsonify`.
    """
    
    if mode == "web_only":
        print(f"Running WEB-ONLY mode for claim: {claim}")
        print(f"Using {'cross-encoder' if use_cross_encoder else 'bi-encoder'} for evidence ranking")
        
        # Pass ranking preference to WebEvidenceRetriever
        web_ret = WebEvidenceRetriever(top_k=100, search_engine="serper", use_cross_encoder=use_cross_encoder)
        web_ev = web_ret.retrieve(claim)

        if not web_ev:
            return {
                "claim": claim,
                "label": "Not Enough Info",
                "reason": "Web search produced no usable evidence.",
                "evidence": [],
                "mode": "web_only",
            }

        # Evidence is already ranked by WebEvidenceRetriever, so we can skip synthesise here
        # Or apply final synthesis if you want double-ranking
        syn_ev = web_ev  # Already synthesised in retrieve()
        nli_out = batch_nli(claim, [e["snippet"] for e in syn_ev])
        lbl, conf, annotated_ev = aggregate(syn_ev, nli_out)

        return {
            "claim": claim,
            "label": lbl,
            "confidence": conf,
            "evidence": annotated_ev,
            "mode": "web_only",
            "evidence_count": len(syn_ev),
            "ranking_method": "cross_encoder" if use_cross_encoder else "bi_encoder"
        }
    
    elif mode == "hybrid":
        print(f"Running HYBRID mode for claim: {claim}")
        
        # ---------- 1.  KG AGENT ---------------------------------------- #
        kg_ret = KGEvidenceRetriever()
        uris, paths = kg_ret.retrieve(claim)

        if paths:
            ranker = EvidenceRanker2(claim_text=claim)
            ranked = ranker.top_k(paths, k=3, use_bi_encoder=False)
            edges = _flatten_edges(ranked, k=3)

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
                "mode": "hybrid",
                "kg_success": True,
            }

        # ---------- 3.  FALLBACK → WEB / RAG agent -------------------- #
        print("KG agent returned 'Not Enough Info', falling back to web search...")
        
        web_ret = WebEvidenceRetriever(top_k=100, search_engine="serper")
        web_ev = web_ret.retrieve(claim)

        if not web_ev:
            return {
                "claim": claim,
                "label": "Not Enough Info",
                "reason": "Neither KG nor web search produced usable evidence.",
                "evidence": [],
                "entity_linking": {
                    "candidates": uris,
                },
                "mode": "hybrid",
                "kg_success": False,
            }

        syn_ev = synthesise(claim, web_ev, top_k=100)
        nli_out = batch_nli(claim, [e["snippet"] for e in syn_ev])
        lbl, conf, annotated_ev = aggregate(syn_ev, nli_out)

        return {
            "claim": claim,
            "label": lbl,
            "confidence": conf,
            "evidence": annotated_ev,
            "fallback_used": True,
            "entity_linking": {
                "candidates": uris,
            },
            "mode": "hybrid",
            "kg_success": False,
        }
    
    else:
        # Invalid mode
        return {
            "claim": claim,
            "label": "Not Enough Info",
            "reason": f"Invalid mode '{mode}'. Must be 'hybrid' or 'web_only'.",
            "evidence": [],
            "mode": mode,
            "error": True,
        }