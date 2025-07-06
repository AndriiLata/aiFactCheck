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
from ..ranking.evidence_ranker import EvidenceRanker
from ..verification.structured_verifier import StructuredVerifier
from .synthesiser import synthesise
from .nli import batch_nli
from .verdict import _aggregate as aggregate
from ..verification.snippet_verifier import SnippetVerifier
from ...models import Edge



def _flatten_edges(ranked_paths: List[Tuple[List[Edge], float]], k: int) -> List[Edge]:
    """Take the first *k* individual edges from the ranked paths list."""
    edges: List[Edge] = []
    for path, _ in ranked_paths:
        edges.extend(path)
        if len(edges) >= k:
            break
    return edges[:k]



def verify_claim_crew(claim: str, mode: str = "web_only", use_cross_encoder: bool = True, classifierDbpedia:str ="LLM", classifierBackup:str ="LLM") -> Dict:
    """
    Multi-agent reasoning wrapper with ranking method support.
    
    Parameters:
    - claim: The claim to verify
    - mode: "hybrid" (KG first, Web fallback) or "web_only" (Web only)
    
    Returns a JSON-serialisable dict ready for `Flask.jsonify`.
    """
    
    if mode == "web_only":
        print(f"Running WEB-ONLY mode for claim: {claim}")

        # Pass ranking preference to WebEvidenceRetriever
        web_ret = WebEvidenceRetriever(search_k= 100, top_k=5, search_engine="serper", use_cross_encoder=use_cross_encoder)
        web_ev = web_ret.retrieve(claim)

        if not web_ev:
            print("No web evidence found.")
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
        print(f"Synthesised evidence: {syn_ev}")

        if classifierBackup=="DEBERTA":
            nli_out = batch_nli(claim, [e["snippet"] for e in syn_ev])
            lbl, conf, annotated_ev = aggregate(syn_ev, nli_out, threshold=0.01)
            print("Label: ", lbl)

            return {
                "claim": claim,
                "label": lbl,
                "confidence": conf,
                "evidence": annotated_ev,
                "mode": "web_only, DEBERTA",
                "evidence_count": len(syn_ev),
                "ranking_method": "cross_encoder" if use_cross_encoder else "bi_encoder"
            }
        else:
            v=SnippetVerifier()
            LLM_ev=[e["snippet"] for e in syn_ev]
            LLM_ev=LLM_ev[0:min(3,len(LLM_ev))]
            lbl, annotated_ev=v.classify(claim, LLM_ev)
            print("Label: ", lbl)

            return {
                "claim": claim,
                "label": lbl,
                "evidence": syn_ev,
                "reason": annotated_ev,
                "mode": "web_only,LLM",
                "evidence_count": len(syn_ev),
                "ranking_method": "cross_encoder" if use_cross_encoder else "bi_encoder"
            }


    
    elif mode == "hybrid":
        print(f"Running HYBRID mode for claim: {claim}")

        # ---------- 1.  KG AGENT ---------------------------------------- #
        kg_ret = KGEvidenceRetriever()
        uris, paths = kg_ret.retrieve(claim)

        if paths:
            ranker = EvidenceRanker(claim_text=claim)
            ranked = ranker.top_k(paths, k=3, use_bi_encoder=False)
            edges = _flatten_edges(ranked, k=3)

            if classifierDbpedia=="DEBERTA":
                evidence = [edge for path, _ in ranked for edge in path]
                evidence = evidence[0:3]

                def _format_name(name: str) -> str:
                    return name.split("/")[-1].replace("_", " ").strip()

                ev_list = [
                    {
                        "snippet": f"{_format_name(e.subject)} → {_format_name(e.predicate)} → {_format_name(e.object)}",
                        "trust": 1.0,  # Standard-Vertrauenswert
                        "source": "knowledge_graph"  # Quelle der Information
                    }
                    for e in evidence
                ]



                nli_out = batch_nli(claim, [e["snippet"] for e in ev_list])
                lbl, conf, annotated_ev = aggregate(ev_list, nli_out, threshold=0.6)

                if lbl in ("Supported", "Refuted"):
                    print("Label: ", lbl)
                    return {
                        "claim": claim,
                        "label": lbl,
                        "reason": annotated_ev,
                        "evidence": [
                            [e.__dict__ for e in p] for p, _ in ranked
                        ],
                        "entity_linking": {
                            "candidates": uris,
                        },
                        "mode": "hybrid, "+classifierDbpedia+", "+classifierBackup,
                        "kg_success": True,
                    }

            else:
                verifier = StructuredVerifier()
                label, reason = verifier.classify(claim, edges)

                if label in ("Supported", "Refuted"):
                    print("Label: ", label)
                    return {
                        "claim": claim,
                        "label": label,
                        "reason": reason,
                        "evidence": [
                            [e.__dict__ for e in p] for p, _ in ranked
                        ],
                        "entity_linking": {
                            "candidates": uris,
                        },
                        "mode": "hybrid, "+classifierDbpedia+", "+classifierBackup,
                        "kg_success": True,
                    }


        # ---------- 3.  FALLBACK → WEB / RAG agent -------------------- #
        print("KG agent returned 'Not Enough Info', falling back to web search...")

        web_ret = WebEvidenceRetriever(search_k=100, top_k=5, search_engine="serper",
                                       use_cross_encoder=use_cross_encoder)
        web_ev = web_ret.retrieve(claim)

        if not web_ev:
            print("No web evidence found.")
            return {
                "claim": claim,
                "label": "Not Enough Info",
                "reason": "Neither KG nor web search produced usable evidence.",
                "evidence": [],
                "entity_linking": {
                    "candidates": uris,
                },
                "mode": "hybrid, "+classifierDbpedia+", "+classifierBackup,
                "kg_success": False,
            }

        # Evidence is already ranked by WebEvidenceRetriever, so we can skip synthesise here
        # Or apply final synthesis if you want double-ranking
        syn_ev = web_ev  # Already synthesised in retrieve()
        print(f"Synthesised evidence: {syn_ev}")

        if classifierBackup == "DEBERTA":
            nli_out = batch_nli(claim, [e["snippet"] for e in syn_ev])
            lbl, conf, annotated_ev = aggregate(syn_ev, nli_out, threshold=0.01)
            print("Label: ", lbl)

            return {
                "claim": claim,
                "label": lbl,
                "confidence": conf,
                "evidence": annotated_ev,
                "mode": "hybrid, "+classifierDbpedia+", "+classifierBackup,
                "evidence_count": len(syn_ev),
                "ranking_method": "cross_encoder" if use_cross_encoder else "bi_encoder",
                "entity_linking": {
                    "candidates": uris,
                },
                "kg_success": False,
            }


        else:
            v = SnippetVerifier()
            LLM_ev = [e["snippet"] for e in syn_ev]
            LLM_ev = LLM_ev[0:min(3, len(LLM_ev))]
            lbl, annotated_ev = v.classify(claim, LLM_ev)
            print("Label: ", lbl)

            return {
                "claim": claim,
                "label": lbl,
                "evidence": syn_ev,
                "reason": annotated_ev,
                "mode": "hybrid, "+classifierDbpedia+", "+classifierBackup,
                "evidence_count": len(syn_ev),
                "ranking_method": "cross_encoder" if use_cross_encoder else "bi_encoder",
                "entity_linking": {
                    "candidates": uris,
                },
                "kg_success": False,
            }
    return None
