"""
Merge + score evidence lists.
Uses MiniLM cosine sim (bi-encoder) or cross-encoder for ranking against the *claim*.
"""
from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer, util
from ..ranking.evidence_ranker2 import EvidenceRanker2

# Keep the bi-encoder for backward compatibility and speed
_BI_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def synthesise(claim: str, evidence: List[Dict], top_k: int = 100, use_cross_encoder: bool = True) -> List[Dict]:
    """
    Rank evidence by relevance to claim using bi-encoder or cross-encoder.
    
    Args:
        claim: The claim to verify
        evidence: List of evidence dictionaries
        top_k: Number of top evidence items to return
        use_cross_encoder: If True, use cross-encoder; if False, use bi-encoder
    
    Returns:
        List of ranked evidence dictionaries
    """
    if not evidence:
        return []

    print(f"Synthesising {len(evidence)} evidence items using {'cross-encoder' if use_cross_encoder else 'bi-encoder'}")

    if use_cross_encoder:
        return _synthesise_cross_encoder(claim, evidence, top_k)
    else:
        return _synthesise_bi_encoder(claim, evidence, top_k)


def _synthesise_bi_encoder(claim: str, evidence: List[Dict], top_k: int) -> List[Dict]:
    """Rank using bi-encoder (faster, current method)."""
    claim_emb = _BI_MODEL.encode(claim, convert_to_tensor=True)
    snippets = [e["snippet"] for e in evidence]
    embeds = _BI_MODEL.encode(snippets, convert_to_tensor=True, batch_size=64)
    sims = util.pytorch_cos_sim(claim_emb, embeds)[0]

    for ev, sim in zip(evidence, sims):
        ev["similarity"] = float(sim)
        ev["ranking_method"] = "bi_encoder"

    # Rank by trust * similarity (existing logic)
    scored = sorted(
        evidence,
        key=lambda e: e["trust"] * e["similarity"],
        reverse=True,
    )

    for ev, score in zip(evidence, sim):
        ev["bi_encoder_score"] = float(sim)
        ev["ranking_method"] = "bi_encoder"
        trust = ev.get("trust", 0.0)
        # 80% cross‐encoder, 20% trust
        ev["combined"] = (ev["bi_encoder_score"] * 0.8 + trust * 0.2)

        # now sort by that composite_score
    scored = sorted(
        evidence,
        key=lambda e: e["combined"],
        reverse=True,
    )

    return scored[:top_k]


def _synthesise_cross_encoder(claim: str, evidence: List[Dict], top_k: int) -> List[Dict]:
    """Rank using cross-encoder from EvidenceRanker2 (more accurate)."""
    try:
        # Create claim-snippet pairs for cross-encoder
        pairs = [(claim, ev["snippet"]) for ev in evidence]
        
        # Use the existing EvidenceRanker2 cross-encoder
        cross_scores = EvidenceRanker2._cross_encoder.predict(pairs)
        
        # Add cross-encoder scores to evidence
        for ev, score in zip(evidence, cross_scores):
            ev["cross_encoder_score"] = float(score)
            ev["ranking_method"] = "cross_encoder"
            trust = ev.get("trust", 0.0)
            # 80% cross‐encoder, 20% trust
            ev["combined"] = (ev["cross_encoder_score"] * 0.8 + trust * 0.2)

        # now sort by that composite_score
        scored = sorted(
            evidence,
            key=lambda e: e["combined"],
            reverse=True,
        )
        
        print(f"Cross-encoder ranking complete. Top score: {scored[0]['cross_encoder_score']:.3f}")
        return scored[:top_k]
        
    except Exception as e:
        print(f"Cross-encoder ranking failed: {e}, falling back to bi-encoder")
        return _synthesise_bi_encoder(claim, evidence, top_k)
        
