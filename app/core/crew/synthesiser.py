"""
Merge + score evidence lists.
Uses MiniLM cosine sim to the *claim* to keep top-k.
"""
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def synthesise(claim: str, evidence: List[Dict], top_k: int = 100) -> List[Dict]:
    if not evidence:
        return []

    claim_emb = _MODEL.encode(claim, convert_to_tensor=True)
    snippets = [e["snippet"] for e in evidence]
    embeds = _MODEL.encode(snippets, convert_to_tensor=True, batch_size=64)
    sims = util.pytorch_cos_sim(claim_emb, embeds)[0]

    for ev, sim in zip(evidence, sims):
        ev["similarity"] = float(sim)

    # rank by trust * similarity
    scored = sorted(
        evidence,
        key=lambda e: e["trust"] * e["similarity"],
        reverse=True,
    )
    return scored[: top_k]
