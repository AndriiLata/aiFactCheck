"""
Single public function: verify_claim_crew(claim:str) -> dict
Glue:
  • async retrieval  (retrievers.py)
  • synthesise       (synthesiser.py)
  • batch NLI        (nli.py)
  • verdict          (verdict.py)

"""
import asyncio
from typing import Dict

from .retrievers import gather_evidence
from .synthesiser import synthesise
from .nli import batch_nli
from .verdict import _aggregate as aggregate


async def _run_async(claim: str) -> Dict:
    retrieved = await gather_evidence(claim)
    triple = retrieved.pop("triple", None)

    all_ev = retrieved["wikidata"] + retrieved["dbpedia"] + retrieved["web"]
    syn_ev = synthesise(claim, all_ev, top_k=100)

    nli_out = batch_nli(claim, [e["snippet"] for e in syn_ev])
    label, conf, annotated_evidence = aggregate(syn_ev, nli_out)

    return {
        "claim": claim,
        "triple": triple.__dict__ if triple else None,
        "label": label,
        "confidence": conf,
        "evidence": annotated_evidence,
    }


def verify_claim_crew(claim: str) -> Dict:
    """Sync wrapper – call from Flask or scripts."""
    return asyncio.run(_run_async(claim))
