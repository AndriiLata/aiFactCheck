"""
End-to-end verification pipeline that *only* uses web evidence gathered via
the Claim-Paraphraser.

Public call-signature matches the old version:

    verify_claim_paraphrased(claim: str) -> dict
"""
from __future__ import annotations

import asyncio
from typing import Dict

from .retrievers_web_only import gather_evidence_web_only
from .synthesiser import synthesise
from .nli import batch_nli
from .verdict import _aggregate as aggregate


async def _run_async(claim: str) -> Dict:
    retrieved = await gather_evidence_web_only(claim)
    web_ev = retrieved["web"]

    syn_ev = synthesise(claim, web_ev, top_k=100)

    nli_out = batch_nli(claim, [e["snippet"] for e in syn_ev])
    label, conf, annotated_evidence = aggregate(syn_ev, nli_out)

    return {
        "claim": claim,
        "label": label,
        "confidence": conf,
        "evidence": annotated_evidence,
    }


# ------------------------------------------------------------------ #
# Public helper
# ------------------------------------------------------------------ #
def verify_claim_paraphrased(claim: str) -> Dict:
    """Sync wrapper – can be called from Flask or scripts."""
    return asyncio.run(_run_async(claim))
