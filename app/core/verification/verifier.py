from __future__ import annotations

import json
from typing import List, Tuple

from ...infrastructure.llm.llm_client import chat
from ...models import Triple, Edge

LABELS = ("Supported", "Refuted", "Not Enough Info")


class Verifier:
    """
    Single GPT call that reasons over claim + evidence.
    """

    def classify(
        self,
        claim: str,
        triple: Triple,
        evidence: List[Edge],
        support_score: float,
    ) -> Tuple[str, str]:
        ev = "\n".join(
            f"{e.subject.split('/')[-1]} —[{e.predicate.split('/')[-1]}]→ {e.object.split('/')[-1]}"
            for e in evidence
        ) or "No evidence retrieved."

        system = {
            "role": "system",
            "content": (
                "You are a factual consistency expert. Decide whether the claim "
                "is Supported, Refuted, or Not Enough Info based ONLY on the "
                "evidence."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n\n"
                f"Triple: (S='{triple.subject}', P='{triple.predicate}', O='{triple.object}')\n"
                f"Evidence paths:\n{ev}\n\n"
                "Respond with a JSON object {\"label\": ..., \"reason\": ...} "
                "where label ∈ {Supported, Refuted, Not Enough Info}.  "
                "Keep the reason to one short paragraph."
            ),
        }

        reply = chat([system, user])

        try:
            data = json.loads(reply.content.strip())
            if data.get("label") in LABELS:
                return data["label"], data.get("reason", "")
        except Exception:
            pass

        # fallback – try to detect label in plain text
        txt = reply.content.strip()
        for lbl in LABELS:
            if lbl.lower() in txt.lower():
                return lbl, txt
        return "Not Enough Info", "The LLM could not determine a definitive answer."
