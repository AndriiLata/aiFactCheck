from __future__ import annotations

import json
from typing import List, Tuple

from ...infrastructure.llm.llm_client import chat
from ...models import Edge

LABELS = ("Supported", "Refuted", "Not Enough Info")


class Verifier2:
    """
    Single GPT call that reasons over claim + evidence.
    """

    def classify(self, claim: str, evidence: List[Edge]) -> Tuple[str, str]:
        def _format_name(name: str) -> str:
            return name.split("/")[-1].replace("_", " ").strip()

        ev = "\n\n".join(
            f"[{i + 1:2}] {_format_name(e.subject)} → {_format_name(e.predicate)} → {_format_name(e.object)}"
            for i, e in enumerate(evidence)
        ) or "No evidence retrieved."

        system = {
            "role": "system",
            "content": (
                "You are a factual consistency expert. Decide whether the claim "
                "is Supported, Refuted, or Not Enough Info based ONLY on the "
                "evidence. Only return Not Enough Info if you absolutely have to. "
                "The original labels are only Supported and Refuted."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n\n"
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
