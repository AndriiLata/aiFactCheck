from __future__ import annotations

import json
import requests
from typing import List, Tuple
from ...config import Settings

from ...infrastructure.llm.llm_client import chat
from ...models import Triple, Edge

settings = Settings()

LABELS = ("Supported", "Refuted", "Not Enough Info")


class Verifier:
    """
    Single GPT call that reasons over claim + evidence.
    """

    def classify(
        self,
        claim: str,
        triple: Triple,
        evidence_paths: List[List[Edge]]
    ) -> Tuple[str, str]:
        ev = self.format_paths(evidence_paths)

        system = {
            "role": "system",
            "content": (
                "You are a factual verification expert. Base your judgment ONLY on the listed KG paths."
                "Do not use prior knowledge. If the evidence is weak or unrelated, answer 'Not Enough Info'."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n\n"
                f"Triple: (S='{triple.subject}', P='{triple.predicate}', O='{triple.object}')\n"
                f"Evidence paths (Top 5):\n{ev}\n\n"
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
    
    @staticmethod
    def format_paths(paths: List[List[Edge]]) -> str:
        out = []
        for i, path in enumerate(paths):
            formatted = " -> ".join(
                f"{e.subject.split('/')[-1]} —[{e.predicate.split('/')[-1]}]→ {e.object.split('/')[-1]}"
                for e in path
            )
            out.append(f"Path {i+1}: {formatted}")
        return "\n".join(out) or "No evidence retrieved."


