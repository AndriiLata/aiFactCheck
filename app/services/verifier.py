"""
LLM-based generator + verifier
"""
from __future__ import annotations

import json
from typing import List, Tuple

from .openai_client import chat
from .models        import Triple, Edge

LABELS = ("Supported", "Refuted", "Not Enough Info")


class Verifier:
    """
    Single GPT call that: (1) reasons over claim & evidence,
    (2) outputs a JSON dict with {label, reason}.
    """

    def classify(
        self,
        claim: str,
        triple: Triple,
        evidence: List[Edge],
        support_score: float,
    ) -> Tuple[str, str]:
        # ----------------------------------------------------------------- #
        # Build a compact evidence string
        # ----------------------------------------------------------------- #
        ev_lines = []
        for edge in evidence:
            s = edge.subject.split("/")[-1]
            p = edge.predicate.split("/")[-1]
            o = edge.object.split("/")[-1]
            ev_lines.append(f"{s} —[{p}]→ {o}")
        evidence_txt = "\n".join(ev_lines) if ev_lines else "No evidence retrieved."

        # ----------------------------------------------------------------- #
        # Prompt
        # ----------------------------------------------------------------- #
        system = {
            "role": "system",
            "content": (
                "You are a factual consistency expert. Given a user claim, a semantic triple, "
                "and KB evidence paths, decide whether the claim is supported, refuted, or if "
                "there is not enough information."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n\n"
                f"Triple: (S='{triple.subject}', P='{triple.predicate}', O='{triple.object}')\n"
                f"Evidence paths (from DBpedia):\n{evidence_txt}\n\n"
                "Respond with a JSON object {{\"label\": ..., \"reason\": ...}} "
                "where label ∈ {Supported, Refuted, Not Enough Info}. "
                "The reason must fit in one short paragraph."
            ),
        }

        reply = chat([system, user])
        # ----------------------------------------------------------------- #
        # Robust JSON read
        # ----------------------------------------------------------------- #
        label, reason = "Not Enough Info", "The system could not determine a definitive answer."
        try:
            data = json.loads(reply.content.strip())
            if data.get("label") in LABELS:
                label  = data["label"]
                reason = data.get("reason", reason)
        except Exception:
            # fallback: try to parse plain text label
            txt = reply.content.strip()
            for lbl in LABELS:
                if lbl.lower() in txt.lower():
                    label = lbl
                    reason = txt
                    break

        return label, reason
