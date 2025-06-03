"""
LLM-based generator + verifier
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

from .openai_client import chat
from .azure_client import chat_a
from .models        import Triple, Edge

from ..config      import Settings

settings = Settings()

LABELS = ("Supported", "Refuted", "Not Enough Info")

def get_chat_func():
    return chat_a if settings.PROVIDER_IN_USE.lower() == "azure" else chat

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
                "there is not enough information. Do not use any other information than what is given in the evidence."
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

        chat_func = get_chat_func()
        reply = chat_func([system, user])
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
    
    def llm_fallback_classify(self, claim: str, triple: Triple) -> tuple[str, str]:
        chat_func = get_chat_func()
        system = {
            "role": "system",
            "content": (
                "You are an advanced fact-checking assistant with access to a large knowledge base. "
                "When given a claim and its semantic triple, use your internal knowledge and logical reasoning "
                "to determine if the claim is Supported, Refuted, or Not Enough Info. "
                "Apply retrieval-augmented generation (RAG) techniques: recall relevant facts, reason step by step, "
                "and make a best-effort judgment even if evidence is partial or indirect. "
                "If you cannot find enough information after careful consideration, only then respond with Not Enough Info. "
                "Always explain your reasoning in a concise paragraph."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n"
                f"Triple: (S='{triple.subject}', P='{triple.predicate}', O='{triple.object}')\n\n"
                "Respond with a JSON object {\"label\": ..., \"reason\": ...} "
                "where label ∈ {Supported, Refuted, Not Enough Info}. "
                "The reason must fit in one short paragraph and include your logical steps."
            ),
        }
        reply = chat_func([system, user])
        content = reply.content.strip()

        # Try to extract JSON from code block if present
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
        try:
            data = json.loads(content.replace("'", '"'))
            return data.get("label", "Not Enough Info"), data.get("reason", "")
        except Exception:
            # Try to extract label from plain text
            for lbl in ("Supported", "Refuted", "Not Enough Info"):
                if lbl.lower() in content.lower():
                    return lbl, content
            return "Not Enough Info", "LLM could not provide a structured answer."