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

        """
        #Gives Less NEI
        system = {
            "role": "system",
            "content": (
                "You are a world‐class fact-verification assistant.\n"
                "Given a claim and a small list of numbered evidence paths, decide exactly one label:\n"
                "Supported, Refuted, or Not Enough Info.\n"
                "Use ONLY the provided evidence; do not invent facts or call the web.\n"
                "Return Not Enough Info only if no path could possibly support or refute.\n"
                "Keep your reasoning private—do NOT expose chain-of-thought.\n"
                "Output MUST be a single JSON object matching this schema:\n"
                '{"label": <Supported|Refuted|Not Enough Info>, "reason": <one short sentence referring to path number(s)>}'
            ),
        }

        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n\n"
                f"Evidence paths:\n{ev}\n\n"
                "Instruction:\n"
                "Decide if the claim is Supported, Refuted, or Not Enough Info based only on these paths.\n"
                "Respond *only* with JSON:\n"
                '{"label": <Supported|Refuted|Not Enough Info>, '
                '"reason": "<one short sentence referring to path number(s)>"}\n\n'
                "Example Supported:\n"
                "Claim: “Alice’s birthplace is Canada.”\n"
                "1. Alice → birthPlace → Canada\n\n"
                "Output:\n"
                "{\"label\":\"Supported\",\"reason\":\"Path 1 shows Alice’s birthplace is Canada.\"}\n\n"
                "Example Refuted:\n"
                "Claim: “Bob was born in France.”\n"
                "1. Bob → birthPlace → Germany\n\n"
                "Output:\n"
                "{\"label\":\"Refuted\",\"reason\":\"Path 1 shows Bob’s birthplace is Germany, not France.\"}"
            ),
        }
        
    
        """
        system = {
            "role": "system",
            "content": (
                "You are a world-class fact-verification assistant.\n"
                "Given a claim and a numbered list of evidence paths, choose exactly one label:\n"
                "  • Supported      – at least one path exactly affirms the claim’s assertion.\n"
                "  • Refuted        – at least one path *explicitly* contradicts it (e.g. predicate like “is not”).\n"
                "  • Not Enough Info – otherwise.\n"
                "\n"
                "Rules:\n"
                "1. If any path affirms the claim’s predicate+object, label Supported.\n"
                "2. Only label Refuted if a path uses negation or clear contradiction.\n"
                "3. Otherwise label Not Enough Info.\n"
                "4. Use *only* the provided paths; do NOT invent facts.\n"
                "5. Keep reasoning private—do NOT show chain-of-thought.\n"
                "6. Output *only* a single JSON object:\n"
                "{\n"
                "  \"label\": <Supported|Refuted|Not Enough Info>,\n"
                "  \"reason\": <one concise sentence citing path number(s)> \n"
                "}"
            ),
        }


        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n\n"
                f"Evidence paths:\n{ev}\n\n"
                "Instruction:\n"
                "- Label Supported if any path’s predicate and object exactly match the claim.\n"
                "- Label Refuted only if a path explicitly contradicts (uses “not”, “no”, etc.).\n"
                "- Otherwise label Not Enough Info.\n\n"
                "Examples:\n\n"
                "1) Supported\n"
                "Claim: “Alice’s birthplace is Canada.”\n"
                "1. Alice → birthPlace → Canada\n\n"
                "Output:\n"
                "{\"label\":\"Supported\",\"reason\":\"Path 1 exactly matches birthPlace→Canada.\"}\n\n"
                "2) Refuted\n"
                "Claim: “Bob is an exponent of Doom metal.”\n"
                "1. Bob → is not an exponent of → Doom_metal\n\n"
                "Output:\n"
                "{\"label\":\"Refuted\",\"reason\":\"Path 1 explicitly states ‘is not an exponent of Doom metal’.\"}\n\n"
                "3) Not Enough Info\n"
                "Claim: “Carol’s nationality is Spanish.”\n"
                "1. Carol → birthPlace → Barcelona\n\n"
                "Output:\n"
                "{\"label\":\"Not Enough Info\",\"reason\":\"Path 1 does not confirm nationality.\"}"
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
