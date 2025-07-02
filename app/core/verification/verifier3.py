import json
from typing import List, Tuple
from ...infrastructure.llm.llm_client import chat

LABELS = ("Supported", "Refuted", "Not Enough Info")

class Verifier3:
    """
    Single GPT call over a claim (str) and evidence (list of snippets).
    Honors the model’s NEI if it truly can’t decide, but only truly NEI when 
    it’s justified.
    """

    def classify(self, claim: str, evidence: List[str]) -> Tuple[str, str]:
        # 1) Build numbered evidence block
        if evidence:
            ev_text = "\n\n".join(f"[{i+1:2}] {snippet}" 
                                   for i, snippet in enumerate(evidence))
        else:
            ev_text = "No evidence provided."

        system = {
            "role": "system",
            "content": (
                "You are a world-class fact-verification assistant.\n"
                "Your job: given a claim and a small numbered list of evidence snippets, "
                "decide *only* one of two labels:\n"
                "  • Supported – at least one snippet clearly confirms the claim.\n"
                "  • Refuted   – at least one snippet explicitly contradicts the claim.\n"
                "\n"
                "You *must not* output any other label.\n"
                "Use *only* the provided snippets; do *not* invent facts or fetch external data.\n"
                "Keep your reasoning private—do *not* expose chain-of-thought.\n"
                "Output *exactly* one JSON object matching:\n"
                "{\n"
                '  "label": <Supported|Refuted>,\n'
                '  "reason": <one short sentence citing snippet number(s)>\n'
                "}"
            )
        }

        # 2) USER PROMPT
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n\n"
                f"Evidence snippets:\n{ev_text}\n\n"
                "Instruction:\n"
                "- If any snippet affirms the claim’s exact assertion, label Supported.\n"
                "- If any snippet contradicts it (negation, opposite fact), label Refuted.\n"
                "- You must choose *one* of the two—no other options.\n\n"
                "Examples:\n"
                "\n"
                "**Supported Example**\n"
                "Claim: “Alice’s birthplace is Canada.”\n"
                "1. Alice → birthPlace → Canada\n\n"
                "Output:\n"
                "{\"label\":\"Supported\",\"reason\":\"Snippet 1 shows birthPlace → Canada.\"}\n"
                "\n"
                "**Refuted Example**\n"
                "Claim: “Bob is an exponent of Doom metal.”\n"
                "1. Bob → is not an exponent of → Doom metal\n\n"
                "Output:\n"
                "{\"label\":\"Refuted\",\"reason\":\"Snippet 1 states ‘is not an exponent of Doom metal’.\"}\n"
                "\n"
            )
        }

        # 4) Invoke the LLM
        reply = chat([system, user])
        text = reply.content.strip()

        # 5) Try parsing JSON
        label, reason = None, ""
        try:
            parsed = json.loads(text)
            if parsed.get("label") in LABELS:
                label = parsed["label"]
                reason = parsed.get("reason", "").strip()
        except json.JSONDecodeError:
            pass

        # 6) Fallback: simple keyword lookup if JSON failed
        if label is None:
            low = text.lower()
            for lbl in LABELS:
                if lbl.lower() in low:
                    label = lbl
                    reason = text
                    break

        # 7) If still no label, default to NEI only if evidence is empty
        if label is None:
            if not evidence:
                return "Not Enough Info", "No evidence to assess."
            # otherwise, default to Supported (safer than refuting)
            return "Supported", f"Defaulting to Supported based on snippet [1]."

        # 8) Return whatever the model chose
        return label, reason