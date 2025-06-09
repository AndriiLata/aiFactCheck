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
    
    def llm_fallback_classify(self, claim: str, triple: Triple) -> tuple[str, str, list]:
        """
        RAG fallback: Search the web for the triple, use findings as context for LLM verification.
        Returns (label, reason, evidence_list)
        """
        serp_key = settings.SERPAPI_KEY

        # 1. Search the web for the triple (subject, predicate, object)
        query = f'"{triple.subject}" "{triple.predicate}" "{triple.object}"'
        search_results = []
        if serp_key:
            params = {
                "q": query,
                "api_key": serp_key,
                "engine": "google",
                "num": 5,
            }
            resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
            if resp.ok:
                data = resp.json()
                search_results = [
                    {
                        "title": r.get("title"),
                        "snippet": r.get("snippet"),
                        "link": r.get("link"),
                    }
                    for r in data.get("organic_results", [])
                ]
        else:
            # If no API key, skip search
            search_results = []

        # 2. Build context for LLM
        context = ""
        evidence_list = []
        if search_results:
            for idx, r in enumerate(search_results):
                context += f"[{idx+1}] {r['title']}: {r['snippet']} (Source: {r['link']})\n"
                evidence_list.append({"title": r["title"], "snippet": r["snippet"], "link": r["link"]})

        # 3. Prompt LLM with claim, triple, and context
        system = {
            "role": "system",
            "content": (
                "You are a fact-checking assistant. Given a claim, its semantic triple, and web search findings, "
                "decide if the claim is Supported, Refuted, or Not Enough Info. Use the findings as evidence. "
                "If you cannot find enough information, respond with Not Enough Info. "
                "Respond with a JSON object: {\"label\": ..., \"reason\": ...}."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n"
                f"Triple: (S='{triple.subject}', P='{triple.predicate}', O='{triple.object}')\n\n"
                f"Web findings:\n{context}\n"
                "Based on the above, is the claim Supported, Refuted, or Not Enough Info? "
                "Explain your reasoning in one paragraph."
            ),
        }
        reply = chat([system, user])
        content = reply.content.strip()
        try:
            data = json.loads(content)
            label = data.get("label", "Not Enough Info")
            reason = data.get("reason", "")
        except Exception:
            label = "Not Enough Info"
            reason = "LLM could not provide a structured answer."

        return label, reason, evidence_list
