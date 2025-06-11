from __future__ import annotations

import json
import re
import requests
from typing import List, Tuple, Dict

from ...config import Settings
from ...infrastructure.llm.llm_client import chat
from ...models import Triple

# Initialize settings to fetch the API keys
settings = Settings()


class WebVerifier:
    """
    Verifies a claim by searching the web for evidence using SerpAPI.
    The pipeline relies on the organic‐result *snippets* returned by SerpAPI. This saves both latency and
    scraping costs while still giving the LLM enough context.
    """

    def __init__(self, num_results: int = 100):
        self.serp_api_key = settings.SEARCHAPI_KEY
        self.num_results = num_results


    def verify(self, claim: str, triple: Triple) -> Tuple[str, str, List[Dict]]:
        """Run the full verification pipeline: search → build context → classify."""
        if not self.serp_api_key:
            return (
                "Not Enough Info",
                "Web search is disabled; no SerpAPI key provided.",
                [],
            )

        # 1. Search for evidence
        query = f'"{triple.subject}" "{triple.predicate}" "{triple.object}"'
        search_results = self._search(query)
        if not search_results:
            return "Not Enough Info", "No relevant search results found on the web.", []

        # 2. Build the LLM context string from snippets only
        context, evidence_list = self._build_context(search_results)

        # 3. Prompt LLM for final verification
        label, reason = self._classify_with_llm(claim, triple, context)
        return label, reason, evidence_list


    def _search(self, query: str) -> List[Dict]:
        """Query Google via SerpAPI and return the organic results as a list."""
        print(f"Searching web for: {query}")
        params = {
            "q": query,
            "api_key": self.serp_api_key,
            "engine": "google",
            "num": self.num_results,
        }
        try:
            resp = requests.get("https://www.searchapi.io/api/v1/search", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic_results", [])
            print(f"Search successful. Found {len(results)} results.")
            return [
                {
                    "title": r.get("title"),
                    "snippet": r.get("snippet"),
                    "link": r.get("link"),
                }
                for r in results
            ]
        except requests.exceptions.RequestException as e:
            print(f"Error during web search: {e}")
            return []

    @staticmethod
    def _build_context(search_results: List[Dict]) -> Tuple[str, List[Dict]]:
        """Turn SerpAPI snippets into the context string expected by the LLM."""
        context_lines = []
        evidence_list = []
        for idx, r in enumerate(search_results):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")

            context_lines.append(f"[{idx + 1}] {title}: {snippet} (Source: {link})")
            evidence_list.append({"title": title, "snippet": snippet, "link": link})

        return "\n\n".join(context_lines), evidence_list

    @staticmethod
    def _extract_json(content: str) -> Tuple[str, str]:
        """Extract and parse a JSON object from an LLM response that may be wrapped in code fences."""
        # First try a direct parse (fast path)
        try:
            data = json.loads(content)
            return data.get("label", "Not Enough Info"), data.get("reason", "")
        except json.JSONDecodeError:
            pass

        # If that fails, attempt to strip markdown code fences
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Fallback: grab first {...} block
            brace_match = re.search(r"\{.*\}", content, re.S)
            if not brace_match:
                return (
                    "Not Enough Info",
                    f"The LLM returned a non‑JSON response: {content}",
                )
            json_str = brace_match.group(0)

        try:
            data = json.loads(json_str)
            return data.get("label", "Not Enough Info"), data.get("reason", "")
        except json.JSONDecodeError:
            return (
                "Not Enough Info",
                f"The LLM returned a malformed JSON response: {content}",
            )

    def _classify_with_llm(self, claim: str, triple: Triple, context: str) -> Tuple[str, str]:
        """Prompt the LLM and return (label, reason)."""
        system = {
            "role": "system",
            "content": (
                "You are a fact‑checking assistant. Given a claim and web search findings, "
                "decide if the claim is Supported, Refuted, or Not Enough Info. "
                "Use ONLY the findings as evidence. "
                "Respond with a JSON object: {\"label\": ..., \"reason\": ...}."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Claim: {claim}\n"
                f"Triple: (S='{triple.subject}', P='{triple.predicate}', O='{triple.object}')\n\n"
                f"Web findings:\n{context}"
            ),
        }

        reply = chat([system, user])
        content = reply.content.strip()
        return self._extract_json(content)
