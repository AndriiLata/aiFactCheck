from __future__ import annotations

import json
import re
import requests
import http.client
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

    def __init__(self, num_results: int = 100, search_engine: str = "serper") -> None:
        # Load all API keys
        self.serp_api_key = getattr(settings, "SERPAPI_KEY", None)
        self.brave_api_key = getattr(settings, "BRAVE_API_KEY", None) 
        self.serper_api_key = getattr(settings, "SERPER_API_KEY", None)
        
        # Set search engine and adjust num_results based on engine limits
        self.search_engine = search_engine.lower()
        
        if self.search_engine == "brave":
            self.num_results = min(num_results, 20)  # Brave limit
        elif self.search_engine == "serper":
            self.num_results = min(num_results, 100)  # Serper limit
        else:  # serpapi
            self.num_results = num_results


    def verify(self, claim: str, triple: Triple) -> Tuple[str, str, List[Dict]]:
        """Run the full verification pipeline: search → build context → classify."""
        # Check API key availability based on selected engine
        if self.search_engine == "brave" and not self.brave_api_key:
            return (
                "Not Enough Info",
                "Brave search is disabled; no Brave API key provided.",
                [],
            )
        elif self.search_engine == "serper" and not self.serper_api_key:
            return (
                "Not Enough Info", 
                "Serper search is disabled; no Serper API key provided.",
                [],
            )
        elif self.search_engine == "serpapi" and not self.serp_api_key:
            return (
                "Not Enough Info",
                "SerpAPI search is disabled; no SerpAPI key provided.",
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
        """Route to the appropriate search engine."""
        print(f"Searching web for: {query} using {self.search_engine}")
        
        if self.search_engine == "brave":
            return self._search_brave(query)
        elif self.search_engine == "serper":
            return self._search_serper(query)
        else:  # Default to serpapi
            return self._search_serpapi(query)

    def _search_serpapi(self, query: str) -> List[Dict]:
        """Query Google via SerpAPI and return the organic results as a list."""
        params = {
            "q": query,
            "api_key": self.serp_api_key,
            "engine": "google",
            "num": self.num_results,
        }
        try:
            resp = requests.get("https://serpapi.com/search", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic_results", [])
            print(f"SerpAPI search successful. Found {len(results)} results.")
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "link": r.get("link", ""),
                }
                for r in results
                if r.get("snippet")  # Only include results with snippets
            ]
        except requests.exceptions.RequestException as e:
            print(f"Error during SerpAPI web search: {e}")
            return []

    def _search_brave(self, query: str) -> List[Dict]:
        """Search using Brave API."""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }
        params = {
            "q": query,
            "count": self.num_results,
            "safesearch": "moderate",
            "freshness": "all"
        }
        
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            print(f"Brave API response status: {resp.status_code}")
            
            if resp.status_code == 422:
                print(f"Brave API 422 error. Trying with simplified parameters.")
                params = {"q": query, "count": min(self.num_results, 10)}
                resp = requests.get(url, headers=headers, params=params, timeout=20)
            
            resp.raise_for_status()
            data = resp.json()
            
            if "error" in data:
                print(f"Brave API error: {data['error']}")
                return []
            
            results = data.get("web", {}).get("results", [])
            print(f"Brave search successful. Found {len(results)} results.")
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("description", ""),
                    "link": r.get("url", ""),
                }
                for r in results
                if r.get("description")  # Only include results with descriptions
            ]
        except requests.exceptions.RequestException as e:
            print(f"Error during Brave web search: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return []

    def _search_serper(self, query: str) -> List[Dict]:
        """Search using Serper.dev API."""
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            
            payload = json.dumps({
                "q": query,
                "num": self.num_results,
                "gl": "us",  # Geographic location
                "hl": "en"   # Language
            })
            
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            conn.request("POST", "/search", payload, headers)
            response = conn.getresponse()
            
            if response.status != 200:
                print(f"Serper API error: {response.status} - {response.reason}")
                return []
            
            data = response.read()
            result = json.loads(data.decode("utf-8"))
            
            # Extract organic results
            organic_results = result.get("organic", [])
            print(f"Serper search successful. Found {len(organic_results)} results.")
            
            # Convert to our expected format
            formatted_results = []
            for item in organic_results:
                snippet = item.get("snippet", "")
                if snippet:  # Only include results with snippets
                    formatted_results.append({
                        "title": item.get("title", ""),
                        "snippet": snippet,
                        "link": item.get("link", ""),
                        "position": item.get("position", 0)
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during Serper search: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()

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