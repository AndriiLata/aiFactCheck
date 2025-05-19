"""
DBpedia Spotlight wrapper with Redis caching.
Call: EntityLinker().link("Angela Merkel")  ->  [("http://dbpedia.org/resource/Angela_Merkel", 0.97), ...]
"""
from __future__ import annotations

import json, re, time
from typing import List, Tuple

import requests

from .redis_client import rdb     # same shared Redis instance


class EntityLinker:
    _API_URL   = "https://api.dbpedia-spotlight.org/en/annotate"
    _TIMEOUT_S = 5

    def __init__(self, confidence: float = 0.35) -> None:
        self.confidence = confidence

    # ------------------------------------------------------------------ #
    def _spotlight(self, surface: str) -> List[Tuple[str, float]]:
        """Raw API call (no cache)."""
        params  = {"text": surface, "confidence": self.confidence}
        headers = {"Accept": "application/json"}
        r = requests.get(self._API_URL, params=params,
                         headers=headers, timeout=self._TIMEOUT_S)
        r.raise_for_status()
        data = r.json().get("Resources", [])
        # Keep only exact-surface matches, sort by similarity
        out: List[Tuple[str, float]] = []
        for ent in data:
            if re.fullmatch(re.escape(surface), ent["@surfaceForm"], flags=re.I):
                out.append((ent["@URI"], float(ent["@similarityScore"])))
        return sorted(out, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------ #
    def link(self, surface: str, *, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return ≤ top_k (uri, score) pairs; fall back to heuristic if none."""
        key = f"spotlight:{surface.lower()}"
        if cached := rdb.get(key):
            return json.loads(cached)

        try:
            candidates = self._spotlight(surface)[:top_k]
        except Exception:
            candidates = []

        # Heuristic fallback → URI-encode the surface directly
        if not candidates:
            uri = f"http://dbpedia.org/resource/{surface.replace(' ', '_')}"
            candidates = [(uri, 0.20)]            # very low confidence

        # Cache 24 h
        rdb.setex(key, 86_400, json.dumps(candidates))
        return candidates
