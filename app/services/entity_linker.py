"""
1. **Search Wikidata** using the public `wbsearchentities` API.
2. For each Q‑ID returned, query the DBpedia SPARQL endpoint for a
   resource that is `owl:sameAs` the Wikidata entity.
3. Return the first *k* `(dbpedia_uri, score)` pairs.
4. If no mapping is found, fall back to **DBpedia Lookup** (Lucene),
   then to **DBpedia Spotlight**, and finally to a heuristic URI.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from typing import List, Tuple

import requests

# from .redis_client import rdb  # caching disabled for now


class EntityLinker:

    # Endpoints
    _WD_SEARCH_API = "https://www.wikidata.org/w/api.php"
    _DBP_SPARQL    = "https://dbpedia.org/sparql"
    _LOOKUP_API    = "https://lookup.dbpedia.org/api/search"  # official Lucene instance
    _SPOTLIGHT_API = "https://api.dbpedia-spotlight.org/en/annotate"

    _TIMEOUT_S = 6  # generous default

    def __init__(self, confidence: float = 0.35) -> None:
        self.confidence = confidence


    # Wikidata →DBpedia mapping
    def _wikidata_search(self, surface: str, *, max_hits: int) -> List[str]:
        """Return up to *max_hits* Wikidata Q‑IDs for *surface*."""
        params = {
            "action": "wbsearchentities",
            "search": surface,
            "language": "en",
            "limit": max_hits,
            "format": "json",
        }
        headers = {"User-Agent": "entity-linker/1.0 (wd-search)"}
        r = requests.get(
            self._WD_SEARCH_API, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        results = r.json().get("search", [])
        return [item["id"] for item in results if "id" in item]

    def _wikidata_to_dbpedia(self, qid: str) -> str | None:
        """Return a DBpedia resource URI mapped to *qid*, if any."""
        query = (
            "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
            "SELECT ?dbp WHERE {\n"
            f"  ?dbp owl:sameAs <http://www.wikidata.org/entity/{qid}> .\n"
            "  FILTER(STRSTARTS(STR(?dbp), \"http://dbpedia.org/resource/\"))\n"
            "} LIMIT 1"
        )
        params = {"query": query, "format": "application/json"}
        headers = {"User-Agent": "entity-linker/1.0 (sparql)"}
        r = requests.get(
            self._DBP_SPARQL, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        bindings = r.json().get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["dbp"]["value"]
        return None

    def _wikidata_lookup(self, surface: str, *, max_hits: int) -> List[Tuple[str, float]]:
        """Full WD → DBpedia pipeline. Returns at most *max_hits* URIs."""
        out: List[Tuple[str, float]] = []
        try:
            qids = self._wikidata_search(surface, max_hits=max_hits)
        except Exception:
            return out

        for rank, qid in enumerate(qids):
            try:
                uri = self._wikidata_to_dbpedia(qid)
            except Exception:
                uri = None
            if uri:
                score = 1.0 - rank / max(max_hits, 1)  # simple position‑based score
                out.append((uri, score))
            if len(out) >= max_hits:
                break
        return out


    # DBpedia Lookup fallback
    def _dbpedia_lookup(self, surface: str, *, max_hits: int) -> List[Tuple[str, float]]:
        params = {
            "query": surface,
            "maxResults": max_hits,
            "format": "JSON",
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "entity-linker/1.0 (dbp-lookup)",
        }
        r = requests.get(
            self._LOOKUP_API, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        docs = r.json().get("docs", [])
        out: List[Tuple[str, float]] = []
        for rank, ent in enumerate(docs):
            resource_list = ent.get("resource") or ent.get("uri") or ent.get("id") or []
            if not resource_list:
                continue
            uri = resource_list[0]
            # Score: prefer explicit score field, else refCount, else rank
            score_val = None
            if (s := ent.get("score")):
                try:
                    score_val = float(s[0]) if isinstance(s, list) else float(s)
                except (ValueError, TypeError):
                    score_val = None
            if score_val is None and (rc := ent.get("refCount")):
                try:
                    score_val = float(rc[0]) if isinstance(rc, list) else float(rc)
                except (ValueError, TypeError):
                    score_val = None
            if score_val is None:
                score_val = 1.0 - rank / max(max_hits, 1)
            out.append((uri, float(score_val)))
        return out


    # Spotlight fallback
    def _spotlight(self, surface: str, *, max_hits: int) -> List[Tuple[str, float]]:
        params = {"text": surface, "confidence": self.confidence}
        headers = {
            "Accept": "application/json",
            "User-Agent": "entity-linker/1.0 (spotlight)",
        }
        r = requests.get(
            self._SPOTLIGHT_API, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        resources = r.json().get("Resources", [])  # type: ignore[index]
        out: List[Tuple[str, float]] = []
        for ent in resources:
            if re.fullmatch(re.escape(surface), ent.get("@surfaceForm", ""), flags=re.I):
                out.append((ent["@URI"], float(ent["@similarityScore"])))
        return sorted(out, key=lambda x: x[1], reverse=True)[:max_hits]


    # Public facade
    def link(self, surface: str, *, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return up to *top_k* `(DBpedia URI, score)` pairs.

        Resolution order:
        1. Wikidata → DBpedia mapping
        2. DBpedia Lookup
        3. DBpedia Spotlight
        4. Heuristic URI
        """
        surface = surface.strip()
        if not surface:
            return []

        # key = f"link:{surface.lower()}:{top_k}"
        # if cached := rdb.get(key):
        #     return json.loads(cached)

        candidates: List[Tuple[str, float]] = []

        # 1) Wikidata pipeline
        try:
            candidates = self._wikidata_lookup(surface, max_hits=top_k)

            print(f"[EntityLinker] Candidates: {candidates}")

        except Exception:
            candidates = []

        # 2) DBpedia Lookup fallback
        if not candidates:
            try:
                candidates = self._dbpedia_lookup(surface, max_hits=top_k)
            except Exception:
                candidates = []

        # 3) Spotlight fallback
        if not candidates:
            try:
                candidates = self._spotlight(surface, max_hits=top_k)
            except Exception:
                candidates = []

        # 4) Heuristic
        if not candidates:
            uri = f"http://dbpedia.org/resource/{urllib.parse.quote(surface.replace(' ', '_'))}"
            candidates = [(uri, 0.20)]


        # rdb.setex(key, 86_400, json.dumps(candidates))

        return candidates[:top_k]
