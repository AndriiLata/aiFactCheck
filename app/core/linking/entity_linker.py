from __future__ import annotations

import urllib.parse
from typing import List, Tuple, Optional

import requests

from ...models import EntityCandidate


class EntityLinker:
    """
    Resolve a surface form to Wikidata / DBpedia URIs with simple heuristics.
    """

    _WD_API = "https://www.wikidata.org/w/api.php"
    _DBP_SPARQL = "https://dbpedia.org/sparql"
    _DBP_LOOKUP = "https://lookup.dbpedia.org/api/search"
    _SPOTLIGHT = "https://api.dbpedia-spotlight.org/en/annotate"
    _TIMEOUT = 6  # s

    # ------------------------------------------------------------------ #
    # Wikidata helpers
    # ------------------------------------------------------------------ #
    def _wd_search(self, surface: str, *, max_hits: int) -> List[str]:
        params = dict(
            action="wbsearchentities",
            search=surface,
            language="en",
            limit=max_hits,
            format="json",
        )
        r = requests.get(self._WD_API, params=params, timeout=self._TIMEOUT)
        r.raise_for_status()
        return [item["id"] for item in r.json().get("search", [])]

    def _wd_to_dbp(self, qid: str) -> Optional[str]:
        query = (
            "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
            "SELECT ?dbp WHERE {\n"
            f"  ?dbp owl:sameAs <http://www.wikidata.org/entity/{qid}> .\n"
            "  FILTER(STRSTARTS(STR(?dbp), \"http://dbpedia.org/resource/\"))\n"
            "} LIMIT 1"
        )
        r = requests.get(
            self._DBP_SPARQL,
            params={"query": query, "format": "application/json"},
            timeout=self._TIMEOUT,
        )
        r.raise_for_status()
        bindings = r.json().get("results", {}).get("bindings", [])
        return bindings[0]["dbp"]["value"] if bindings else None

    def _wd_lookup(self, surface: str, *, max_hits: int) -> List[EntityCandidate]:
        out: List[EntityCandidate] = []
        for rank, qid in enumerate(self._wd_search(surface, max_hits=max_hits)):
            wikidata_uri = f"http://www.wikidata.org/entity/{qid}"
            dbpedia_uri = self._wd_to_dbp(qid)
            score = 1.0 - rank / max(max_hits, 1)
            out.append(
                EntityCandidate(
                    surface_form=surface,
                    dbpedia_uri=dbpedia_uri,
                    wikidata_uri=wikidata_uri,
                    score=score,
                )
            )
            if len(out) >= max_hits:
                break
        return out

    # ------------------------------------------------------------------ #
    # DBpedia Lookup
    # ------------------------------------------------------------------ #
    def _dbp_lookup(self, surface: str, *, max_hits: int) -> List[EntityCandidate]:
        r = requests.get(
            self._DBP_LOOKUP,
            params=dict(query=surface, maxResults=max_hits, format="JSON"),
            timeout=self._TIMEOUT,
        )
        r.raise_for_status()
        docs = r.json().get("docs", [])
        out: List[EntityCandidate] = []

        for rank, ent in enumerate(docs):
            # ---- extract URI ------------------------------------------------
            uri_list = ent.get("resource") or ent.get("uri") or ent.get("id") or []
            if not uri_list:
                continue
            uri = uri_list[0]

            # ---- extract / normalise score ---------------------------------
            raw_score = ent.get("score")
            if isinstance(raw_score, list) and raw_score:
                raw_score = raw_score[0]
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 1.0 - rank / max(max_hits, 1)  # fallback

            out.append(EntityCandidate(surface_form=surface, dbpedia_uri=uri, score=score))

        return out

    def _spotlight(self, surface: str, *, max_hits: int) -> List[EntityCandidate]:
        r = requests.get(
            self._SPOTLIGHT,
            params=dict(text=surface, confidence=0.35),
            headers={"Accept": "application/json"},
            timeout=self._TIMEOUT,
        )
        r.raise_for_status()
        resources = r.json().get("Resources", [])
        cands: List[Tuple[str, float]] = [
            (res["@URI"], float(res["@similarityScore"]))
            for res in resources
            if res.get("@surfaceForm", "").lower() == surface.lower()
        ]
        cands.sort(key=lambda x: x[1], reverse=True)
        return [
            EntityCandidate(surface_form=surface, dbpedia_uri=uri, score=score)
            for uri, score in cands[:max_hits]
        ]

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def link(self, surface: str, *, top_k: int = 3) -> List[EntityCandidate]:
        surface = surface.strip()
        if not surface:
            return []

        # 1. Wikidata pipeline
        cands = self._wd_lookup(surface, max_hits=top_k)

        # 2. DBpedia Lookup fallback
        if not cands:
            cands = self._dbp_lookup(surface, max_hits=top_k)

        # 3. Spotlight fallback
        if not cands:
            cands = self._spotlight(surface, max_hits=top_k)

        # 4. Heuristic – last resort
        if not cands or not any(c.dbpedia_uri for c in cands):
            uri = f"http://dbpedia.org/resource/{urllib.parse.quote(surface.replace(' ', '_'))}"
            cands.append(EntityCandidate(surface_form=surface, dbpedia_uri=uri, score=0.2))

        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:top_k]
