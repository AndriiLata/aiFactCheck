# app/services/entity_linker.py
from __future__ import annotations

import json
import re
import urllib.parse
from typing import List, Tuple, Optional # Added Optional

import requests

from .models import EntityCandidate # Changed import

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
        headers = {"User-Agent": "entity-linker/1.0 (wd-search; your-contact@example.com)"} # Added contact
        r = requests.get(
            self._WD_SEARCH_API, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        results = r.json().get("search", [])
        return [item["id"] for item in results if "id" in item]

    def _wikidata_to_dbpedia(self, qid: str) -> Optional[str]: # Return type changed
        """Return a DBpedia resource URI mapped to *qid*, if any."""
        query = (
            "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
            "SELECT ?dbp WHERE {\n"
            f"  ?dbp owl:sameAs <http://www.wikidata.org/entity/{qid}> .\n"
            "  FILTER(STRSTARTS(STR(?dbp), \"http://dbpedia.org/resource/\"))\n"
            "} LIMIT 1"
        )
        params = {"query": query, "format": "application/json"}
        headers = {"User-Agent": "entity-linker/1.0 (sparql-wd2dbp; your-contact@example.com)"} # Added contact
        r = requests.get(
            self._DBP_SPARQL, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        bindings = r.json().get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["dbp"]["value"]
        return None

    def _wikidata_lookup(self, surface: str, *, max_hits: int) -> List[EntityCandidate]: # Return type changed
        """Full WD → DBpedia pipeline. Returns at most *max_hits* EntityCandidates."""
        out: List[EntityCandidate] = []
        try:
            qids = self._wikidata_search(surface, max_hits=max_hits)
        except requests.exceptions.RequestException as e:
            print(f"[EntityLinker] Wikidata search failed for '{surface}': {e}")
            return out

        for rank, qid in enumerate(qids):
            wikidata_uri = f"http://www.wikidata.org/entity/{qid}"
            dbpedia_uri: Optional[str] = None
            try:
                dbpedia_uri = self._wikidata_to_dbpedia(qid)
            except requests.exceptions.RequestException as e:
                print(f"[EntityLinker] DBpedia mapping failed for QID {qid}: {e}")
            
            # Create candidate if at least one URI is found (Wikidata URI should always exist if qid is valid)
            score = 1.0 - rank / max(max_hits, 1)  # simple position‑based score
            out.append(EntityCandidate(
                surface_form=surface,
                dbpedia_uri=dbpedia_uri,
                wikidata_uri=wikidata_uri,
                score=score
            ))
            if len(out) >= max_hits:
                break
        return out


    # DBpedia Lookup fallback
    def _dbpedia_lookup(self, surface: str, *, max_hits: int) -> List[EntityCandidate]: # Return type changed
        params = {
            "query": surface,
            "maxResults": max_hits,
            "format": "JSON",
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "entity-linker/1.0 (dbp-lookup; your-contact@example.com)", # Added contact
        }
        r = requests.get(
            self._LOOKUP_API, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        docs = r.json().get("docs", [])
        out: List[EntityCandidate] = []
        for rank, ent in enumerate(docs):
            resource_list = ent.get("resource") or ent.get("uri") or ent.get("id") or []
            if not resource_list:
                continue
            uri = resource_list[0]
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
                score_val = 1.0 - rank / max(max_hits, 1) # Fallback score
            
            out.append(EntityCandidate(
                surface_form=surface,
                dbpedia_uri=uri,
                wikidata_uri=None, # DBpedia Lookup doesn't directly provide Wikidata URI
                score=float(score_val)
            ))
        return out


    # Spotlight fallback
    def _spotlight(self, surface: str, *, max_hits: int) -> List[EntityCandidate]: # Return type changed
        params = {"text": surface, "confidence": self.confidence}
        headers = {
            "Accept": "application/json",
            "User-Agent": "entity-linker/1.0 (spotlight; your-contact@example.com)", # Added contact
        }
        r = requests.get(
            self._SPOTLIGHT_API, params=params, headers=headers, timeout=self._TIMEOUT_S
        )
        r.raise_for_status()
        resources = r.json().get("Resources", [])
        
        candidates_spotlight: List[Tuple[str, float]] = []
        for ent in resources:
            # Ensure the surface form matches to avoid linking parts of a larger phrase
            # The API sometimes returns entities for sub-parts of the 'text' param.
            # A more robust check might be needed depending on Spotlight's behavior for your inputs.
            if ent.get("@surfaceForm", "").lower() == surface.lower():
                 candidates_spotlight.append((ent["@URI"], float(ent["@similarityScore"])))
        
        # Sort by score and take top N
        candidates_spotlight.sort(key=lambda x: x[1], reverse=True)
        
        out: List[EntityCandidate] = []
        for uri, score in candidates_spotlight[:max_hits]:
            out.append(EntityCandidate(
                surface_form=surface,
                dbpedia_uri=uri,
                wikidata_uri=None, # Spotlight provides DBpedia URIs
                score=score
            ))
        return out


    # Public facade
    def link(self, surface: str, *, top_k: int = 3) -> List[EntityCandidate]: # Return type changed
        """Return up to *top_k* `EntityCandidate` objects.

        Resolution order:
        1. Wikidata → DBpedia mapping (provides both Wikidata and potentially DBpedia URIs)
        2. DBpedia Lookup (provides DBpedia URIs)
        3. DBpedia Spotlight (provides DBpedia URIs)
        4. Heuristic URI (provides a DBpedia URI)
        """
        surface = surface.strip()
        if not surface:
            return []

        # key = f"link:{surface.lower()}:{top_k}" # Consider separate cache for EntityCandidate
        # if cached := rdb.get(key):
        #     # Need to handle deserialization into List[EntityCandidate]
        #     # For simplicity, caching is kept disabled as in original.
        #     pass


        candidates: List[EntityCandidate] = []

        # 1) Wikidata pipeline
        try:
            candidates = self._wikidata_lookup(surface, max_hits=top_k)
            # Filter out candidates that don't have at least one URI, though _wikidata_lookup should ensure wikidata_uri
            candidates = [c for c in candidates if c.dbpedia_uri or c.wikidata_uri]
            print(f"[EntityLinker] Wikidata Lookup for '{surface}': {len(candidates)} candidates found.")
        except Exception as e:
            print(f"[EntityLinker] Error in _wikidata_lookup for '{surface}': {e}")
            candidates = []

        # 2) DBpedia Lookup fallback
        if not candidates: # Or if candidates lack dbpedia_uri and it's preferred for some reason
            try:
                dbp_candidates = self._dbpedia_lookup(surface, max_hits=top_k)
                # Merge or replace: for now, simple fallback
                candidates.extend(c for c in dbp_candidates if c.dbpedia_uri) # Ensure dbpedia_uri exists
                print(f"[EntityLinker] DBpedia Lookup for '{surface}': {len(dbp_candidates)} candidates found.")
            except Exception as e:
                print(f"[EntityLinker] Error in _dbpedia_lookup for '{surface}': {e}")

        # 3) Spotlight fallback
        if not candidates: # Or if candidates lack dbpedia_uri
            try:
                spotlight_candidates = self._spotlight(surface, max_hits=top_k)
                candidates.extend(c for c in spotlight_candidates if c.dbpedia_uri)
                print(f"[EntityLinker] Spotlight for '{surface}': {len(spotlight_candidates)} candidates found.")
            except Exception as e:
                print(f"[EntityLinker] Error in _spotlight for '{surface}': {e}")

        # Deduplicate candidates based on URIs, keeping the one with the highest score
        # This is a bit more complex if merging strategies are involved.
        # For now, the sequential fallback mostly handles this.
        # A more robust deduplication might look at (dbpedia_uri, wikidata_uri) pair.

        # 4) Heuristic (only if still no candidates, and specifically for DBpedia)
        if not candidates or not any(c.dbpedia_uri for c in candidates):
            heuristic_dbp_uri = f"http://dbpedia.org/resource/{urllib.parse.quote(surface.replace(' ', '_'))}"
            # Check if this heuristic URI is already present from a previous step (unlikely if list is empty)
            is_present = False
            for cand in candidates:
                if cand.dbpedia_uri == heuristic_dbp_uri:
                    is_present = True
                    break
            if not is_present:
                 candidates.append(EntityCandidate(
                    surface_form=surface,
                    dbpedia_uri=heuristic_dbp_uri,
                    wikidata_uri=None,
                    score=0.20 # Low confidence score
                ))
                 print(f"[EntityLinker] Heuristic URI for '{surface}' added.")


        # Sort final candidates by score and return top_k
        # If merging from multiple sources, ensure scores are comparable or re-evaluate.
        # Current logic mostly relies on the first successful method's candidates.
        # A more sophisticated merging would collect all, then rank/filter.
        # For simplicity, we take what we have and ensure top_k.
        
        # Example simple sort and top_k if multiple strategies contributed without prior top_k limit each time
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        # rdb.setex(key, 86_400, json.dumps([c.__dict__ for c in candidates[:top_k]]))
        final_candidates = candidates[:top_k]
        print(f"[EntityLinker] Final candidates for '{surface}': {[(c.dbpedia_uri, c.wikidata_uri, c.score) for c in final_candidates]}")
        return final_candidates